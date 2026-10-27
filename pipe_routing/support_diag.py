"""서포트 부재 막힘 진단 (S1-6 판정 후속, 결정 전 분석) — 출력 JSON 재분석, 새 라우팅 없음.

사용: python -m pipe_routing.support_diag <output.json>... -o diag.json

support 위반 배관 P 마다:
  - 막는 배관 Q = P 근처(영향 범위 안) 놓인 배관 중 **그 하나만 빼면** P 의 support 판정이 통과하는 배관.
    하나로 안 풀리면 "복수" — 빼면 서포트 불가 구간이 줄어드는 배관들을 적는다.
  - 순서: 최종 배치 시점(순차 배치 + rip-up 재배치 반영)으로 (가) Q 가 P 보다 나중 / (나) Q 가 먼저.
  - 막힌 부재: 다른 배관을 모두 뺀 단독 배치에서 그 서포트 불가 구간에 놓이는 서포트의 형식·면.
  - 거리: P·Q 중심선 최소거리와 표면 간격(거리 − r_P − r_Q).
"""
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from .scenario import load
from .support_std import load_standard
from .verifier.core import MODULES, Context, PipeRoute, centerlines_of
from .verifier.geom import segment_segment_distance
from .verifier.session import _reach, _segs


def scenario_path(stem: str) -> str:
    if stem.startswith("dense"):
        return f"scenarios/dense/{stem}.json"
    if stem.startswith("manual"):
        return f"scenarios/manual/{stem}.json"
    if stem.startswith("stair"):
        return f"scenarios/stair/{stem}.json"
    if stem.startswith("pair"):
        return f"scenarios/pair/{stem}.json"
    return f"scenarios/procedural/{stem}.json"


def placement_time(order: list, events: list, routed: set) -> dict:
    """최종 배치 시점. 순차 배치 후 rip-up(되돌리지 않은 것)은 실패 배관 → 걷어낸 배관 순으로 끝에 다시 놓는다."""
    seq = list(order)
    for e in events:
        m = re.match(r"rip-up \d+: (\S+) 위해 \[(.*?)\] 걷어냄 → 성공 \d+→\d+$", e)
        if not m:
            continue
        f = m.group(1)
        removed = [s.strip().strip("'") for s in m.group(2).split(",") if s.strip()]
        for pid in [f] + removed:
            seq.remove(pid)
            seq.append(pid)
    return {pid: i for i, pid in enumerate(p for p in seq if p in routed)}


def _support_of(sc, pipes, routes, cls, std, pid, exclude=()):
    rmap = {q: routes[q] for q in routes if q not in exclude}
    ctx = Context(sc, pipes, rmap, {q: cls[q] for q in rmap}, std=std)
    res = MODULES["support"](ctx, routes[pid])
    return res, ctx.cache[("supports", pid)]


def _arclen(points, q):
    """중심선 점열 위 q 의 호 길이 (가장 가까운 조각에 투영)."""
    best, s0, out = math.inf, 0.0, 0.0
    for a, b in zip(points[:-1], points[1:]):
        d = b - a
        L = float(np.linalg.norm(d))
        t = 0.0 if L == 0 else float(np.clip(np.dot(q - a, d) / (L * L), 0, 1))
        dist = float(np.linalg.norm(a + t * d - q))
        if dist < best:
            best, out = dist, s0 + t * L
        s0 += L
    return out


def _dir_kind(u) -> str:
    u = np.asarray(u, float)
    if abs(u[2]) > 1e-6:
        return "vertical" if abs(abs(u[2]) - 1) < 1e-6 else "inclined45"
    return "axis" if abs(np.abs(u).max() - 1) < 1e-6 else "horizontal45"


def relation(P, Q, clP, clQ, lo, hi) -> dict:
    """R7 막힘 유형 (S1-6c): P 의 서포트 불가 구간 [lo, hi] 중점에서 P 방향과, 막는 배관 Q 의 가장 가까운 직관 방향.
    p_vertical_45 = P 구간이 수직·45° (D89 대상 아님) / q_vertical_45 = Q 가 수직·45° / crossing = 둘 다 축 정렬
    수평이고 축이 다름 / parallel_* = 같은 축 수평 평행, 지지면 쪽 표면 높이 차(바닥·천장 중 작은 쪽) ≤ 50 · 50~300 · > 300."""
    pts = clP.points
    seg = pts[1:] - pts[:-1]
    lens = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0], np.cumsum(lens)])
    s = (lo + hi) / 2
    i = int(min(np.searchsorted(cum, s, side="right") - 1, len(lens) - 1))
    uP = seg[i] / lens[i]
    p = pts[i] + (s - cum[i]) * uP
    A, B, _, _ = clQ.arrays(("straight",))
    d = B - A
    L = np.linalg.norm(d, axis=1)
    t = np.clip(((p - A) * d).sum(1) / np.maximum(L * L, 1e-12), 0, 1)
    k = int(np.argmin(np.linalg.norm(A + t[:, None] * d - p, axis=1)))
    q = A[k] + t[k] * d[k]
    uQ = d[k] / L[k]
    kp, kq = _dir_kind(uP), _dir_kind(uQ)
    out = {"p_dir": kp, "q_dir": kq, "dz_mm": round(float(q[2] - p[2]), 1)}
    if kp != "axis":
        out["kind"] = "p_vertical_45"
    elif kq != "axis":
        out["kind"] = "q_vertical_45"
    elif int(np.argmax(np.abs(uP))) != int(np.argmax(np.abs(uQ))):
        out["kind"] = "crossing"
    else:
        sd = min(abs((p[2] - P.od / 2) - (q[2] - Q.od / 2)), abs((p[2] + P.od / 2) - (q[2] + Q.od / 2)))
        out["surface_diff_mm"] = round(float(sd), 1)
        out["kind"] = "parallel_le50" if sd <= 50 else ("parallel_50_300" if sd <= 300 else "parallel_gt300")
    return out


def diagnose(out_path: str, std) -> list:
    out = json.loads(Path(out_path).read_text())
    stem = Path(out_path).name[: -len("_output.json")]
    if not any(r["waypoints"] and r["layer0"].get("support") is False for r in out["routes"]):
        return []
    return diagnose_out(out, load(scenario_path(stem)), std, stem)


def diagnose_out(out: dict, sc, std, stem: str = "") -> list:
    """출력 dict 하나의 부재 막힘 진단 (pipeline global.member_blockage·R7 목록 공용)."""
    from . import verifier  # noqa: F401  (모듈 등록)
    stem = stem or out.get("scenario") or ""
    bad = [r for r in out["routes"] if r["waypoints"] and r["layer0"].get("support") is False]
    if not bad:
        return []
    pipes = {p.id: p for p in sc.pipes}
    routes = {r["pipe_id"]: PipeRoute(r["pipe_id"], r["waypoints"]) for r in out["routes"] if r["waypoints"]}
    cls = {pid: centerlines_of(pipes[pid], r) for pid, r in routes.items()}
    segs = {pid: _segs(c) for pid, c in cls.items()}
    g = out["global"]
    tpos = ({pid: i for i, pid in enumerate(p for p in g["placement_order"] if p in routes)}
            if g.get("placement_order") else placement_time(g["order"], g.get("planner_events") or [], set(routes)))
    reach = _reach(std)
    rows = []
    for r in bad:
        P = r["pipe_id"]
        A0, A1, AS = segs[P]
        near, dist = [], {}
        for Q in routes:
            if Q == P:
                continue
            B0, B1, BS = segs[Q]
            d, _, _ = segment_segment_distance(A0[:, None], A1[:, None], B0[None], B1[None])
            dmin = float((d - AS[:, None] - BS[None]).min())
            dist[Q] = dmin
            if dmin <= pipes[P].radius + pipes[Q].radius + reach:
                near.append(Q)
        base_v, _ = _support_of(sc, pipes, routes, cls, std, P)
        n0 = len(base_v)
        single, reducing = [], []
        for Q in sorted(near, key=lambda q: dist[q]):
            v, _ = _support_of(sc, pipes, routes, cls, std, P, exclude=(Q,))
            if not v:
                single.append(Q)
            elif len(v) < n0:
                reducing.append(Q)
        solo_v, solo_sup = _support_of(sc, pipes, routes, cls, std, P, exclude=tuple(q for q in routes if q != P))
        pts = cls[P][0].points
        blocked = []
        for v in base_v:
            m = re.match(r"([\d.]+)~([\d.]+) m", v.message)
            lo, hi = float(m.group(1)) * 1000, float(m.group(2)) * 1000
            for s in solo_sup:
                sa = _arclen(pts, np.array(s["pos"], dtype=float))
                if lo - 1 <= sa <= hi + 1:
                    blocked.append({"form": s["form"], "face": s["face"], "variant": s["variant"], "L": s["L"],
                                    "gap_m": [lo / 1000, hi / 1000]})
        blockers = single or reducing
        rels = []
        for v in base_v:
            m = re.match(r"([\d.]+)~([\d.]+) m", v.message)
            lo, hi = float(m.group(1)) * 1000, float(m.group(2)) * 1000
            if blockers:
                Q = min(blockers, key=lambda q: dist[q])
                rels.append(dict(relation(pipes[P], pipes[Q], cls[P][0], cls[Q][0], lo, hi), blocker=Q))
        rel = sorted({"after" if tpos[Q] > tpos[P] else "before" for Q in blockers})
        rows.append({
            "scenario": stem, "pipe": P, "size": pipes[P].nominal_size, "type": pipes[P].type_id,
            "gaps": n0, "solo_pass": not solo_v,
            "blockers": blockers, "blocker_kind": "single" if single else ("multiple" if reducing else "none"),
            "order": rel,   # after = (가) 막는 배관이 나중, before = (나) 먼저
            "t_pipe": tpos[P], "t_blockers": {Q: tpos[Q] for Q in blockers},
            "blocker_sizes": {Q: pipes[Q].nominal_size for Q in blockers},
            "center_clearance_mm": {Q: round(dist[Q], 1) for Q in blockers},
            "surface_gap_mm": {Q: round(dist[Q] - pipes[P].radius - pipes[Q].radius, 1) for Q in blockers},
            "blocked_supports": blocked,
            "relations": rels,   # R7 막힘 유형 (가장 가까운 막는 배관 기준, 불가 구간마다)
            "relation": (max(sorted({x["kind"] for x in rels}), key=[x["kind"] for x in rels].count)
                         if rels else "none"),
        })
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("outputs", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)
    std = load_standard()
    rows = []
    for f in a.outputs:
        rs = diagnose(f, std)
        for x in rs:
            print(f"{x['scenario']} {x['pipe']} {x['size']} {x['blocker_kind']} {x['blockers']} {x['order']} "
                  f"{[(b['form'], b['face']) for b in x['blocked_supports']]}", flush=True)
        rows += rs
    Path(a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
