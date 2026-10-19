"""D 묶음 연구 (S1-6 후속 3, 2026-10-18) — 출력 JSON 재분석, 새 라우팅 없음.

사용: python -m pipe_routing.support_group_study <output.json>... [--diag out/diag/diag.json] -o study.json

(i)   D82 ③ ±250 → ±500 의 효과: 같은 경로를 두 창으로 검증해 묶음 수·서포트 무게·통과 수·J_total 비교.
(ii)  묶음 정렬(D88 설계안) 상한: 표면 높이 차 허용을 300 mm 로 넓혀 묶음을 세운다 — "작은 배관을 기준 배관 높이로
      옮겼다면" 의 가상 상한 (경로·높이는 실제로 바꾸지 않으므로 상한일 뿐, 재라우팅 실패·간섭은 반영 안 함).
(iii) (c) "막히면 함께 받친다"(D89 설계안): 부재 막힘 support 위반(support_diag) 의 불가 구간 안에서, 막는 배관 Q 를
      함께 얹는 D(P, Q) 가 D82 조건(표면 차 ≤ 50 mm 포함)·D 표 범위·부재 간섭 없이 가능한 위치가 있는지.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import support_std as ss
from .pipeline import j_breakdown
from .scenario import load
from .support_diag import scenario_path
from .support_std import load_standard
from .verifier import support_group as sg
from .verifier.core import Context, PipeRoute, centerlines_of, verify


def _score(sc, rep):
    pipes = {p.id: p for p in sc.pipes}
    sup_kg, groups, gkg, n_pass, jt = 0.0, set(), 0.0, 0, 0.0
    for pid, r in rep["pipes"].items():
        if not r.routed:
            continue
        for s in r.supports:
            if s.get("group"):
                if s["group"] not in groups:
                    groups.add(s["group"]); gkg += s["kg_total"]
            else:
                sup_kg += s["kg"]
        if r.success:
            n_pass += 1
    for pid, r in rep["pipes"].items():
        if r.success:
            jt += j_breakdown(pipes[pid], rep["_wps"][pid], r.fittings, r.supports)["total"]
    return {"groups": len(groups), "grouped_supports": sum(1 for r in rep["pipes"].values() for s in r.supports
                                                         if s.get("group")),
            "support_total_kg": round(sup_kg + gkg, 2), "shared_kg": round(gkg, 2), "pass": n_pass,
            "J_total": round(jt, 2)}


def _verify(sc, wps, std, window, tol):
    old = (sg.WINDOW_MM, sg.SURFACE_TOL_MM)
    sg.WINDOW_MM, sg.SURFACE_TOL_MM = window, tol
    try:
        rep = verify(sc, [PipeRoute(p.id, wps[p.id]) for p in sc.pipes if p.id in wps], std=std)
    finally:
        sg.WINDOW_MM, sg.SURFACE_TOL_MM = old
    rep["_wps"] = wps
    return rep


def _point_on(cls, ax, a, near):
    """배관 중심선의 축 ax 정렬 수평 직관 중 좌표 a 를 지나는 점 (near 에 가장 가까운 것)."""
    best = None
    for cl in cls:
        A, B, _, pieces = cl.arrays(("straight",))
        for p0, p1 in zip(A, B):
            d = p1 - p0
            L = float(np.linalg.norm(d))
            if L < 1e-6 or abs(abs(d[ax]) - L) > 1e-6 * L:
                continue
            lo, hi = sorted((p0[ax], p1[ax]))
            if not (lo - 1e-6 <= a <= hi + 1e-6):
                continue
            q = p0 + (a - p0[ax]) / d[ax] * d
            dist = float(np.linalg.norm(q - near))
            if best is None or dist < best[0]:
                best = (dist, q)
    return None if best is None else best[1]


def together(sc, wps, std, row, step=50.0):
    """(iii): 위반 배관 P 의 불가 구간마다 D(P, Q) 가능 위치가 있는가 (Q = 진단의 막는 배관)."""
    pipes = {p.id: p for p in sc.pipes}
    routes = {pid: PipeRoute(pid, w) for pid, w in wps.items()}
    cls = {pid: centerlines_of(pipes[pid], r) for pid, r in routes.items()}
    ctx = Context(sc, pipes, routes, cls, std=std)
    P = row["pipe"]
    main = cls[P][0]
    pts = main.points
    seg = pts[1:] - pts[:-1]
    lens = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0], np.cumsum(lens)])
    gaps = row["gap_intervals"]
    solved = []
    for lo, hi in gaps:
        ok = None
        for s in np.arange(hi, lo, -step):
            i = int(min(np.searchsorted(cum, s, side="right") - 1, len(lens) - 1))
            u = seg[i] / lens[i]
            if abs(u[2]) > 1e-9:
                continue
            ax = int(np.argmax(np.abs(u)))
            if abs(abs(u[ax]) - 1) > 1e-9:
                continue
            p = pts[i] + (s - cum[i]) * u
            for Q in row["blockers"]:
                q = _point_on(cls[Q], ax, p[ax], p)
                if q is None:
                    continue
                for face in ("ceiling", "floor"):
                    sp = p[2] - pipes[P].od / 2 if face == "floor" else p[2] + pipes[P].od / 2
                    sq = q[2] - pipes[Q].od / 2 if face == "floor" else q[2] + pipes[Q].od / 2
                    if abs(sp - sq) > sg.SURFACE_TOL_MM:
                        continue
                    grp = [{"pid": P, "ax": ax, "p": p, "od": pipes[P].od, "surf": sp, "nd": pipes[P].nominal_size},
                           {"pid": Q, "ax": ax, "p": q, "od": pipes[Q].od, "surf": sq, "nd": pipes[Q].nominal_size}]
                    r = sg._try(ctx, grp, face)
                    if r is not None:
                        ok = {"s_m": round(s / 1000, 2), "with": Q, "face": face, "kg": round(r[0], 2)}
                        break
                if ok:
                    break
            if ok:
                break
        solved.append(ok)
    return solved


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("outputs", nargs="+")
    ap.add_argument("--diag", default=None)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)
    std = load_standard()
    diag = json.loads(Path(a.diag).read_text()) if a.diag else []
    rows = []
    for f in a.outputs:
        out = json.loads(Path(f).read_text())
        stem = Path(f).name[: -len("_output.json")]
        sc = load(scenario_path(stem))
        wps = {r["pipe_id"]: r["waypoints"] for r in out["routes"] if r["waypoints"]}
        res = {"scenario": stem}
        for name, (w, t) in {"w250": (250.0, 50.0), "w500": (500.0, 50.0), "align300": (500.0, 300.0)}.items():
            rep = _verify(sc, wps, std, w, t)
            res[name] = _score(sc, rep)
            if name == "w500":
                viol = {pid: [v.message for v in r.violations if v.module == "support"]
                        for pid, r in rep["pipes"].items() if r.layer0.get("support") is False}
        res["together"] = []
        for d in diag:
            if d["scenario"] != stem or d["pipe"] not in viol:
                continue
            gi = []
            for m in viol[d["pipe"]]:
                lo, hi = m.split(" m")[0].split("~")
                gi.append((float(lo) * 1000, float(hi) * 1000))
            row = dict(d, gap_intervals=gi)
            sol = together(sc, wps, std, row) if d["blockers"] else [None] * len(gi)
            res["together"].append({"pipe": d["pipe"], "order": d["order"], "gaps": len(gi),
                                    "solved_gaps": sum(1 for x in sol if x), "detail": sol})
        rows.append(res)
        t = res["together"]
        print(f"{stem}: 묶음 {res['w250']['groups']}→{res['w500']['groups']} (정렬 상한 {res['align300']['groups']}) "
              f"통과 {res['w250']['pass']}→{res['w500']['pass']}  (c) {sum(1 for x in t if x['solved_gaps'] == x['gaps'])}"
              f"/{len(t)}", flush=True)
    Path(a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
