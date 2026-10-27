"""서포트 힌트 (D100) 대조·호출 비용 측정 — 출력 JSON 의 경로로 검증기를 다시 돌리고, 검증기가 놓은 서포트마다 힌트를 묻는다.

대조: A·B·C 서포트 → form_at 선택 같음 (형식·면·보강·규격·무게·L·부재 정확히 — 검증기가 판정에 쓴 반올림 전 위치·축으로 묻는다) · D 묶음 → groupable ok·같은 묶음 무게 · D89 실물 → form_at None·together_at 같은 D.
비용: 호출마다 벽시계 (힌트 객체 생성 포함 따로). 출력에 선정표 값은 없다 (형식·개수·시간만, D78).

python -m pipe_routing.verifier.hint_bench out/run/proc_000_output.json ... -o out/hint_bench.json
"""
import argparse
import json
import statistics
import time
from pathlib import Path

from ..scenario import load
from ..support_diag import scenario_path
from ..support_std import load_standard
from . import PipeRoute, verify
from .hint import SupportHint

KEYS_C = ("form", "face", "variant", "section", "kg", "L")   # + members 정확히 (반올림 전 위치·축으로 묻는다)
KEYS_T = ("group", "face", "section", "kg_total", "carried", "shares")
MEMBER_TOL_MM = 1.0   # 출력 좌표·축은 반올림돼 있다 (축 0.707 등) — 45° 배관의 부재 끝점은 1 mm 미만 어긋난다


def _members_close(a, b) -> bool:
    if len(a) != len(b):
        return False
    return all(x["role"] == y["role"] and max(abs(u - v) for u, v in zip(x["a"] + x["b"], y["a"] + y["b"])) <= MEMBER_TOL_MM
               for x, y in zip(a, b))


def _verify_recording(sc, routes, std):
    """검증기를 돌리면서 서포트 후보 판정에 실제로 넘긴 정확한 (위치, 축) 을 고른 서포트 dict 에 붙여 둔다
    (출력 좌표·축은 반올림돼 있어 45° 배관에서 끝자리가 달라진다 — 대조는 반올림 전 값으로 정확히 한다)."""
    from . import modules as mod
    orig = mod.support_candidates

    def rec(ctx, pid, pts, dirs):
        res = orig(ctx, pid, pts, dirs)
        for i, c in enumerate(res):
            if c is not None:
                c["_exact"] = (pts[i].copy(), dirs[i].copy())
        return res
    mod.support_candidates = rec
    try:
        return verify(sc, routes, std=std)["pipes"]
    finally:
        mod.support_candidates = orig


def _exact(s):
    return s.get("_exact") or (s["pos"], s["axis"])


def bench(out_path: str, std) -> dict:
    out = json.loads(Path(out_path).read_text(encoding="utf-8"))
    stem = Path(out_path).name[: -len("_output.json")]
    sc = load(scenario_path(stem))
    routes = [PipeRoute(x["pipe_id"], x["waypoints"]) for x in out["routes"] if x["waypoints"]]
    reps = _verify_recording(sc, routes, std)
    t0 = time.perf_counter()
    h = SupportHint(sc, routes, std=std)
    t_new = time.perf_counter() - t0
    tf, tg, tt = [], [], []
    bad = {"form_at": 0, "groupable": 0, "together_at": 0}
    n = {"form_at": 0, "groupable": 0, "together_at": 0}
    groups = {}
    for pid, rep in reps.items():
        for s in rep.supports:
            if s.get("origin") == "D89":
                if s.get("attached"):
                    continue
                t1 = time.perf_counter()
                c = h.form_at(pid, s["pos"], s["axis"])["chosen"]
                t = h.together_at(pid, s["pos"], s["axis"])
                tt.append(time.perf_counter() - t1)
                n["together_at"] += 1
                bad["together_at"] += int(c is not None or t is None or any(t[k] != s[k] for k in KEYS_T)
                                         or not _members_close(t["members"], s["members"]))
                continue
            if s["form"] == "D":
                groups.setdefault(s["group"], []).append((pid, s))
                continue
            t1 = time.perf_counter()
            c = h.form_at(pid, *_exact(s))["chosen"]
            tf.append(time.perf_counter() - t1)
            n["form_at"] += 1
            bad["form_at"] += int(c is None or any(c[k] != s[k] for k in KEYS_C) or c["members"] != s["members"])
    for gid, lst in groups.items():
        if len(lst) != 2:     # 3 배관 이상 묶음은 groupable(두 위치) 의 범위 밖 — 개수만 센다
            n.setdefault("group_3plus", 0)
            n["group_3plus"] += 1
            continue
        (pa, sa), (pb, sb) = lst
        t1 = time.perf_counter()
        g = h.groupable((pa, sa["pos"], sa["axis"]), (pb, sb["pos"], sb["axis"]), face=sa["face"])
        tg.append(time.perf_counter() - t1)
        n["groupable"] += 1
        bad["groupable"] += int(not g["ok"] or abs(g["kg"] - sa["kg_total"]) > 1e-4)
    # 묶음 후보 쌍 (실제 세트에는 D 묶음이 거의 없다): 같은 면 C · 같은 수평 축 · 축 위치 ±500 mm · 가로 거리 ≤ L1 표 끝
    from . import support_group as sg
    cs = [(pid, s) for pid, rep in reps.items() for s in rep.supports if s["form"] == "C" and not s.get("origin")]
    lim = sg.l1_max(std)
    pairs = []
    for i, (pa, sa) in enumerate(cs):
        ua = sa["axis"]
        if abs(ua[2]) > 1e-9 or max(abs(v) for v in ua) < 1 - 1e-9:
            continue
        ax = max(range(3), key=lambda k: abs(ua[k]))
        for pb, sb in cs[i + 1:]:
            if pb == pa or sb["face"] != sa["face"] or sb["axis"] != ua or \
                    abs(sa["pos"][ax] - sb["pos"][ax]) > sg.WINDOW_MM or abs(sa["pos"][1 - ax] - sb["pos"][1 - ax]) > lim:
                continue
            pairs.append(((pa, *_exact(sa)), (pb, *_exact(sb)), sa["face"]))
    tg2, ok2, why = [], 0, {}
    for a_, b_, f in pairs:
        t1 = time.perf_counter()
        g = h.groupable(a_, b_, face=f)
        tg2.append(time.perf_counter() - t1)
        ok2 += int(g["ok"])
        if not g["ok"]:
            why[g["reason"]] = why.get(g["reason"], 0) + 1
    n["groupable_pairs"] = len(pairs)
    n["groupable_pairs_ok"] = ok2
    tg += tg2
    return {"why_not_grouped": why, "scenario": stem, "pipes": len(sc.pipes), "routed": len(routes), "n": n, "mismatch": bad,
            "init_sec": t_new, "form_at_sec": tf, "groupable_sec": tg, "together_at_sec": tt}


def _q(x, p):
    x = sorted(x)
    return x[min(len(x) - 1, int(p * (len(x) - 1) + 0.5))] if x else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("outputs", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)
    std = load_standard()
    rows = []
    for f in a.outputs:
        r = bench(f, std)
        rows.append(r)
        print(f"{r['scenario']}: n {r['n']} 불일치 {r['mismatch']} · 생성 {r['init_sec']:.3f}s · form_at 중앙 "
              f"{statistics.median(r['form_at_sec']) * 1e3 if r['form_at_sec'] else 0:.1f} ms", flush=True)
    summ = {k: {"median_ms": round(_q(v, .5) * 1e3, 2) if v else None, "p90_ms": round(_q(v, .9) * 1e3, 2) if v else None,
                "max_ms": round(max(v) * 1e3, 2) if v else None, "calls": len(v)}
            for k, v in (("init", [r["init_sec"] for r in rows]),
                         ("form_at", [t for r in rows for t in r["form_at_sec"]]),
                         ("groupable", [t for r in rows for t in r["groupable_sec"]]),
                         ("together_at", [t for r in rows for t in r["together_at_sec"]]))}
    Path(a.out).write_text(json.dumps({"support_standard": std.info(), "summary": summ,
                                       "mismatch": {k: sum(r["mismatch"][k] for r in rows) for k in rows[0]["mismatch"]},
                                       "calls": {k: sum(r["n"].get(k, 0) for r in rows) for k in
                                                 ("form_at", "groupable", "together_at", "group_3plus", "groupable_pairs", "groupable_pairs_ok")},
                                       "why_not_grouped": {k: sum(r["why_not_grouped"].get(k, 0) for r in rows)
                                                           for k in sorted({k for r in rows for k in r["why_not_grouped"]})},
                                       "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
