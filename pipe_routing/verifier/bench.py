"""검증기 호출 비용 측정 (D84) · 증분 검증 측정 (D85).

사용: python -m pipe_routing.verifier.bench <scenario.json>=<output.json> ... [--incremental N] [--check K] -o out.json
  - 전체 검증: 출력 JSON 의 경로로 verify() 1회 — 시나리오 시간, 모듈별 시간 분해 (verify()["timing"]).
  - --incremental N: Session 에 전 배관을 무작위 순서로 add 한 뒤, 무작위 배관 N 개에 대해 remove → add (같은 경로).
    호출별 시간과 다시 판정한 배관 수를 기록. --check K: K 번째 호출마다 전체 verify 와 비트 동일 확인.
"""
import argparse
import json
import random
import statistics
import time
from pathlib import Path

from ..scenario import load
from ..support_std import load_standard
from .core import PipeRoute, verify
from .session import Session


def _dump(reps) -> str:
    return json.dumps({k: v.to_dict() for k, v in reps.items()}, sort_keys=True, default=str)


def bench(scn_path, out_path, n_inc=0, check=0, seed=0) -> dict:
    sc = load(scn_path)
    out = json.loads(Path(out_path).read_text())
    wps = {r["pipe_id"]: r["waypoints"] for r in out["routes"] if r["waypoints"]}
    std = load_standard()
    routes = [PipeRoute(p.id, wps[p.id]) for p in sc.pipes if p.id in wps]
    t = time.perf_counter()
    rep = verify(sc, routes, std=std)
    full = time.perf_counter() - t
    res = {"scenario": Path(scn_path).stem, "pipes": len(sc.pipes), "routed": len(routes), "full_sec": full,
           "timing": rep["timing"]}
    if n_inc:
        rng = random.Random(seed)
        sess = Session(sc, std=std)
        ids = sorted(wps)
        rng.shuffle(ids)
        build = []
        for pid in ids:
            r = sess.add(pid, wps[pid])
            build.append((r["sec"], len(r["recomputed"])))
        calls, ok = [], True
        for k in range(n_inc):
            pid = rng.choice(ids)
            r1 = sess.remove(pid)
            r2 = sess.add(pid, wps[pid])
            calls += [(r1["sec"], len(r1["recomputed"])), (r2["sec"], len(r2["recomputed"]))]
            if check and (k + 1) % check == 0:
                ok &= _dump(sess.reports()) == _dump(verify(sc, routes, std=std)["pipes"])
        if check:
            ok &= _dump(sess.reports()) == _dump(rep["pipes"])
        res.update({"build_sec": [s for s, _ in build], "inc_sec": [s for s, _ in calls],
                    "inc_recomputed": [n for _, n in calls], "bit_identical": ok if check else None})
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+", help="scenario.json=output.json")
    ap.add_argument("--incremental", type=int, default=0)
    ap.add_argument("--check", type=int, default=0)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)
    rows = []
    for pair in a.pairs:
        s, o = pair.split("=")
        r = bench(s, o, a.incremental, a.check)
        rows.append(r)
        inc = r.get("inc_sec") or [0]
        print(f"{r['scenario']}: 배관 {r['routed']}/{r['pipes']} 전체 {r['full_sec']:.2f}s"
              + (f" · 증분 중앙 {statistics.median(inc):.3f}s 비트동일 {r['bit_identical']}" if a.incremental else ""),
              flush=True)
    Path(a.out).write_text(json.dumps(rows, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
