"""D128 `support_blockers` 대조·호출 비용 — 출력 JSON 의 경로로 힌트를 만들고, 경로 있는 배관마다 support_blockers 를 부른다.

대조: support 위반 배관마다 R7 진단 (`support_diag.diagnose_out` — 막는 배관 하나를 빼면 support 가 통과하는가) 의 막는 배관과
support_blockers 의 kind = gap 막는 배관을 비교한다 (`docs/reports/r7_cases.md` 와 같은 정의).
비용: 배관마다 벽시계 (힌트 객체 생성 따로). 출력에 선정표 값은 없다 (형식·면·개수·시간만, D78).

python -m pipe_routing.verifier.blockers_bench out/run/*_output.json -o out/blockers_bench.json
"""
import argparse
import json
import time
from pathlib import Path

from ..scenario import load
from ..support_diag import diagnose_out, scenario_path
from ..support_std import load_standard
from . import PipeRoute
from .hint import SupportHint


def bench(out_path: str, std) -> dict:
    out = json.loads(Path(out_path).read_text(encoding="utf-8"))
    stem = Path(out_path).name[: -len("_output.json")]
    sc = load(scenario_path(stem))
    routes = [PipeRoute(x["pipe_id"], x["waypoints"]) for x in out["routes"] if x["waypoints"]]
    t0 = time.perf_counter()
    h = SupportHint(sc, routes, std=std)
    init = time.perf_counter() - t0
    calls, rows = [], {}
    for r in routes:
        t0 = time.perf_counter()
        rows[r.pipe_id] = h.support_blockers(r.pipe_id)
        calls.append(time.perf_counter() - t0)
    r7 = {x["pipe"]: x for x in diagnose_out(out, sc, std, stem)}
    cmp = []
    for pid, x in r7.items():
        gap = sorted({b["blocker"] for b in rows[pid] if b["kind"] == "gap"})
        cmp.append({"pipe": pid, "r7_blockers": sorted(x["blockers"]), "r7_kind": x["blocker_kind"], "gap_blockers": gap,
                    "same": sorted(x["blockers"]) == gap,
                    "r7_subset": set(x["blockers"]) <= set(gap),
                    "r7_blocked_forms": sorted({f"{b['form']}/{b['face']}" for b in x["blocked_supports"]}),
                    "gap_forms": sorted({f"{b['form']}/{b['face']}" for b in rows[pid] if b["kind"] == "gap"})})
    support_fail = {x["pipe_id"] for x in out["routes"] if x["waypoints"] and x["layer0"].get("support") is False}
    return {"scenario": stem, "pipes": len(routes), "init_sec": init, "call_sec": calls,
            "with_blockers": sum(1 for v in rows.values() if v),
            "rows": sum(len(v) for v in rows.values()),
            "kinds": {k: sum(1 for v in rows.values() for b in v if b["kind"] == k) for k in ("gap", "displaced")},
            "gap_pipes_not_support_fail": sorted(p for p, v in rows.items()
                                                 if any(b["kind"] == "gap" for b in v) and p not in support_fail),
            "r7": cmp}


def _q(x, p):
    x = sorted(x)
    return x[min(len(x) - 1, int(p * (len(x) - 1) + 0.5))] if x else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("outputs", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)
    std = load_standard()
    res = []
    for f in a.outputs:
        r = bench(f, std)
        res.append(r)
        print(f"{r['scenario']}: 배관 {r['pipes']} · 막힘 있는 배관 {r['with_blockers']} · R7 {len(r['r7'])} "
              f"(같음 {sum(c['same'] for c in r['r7'])}) · 호출 중앙 {(_q(r['call_sec'], .5) or 0) * 1e3:.0f} ms", flush=True)
    calls = [t for r in res for t in r["call_sec"]]
    summ = {"support_standard": std.info(),
            "call_ms": {"median": round(_q(calls, .5) * 1e3, 1), "p90": round(_q(calls, .9) * 1e3, 1),
                        "max": round(max(calls) * 1e3, 1), "n": len(calls)} if calls else None,
            "init_ms": {"median": round(_q([r["init_sec"] for r in res], .5) * 1e3, 1)},
            "r7_cases": sum(len(r["r7"]) for r in res), "r7_same": sum(c["same"] for r in res for c in r["r7"]),
            "r7_subset": sum(c["r7_subset"] for r in res for c in r["r7"]),
            "rows": res}
    Path(a.out).write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summ.items() if k != "rows"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
