"""M18 스케일링 실측: 놓인 배관 수 k 에 따른 다음 배관의 그래프 크기·생성·탐색 시간.

시나리오마다 순차 배치 결과(pipeline 출력 JSON)의 배치 순서를 따른다. 대상 배관 = 배치 순서의 마지막 배관(고정),
놓인 배관 = 순서상 앞에서부터 경로가 있는 배관 k 개. 대상이 같으므로 k 의 효과만 본다.
표현: current (S0 escape graph, 놓인 배관 격자선), A (고정 레이어 + 표시), B (A + 국소 이격선).

python -m pipe_routing.scaling scenarios/procedural/proc_000.json ... --runs DIR --json out/scaling.jsonl
"""
import argparse
import json
import time
from pathlib import Path

from .escape_graph import EscapeGraph
from .layered import LayeredGraph, _route, clear_cache
from .router_astar import astar_route
from .scenario import load

KS = (0, 1, 3, 5, 7, 9)


def measure(sc, out: dict, ks=KS, time_limit=60.0, reps=("current", "A", "B")):
    pipes = {p.id: p for p in sc.pipes}
    order = out["global"]["order"]
    wp = {x["pipe_id"]: x["waypoints"] for x in out["routes"] if x["waypoints"]}
    target = pipes[order[-1]]
    pool = [(pipes[q], wp[q]) for q in order[:-1] if q in wp]
    rows = []
    for k in ks:
        if k > len(pool):
            break
        placed = pool[:k]
        for rep in reps:
            if rep == "current":
                t0 = time.perf_counter()
                g = EscapeGraph(sc, target, others=placed)
                gsec = time.perf_counter() - t0
                s = g.stats()
                r = astar_route(g, target, time_limit)
                row = {"grid": s["grid"], "grid_points": s["grid_points"], "nodes": s["nodes"], "edges": s["edges"]}
            else:
                clear_cache()   # 고정 레이어 구성 시간을 따로 보기 위해 매번 새로
                t0 = time.perf_counter()
                g = LayeredGraph(sc, target, placed, local_lines=(rep == "B"))
                gsec = time.perf_counter() - t0
                r = _route(g, target, time_limit)
                row = {"grid": list(g.shape), "nodes": int(g.node_valid.sum()), "patch_nodes": g.n_patch_nodes,
                       "edges": int(g.n_edges), "fixed_sec": round(g.fixed_sec, 3)}
            row.update({"scenario": sc.meta.get("name", ""), "pipe": target.id, "size": target.nominal_size,
                        "k": k, "rep": rep, "graph_sec": round(gsec, 3), "status": r.status,
                        "expanded": r.expanded, "search_sec": round(r.search_sec, 3),
                        "J": round(r.J, 3) if r.J is not None else None})
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="M18 스케일링 실측")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--runs", required=True, help="순차 배치 pipeline 출력 폴더 (<stem>_output.json)")
    ap.add_argument("--json", help="결과 JSON 줄 저장")
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--reps", default="current,A,B")
    args = ap.parse_args(argv)
    for path in args.paths:
        sc = load(path)
        out = json.loads((Path(args.runs) / f"{Path(path).stem}_output.json").read_text(encoding="utf-8"))
        rows = measure(sc, out, time_limit=args.time_limit, reps=tuple(args.reps.split(",")))
        if args.json:
            with open(args.json, "a", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
