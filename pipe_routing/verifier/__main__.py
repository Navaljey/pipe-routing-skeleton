"""python -m pipe_routing.verifier <scenario.json>... [--routes FILE] [--viz DIR] [--json FILE]

경로 입력: --routes 로 JSON ({"routes": [{"pipe_id", "waypoints", "branches"?}]}, §7.2 형식의 routes 와 호환)을 주거나,
생략하면 S0 A* (4단계)로 경로를 만든 뒤 검증한다.
"""
import argparse
import collections
import json
from pathlib import Path

from ..scenario import load
from . import MODULE_ORDER, PipeRoute, verify


def astar_routes(sc):
    from ..escape_graph import EscapeGraph
    from ..router_astar import astar_route
    out = []
    for p in sc.pipes:
        r = astar_route(EscapeGraph(sc, p), p)
        out.append(PipeRoute(p.id, r.waypoints if r.status == "ok" else []))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="검증기 Layer 0 (5단계)")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--routes", help="경로 JSON (시나리오 1개일 때)")
    ap.add_argument("--viz", metavar="DIR", help="위반 위치를 그린 HTML 저장 폴더")
    ap.add_argument("--json", help="배관별 판정을 JSON 줄 단위로 저장 (append)")
    args = ap.parse_args(argv)
    total = collections.Counter()
    for path in args.paths:
        sc = load(path)
        if args.routes:
            data = json.loads(Path(args.routes).read_text(encoding="utf-8"))
            routes = [PipeRoute(r["pipe_id"], r.get("waypoints") or [], r.get("branches", [])) for r in data["routes"]]
        else:
            routes = astar_routes(sc)
        rep = verify(sc, routes)
        print(f"== {path}  (검증 {rep['sec']:.2f}s)")
        for pid, pr in rep["pipes"].items():
            flags = " ".join(f"{m}:{'-' if pr.layer0.get(m) is None else ('O' if pr.layer0[m] else 'X')}"
                             for m in MODULE_ORDER) if pr.routed else "경로 없음"
            cnt = collections.Counter(v.module for v in pr.violations)
            print(f"  {pid:5s} {'PASS' if pr.success else 'FAIL'}  {flags}  {dict(cnt) if cnt else ''}")
            total["pipes"] += 1
            total["pass"] += pr.success
            for m in MODULE_ORDER:
                total[f"fail_{m}"] += pr.layer0.get(m) is False
            if args.json:
                with open(args.json, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"scenario": sc.meta.get("name", path), "type_id": sc.pipes[[p.id for p in sc.pipes].index(pid)].type_id,
                                        **pr.to_dict()}, ensure_ascii=False) + "\n")
        if args.viz:
            from ..router_astar import RouteResult
            from ..viz import add_routes, add_verification, scenario_figure, write_html
            out = Path(args.viz)
            out.mkdir(parents=True, exist_ok=True)
            fig = scenario_figure(sc, connect=False)
            add_routes(fig, sc, [RouteResult(r.pipe_id, "ok" if r.waypoints else "unreachable",
                                             waypoints=r.waypoints) for r in routes])
            add_verification(fig, rep)
            dst = out / (Path(path).stem + "_verify.html")
            write_html(fig, dst)
            print(f"  → {dst}")
    if len(args.paths) > 1:
        print(f"\n합계: 통과 {total['pass']}/{total['pipes']}  " +
              "  ".join(f"{m} 위반 {total[f'fail_{m}']}" for m in MODULE_ORDER))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
