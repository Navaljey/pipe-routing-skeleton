"""6단계: 입력 → 라우팅 → 검증 → 지표·J 분해 → 출력 JSON(§7.2)·리포트 (D49).

    python -m pipe_routing.pipeline <scenario.json>... [-o out/run] [--time-limit 60]

라우터 슬롯 (§4.1, D3): router(scenario, pipe, time_limit) → RouteResult 호환 객체
(status, waypoints, J, expanded, search_sec, graph_sec). 기본값은 S0 A* (escape graph). V3 는 같은 형태의 함수를 넘기면 된다.
"""
import argparse
import json
import math
import time
from pathlib import Path
from typing import Callable

import jsonschema

from .constants import FITTINGS, PIPE_SPECS
from .scenario import Pipe, Scenario, load
from .verifier import MODULE_ORDER, PipeRoute, verify

OUTPUT_SCHEMA = Path(__file__).resolve().parent.parent / "schemas" / "scenario_output.schema.json"


# ---------------------------------------------------------------- 라우터 슬롯

def astar_router(sc: Scenario, pipe: Pipe, time_limit: float):
    """S0 기본 라우터: 배관마다 escape graph 를 만들고(D11) A* (D40·D41)."""
    from .escape_graph import EscapeGraph
    from .router_astar import astar_route
    g = EscapeGraph(sc, pipe)
    r = astar_route(g, pipe, time_limit)
    r.graph_sec = g.build_sec
    return r


# ---------------------------------------------------------------- J (§5, D49)

def polyline_length_mm(wps) -> float:
    return sum(math.dist(a, b) for a, b in zip(wps, wps[1:]))


def j_breakdown(pipe: Pipe, waypoints, fittings: list, supports: list) -> dict:
    """§5 J 분해 (kg). 직관 = 꺾임점 사이 중심선 길이 합 × kg/m — 라우터 비용(D37·D41)과 같은 정의 (D49).

    엘보·티·밸브 = 검증기가 보고한 관이음 목록 × §3.2 중량 (135° = 90° + 45°, D25), 서포트 = 검증기 서포트 kg 합 (D46).
    """
    f = FITTINGS[pipe.nominal_size]
    unit = {"elbow_90": f.elbow90, "elbow_45": f.elbow45, "tee": f.tee, "gate_valve": f.gate_valve}
    by = {k: sum(unit[x["type"]] for x in fittings if x["type"] == k) for k in unit}
    J = {
        "pipe": polyline_length_mm(waypoints) / 1000 * PIPE_SPECS[pipe.nominal_size].kg_per_m,
        "elbow": by["elbow_90"] + by["elbow_45"],
        "tee": by["tee"],
        "valve": by["gate_valve"],
        "support": sum(s["kg"] for s in supports),
    }
    J["total"] = sum(J.values())
    return {k: round(v, 4) for k, v in J.items()}


# ---------------------------------------------------------------- 실행

def run(sc: Scenario, router: Callable = astar_router, time_limit: float = 60.0) -> dict:
    """시나리오 1개 실행 → §7.2 출력 dict."""
    t0 = time.perf_counter()
    results = {}
    for p in sc.pipes:
        results[p.id] = router(sc, p, time_limit)
    route_sec = time.perf_counter() - t0
    rep = verify(sc, [PipeRoute(pid, r.waypoints if r.status == "ok" else []) for pid, r in results.items()])

    routes = []
    for p in sc.pipes:
        r, v = results[p.id], rep["pipes"][p.id]
        entry = {
            "pipe_id": p.id,
            "type_id": p.type_id,
            "nominal_size": p.nominal_size,
            "success": v.success,
            "waypoints": [[round(c, 3) for c in w] for w in r.waypoints] if r.status == "ok" else [],
            "fittings": v.fittings,
            "supports": v.supports,
            "layer0": {m: v.layer0.get(m) for m in MODULE_ORDER} if v.routed else {},
            "J": j_breakdown(p, r.waypoints, v.fittings, v.supports) if v.routed else None,
            "fail_causes": fail_causes(r.status, v),
            "violations": [{"module": x.module, "pos": x.pos, "message": x.message, "other": x.other}
                           for x in v.violations],
            "metrics": metrics(r, v),
            "router": {"status": r.status, "expanded": r.expanded, "search_sec": round(r.search_sec, 3),
                       "graph_sec": round(getattr(r, "graph_sec", 0.0), 3)},
        }
        routes.append(entry)
    total_sec = time.perf_counter() - t0
    return {"scenario": sc.meta.get("name", ""), "routes": routes,
            "global": global_summary(routes, total_sec, route_sec, rep["sec"])}


def fail_causes(router_status: str, v) -> list:
    """§9-2 실패 원인: 라우터 실패(timeout/unreachable, D34·D40) 또는 위반 모듈 목록."""
    if router_status != "ok":
        return [f"router:{router_status}"]
    return [m for m in MODULE_ORDER if v.layer0.get(m) is False]


def metrics(r, v) -> dict:
    """§6.2 보조 지표."""
    if r.status != "ok":
        return {}
    n = {"elbow_45": 0, "elbow_90": 0, "tee": 0, "gate_valve": 0}
    for f in v.fittings:
        n[f["type"]] += 1
    return {"length_m": round(polyline_length_mm(r.waypoints) / 1000, 3), "n_elbow_90": n["elbow_90"],
            "n_elbow_45": n["elbow_45"], "n_tee": n["tee"], "n_valve": n["gate_valve"],
            "n_supports": len(v.supports)}


def global_summary(routes: list, total_sec: float, route_sec: float, verify_sec: float) -> dict:
    routed = [x for x in routes if x["J"] is not None]
    ok = [x for x in routes if x["success"]]
    causes = {}
    for x in routes:
        for c in x["fail_causes"]:
            causes[c] = causes.get(c, 0) + 1
    pairwise = sum(1 for x in routes for v in x["violations"]
                   if v["module"] in ("collision", "bend") and v["other"] and not v["other"].startswith("OBS"))
    jsum = lambda xs, k: round(sum(x["J"][k] for x in xs), 4)
    return {
        "total_pipes": len(routes),
        "routed_count": len(routed),
        "success_count": len(ok),
        "success_rate": round(len(ok) / len(routes), 4) if routes else None,
        "success_rate_routed": round(len(ok) / len(routed), 4) if routed else None,   # D34 기준선 도달 가능 부분집합
        "J_total": jsum(ok, "total"),                                                 # §6.3 성공 배관 합
        "J_breakdown_success": {k: jsum(ok, k) for k in ("pipe", "elbow", "tee", "valve", "support")},
        "J_total_routed": jsum(routed, "total"),
        "fail_causes": dict(sorted(causes.items())),
        "pairwise_violation": pairwise // 2,                                          # 배관 쌍마다 양쪽에 기록됨
        "computation_time_sec": round(total_sec, 3),
        "routing_time_sec": round(route_sec, 3),
        "verify_time_sec": round(verify_sec, 3),
    }


def validate_output(out: dict) -> None:
    schema = json.loads(OUTPUT_SCHEMA.read_text(encoding="utf-8"))
    jsonschema.validate(out, schema)


# ---------------------------------------------------------------- 리포트

def report_md(out: dict, scenario_path: str = "") -> str:
    g = out["global"]
    L = [f"# S0 실행 리포트 — {out['scenario'] or scenario_path}", ""]
    L += ["## 요약", "",
          "| 항목 | 값 |", "|---|---|",
          f"| 배관 | {g['total_pipes']} (경로 있음 {g['routed_count']}) |",
          f"| Layer 0 전체 통과 | {g['success_count']} ({g['success_rate']:.0%}, 경로 있는 배관 기준 {g['success_rate_routed'] or 0:.0%}) |",
          f"| J_total (성공 배관 합) | {g['J_total']:,.2f} kg |",
          "| J 분해 (성공 배관) | " + " · ".join(f"{k} {v:,.2f}" for k, v in g["J_breakdown_success"].items()) + " |",
          f"| J (경로 있는 배관 전체) | {g['J_total_routed']:,.2f} kg |",
          f"| 실패 원인 | {', '.join(f'{k} {v}' for k, v in g['fail_causes'].items()) or '없음'} |",
          f"| 배관 간 이격 위반 쌍 | {g['pairwise_violation']} |",
          f"| 계산 시간 | {g['computation_time_sec']:.2f} s (라우팅 {g['routing_time_sec']:.2f} · 검증 {g['verify_time_sec']:.2f}) |",
          ""]
    L += ["## 배관별", "",
          "| 배관 | 구경 | 타입 | 라우터 | 탐색 s | 판정 | 실패 원인 | J total | 직관 | 엘보 | 서포트 | 길이 m | 엘보 90/45 | 서포트 수 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in out["routes"]:
        J, m = x["J"] or {}, x["metrics"]
        fmt = lambda k: f"{J[k]:,.2f}" if J else "-"
        L.append(f"| {x['pipe_id']} | {x['nominal_size']} | {x['type_id']} | {x['router']['status']} | "
                 f"{x['router']['search_sec']:.2f} | {'PASS' if x['success'] else 'FAIL'} | "
                 f"{', '.join(x['fail_causes']) or '-'} | {fmt('total')} | {fmt('pipe')} | {fmt('elbow')} | "
                 f"{fmt('support')} | {m.get('length_m', '-')} | "
                 f"{m.get('n_elbow_90', '-')}/{m.get('n_elbow_45', '-')} | {m.get('n_supports', '-')} |")
    L += ["", "## 위반 상세", ""]
    any_v = False
    for x in out["routes"]:
        for v in x["violations"]:
            any_v = True
            L.append(f"- **{x['pipe_id']} {v['module']}** @ {[round(c) for c in v['pos']]}: {v['message']}")
    if not any_v:
        L.append("- 없음")
    L += ["", "Layer 0 모듈: " + " · ".join(MODULE_ORDER) + " (§6.1, D44). J 정의: §5, D49.", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="S0 파이프라인: 입력 → 라우팅 → 검증 → 출력 JSON·리포트 (6단계)")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("-o", "--out", default="out/run")
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--no-viz", action="store_true", help="HTML 리포트(경로·위반 3D) 생략")
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.paths:
        sc = load(path)
        out = run(sc, time_limit=args.time_limit)
        validate_output(out)
        stem = Path(path).stem
        (out_dir / f"{stem}_output.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out_dir / f"{stem}_report.md").write_text(report_md(out, path), encoding="utf-8")
        if not args.no_viz:
            write_report_html(sc, out, out_dir / f"{stem}_report.html")
        g = out["global"]
        print(f"{path}: 통과 {g['success_count']}/{g['total_pipes']}  J_total {g['J_total']:,.2f} kg  "
              f"{g['computation_time_sec']:.1f}s → {out_dir}/{stem}_output.json, _report.md")
    return 0


def write_report_html(sc: Scenario, out: dict, path) -> None:
    from .router_astar import RouteResult
    from .verifier.core import PipeReport, Violation
    from .viz import add_routes, add_verification, scenario_figure, write_html
    fig = scenario_figure(sc, connect=False)
    add_routes(fig, sc, [RouteResult(x["pipe_id"], "ok" if x["waypoints"] else x["router"]["status"],
                                     J=(x["J"] or {}).get("total"), waypoints=x["waypoints"]) for x in out["routes"]])
    reps = {x["pipe_id"]: PipeReport(x["pipe_id"], bool(x["waypoints"]),
                                     violations=[Violation(v["module"], x["pipe_id"], v["pos"], v["message"])
                                                 for v in x["violations"]],
                                     supports=x["supports"]) for x in out["routes"]}
    add_verification(fig, {"pipes": reps})
    g = out["global"]
    fig.update_layout(title=dict(text=fig.layout.title.text +
                                 f"<br>통과 {g['success_count']}/{g['total_pipes']} · J_total {g['J_total']:,.1f} kg"))
    write_html(fig, path)


if __name__ == "__main__":
    raise SystemExit(main())
