"""6단계: 입력 → 라우팅 → 검증 → 지표·J 분해 → 출력 JSON(§7.2)·리포트 (D49).

    python -m pipe_routing.pipeline <scenario.json>... [-o out/run] [--time-limit 60]

라우터 슬롯 (§4.1, D3): router(scenario, pipe, time_limit, placed) → RouteResult 호환 객체
(status, waypoints, J, expanded, search_sec, graph_sec). placed = 이미 놓인 배관 [(Pipe, waypoints)] (D50②).
다중 배관 슬롯 (D50⑧): planner(scenario, router, time_limit) → multi.PlanResult. 기본값은 순서 + rip-up (D50),
6단계 방식(배관 단독)은 multi.independent_planner. V3 는 같은 형태의 함수를 넘기면 된다.
"""
import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Callable

import jsonschema

from .constants import FITTINGS, PIPE_SPECS
from .layered import layered_router_a, make_layered_router
from .multi import FAIL_CLASS_KO, FAIL_CLASSES, astar_router, independent_planner, sequential_ripup_planner
from .scenario import Pipe, Scenario, load
from .verifier import MODULE_ORDER, PipeRoute, verify

OUTPUT_SCHEMA = Path(__file__).resolve().parent.parent / "schemas" / "scenario_output.schema.json"


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


def real_weight(pipe: Pipe, waypoints, J: dict) -> dict:
    """D49 보완 — 보조 지표 실중량: 직관 = (꺾임점 사이 길이 − 엘보마다 접선 2t) × kg/m + 엘보·티·밸브·서포트.
    J 계산과 최적화에는 쓰지 않는다."""
    from .constants import ANGLE_TOL_DEG, elbow_radius
    from .verifier.geom import build_centerline
    cl = build_centerline(waypoints, elbow_radius(pipe.nominal_size), ANGLE_TOL_DEG)
    straight = polyline_length_mm(waypoints) - sum(2 * b.tangent for b in cl.bends if math.isfinite(b.tangent))
    out = {"pipe": straight / 1000 * PIPE_SPECS[pipe.nominal_size].kg_per_m,
           "elbow": J["elbow"], "tee": J["tee"], "valve": J["valve"], "support": J["support"]}
    out["total"] = sum(out.values())
    return {k: round(v, 4) for k, v in out.items()}


# ---------------------------------------------------------------- 실행

def run(sc: Scenario, router: Callable = layered_router_a, time_limit: float = 60.0,
        planner: Callable = sequential_ripup_planner) -> dict:
    """시나리오 1개 실행 → §7.2 출력 dict."""
    t0 = time.perf_counter()
    plan = planner(sc, router, time_limit)
    results = plan.routes
    route_sec = time.perf_counter() - t0
    rep = verify(sc, [PipeRoute(pid, r.waypoints if r.status == "ok" else []) for pid, r in results.items()])

    routes = []
    for p in sc.pipes:
        r, v = results[p.id], rep["pipes"][p.id]
        J = j_breakdown(p, r.waypoints, v.fittings, v.supports) if v.routed else None
        tries = plan.attempts.get(p.id, [])
        entry = {
            "pipe_id": p.id,
            "type_id": p.type_id,
            "nominal_size": p.nominal_size,
            "success": v.success,
            "waypoints": [[round(c, 3) for c in w] for w in r.waypoints] if r.status == "ok" else [],
            "fittings": v.fittings,
            "supports": v.supports,
            "layer0": {m: v.layer0.get(m) for m in MODULE_ORDER} if v.routed else {},
            "J": J,
            "real_weight": real_weight(p, r.waypoints, J) if J else None,
            "fail_causes": fail_causes(r.status, v),
            "fail_class": plan.fail_class.get(p.id),
            "fail_detail": plan.fail_detail.get(p.id),
            "violations": [{"module": x.module, "pos": x.pos, "message": x.message, "other": x.other}
                           for x in v.violations],
            "metrics": metrics(r, v),
            "router": {"status": r.status, "expanded": r.expanded, "search_sec": round(r.search_sec, 3),
                       "name": getattr(router, "__name__", ""),
                       "support_est": round(getattr(r, "J_support_est", 0.0), 3) if r.status == "ok" else None,
                       "graph_sec": round(getattr(r, "graph_sec", 0.0), 3),
                       "attempts": len(tries),
                       "graph_sec_total": round(sum(a["graph_sec"] for a in tries), 3),
                       "search_sec_total": round(sum(a["search_sec"] for a in tries), 3)},
        }
        routes.append(entry)
    total_sec = time.perf_counter() - t0
    g = global_summary(routes, total_sec, route_sec, rep["sec"])
    g.update({
        "planner": plan.planner,
        "order": plan.order,
        "reroute_count": plan.ripups,          # §6.3 reroute_count (rip-up 횟수, 되돌린 것 포함)
        "reverted_count": plan.reverted,
        "fail_class": {c: sum(1 for v in plan.fail_class.values() if v == c) for c in FAIL_CLASSES},
        # timeout·unreachable 구분 유지: 개별 경로는 단독 결과, 간섭은 순차 배치 결과의 상태
        "fail_class_status": dict(sorted(Counter(
            f"{d['class']}:{d['solo_status'] if d['class'] == 'individual' else d['seq_status']}"
            for d in plan.fail_detail.values()).items())),
        "timeout_count": sum(1 for x in routes if x["router"]["status"] == "timeout"),
        "graph_regen_sec_total": round(sum(a["graph_sec"] for t in plan.attempts.values() for a in t), 3),
        "routing_attempts": sum(len(t) for t in plan.attempts.values()),
        "planner_events": plan.events,
        "peak_rss_mb": _peak_rss_mb(),   # 프로세스 최대 메모리 (8단계 밀집 시험)
    })
    return {"scenario": sc.meta.get("name", ""), "routes": routes, "global": g}


def _peak_rss_mb():
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)   # Linux: KB
    except Exception:
        return None


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
    # D57·§9-6: 라우터 추정 서포트 vs 검증기 서포트 (경로 있는 배관)
    est = [(x["router"].get("support_est") or 0.0, x["J"]["support"]) for x in routed]
    rel = sorted((e - v) / v for e, v in est if v > 0)
    sup_err = {
        "est_total": round(sum(e for e, _ in est), 4), "verifier_total": round(sum(v for _, v in est), 4),
        "rel_median": round(rel[len(rel) // 2], 4) if rel else None,
        "rel_max_abs": round(max(abs(x) for x in rel), 4) if rel else None,
        "over": sum(1 for x in rel if x > 0), "under": sum(1 for x in rel if x < 0),
    }
    return {
        "total_pipes": len(routes),
        "routed_count": len(routed),
        "success_count": len(ok),
        "success_rate": round(len(ok) / len(routes), 4) if routes else None,
        "success_rate_routed": round(len(ok) / len(routed), 4) if routed else None,   # D34 기준선 도달 가능 부분집합
        "J_total": jsum(ok, "total"),                                                 # §6.3 성공 배관 합
        "J_breakdown_success": {k: jsum(ok, k) for k in ("pipe", "elbow", "tee", "valve", "support")},
        "J_total_routed": jsum(routed, "total"),
        "real_weight_total": round(sum(x["real_weight"]["total"] for x in ok), 4),          # D49 보완, 성공 배관 합
        "real_weight_total_routed": round(sum(x["real_weight"]["total"] for x in routed), 4),
        "fail_causes": dict(sorted(causes.items())),
        "pairwise_violation": pairwise // 2,                                          # 배관 쌍마다 양쪽에 기록됨
        "computation_time_sec": round(total_sec, 3),
        "routing_time_sec": round(route_sec, 3),
        "verify_time_sec": round(verify_sec, 3),
        "support_estimate": sup_err,
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
          f"| 실중량 (성공 배관 / 경로 있는 배관, 보조 지표) | {g['real_weight_total']:,.2f} / {g['real_weight_total_routed']:,.2f} kg |",
          f"| 다중 배관 | {g.get('planner', '-')} · rip-up {g.get('reroute_count', 0)} (되돌림 {g.get('reverted_count', 0)}) · "
          f"timeout {g.get('timeout_count', 0)} · 라우팅 시도 {g.get('routing_attempts', '-')} · 그래프 재생성 합 {g.get('graph_regen_sec_total', 0):.1f} s |",
          "| 실패 분류 (D51) | " + " · ".join(f"{FAIL_CLASS_KO[c]} {g.get('fail_class', {}).get(c, 0)}" for c in FAIL_CLASSES)
          + (" (" + ", ".join(f"{k} {v}" for k, v in g.get("fail_class_status", {}).items()) + ")"
             if g.get("fail_class_status") else "") + " |",
          f"| 실패 원인 | {', '.join(f'{k} {v}' for k, v in g['fail_causes'].items()) or '없음'} |",
          f"| 배관 간 이격 위반 쌍 | {g['pairwise_violation']} |",
          "| 라우터 추정 서포트 / 검증기 서포트 (D57) | "
          + (f"{g['support_estimate']['est_total']:,.2f} / {g['support_estimate']['verifier_total']:,.2f} kg, "
             f"배관별 상대 오차 중앙 {g['support_estimate']['rel_median']} · 최대 |{g['support_estimate']['rel_max_abs']}| · "
             f"과대 {g['support_estimate']['over']} / 과소 {g['support_estimate']['under']}" if g.get("support_estimate") else "-")
          + " |",
          f"| 계산 시간 | {g['computation_time_sec']:.2f} s (라우팅 {g['routing_time_sec']:.2f} · 검증 {g['verify_time_sec']:.2f}) |",
          ""]
    L += ["## 배관별", "",
          "| 배관 | 구경 | 타입 | 라우터 | 탐색 s | 그래프 s | 판정 | 실패 원인 | 분류 | J total | 직관 | 엘보 | 서포트 | 실중량 | 길이 m | 엘보 90/45 | 서포트 수 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in out["routes"]:
        J, m = x["J"] or {}, x["metrics"]
        fmt = lambda k: f"{J[k]:,.2f}" if J else "-"
        rw = x.get("real_weight") or {}
        cls = FAIL_CLASS_KO.get(x.get("fail_class"), "-")
        rw_s = f"{rw['total']:,.2f}" if rw else "-"
        L.append(f"| {x['pipe_id']} | {x['nominal_size']} | {x['type_id']} | {x['router']['status']} | "
                 f"{x['router']['search_sec']:.2f} | {x['router']['graph_sec']:.2f} | {'PASS' if x['success'] else 'FAIL'} | "
                 f"{', '.join(x['fail_causes']) or '-'} | {cls} | {fmt('total')} | {fmt('pipe')} | {fmt('elbow')} | "
                 f"{fmt('support')} | {rw_s} | {m.get('length_m', '-')} | "
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
    ap.add_argument("--planner", choices=("sequential", "independent"), default="sequential",
                    help="다중 배관 슬롯: sequential = 순서 + rip-up (D50), independent = 배관 단독 (6단계 비교용)")
    ap.add_argument("--router", choices=("layered-a", "layered-b", "astar"), default="layered-a",
                    help="라우터 슬롯: layered-a = 기본 (D55 2층 구조 A), layered-b = 국소 이격선 (비교용), "
                         "astar = 배관마다 escape graph 재생성 (7단계 방식, 비교용)")
    ap.add_argument("--no-support-cost", action="store_true",
                    help="layered 라우터 비용에서 D57 추정 서포트를 뺀다 (M4 이전 비교용)")
    ap.add_argument("--html-only", action="store_true",
                    help="다시 계산하지 않고 -o 폴더의 기존 출력 JSON 으로 3D 리포트 HTML 만 다시 만든다 "
                         "(plotly.js 포함 — 보고 첨부용, §0)")
    args = ap.parse_args(argv)
    if args.html_only:
        for path in args.paths:
            stem = Path(path).stem
            out = json.loads((Path(args.out) / f"{stem}_output.json").read_text(encoding="utf-8"))
            write_report_html(load(path), out, Path(args.out) / f"{stem}_report.html", offline=True)   # 첨부용 (§0)
            print(f"{path}: → {args.out}/{stem}_report.html")
        return 0
    if args.router == "astar":
        router = astar_router
    else:
        router = make_layered_router(local_lines=args.router == "layered-b", support_cost=not args.no_support_cost)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.paths:
        sc = load(path)
        out = run(sc, router=router, time_limit=args.time_limit,
                  planner=sequential_ripup_planner if args.planner == "sequential" else independent_planner)
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


def write_report_html(sc: Scenario, out: dict, path, offline: bool = False) -> None:
    from .router_astar import RouteResult
    from .verifier.core import PipeReport, Violation
    from .viz import add_pipe_tubes, add_routes, add_verification, scenario_figure, write_html
    fig = scenario_figure(sc, connect=False)
    add_pipe_tubes(fig, sc, {x["pipe_id"]: x["waypoints"] for x in out["routes"] if x["waypoints"]})   # §0 보고 규칙
    add_routes(fig, sc, [RouteResult(x["pipe_id"], "ok" if x["waypoints"] else x["router"]["status"],
                                     J=(x["J"] or {}).get("total"), waypoints=x["waypoints"]) for x in out["routes"]],
               line_width=2)   # 중심선은 가늘게 — 관(실제 굵기)이 보이도록
    reps = {x["pipe_id"]: PipeReport(x["pipe_id"], bool(x["waypoints"]),
                                     violations=[Violation(v["module"], x["pipe_id"], v["pos"], v["message"])
                                                 for v in x["violations"]],
                                     supports=x["supports"]) for x in out["routes"]}
    add_verification(fig, {"pipes": reps})
    g = out["global"]
    fig.update_layout(title=dict(text=fig.layout.title.text +
                                 f"<br>통과 {g['success_count']}/{g['total_pipes']} · J_total {g['J_total']:,.1f} kg"))
    write_html(fig, path, offline=offline)


if __name__ == "__main__":
    raise SystemExit(main())
