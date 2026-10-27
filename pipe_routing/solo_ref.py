"""D101 Σ 단독 J 참고치 — 다중 배치 결과와 비교할 배관별 단독 경로 J (보고용, 예산 밖).

정의: 각 배관을 다른 배관·단자 예약(D59)·부재 예약(D90) 없이 단독으로 기준선 A*(그 세트의 상한, D73·D92 ③)로
라우팅하고, 그 경로를 단독으로(다른 배관 없이) 검증해 Layer 0·J 를 구한다.

보고: 배관별 solo_J · 간격 = Σ J_다중 / Σ J_단독 − 1 ("단독 통과 ∧ 다중 통과" 배관만) · 단독 실패 수.
공유 서포트(D 묶음 D82 · 함께 받치기 D89)가 다중 결과에 있으면 "하한 아님, 참고치" 로 표시한다 — 공유 서포트는
배관별 J 를 단독 최경량 형식보다 낮출 수 있다. 공유 서포트가 없어도 엄밀한 하한은 아니다: 기준선 A* 는 라우터 비용
(D80 ⑨ 서포트 추정 포함) 최적이고, 판정 J 는 검증기 값이다 (D99 와 같은 이유).

결정적이다 (A* 결정적, 예산 = 확장 수). 실패 분류(D74)에서 이미 구한 예약 없는 단독 결과는 결과 불변일 때만 다시 쓴다:
그 결과가 상한 소진이 아니고 확장 수가 이 세트 상한보다 작으면, 상한이 달라도 같은 탐색이 같은 곳에서 끝난다.

python -m pipe_routing.solo_ref out/run/*_output.json   # 기존 출력에 사후 추가 (출력 파일을 고쳐 쓴다)
"""
import statistics
import time

from .layered import default_budget


def solo_routes(sc, router, reuse: dict = None) -> tuple:
    """배관마다 예약 없는 단독 경로. 반환 ({pipe_id: RouteResult}, 재사용 수)."""
    alt = getattr(router, "without_reservations", router)
    cap = default_budget(sc)
    out, reused = {}, 0
    for p in sc.pipes:
        r = (reuse or {}).get(p.id)
        if r is not None and r.status != "timeout" and int(r.expanded or 0) < cap:
            out[p.id] = r
            reused += 1
            continue
        out[p.id] = alt(sc, p, 60.0, [])
    return out, reused


def solo_scores(sc, solo: dict, std) -> dict:
    """단독 경로를 단독으로 검증 → {pipe_id: {status, expanded, pass, J}} (J = §5 분해 total, 경로 없으면 None)."""
    from .pipeline import j_breakdown
    from .verifier import PipeRoute, verify
    pipes = {p.id: p for p in sc.pipes}
    out = {}
    for pid, r in solo.items():
        row = {"status": r.status, "expanded": int(r.expanded or 0), "pass": False, "J": None}
        if r.status == "ok":
            v = verify(sc, [PipeRoute(pid, r.waypoints)], std=std)["pipes"][pid]
            row["pass"] = bool(v.success)
            row["J"] = j_breakdown(pipes[pid], r.waypoints, v.fittings, v.supports, sc)["total"]
        out[pid] = row
    return out


def summary(routes: list, scores: dict, shared: bool, sec: float, reused: int) -> dict:
    """global.solo_reference. routes = 출력 routes[] (다중 결과)."""
    both = [x for x in routes if x["success"] and scores[x["pipe_id"]]["pass"]]
    jm = sum(x["J"]["total"] for x in both)
    js = sum(scores[x["pipe_id"]]["J"] for x in both)
    ratios = sorted(x["J"]["total"] / scores[x["pipe_id"]]["J"] - 1 for x in both if scores[x["pipe_id"]]["J"] > 0)

    def q(p):
        return round(ratios[min(len(ratios) - 1, int(p * (len(ratios) - 1) + 0.5))], 4) if ratios else None
    return {
        "definition": "D101",
        "note": "하한 아님, 참고치 (공유 서포트 있음)" if shared else "참고치 (라우터 비용 최적 단독 경로의 검증기 J)",
        "shared_support": shared,
        "both_pass": len(both),
        "J_multi": round(jm, 3), "J_solo": round(js, 3),
        "gap": round(jm / js - 1, 4) if js > 0 else None,
        "gap_pipe": {"median": round(statistics.median(ratios), 4) if ratios else None, "p10": q(0.1), "p90": q(0.9),
                     "min": ratios[0] if ratios else None, "max": ratios[-1] if ratios else None,
                     "negative": sum(1 for x in ratios if x < -1e-9)},
        "solo_fail": sum(1 for s in scores.values() if not s["pass"]),
        "solo_fail_router": sum(1 for s in scores.values() if s["status"] != "ok"),
        "solo_fail_status": {k: sum(1 for s in scores.values() if s["status"] == k) for k in ("unreachable", "timeout")},
        "multi_pass_solo_fail": sum(1 for x in routes if x["success"] and not scores[x["pipe_id"]]["pass"]),
        "sec": round(sec, 3), "reused": reused,
    }


def attach(out: dict, sc, router, std, reuse: dict = None) -> None:
    """출력 dict 에 routes[].solo (배관별) 와 global.solo_reference 를 더한다."""
    t0 = time.perf_counter()
    solo, reused = solo_routes(sc, router, reuse)
    scores = solo_scores(sc, solo, std)
    sec = time.perf_counter() - t0
    for x in out["routes"]:
        s = scores[x["pipe_id"]]
        x["solo"] = {"status": s["status"], "expanded": s["expanded"], "pass": s["pass"],
                     "J": round(s["J"], 4) if s["J"] is not None else None}
    g = out["global"]
    shared = bool(g.get("shared_support_count")) or any(
        s.get("origin") == "D89" for x in out["routes"] for s in x["supports"])
    g["solo_reference"] = summary(out["routes"], scores, shared, sec, reused)


def main(argv=None) -> int:
    import argparse
    import json
    import sys
    from pathlib import Path

    from .layered import make_layered_router, parse_grid
    from .scenario import load
    from .support_diag import scenario_path
    from .support_std import load_standard
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("outputs", nargs="+")
    ap.add_argument("--grid", default=None, help="격자 변형 (기본 = 출력 global.grid_variant)")
    a = ap.parse_args(argv)
    std = load_standard()
    if std.source != "private":
        print("경고: 서포트 선정표 = 공개 임시 표 (PIPE_SUPPORT_STANDARD 없음). 측정·비교용 결과가 아니다.", file=sys.stderr)
    routers = {}
    for f in a.outputs:
        out = json.loads(Path(f).read_text(encoding="utf-8"))
        grid = a.grid or out["global"].get("grid_variant") or "g0"
        router = routers.setdefault(grid, make_layered_router(grid=parse_grid(grid)))
        stem = Path(f).name[: -len("_output.json")]
        sc = load(scenario_path(stem))
        attach(out, sc, router, std)
        Path(f).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        s = out["global"]["solo_reference"]
        print(f"{stem}: 간격 {s['gap']} (배관 중앙 {s['gap_pipe']['median']}, {s['both_pass']}배관) · 단독 실패 "
              f"{s['solo_fail']} (경로 없음 {s['solo_fail_router']}) · {s['sec']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
