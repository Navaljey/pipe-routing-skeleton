"""S1-5 (D68): 시간-품질 곡선 도구 + anytime 기준선 플래너.

anytime 기준선 = 순차 배치(D50, first-solution 기준선)로 첫 해 → 남은 예산 동안 무튜닝 범용 개선을 반복하고 최선 해를
유지한다. rip-up 횟수 상한 없음. 무작위 요소는 모두 시드 고정 (D72).

개선 이동 (`추정:` 무튜닝 범용 규칙 — 확률·순서는 고정값, 튜닝하지 않는다):
  ① 실패 배관 rip-up: 실패 배관 하나(무작위)의 단독 경로와 충돌하는 놓인 배관을 걷어내고, 실패 배관 → 걷어낸 배관(무작위 순서)
     으로 다시 깐다 (막는 배관이 없으면 그 배관만 지금 배치에서 다시 깐다)
  ② 최대 J 배관 rip-up: 놓인 배관 중 라우터 비용(J + 추정 서포트)이 가장 큰 배관(이번 상태에서 아직 안 해 본 것)을 같은 방식으로
     — 단독 경로가 지금보다 싸면 그 단독 경로를 막는 배관을 걷어내고 다시 깐다
  ③ 순서 무작위 재시작: 무작위 순서로 처음부터 순차 배치 (rip-up 없음)
  고르는 규칙: 실패 배관이 있으면 ①, 없으면 ②, ①·② 할 것이 없으면 ③. 그 밖에 이동 8번마다 한 번은 ③.
  받아들이는 규칙: 후보가 지금 최선보다 좋으면 (경로 배관 수 많은 쪽, 같으면 라우터 비용 합 작은 쪽) 최선 해로 바꾼다.

두 축 (D68 보완): (i) 결정적 예산 = 라우터 확장 상태 수 누적 (재현용, D72) (ii) 벽시계 초 (비교용, 고정 조건에서만).
궤적 = 최선 해가 바뀔 때마다 (누적 확장, 벽시계, 해). 품질 지표는 끝난 뒤 궤적의 해마다 검증기로 계산한다
(평가 비용은 예산에 넣지 않는다).
"""
import random
import time
from dataclasses import dataclass, field

from .multi import default_order, routes_conflict, sequential_ripup_planner

RESTART_EVERY = 8   # `추정:` 이동 몇 번마다 한 번 순서 무작위 재시작


@dataclass
class Trajectory:
    seed: int
    t0_sec: float = 0.0           # 첫 해(first-solution 기준선) 시간
    e0: int = 0                   # 첫 해까지 누적 확장
    points: list = field(default_factory=list)   # [(누적 확장, 벽시계 초, {pipe_id: RouteResult}, 이동 종류)]
    moves: int = 0
    total_expanded: int = 0
    total_sec: float = 0.0
    stop: str = ""


def _cost(r) -> float:
    return (r.J or 0.0) + (getattr(r, "J_support_est", 0.0) or 0.0)


def _objective(routes: dict):
    ok = [r for r in routes.values() if r.status == "ok"]
    return (len(ok), -sum(_cost(r) for r in ok))


def anytime_run(sc, router, time_limit: float = 60.0, seed: int = 0, max_expanded: int = None,
                max_sec: float = None) -> Trajectory:
    """anytime 기준선 실행. 멈춤 = 누적 확장 ≥ max_expanded (결정적) 또는 벽시계 ≥ max_sec (안전 상한, 결정적이지
    않지만 궤적의 각 점은 결정적 — 같은 시드면 누적 확장 축에서 같은 궤적의 앞부분)."""
    rng = random.Random(seed)
    pipes = {p.id: p for p in sc.pipes}
    base_order = [p.id for p in default_order(sc)]
    tr = Trajectory(seed=seed)
    t0 = time.perf_counter()
    count = [0]

    def call(pid, placed_items):
        r = router(sc, pipes[pid], time_limit, placed_items)
        count[0] += int(r.expanded or 0)
        return r

    def counting_router(sc_, pipe, tl, placed=()):
        r = router(sc_, pipe, tl, placed)
        count[0] += int(r.expanded or 0)
        return r
    if hasattr(router, "without_reservations"):
        counting_router.without_reservations = router.without_reservations

    def done():
        return ((max_expanded is not None and count[0] >= max_expanded)
                or (max_sec is not None and time.perf_counter() - t0 >= max_sec))

    # 첫 해 = first-solution 기준선 (D50, rip-up 3회)
    first = sequential_ripup_planner(sc, router=counting_router, time_limit=time_limit)
    best = dict(first.routes)
    tr.t0_sec, tr.e0 = time.perf_counter() - t0, count[0]
    tr.points.append((count[0], tr.t0_sec, dict(best), "first"))
    solo: dict = {}

    def get_solo(pid):
        if pid not in solo:
            solo[pid] = call(pid, [])
        return solo[pid]

    def items(state, order, exclude=()):
        return [(pipes[q], state[q].waypoints) for q in order if q in state and state[q].status == "ok"
                and q not in exclude]

    def lay(state, seq, order):
        for q in seq:
            if done():
                return False
            state[q] = call(q, items(state, order, exclude=(q,)))
        return True

    tried_max, tried_failed = set(), set()
    while not done():
        tr.moves += 1
        failed = [q for q in base_order if best[q].status != "ok" and q not in tried_failed]
        cand, kind = None, None
        if tr.moves % RESTART_EVERY != 0:
            if failed:
                f = failed[rng.randrange(len(failed))]
                tried_failed.add(f)        # 이번 상태에서 같은 배관을 다시 고르지 않는다 (같은 결과)
                s = get_solo(f)
                if s.status == "ok":
                    kind, target, route = "ripup_failed", f, s
            if kind is None:   # 비용 큰 순서로, 이번 상태에서 아직 안 본 배관 중 단독 경로가 더 싼 첫 배관
                placed = sorted((q for q in base_order if best[q].status == "ok" and q not in tried_max),
                                key=lambda x: (-_cost(best[x]), x))
                for q in placed:
                    tried_max.add(q)
                    s = get_solo(q)
                    if s.status == "ok" and _cost(s) < _cost(best[q]) - 1e-9:
                        kind, target, route = "ripup_maxJ", q, s
                        break
                    if done():
                        break
        if kind is not None:
            blockers = [q for q in base_order if q != target and best[q].status == "ok"
                        and routes_conflict(pipes[target], route.waypoints, pipes[q], best[q].waypoints)]
            cand = dict(best)           # 막는 배관이 없으면 대상만 지금 배치에서 다시 깐다 (예약이 풀렸거나 이웃이 바뀌었을 수 있다)
            cand[target] = _Removed(target)
            for q in blockers:
                cand[q] = _Removed(q)
            rng.shuffle(blockers)
            seq = [target] + blockers
            if not lay(cand, seq, base_order):
                break
        else:
            kind = "restart"
            order = list(base_order)
            rng.shuffle(order)
            cand = {q: _Removed(q) for q in order}
            if not lay(cand, order, order):
                break
        if _objective(cand) > _objective(best):
            best = cand
            tried_max.clear()
            tried_failed.clear()
            tr.points.append((count[0], time.perf_counter() - t0, dict(best), kind))
    tr.total_expanded = count[0]
    tr.total_sec = time.perf_counter() - t0
    tr.stop = "expansions" if (max_expanded is not None and count[0] >= max_expanded) else "wall_clock"
    return tr


class _Removed:
    """걷어낸(아직 다시 깔지 않은) 배관 자리표시."""
    status = "removed"
    waypoints = []
    J = None
    J_support_est = 0.0
    expanded = 0

    def __init__(self, pid):
        self.pipe_id = pid


# ---------------------------------------------------------------- 평가·곡선

def quality(sc, routes: dict, solo_route) -> dict:
    """궤적 한 점의 품질 (검증기 기준, D44·D49): 경로 배관 수 · Layer 0 통과 수 · success_rate · J_total(통과 배관 합) ·
    간섭-차단 수(실패 배관 중 단독 경로가 놓인 배관과 충돌, D51). solo_route(pid) = 단독 경로 (평가용, 예산 밖)."""
    from .pipeline import j_breakdown
    from .verifier import PipeRoute, verify
    pipes = {p.id: p for p in sc.pipes}
    rep = verify(sc, [PipeRoute(pid, r.waypoints if r.status == "ok" else []) for pid, r in routes.items()])
    success, J = 0, 0.0
    for pid, v in rep["pipes"].items():
        if v.success:
            success += 1
            J += j_breakdown(pipes[pid], routes[pid].waypoints, v.fittings, v.supports)["total"]
    ok = [pid for pid, r in routes.items() if r.status == "ok"]
    block = 0
    for pid, r in routes.items():
        if r.status == "ok":
            continue
        s = solo_route(pid)
        if s.status == "ok" and any(routes_conflict(pipes[pid], s.waypoints, pipes[q], routes[q].waypoints) for q in ok):
            block += 1
    n = len(sc.pipes)
    return {"routed": len(ok), "success": success, "success_rate": round(success / n, 4) if n else 0.0,
            "J_total": round(J, 3), "interference_block": block}


def machine_info() -> dict:
    import os
    import platform
    import numpy
    info = {"cpu": platform.processor() or platform.machine(), "cores": os.cpu_count(), "python": platform.python_version(),
            "numpy": numpy.__version__}
    try:
        import numba
        info["numba"] = numba.__version__
    except ImportError:
        info["numba"] = None
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return info


def main(argv=None) -> int:
    import argparse
    import json
    from pathlib import Path

    from .layered import make_layered_router, parse_grid
    from .multi import PlanResult
    from .pipeline import report_md, run, validate_output
    from .scenario import load

    ap = argparse.ArgumentParser(description="S1-5 anytime 기준선 — 시간-품질 궤적 (D68)")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("-o", "--out", default="out/anytime")
    ap.add_argument("--grid", default="g0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-sec", type=float, default=3600.0, help="벽시계 상한 (안전 상한, 기본 60분)")
    ap.add_argument("--max-expanded", type=int, default=None, help="누적 확장 상한 (결정적 멈춤)")
    ap.add_argument("--concurrency", type=int, default=None, help="측정 조건 기록용 (동시 실행 수)")
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    router = make_layered_router(grid=parse_grid(args.grid))
    for path in args.paths:
        sc = load(path)
        stem = Path(path).stem
        tr = anytime_run(sc, router, seed=args.seed, max_expanded=args.max_expanded, max_sec=args.max_sec)
        solo_cache: dict = {}

        def solo_route(pid):
            if pid not in solo_cache:
                p = next(q for q in sc.pipes if q.id == pid)
                solo_cache[pid] = getattr(router, "diagnostic", router)(sc, p, 60.0, [])   # D74 진단 예산
            return solo_cache[pid]
        points = []
        for e, t, routes, kind in tr.points:
            q = quality(sc, routes, solo_route)
            q.update({"expanded": e, "sec": round(t, 3), "move": kind})
            points.append(q)
        best = tr.points[-1][2]
        out = run(sc, router=router, planner=lambda sc_, r_, tl: PlanResult(routes=best, order=list(best),
                                                                             planner="anytime_best"))
        validate_output(out)
        curve = {"scenario": stem, "n_pipes": len(sc.pipes), "grid_variant": args.grid, "seed": args.seed, "search_budget": router.search_budget,
                 "stop": tr.stop, "max_sec": args.max_sec, "max_expanded": args.max_expanded,
                 "t0_sec": round(tr.t0_sec, 3), "e0": tr.e0, "moves": tr.moves, "total_expanded": tr.total_expanded,
                 "total_sec": round(tr.total_sec, 3), "concurrency": args.concurrency, "machine": machine_info(),
                 "points": points}
        (out_dir / f"{stem}_curve.json").write_text(json.dumps(curve, ensure_ascii=False, indent=1), encoding="utf-8")
        (out_dir / f"{stem}_output.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        (out_dir / f"{stem}_report.md").write_text(report_md(out, path), encoding="utf-8")
        print(f"{path}: t0 {tr.t0_sec:.0f}s · 점 {len(points)} · 이동 {tr.moves} · 확장 {tr.total_expanded} · "
              f"최종 경로 {points[-1]['routed']} 통과 {points[-1]['success']} J {points[-1]['J_total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
