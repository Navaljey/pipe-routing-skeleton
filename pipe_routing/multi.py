"""7단계: 다중 배관 슬롯 (D50, D3).

planner(scenario, router, time_limit) → PlanResult. 교체 가능한 함수 자리 — V3 가 순서·rip-up 판단을 대신할 수 있다.
router(scenario, pipe, time_limit, placed) → RouteResult 호환. placed = 이미 놓인 배관 [(Pipe, waypoints), ...]
"""
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .constants import ANGLE_TOL_DEG, elbow_radius
from .scenario import Pipe, Scenario
from .verifier.geom import EPS, build_centerline, segment_segment_distance

MAX_RIPUPS = 3   # D50④ 시나리오당
# D51 실패 3분류: 간섭-차단 (단독 경로가 놓인 배관과 충돌) / 간섭-탐색 (충돌 없는데 실패, 성능) / 개별 경로 (단독 실패)
FAIL_CLASSES = ("interference_block", "interference_search", "individual", "router_mismatch")
FAIL_CLASS_KO = {"interference_block": "간섭-차단", "interference_search": "간섭-탐색", "individual": "개별 경로",
                 "router_mismatch": "라우터 규칙 불일치"}   # D60: 간섭-탐색은 timeout 만


@dataclass
class PlanResult:
    routes: dict                                  # pipe_id → RouteResult (최종)
    order: list                                   # 배관 id 순서 (D50①)
    ripups: int = 0                               # 실행한 rip-up 횟수 (되돌린 것 포함)
    reverted: int = 0                             # 되돌린 횟수
    events: list = field(default_factory=list)    # 로그
    fail_class: dict = field(default_factory=dict)   # 최종 실패 배관 → FAIL_CLASSES 중 하나 (D50⑥, D51)
    fail_detail: dict = field(default_factory=dict)  # 최종 실패 배관 → {class, seq_status, solo_status, blockers}
    solo: dict = field(default_factory=dict)      # 단독 라우팅 결과 (rip-up·분류용)
    attempts: dict = field(default_factory=dict)  # pipe_id → [{"status", "graph_sec", "search_sec"}] (D50③)
    total_sec: float = 0.0
    planner: str = ""


# ---------------------------------------------------------------- 기본 라우터 (escape graph + A*)

def astar_router(sc: Scenario, pipe: Pipe, time_limit: float, placed=()):
    from .escape_graph import EscapeGraph
    from .router_astar import astar_route
    g = EscapeGraph(sc, pipe, others=placed)   # 배관마다 그래프 재생성 (D11, D50③)
    r = astar_route(g, pipe, time_limit)
    r.graph_sec = g.build_sec
    return r


def default_order(sc: Scenario) -> list:
    """D50①: 호칭경 내림차순, 같으면 start–end 맨해튼 거리 내림차순."""
    def key(p: Pipe):
        man = sum(abs(a - b) for a, b in zip(p.start.pos, p.end.pos))
        return (-int(p.nominal_size[:-1]), -man)
    return sorted(sc.pipes, key=key)


# ---------------------------------------------------------------- 기하: 배관 쌍 충돌 (검증기와 같은 판정)

def _pieces(pipe: Pipe, wps):
    cl = build_centerline(wps, elbow_radius(pipe.nominal_size), ANGLE_TOL_DEG)
    A, B, S, _ = cl.arrays()
    return A, B, S


def routes_conflict(p1: Pipe, w1, p2: Pipe, w2) -> bool:
    """두 배관 중심선(직관 + 엘보 호)이 r₁ + r₂ 안으로 들어오는가 (검증기 _pipe_hits 와 같은 식)."""
    A1, B1, S1 = _pieces(p1, w1)
    A2, B2, S2 = _pieces(p2, w2)
    lo1, hi1 = np.minimum(A1, B1).min(0), np.maximum(A1, B1).max(0)
    lo2, hi2 = np.minimum(A2, B2).min(0), np.maximum(A2, B2).max(0)
    need = p1.radius + p2.radius + S1.max(initial=0) + S2.max(initial=0)
    if np.any(lo1 > hi2 + need) or np.any(lo2 > hi1 + need):
        return False
    d, _, _ = segment_segment_distance(A1[:, None], B1[:, None], A2[None], B2[None])
    return bool(np.any(d - S1[:, None] - S2[None] < p1.radius + p2.radius - EPS))


# ---------------------------------------------------------------- 플래너

def _route(res: PlanResult, sc, pipe, router, time_limit, placed_items):
    r = router(sc, pipe, time_limit, placed_items)
    res.attempts.setdefault(pipe.id, []).append(
        {"status": r.status, "graph_sec": round(getattr(r, "graph_sec", 0.0), 3), "search_sec": round(r.search_sec, 3),
         "n_obstacle_pipes": len(placed_items)})
    return r


def sequential_ripup_planner(sc: Scenario, router: Callable = astar_router, time_limit: float = 60.0,
                             max_ripups: int = MAX_RIPUPS, order_fn: Callable = default_order) -> PlanResult:
    """D50: 순서대로 깔고(놓인 배관 = 장애물), 실패 배관은 rip-up & reroute.
    D52: 성공 수(경로가 놓인 배관 수)가 엄격히 늘 때만 유지하고, 같거나 줄면 되돌린다."""
    t0 = time.perf_counter()
    pipes = {p.id: p for p in sc.pipes}
    order = [p.id for p in order_fn(sc)]
    res = PlanResult(routes={}, order=order, planner="sequential_ripup")

    def placed_items(placed: dict, exclude=None):
        return [(pipes[pid], placed[pid].waypoints) for pid in order if pid in placed and pid != exclude]

    def lay(seq, placed: dict, routes: dict):
        for pid in seq:
            r = _route(res, sc, pipes[pid], router, time_limit, placed_items(placed))
            routes[pid] = r
            if r.status == "ok":
                placed[pid] = r

    placed: dict = {}
    routes: dict = {}
    lay(order, placed, routes)

    def solo(pid):
        if pid not in res.solo:
            res.solo[pid] = _route(res, sc, pipes[pid], router, time_limit, [])
        return res.solo[pid]

    tried = set()
    while res.ripups < max_ripups:
        failed = [pid for pid in order if pid not in placed and pid not in tried]
        if not failed:
            break
        f = failed[0]
        tried.add(f)
        s = solo(f)
        if s.status != "ok":
            res.events.append(f"{f}: 단독 실패({s.status}) — rip-up 대상 아님")
            continue
        blockers = [pid for pid in order if pid in placed
                    and routes_conflict(pipes[f], s.waypoints, pipes[pid], placed[pid].waypoints)]
        if not blockers:
            res.events.append(f"{f}: 단독 경로와 충돌하는 배관 없음 — rip-up 생략")
            continue
        res.ripups += 1
        before = len(placed)
        snap = (dict(placed), dict(routes))
        for pid in blockers:
            del placed[pid]
        lay([f] + blockers, placed, routes)       # 실패 배관 먼저, 걷어낸 배관은 원래 순서대로 다시
        after = len(placed)
        if after <= before:                      # D52
            placed, routes = snap
            res.reverted += 1
            res.events.append(f"rip-up {res.ripups}: {f} 위해 {blockers} 걷어냄 → 성공 {before}→{after} 증가 없음, 되돌림")
        else:
            res.events.append(f"rip-up {res.ripups}: {f} 위해 {blockers} 걷어냄 → 성공 {before}→{after}")
            tried.clear()                            # 상태가 바뀌었으니 다른 실패 배관을 다시 시도할 수 있다

    # D50⑥ · D51 · D60: 최종 실패 배관 분류
    for pid in order:
        if pid not in placed:
            s = solo(pid)
            blockers = [] if s.status != "ok" else [
                q for q in order if q in placed
                and routes_conflict(pipes[pid], s.waypoints, pipes[q], placed[q].waypoints)]
            seq = routes[pid].status
            if s.status != "ok":
                cls = "individual"
            elif blockers:
                cls = "interference_block"
            elif seq == "timeout":
                cls = "interference_search"
            else:   # D60: 충돌 없는 단독 경로가 있는데 순차에서 unreachable — 라우터 규칙이 검증기와 어긋남
                cls = "router_mismatch"
            detail = {"class": cls, "seq_status": seq, "solo_status": s.status, "blockers": blockers}
            alt = getattr(router, "without_reservations", None)
            if cls == "individual" and alt is not None:   # D59: 예약을 빼면 단독으로 되는가 = 예약끼리 충돌
                r2 = alt(sc, pipes[pid], time_limit, [])
                detail["reservation_conflict"] = r2.status == "ok"
            res.fail_class[pid] = cls
            res.fail_detail[pid] = detail
    res.routes = routes
    res.total_sec = time.perf_counter() - t0
    return res


def independent_planner(sc: Scenario, router: Callable = astar_router, time_limit: float = 60.0) -> PlanResult:
    """6단계 방식 (비교용): 배관마다 빈 상태에서 단독 라우팅. 배관 간 간섭은 검증기만 본다."""
    t0 = time.perf_counter()
    res = PlanResult(routes={}, order=[p.id for p in sc.pipes], planner="independent")
    for p in sc.pipes:
        r = _route(res, sc, p, router, time_limit, [])
        res.routes[p.id] = r
        res.solo[p.id] = r
    res.total_sec = time.perf_counter() - t0
    return res
