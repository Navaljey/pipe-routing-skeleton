"""S0 라우터: 단일 배관 A* (CLAUDE.md §8 4단계, D40·D41).

라우터 슬롯 구현이다. 표현은 SpaceRepresentation 인터페이스(§4.1, D37)로만 본다 (D9).
  - 비용 g = J (직관 kg + 엘보 kg, §5). 서포트는 검증기에서 사후 산정
  - 휴리스틱 h = 허용(admissible) 하한 (D40①, D41) → 반환 경로는 그 표현 위에서 J 최적
  - 지배 상태 가지치기: 같은 (노드, 방향)에서 직진 길이 ≥ 이고 g ≤ 인 상태가 이미 있으면 버린다 (최적성 보존)
  - 시간 제한 60초 → timeout, 탐색 완료 후 경로 없음 → unreachable (D40②, D34)
"""
import argparse
import heapq
import itertools
import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

from .constants import FITTINGS, PIPE_SPECS
from .scenario import Pipe, load
from .space import DEFLECTION, DIRS, SpaceRepresentation, State

TIME_LIMIT_SEC = 60.0   # D40② (추정: 4단계 실측 후 조정)
_CHECK_EVERY = 2048     # 확장 몇 번마다 시간 확인


@dataclass
class RouteResult:
    pipe_id: str
    status: str                       # "ok" | "unreachable" | "timeout"
    J: Optional[float] = None         # 직관 + 엘보 (kg). 서포트 제외
    J_pipe: Optional[float] = None
    J_elbow: Optional[float] = None
    length_mm: Optional[float] = None
    waypoints: list = field(default_factory=list)   # 꺾이는 점 (mm), start·end 포함
    bends: dict = field(default_factory=dict)       # {"45": n, "90": n, "135": n}
    length_45_mm: float = 0.0                       # 45° 구간 길이 (M9 실측)
    n_segments_45: int = 0
    expanded: int = 0                 # 확장(팝) 상태 수
    generated: int = 0                # 큐에 넣은 상태 수
    search_sec: float = 0.0
    states: list = field(default_factory=list, repr=False)

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("states")
        return d


def heuristic_factory(space: SpaceRepresentation, pipe: Pipe):
    """허용 하한 h(상태) (D41).

    h = 남은 유클리드 직선거리 × kg/m + 남은 최소 꺾임 수 × 45° 엘보 중량
      · 경로 길이 ≥ 직선거리, 엘보 중량 ≥ 45° 엘보 (§3.2 표에서 모든 구경 e45 ≤ e90) → 과대추정 없음
      · 최소 꺾임 수: 현재 방향 ≠ 도착 방향(end dir) 이면 ≥1.
        같은데 목표가 전방 반직선 위에 없으면 벗어났다 돌아와야 하므로 ≥2. 그 밖에는 0
    """
    goal = tuple(map(float, pipe.end.pos))
    kgmm = PIPE_SPECS[pipe.nominal_size].kg_per_m / 1000
    e45 = FITTINGS[pipe.nominal_size].elbow45
    end_dir = tuple(pipe.end.dir)

    def h(state: State) -> float:
        p = space.position(state.node)
        dist = math.dist(p, goal)
        d = DIRS[state.dir]
        if d != end_dir:
            bends = 1
        elif dist == 0:
            bends = 0
        else:
            # 목표가 p + t·d (t>0) 위에 있는가
            v = [g - q for g, q in zip(goal, p)]
            t = sum(a * b for a, b in zip(v, d)) / sum(a * a for a in d)
            on_ray = t > 0 and all(abs(v[i] - t * d[i]) < 1e-6 for i in range(3))
            bends = 0 if on_ray else 2
        return dist * kgmm + bends * e45

    return h


def astar_route(space: SpaceRepresentation, pipe: Pipe, time_limit: float = TIME_LIMIT_SEC,
                heuristic=None) -> RouteResult:
    """heuristic=None 이면 heuristic_factory (D41). 시험용으로 lambda s: 0 (Dijkstra) 을 넣을 수 있다."""
    t0 = time.perf_counter()
    h = heuristic or heuristic_factory(space, pipe)
    start = space.start_state(pipe)
    tie = itertools.count()
    open_heap = [(h(start), next(tie), 0.0, start)]
    g_best = {start: 0.0}
    parent: dict[State, Optional[State]] = {start: None}
    # 지배 가지치기: (노드, 방향) → [(run, bend, g), ...] 확정(팝)된 상태.
    # run 이 크거나 같고 bend(직전 엘보 편향각)가 작거나 같으면 이후 허용 이동이 포함관계로 넓다 (State 설명, D45)
    closed: dict[tuple, list[tuple[float, int, float]]] = {}
    expanded = generated = 0

    def dominated(s: State, g: float) -> bool:
        for run, bend, gc in closed.get((s.node, s.dir), ()):
            if run >= s.run and bend <= s.bend and gc <= g + 1e-9:
                return True
        return False

    while open_heap:
        f, _, g, s = heapq.heappop(open_heap)
        if g > g_best.get(s, math.inf) + 1e-9:
            continue   # 낡은 큐 항목
        if dominated(s, g):
            continue
        closed.setdefault((s.node, s.dir), []).append((s.run, s.bend, g))
        expanded += 1
        if space.is_goal(s, pipe):
            return _finish(space, pipe, s, parent, expanded, generated, time.perf_counter() - t0)
        if expanded % _CHECK_EVERY == 0 and time.perf_counter() - t0 > time_limit:
            return RouteResult(pipe.id, "timeout", expanded=expanded, generated=generated,
                               search_sec=time.perf_counter() - t0)
        for nb in space.neighbors(s, pipe):
            ng = g + space.cost(s, nb, pipe)
            if ng + 1e-9 < g_best.get(nb, math.inf) and not dominated(nb, ng):
                g_best[nb] = ng
                parent[nb] = s
                heapq.heappush(open_heap, (ng + h(nb), next(tie), ng, nb))
                generated += 1
    return RouteResult(pipe.id, "unreachable", expanded=expanded, generated=generated,
                       search_sec=time.perf_counter() - t0)


def _finish(space, pipe, goal_state, parent, expanded, generated, sec) -> RouteResult:
    states = []
    s = goal_state
    while s is not None:
        states.append(s)
        s = parent[s]
    states.reverse()
    kgmm = PIPE_SPECS[pipe.nominal_size].kg_per_m / 1000
    pts = [space.position(s.node) for s in states]
    length = sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
    J = sum(space.cost(a, b, pipe) for a, b in zip(states, states[1:]))
    bends = {"45": 0, "90": 0, "135": 0}
    waypoints = [pts[0]]
    for i in range(1, len(states)):
        defl = DEFLECTION[states[i - 1].dir][states[i].dir]
        if defl:
            bends[str(defl)] += 1
            waypoints.append(pts[i - 1])
    waypoints.append(pts[-1])
    len45 = 0.0
    n45 = 0
    for i in range(1, len(states)):
        if states[i].dir >= 6:
            len45 += math.dist(pts[i - 1], pts[i])
            if states[i - 1].dir != states[i].dir or i == 1:
                n45 += 1
    J_pipe = length * kgmm
    return RouteResult(pipe.id, "ok", J=J, J_pipe=J_pipe, J_elbow=J - J_pipe, length_mm=length,
                       waypoints=[list(p) for p in waypoints], bends=bends, length_45_mm=len45,
                       n_segments_45=n45, expanded=expanded, generated=generated, search_sec=sec,
                       states=states)


def main(argv=None) -> int:
    from .escape_graph import EscapeGraph
    ap = argparse.ArgumentParser(description="단일 배관 A* (4단계) — 배관별 경로·J·탐색 통계")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--pipe", help="이 배관만")
    ap.add_argument("--time-limit", type=float, default=TIME_LIMIT_SEC)
    ap.add_argument("--json", help="결과를 JSON 줄 단위로 저장 (append)")
    ap.add_argument("--viz", metavar="DIR", help="경로를 그린 HTML 을 이 폴더에 저장")
    ap.add_argument("--no-45", action="store_true", help="45° 엣지 없이 (M9 비교 실험)")
    args = ap.parse_args(argv)
    for path in args.paths:
        sc = load(path)
        print(f"== {path}")
        results = []
        for p in sc.pipes:
            if args.pipe and p.id != args.pipe:
                continue
            g = EscapeGraph(sc, p, allow_45=not args.no_45)
            r = astar_route(g, p, args.time_limit)
            results.append(r)
            extra = (f"J {r.J:9.2f} kg (직관 {r.J_pipe:.2f} + 엘보 {r.J_elbow:.2f})  길이 {r.length_mm / 1000:7.2f} m  "
                     f"꺾임 {r.bends}  45° {r.length_45_mm / 1000:.2f} m" if r.status == "ok" else "")
            print(f"  {p.id:5s} {p.nominal_size:>5s} {r.status:11s} 확장 {r.expanded:>8d}  "
                  f"그래프 {g.build_sec:5.2f}s 탐색 {r.search_sec:6.2f}s  {extra}")
            if args.json:
                with open(args.json, "a", encoding="utf-8") as f:
                    row = {"scenario": sc.meta.get("name", path), "size": p.nominal_size,
                           "type_id": p.type_id, "graph_sec": g.build_sec, **g.stats(), **r.summary()}
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if args.viz:
            from pathlib import Path
            from .viz import add_routes, scenario_figure, write_html
            out = Path(args.viz)
            out.mkdir(parents=True, exist_ok=True)
            fig = scenario_figure(sc, connect=False)
            add_routes(fig, sc, results)
            dst = out / (Path(path).stem + "_routes.html")
            write_html(fig, dst)
            print(f"  → {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
