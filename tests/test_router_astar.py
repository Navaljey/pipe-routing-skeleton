import heapq
import itertools
import json
import math
import unittest
from pathlib import Path

from pipe_routing.constants import elbow_kg
from pipe_routing.escape_graph import EscapeGraph
from pipe_routing.router_astar import astar_route, heuristic_factory
from pipe_routing.scenario import from_dict, load
from pipe_routing.space import DIRS

ROOT = Path(__file__).resolve().parent.parent
MANUAL = ROOT / "scenarios" / "manual" / "manual_01.json"


def plain_dijkstra(g, pipe) -> float:
    """가지치기 없는 Dijkstra — A* 최적성 대조용 (D40①)."""
    s0 = g.start_state()
    dist = {s0: 0.0}
    heap = [(0.0, 0, s0)]
    tie = itertools.count(1)
    while heap:
        d, _, s = heapq.heappop(heap)
        if d > dist[s] + 1e-9:
            continue
        if g.is_goal(s):
            return d
        for nb in g.neighbors(s):
            nd = d + g.cost(s, nb)
            if nd + 1e-9 < dist.get(nb, math.inf):
                dist[nb] = nd
                heapq.heappush(heap, (nd, next(tie), nb))
    return math.inf


class AStarTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sc = load(MANUAL)
        cls.graphs = {p.id: EscapeGraph(cls.sc, p) for p in cls.sc.pipes}
        cls.results = {p.id: astar_route(cls.graphs[p.id], p) for p in cls.sc.pipes}

    def test_manual_all_routed(self):
        for pid, r in self.results.items():
            self.assertEqual(r.status, "ok", pid)

    def test_optimal_vs_plain_dijkstra(self):
        """A*(휴리스틱 + 지배 가지치기) J = 가지치기 없는 Dijkstra J."""
        for p in self.sc.pipes:
            self.assertAlmostEqual(self.results[p.id].J, plain_dijkstra(self.graphs[p.id], p), places=6, msg=p.id)

    def test_heuristic_admissible_on_path(self):
        for p in self.sc.pipes:
            g, r = self.graphs[p.id], self.results[p.id]
            h = heuristic_factory(g, p)
            gs = 0.0
            for i, s in enumerate(r.states):
                if i:
                    gs += g.cost(r.states[i - 1], s)
                self.assertLessEqual(h(s), r.J - gs + 1e-6, f"{p.id} 상태 {i}")

    def test_result_consistency(self):
        for p in self.sc.pipes:
            r = self.results[p.id]
            self.assertEqual(r.waypoints[0], list(map(float, p.start.pos)))
            self.assertEqual(r.waypoints[-1], list(map(float, p.end.pos)))
            self.assertEqual(DIRS[r.states[1].dir], p.start.dir)    # D29 출발 방향
            self.assertEqual(DIRS[r.states[-1].dir], p.end.dir)     # D29 도착 방향
            self.assertAlmostEqual(r.J, r.J_pipe + r.J_elbow)
            expect = sum(n * elbow_kg(p.nominal_size, int(a)) for a, n in r.bends.items())
            self.assertAlmostEqual(r.J_elbow, expect, places=6)
            self.assertEqual(len(r.waypoints), 2 + sum(r.bends.values()))

    def test_unreachable(self):
        """블록을 가르는 벽 → 탐색 완료 후 경로 없음 = unreachable (D40②)."""
        data = json.loads(MANUAL.read_text(encoding="utf-8"))
        data["obstacles"] = [{"id": "WALL", "type": "wall", "min": [20000, 0, 0], "max": [20100, 40000, 10000]}]
        data["pipes"] = [data["pipes"][3]]   # P004: x=40000 경계에서 출발 → 벽 건너편 x=0 경계로
        data["pipes"][0]["end"] = {"pos": [0, 15000, 2500], "kind": "boundary", "dir": [-1, 0, 0]}
        sc = from_dict(data)
        r = astar_route(EscapeGraph(sc, sc.pipes[0]), sc.pipes[0])
        self.assertEqual(r.status, "unreachable")
        self.assertGreater(r.expanded, 0)

    def test_timeout(self):
        sc = load(ROOT / "scenarios" / "procedural" / "proc_000.json")
        p = sc.pipes[4]
        r = astar_route(EscapeGraph(sc, p), p, time_limit=0.0)
        self.assertEqual(r.status, "timeout")
        self.assertGreaterEqual(r.expanded, 2048)

    def test_no_45_never_cheaper(self):
        """45° 엣지를 빼면 그래프가 부분집합이므로 최적 J 는 같거나 커진다 (M9 비교 실험의 전제)."""
        for p in self.sc.pipes:
            r = astar_route(EscapeGraph(self.sc, p, allow_45=False), p)
            self.assertEqual(r.length_45_mm, 0)
            self.assertGreaterEqual(r.J + 1e-6, self.results[p.id].J)

    def test_d45_neighbors_respect_elbow_tangents(self):
        """모든 이웃 이동: 꺾을 때 직관 ≥ max(§3.3, t_직전 + t_이번) (D45 ①②)."""
        from pipe_routing.constants import elbow_tangent
        from pipe_routing.space import DEFLECTION
        from collections import deque
        for p in self.sc.pipes:
            g = self.graphs[p.id]
            s0 = g.start_state()
            seen, q = {s0}, deque([s0])
            turns = 0
            while q and len(seen) < 4000:
                st = q.popleft()
                for nb in g.neighbors(st):
                    defl = DEFLECTION[st.dir][nb.dir]
                    if defl:
                        turns += 1
                        need = max(p.min_straight, elbow_tangent(p.nominal_size, st.bend)
                                   + elbow_tangent(p.nominal_size, defl))
                        self.assertGreaterEqual(st.run + 1e-6, need)
                        self.assertEqual(nb.bend, defl)
                    if nb not in seen:
                        seen.add(nb); q.append(nb)
            self.assertGreater(turns, 0)

    def test_d45_goal_needs_tangent(self):
        """end 단자 도착 판정: 마지막 꺾임 이후 직관 ≥ 그 엘보 접선 (D45 ③)."""
        from pipe_routing.constants import elbow_tangent
        from pipe_routing.space import DIR_INDEX, State
        p = self.sc.pipes[0]
        g = self.graphs[p.id]
        d = DIR_INDEX[p.end.dir]
        t = elbow_tangent(p.nominal_size, 90)
        self.assertFalse(g.is_goal(State(g.end_node, d, t - 1, 90)))
        self.assertTrue(g.is_goal(State(g.end_node, d, t, 90)))
        self.assertTrue(g.is_goal(State(g.end_node, d, 0.0, 0)))

    def test_routes_pass_verifier_straight_lengths(self):
        """A* 경로를 검증기(실제 엘보 형상)로 판정해도 직관 길이 위반이 없다 — 라우터·검증기 규칙 정합 (D45)."""
        from pipe_routing.verifier import PipeRoute, verify
        for p in self.sc.pipes:
            r = self.results[p.id]
            rep = verify(self.sc, [PipeRoute(p.id, r.waypoints)], modules=["bend"])["pipes"][p.id]
            self.assertEqual([v.message for v in rep.violations if "직관" in v.message], [], p.id)

    def test_d47_deck_start_turn_needs_r_plus_t(self):
        """경계 단자(하부 데크) start: 첫 꺾임 전 직관 ≥ max(§3.3, r + t) (D47)."""
        from pipe_routing.space import DIR_INDEX, State
        data = json.loads(MANUAL.read_text(encoding="utf-8"))
        data["obstacles"] = []
        d = data["pipes"][0]
        d["nominal_size"] = "25A"   # r 77 + t90 37.5 = 114.5 > §3.3 100
        d["start"] = {"pos": [20000, 5000, 0], "kind": "boundary", "dir": [0, 0, 1]}
        d["end"] = {"pos": [40000, 5000, 5000], "kind": "boundary", "dir": [1, 0, 0]}
        data["pipes"] = [d]
        sc = from_dict(data)
        p = sc.pipes[0]
        g = EscapeGraph(sc, p)
        self.assertAlmostEqual(g.turn_need(g.start_state(), 90), p.radius + 37.5)
        self.assertAlmostEqual(g.turn_need(State(g.start_node, 0, 0.0, 90), 90), 100)   # 첫 꺾임 이후는 D45 그대로
        r = astar_route(g, p)
        self.assertEqual(r.status, "ok")
        first = r.waypoints[1][2] - r.waypoints[0][2]
        self.assertGreaterEqual(first + 1e-6, p.radius + 37.5)

    def test_d48_elbow_into_obstacle_corner_refused(self):
        """직관은 이격을 지키지만 엘보 호가 장애물 안쪽 모서리를 파고드는 꺾임은 neighbors 에서 빠진다 (D48)."""
        from pipe_routing.space import DIR_INDEX, State
        data = json.loads(MANUAL.read_text(encoding="utf-8"))
        d = data["pipes"][0]
        d["start"] = {"pos": [0, 5000, 1000], "kind": "boundary", "dir": [1, 0, 0]}
        d["end"] = {"pos": [40000, 5000, 1000], "kind": "boundary", "dir": [1, 0, 0]}
        data["pipes"] = [d]
        corner = {"id": "C", "type": "box", "min": [15000, 5120, 0], "max": [19880, 9000, 2000]}
        state_dirs = (DIR_INDEX[(1, 0, 0)], DIR_INDEX[(0, 1, 0)])
        far = {"id": "F", "type": "box", "min": [15000, 30000, 0], "max": [19880, 31000, 2000]}   # 같은 x 격자선만 만든다
        for obstacles, allowed in (([corner], False), ([far], True)):
            data["obstacles"] = obstacles
            sc = from_dict(data)
            g = EscapeGraph(sc, sc.pipes[0])
            node = g.node_of((20000, 5000, 1000))
            st = State(node, state_dirs[0], 5000.0, 0)
            dirs = {nb.dir for nb in g.neighbors(st)}
            self.assertEqual(state_dirs[1] in dirs, allowed)

    def test_routes_pass_verifier_boundary_and_bend_obstacles(self):
        """A* 경로를 검증기로 판정: boundary 위반 0, 장애물 기인 bend 위반 0 (D47·D48). 배관 간 간섭은 7단계 대상이라 제외."""
        from pipe_routing.verifier import PipeRoute, verify
        for path in (MANUAL, ROOT / "scenarios" / "procedural" / "proc_000.json"):
            sc = load(path)
            pipes = sc.pipes if path == MANUAL else [p for p in sc.pipes
                                                     if "boundary" in (p.start.kind, p.end.kind)][:3]
            for p in pipes:
                r = astar_route(EscapeGraph(sc, p), p)
                self.assertEqual(r.status, "ok")
                rep = verify(sc, [PipeRoute(p.id, r.waypoints)], modules=["boundary", "bend"])["pipes"][p.id]
                self.assertEqual([v.message for v in rep.violations], [], f"{path.name} {p.id}")


if __name__ == "__main__":
    unittest.main()
