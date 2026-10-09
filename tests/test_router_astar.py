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


if __name__ == "__main__":
    unittest.main()
