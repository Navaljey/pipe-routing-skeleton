"""S1-3 시험: D65 고정 장애물 꺾임점 엘보 호 판정."""
import math
import unittest

from pipe_routing import astar_fast
from pipe_routing.layered import LayeredGraph, _route, clear_cache
from pipe_routing.router_astar import astar_route
from pipe_routing.scenario import from_dict
from pipe_routing.verifier import PipeRoute, verify

Y0, Z0, X1 = 20000, 1500, 20000


def scen(c):
    """100A (r 117.15, t₉₀ 150) 가 x 로 가다가 x = X1 에서 +y 로 꺾는 L 경로. 꺾임 바깥쪽 모서리에 박스 —
    박스 모서리 (X1 + c, Y0 − c): 꺾임점 이격 = c√2."""
    pipe = {"id": "P", "type_id": "T1", "nominal_size": "100A", "gravity_pipe": False, "min_slope": None,
            "insulation_thickness": 50,
            "start": {"pos": [0, Y0, Z0], "kind": "boundary", "dir": [1, 0, 0]},
            "end": {"pos": [X1, 40000, Z0], "kind": "boundary", "dir": [0, 1, 0]},
            "valve_positions": [], "branch_points": []}
    box = {"id": "B", "type": "box", "min": [X1 + c, Y0 - 3000, 0], "max": [X1 + 3000, Y0 - c, 4000]}
    return from_dict({"block": {"width": 40000, "length": 40000, "height": 10000, "unit": "mm"},
                      "obstacles": [box], "pipes": [pipe]})


L_PATH = [[0.0, Y0, Z0], [X1, Y0, Z0], [X1, 40000.0, Z0]]


@unittest.skipUnless(astar_fast.HAVE_NUMBA, "numba 없음")
class CornerArcFixedTest(unittest.TestCase):
    def test_corner_inside_r_but_arc_clear(self):
        """c = 50: 꺾임점 이격 70.7 < r 이지만 엘보 호·깎은 직관은 r 이상 → L 경로 허용, 검증기 위반 0."""
        clear_cache()
        sc = scen(50)
        p = sc.pipes[0]
        g = LayeredGraph(sc, p, support_cost=False)
        corner = g._id_of([X1, Y0, Z0])
        self.assertLess(g.clearance[corner], p.radius)            # 예전 규칙(D38)이면 노드 무효
        self.assertTrue(g.node_valid[corner])
        r = _route(g, p, 60)
        self.assertEqual(r.status, "ok")
        self.assertEqual([list(map(float, w)) for w in r.waypoints], L_PATH)
        rep = verify(sc, [PipeRoute("P", r.waypoints)], modules=["collision", "bend", "boundary"])
        self.assertEqual([v.message for v in rep["pipes"]["P"].violations], [])
        # 꺾지 않고 그 노드를 지나가는 직관은 여전히 r 이상이어야 한다: 노드에서 직진(등급 0)으로 나가는 엣지는 막힘
        d_px = 0   # +x
        self.assertTrue(g.step[corner, d_px] < 0 or (int(g.emask[corner, d_px]) & 0xF) == 0)

    def test_arc_blocked_corner_rejected(self):
        """c = 0: 엘보 호 중점이 박스 모서리에서 R(√2 − 1) ≈ 62 < r → L 경로 불가. 찾은 경로는 검증기 통과."""
        clear_cache()
        sc = scen(0)
        p = sc.pipes[0]
        r = _route(LayeredGraph(sc, p, support_cost=False), p, 60)
        self.assertEqual(r.status, "ok")
        self.assertNotEqual([list(map(float, w)) for w in r.waypoints], L_PATH)
        rep = verify(sc, [PipeRoute("P", r.waypoints)], modules=["collision", "bend", "boundary"])
        self.assertEqual([v.message for v in rep["pipes"]["P"].violations], [])

    def test_matches_dijkstra(self):
        """D65 마스크가 들어간 그래프에서도 컴파일 A* J = 가지치기 없는 파이썬 Dijkstra J (D40①)."""
        for c in (50, 0):
            clear_cache()
            sc = scen(c)
            p = sc.pipes[0]
            g = LayeredGraph(sc, p)
            fast = _route(g, p, 60)
            dij = astar_route(g, p, 300, heuristic=lambda s: 0.0)
            self.assertEqual(fast.status, dij.status)
            self.assertTrue(math.isclose(fast.J, dij.J, rel_tol=1e-9, abs_tol=1e-6), (c, fast.J, dij.J))


if __name__ == "__main__":
    unittest.main()
