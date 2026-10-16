"""S1-6 라우터 서포트 모델 (D80 ⑨) — 격자선 서포트 가능 구간, 상태 u 갱신, 라우터–검증기 정합."""
import math
import unittest

import numpy as np

from pipe_routing import astar_fast
from pipe_routing import support_std as ss
from pipe_routing.constants import SUPPORT_SPACING
from pipe_routing.layered import LayeredGraph, _route, clear_cache
from pipe_routing.router_astar import astar_route
from pipe_routing.scenario import from_dict
from pipe_routing.support_router import NONE, LineSupport, pack, reverse, support_step, unpack
from pipe_routing.verifier import PipeRoute, verify

EXT = [40000.0, 40000.0, 10000.0]
STD = ss.public_approx()   # 범위 3,000 (공개 임시 표)


def ls(boxes=()):
    lo = [b[0] for b in boxes] or np.zeros((0, 3))
    hi = [b[1] for b in boxes] or np.zeros((0, 3))
    return LineSupport(STD, EXT, lo, hi, "100A", 114.3, SUPPORT_SPACING["100A"])


class StepTest(unittest.TestCase):
    def test_pack_roundtrip_and_reverse(self):
        p = pack([10.2], [20.0], [30.0])
        self.assertEqual([int(v[0]) for v in unpack(p)], [11, 20, 30])   # 올림 (보수적)
        self.assertEqual([int(v[0]) for v in unpack(reverse(p))], [20, 11, 30])
        n = pack([NONE], [0], [0])
        self.assertEqual(int(reverse(n)[0]), int(n[0]))

    def test_support_step(self):
        smax = 3000.0
        none = int(pack([NONE], [0], [0])[0])
        self.assertEqual(support_step(0.5, none, 1500, smax), (1.0, True))
        self.assertFalse(support_step(0.6, none, 1500, smax)[1])
        p = int(pack([600], [900], [0])[0])
        self.assertEqual(support_step(0.8, p, 5000, smax), (0.3, True))      # 0.8 + 0.2 ≤ 1 → 새 u = 900/3000
        self.assertFalse(support_step(0.9, p, 5000, smax)[1])
        g = int(pack([0], [0], [3100])[0])
        self.assertFalse(support_step(0.0, g, 8000, smax)[1])               # 엣지 안 불가 구간 > 최대 간격
        self.assertEqual(support_step(0.7, 0, 1000, smax), (0.0, True))      # 0 = 제약 없음


class LineTest(unittest.TestCase):
    def test_band(self):
        L = ls()
        P0 = np.array([[1000, 20000, 1000.0], [1000, 20000, 5000.0]])
        P1 = np.array([[9000, 20000, 1000.0], [9000, 20000, 5000.0]])
        cost, pk = L.axis_edges(0, P0, P1)
        a, b, g = unpack(pk)
        self.assertEqual((int(a[0]), int(b[0]), int(g[0])), (0, 0, 0))     # 바닥 C 범위 안
        self.assertEqual(int(a[1]), NONE)                                     # 중앙 높이: 어느 면도 범위 밖
        self.assertAlmostEqual(cost[0], 8000 / 3000 * 4.43 * (2000 + 314.3) / 1000)

    def test_obstacle_interval_matches_verifier(self):
        """다리가 장비에 닿는 구간 = 불가. 검증기 evaluate 로 같은 선 위 점들을 찍어 경계를 대조한다."""
        box = ([10000, 19700, 0], [12000, 20000, 600])
        L = ls([box])
        cost, pk = L.axis_edges(0, np.array([[5000, 20000, 1000.0]]), np.array([[20000, 20000, 1000.0]]))
        a, b, g = unpack(pk)
        self.assertEqual((int(a[0]), int(b[0])), (0, 0))
        xs = np.arange(5000, 20001, 10.0)
        X = np.stack([xs, np.full_like(xs, 20000), np.full_like(xs, 1000)], -1)
        ev = ss.evaluate(STD, EXT, [box[0]], [box[1]], "100A", 114.3, X, np.tile([1.0, 0, 0], (len(xs), 1)))
        bad = xs[ev["opt"] < 0]
        self.assertTrue(len(bad))
        self.assertLessEqual(bad.max() - bad.min(), g[0])          # 라우터 불가 구간 ⊇ 검증기 불가 점들
        self.assertLessEqual(g[0] - (bad.max() - bad.min()), 30)   # (10 mm 간격 + 올림 오차 안)


@unittest.skipUnless(astar_fast.HAVE_NUMBA, "numba 없음")
class RouterTest(unittest.TestCase):
    def scenario(self, z):
        return from_dict({"block": {"width": 40000, "length": 40000, "height": 10000, "unit": "mm"},
                          "obstacles": [],
                          "pipes": [{"id": "P", "type_id": "T1", "nominal_size": "100A", "gravity_pipe": False,
                                     "min_slope": None, "insulation_thickness": 50,
                                     "start": {"pos": [0, 20000, z], "kind": "boundary", "dir": [1, 0, 0]},
                                     "end": {"pos": [40000, 20000, z], "kind": "boundary", "dir": [1, 0, 0]},
                                     "valve_positions": [], "branch_points": []}]})

    def test_mid_height_pipe_detours_into_band_and_verifies(self):
        """중앙 높이 배관: 곧장 가면 서포트 불가 → 라우터가 표 범위 띠로 내려가고, 검증기 support 통과 (정합)."""
        sc = self.scenario(5000)
        p = sc.pipes[0]
        clear_cache()
        r = _route(LayeredGraph(sc, p), p, 60, 4_194_304)
        if r.status != "ok":
            self.fail(f"서포트 띠로 들어가지 못함: {r.status}")
        self.assertGreater(len(r.waypoints), 2)
        rep = verify(sc, [PipeRoute(p.id, r.waypoints)], std=STD)
        self.assertTrue(rep["pipes"]["P"].layer0["support"], [v.message for v in rep["pipes"]["P"].violations])

    def test_matches_dijkstra_with_support_state(self):
        """상태 u 를 넣은 컴파일 A* J = 가지치기 없는 파이썬 Dijkstra J (D40①, D80 ⑨)."""
        for z in (1000, 5000):
            sc = self.scenario(z)
            p = sc.pipes[0]
            clear_cache()
            g = LayeredGraph(sc, p)
            fast = _route(g, p, 60, 4_194_304)
            dij = astar_route(g, p, 300, heuristic=lambda s: 0.0)
            self.assertEqual(fast.status, dij.status, z)
            if fast.status == "ok":
                self.assertTrue(math.isclose(fast.J, dij.J, rel_tol=1e-9, abs_tol=1e-6), (z, fast.J, dij.J))


if __name__ == "__main__":
    unittest.main()
