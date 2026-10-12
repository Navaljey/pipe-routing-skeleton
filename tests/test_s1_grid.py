"""S1-3 시험: D66 고정층 격자 보강 변형 (G1 중력관 보조선 · G2 일정 간격 보조선)."""
import math
import unittest
from pathlib import Path

from pipe_routing import astar_fast
from pipe_routing.layered import (G0, GridSpec, LayeredGraph, _route, clear_cache, fixed_layer, grid_extra_coords,
                                  make_layered_router, parse_grid)
from pipe_routing.router_astar import astar_route
from pipe_routing.scenario import load
from pipe_routing.verifier import PipeRoute, verify

ROOT = Path(__file__).resolve().parent.parent
MANUAL = ROOT / "scenarios" / "manual" / "manual_01.json"


class GridSpecTest(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_grid("g0"), G0)
        self.assertEqual(parse_grid("g1"), GridSpec(True, 0.0))
        self.assertEqual(parse_grid("g2:2000"), GridSpec(False, 2000.0))
        self.assertEqual(parse_grid("g1+g2:1000").name, "g1+g2:1000")
        with self.assertRaises(ValueError):
            parse_grid("g3")

    def test_extra_coords(self):
        sc = load(MANUAL)
        self.assertIsNone(grid_extra_coords(sc, "65A", True, G0))
        xs, ys, zs = grid_extra_coords(sc, "65A", False, GridSpec(False, 4000.0))
        self.assertEqual(xs, [4000.0 * k for k in range(1, 11)])
        self.assertEqual(zs, [4000.0, 8000.0])
        # G1 은 중력관 층에만: 압력관 층이면 보조선 없음
        self.assertIsNone(grid_extra_coords(sc, "65A", False, GridSpec(True, 0.0)))
        xs, ys, zs = grid_extra_coords(sc, "65A", True, GridSpec(True, 0.0))
        self.assertTrue(zs and all(800 < z < 2000 for z in zs))          # P005: start 2000 → end 800 사이
        self.assertEqual(xs[0], 3760.0)                                    # L_max 7.53 m / 2, 10 mm 내림


@unittest.skipUnless(astar_fast.HAVE_NUMBA, "numba 없음")
class GridRouteTest(unittest.TestCase):
    def setUp(self):
        clear_cache()
        self.sc = load(MANUAL)
        self.p5 = next(p for p in self.sc.pipes if p.id == "P005")

    def test_g1_manual_p005_routes(self):
        """M23 사례: 수작업 P005 65A — G0 은 경로 없음, G1 은 경로를 찾고 검증기 위반 0."""
        self.assertEqual(_route(LayeredGraph(self.sc, self.p5), self.p5, 60).status, "unreachable")
        r = _route(LayeredGraph(self.sc, self.p5, grid=parse_grid("g1")), self.p5, 60)
        self.assertEqual(r.status, "ok")
        rep = verify(self.sc, [PipeRoute("P005", r.waypoints)],
                     modules=["gravity_slope", "collision", "bend", "boundary"])
        self.assertEqual([v.message for v in rep["pipes"]["P005"].violations], [])

    def test_g1_layer_separate_from_pressure(self):
        """G1 중력관 층은 같은 구경 압력관 층과 따로 캐시 — 압력관 그래프는 G0 과 같다."""
        g1 = parse_grid("g1")
        press = next(p for p in self.sc.pipes if not p.gravity_pipe)
        Fp, _ = fixed_layer(self.sc, press, g1)
        clear_cache()
        Fp0, _ = fixed_layer(self.sc, press, G0)
        self.assertEqual(Fp.shape, Fp0.shape)
        Fg, _ = fixed_layer(self.sc, self.p5, g1)
        Fg0, _ = fixed_layer(self.sc, self.p5, G0)
        self.assertGreater(Fg.shape[2], Fg0.shape[2])

    def test_router_name_and_grid(self):
        r = make_layered_router(grid=parse_grid("g2:2000"))
        self.assertTrue(r.__name__.endswith("g2:2000"))
        self.assertEqual(r.without_reservations.grid, r.grid)

    def test_matches_dijkstra_g1(self):
        """G1 중력관 층에서도 컴파일 A* J = 가지치기 없는 파이썬 Dijkstra J (D40①)."""
        g = LayeredGraph(self.sc, self.p5, grid=parse_grid("g1"))
        fast = _route(g, self.p5, 60)
        dij = astar_route(g, self.p5, 900, heuristic=lambda s: 0.0)
        self.assertEqual(fast.status, dij.status)
        self.assertTrue(math.isclose(fast.J, dij.J, rel_tol=1e-9, abs_tol=1e-6), (fast.J, dij.J))

    def test_matches_dijkstra_g2(self):
        p = self.sc.pipes[0]
        g = LayeredGraph(self.sc, p, grid=parse_grid("g2:4000"))
        fast = _route(g, p, 60)
        dij = astar_route(g, p, 900, heuristic=lambda s: 0.0)
        self.assertEqual(fast.status, dij.status)
        self.assertTrue(math.isclose(fast.J, dij.J, rel_tol=1e-9, abs_tol=1e-6), (fast.J, dij.J))


if __name__ == "__main__":
    unittest.main()
