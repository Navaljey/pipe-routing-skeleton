"""S1-2 시험: 중력관 계단식 하향 규칙 — 라우터 (D61·D63) · 실현 가능성 (D64)."""
import math
import unittest
from pathlib import Path

from pipe_routing import astar_fast
from pipe_routing.gravity import (GRAVITY_LMAX_MM, check_waypoints, feasibility, gravity_lmax, min_step_drop,
                                  required_drop)
from pipe_routing.layered import LayeredGraph, _route, clear_cache
from pipe_routing.router_astar import astar_route
from pipe_routing.scenario import from_dict, load
from pipe_routing.verifier import PipeRoute, verify

ROOT = Path(__file__).resolve().parent.parent


def t5(pid, size, start, sdir, end, edir, start_kind="boundary", end_kind="boundary"):
    return {"id": pid, "type_id": "T5", "nominal_size": size, "gravity_pipe": True, "min_slope": 0.01,
            "insulation_thickness": 50,
            "start": {"pos": start, "kind": start_kind, "dir": sdir},
            "end": {"pos": end, "kind": end_kind, "dir": edir},
            "valve_positions": [], "branch_points": []}


def scen(pipes, obstacles=()):
    return from_dict({"block": {"width": 40000, "length": 40000, "height": 10000, "unit": "mm"},
                      "obstacles": [{"id": f"O{i}", "type": "box", "min": lo, "max": hi}
                                    for i, (lo, hi) in enumerate(obstacles)], "pipes": pipes})


# 계단용 z 격자선을 만드는 얇은 장애물들 (경로와 떨어진 곳) — 고정층은 장애물 팽창 면에서만 격자선이 생긴다 (D38)
STAIR_OBS = [([2000, 2000, 400 + 600 * i], [2500, 2500, 450 + 600 * i]) for i in range(8)] + \
            [([5000 * i, 30000, 9000], [5000 * i + 400, 30400, 9400]) for i in range(1, 8)]


class LmaxTableTest(unittest.TestCase):
    def test_table_matches_d62(self):
        self.assertEqual(GRAVITY_LMAX_MM["15A"], 4000)
        self.assertEqual(GRAVITY_LMAX_MM["100A"], 10000)
        self.assertEqual(GRAVITY_LMAX_MM["500A"], 50000)
        # 지정점 사이 선형 보간 (표 값은 소유자가 준 결과 그대로: 25A 4.71 · 65A 7.53 은 반올림)
        for size, val in GRAVITY_LMAX_MM.items():
            a = int(size[:-1])
            pts = [(15, 4000), (100, 10000), (200, 20000), (500, 50000)]
            for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
                if x0 <= a <= x1:
                    self.assertLessEqual(abs(val - (y0 + (y1 - y0) * (a - x0) / (x1 - x0))), 10.0, size)


class FeasibilityTest(unittest.TestCase):
    def test_up_terminal_is_infeasible(self):
        sc = scen([t5("G", "100A", [20000, 20000, 0], [0, 0, 1], [20000, 20000, 10000], [0, 0, 1])])
        f = feasibility(sc.pipes[0])
        self.assertFalse(f["feasible"])
        self.assertIsNone(f["required_mm"])

    def test_required_drop_bound(self):
        """D64 하한: 수평 H, L_max, 단 낙차 d 로 min_k max((k−1)d, H − kL)."""
        sc = scen([t5("G", "100A", [0, 20000, 5000], [1, 0, 0], [40000, 20000, 1000], [1, 0, 0])])
        p = sc.pipes[0]
        req, _ = required_drop(p)
        d, L, H = min_step_drop("100A"), gravity_lmax("100A"), 40000.0
        expect = min(max((k - 1) * d, H - k * L) for k in range(1, 10))
        self.assertAlmostEqual(req, expect)
        self.assertTrue(feasibility(p)["feasible"])
        # 높이 차가 하한보다 작으면 규칙상 불가능
        z_end = 5000 - (math.floor(expect / 10) * 10 - 10)   # 하한보다 10 mm 이상 작은 낙차 (10 mm 격자)
        sc2 = scen([t5("G", "100A", [0, 20000, 5000], [1, 0, 0], [40000, 20000, z_end], [1, 0, 0])])
        self.assertFalse(feasibility(sc2.pipes[0])["feasible"])


@unittest.skipUnless(astar_fast.HAVE_NUMBA, "numba 없음")
class GravityRouterTest(unittest.TestCase):
    def test_routed_gravity_paths_pass_verifier(self):
        """라우터 경로를 검증기로 판정하면 gravity_slope 위반 0 (D63 라우터–검증기 정합)."""
        sc = scen([t5("G", "100A", [0, 20000, 5000], [1, 0, 0], [40000, 15000, 1000], [1, 0, 0])], STAIR_OBS)
        p = sc.pipes[0]
        clear_cache()
        r = _route(LayeredGraph(sc, p), p, 60)
        self.assertEqual(r.status, "ok")
        self.assertEqual(check_waypoints(r.waypoints, gravity_lmax("100A")), [])
        self.assertTrue(all(b[2] <= a[2] for a, b in zip(r.waypoints, r.waypoints[1:])))   # 상향 없음
        rep = verify(sc, [PipeRoute("G", r.waypoints)], modules=["gravity_slope", "bend", "collision"])
        self.assertEqual([v.message for v in rep["pipes"]["G"].violations], [])

    def test_manual_and_procedural_gravity(self):
        """수작업·procedural 중력관: 경로가 있으면 검증기 gravity_slope 통과."""
        clear_cache()
        for path in (ROOT / "scenarios" / "manual" / "manual_01.json", ROOT / "scenarios" / "procedural" / "proc_000.json"):
            sc = load(path)
            for p in sc.pipes:
                if not p.gravity_pipe:
                    continue
                r = _route(LayeredGraph(sc, p), p, 60)
                if r.status != "ok":
                    continue
                rep = verify(sc, [PipeRoute(p.id, r.waypoints)], modules=["gravity_slope"])
                self.assertEqual([v.message for v in rep["pipes"][p.id].violations], [], (path.name, p.id))

    def test_rising_pipe_unreachable(self):
        sc = scen([t5("G", "100A", [0, 20000, 1000], [1, 0, 0], [40000, 20000, 5000], [1, 0, 0])], STAIR_OBS)
        p = sc.pipes[0]
        clear_cache()
        self.assertFalse(feasibility(p)["feasible"])
        self.assertEqual(_route(LayeredGraph(sc, p), p, 60).status, "unreachable")

    def test_dijkstra_with_gravity_state(self):
        """h 상태가 들어간 탐색도 컴파일 A* J = 가지치기 없는 파이썬 Dijkstra J (D63, D40①)."""
        sc = scen([t5("G", "50A", [0, 20000, 5000], [1, 0, 0], [40000, 18000, 2000], [1, 0, 0])], STAIR_OBS)
        p = sc.pipes[0]
        clear_cache()
        g = LayeredGraph(sc, p)
        fast = _route(g, p, 60)
        dij = astar_route(g, p, 600, heuristic=lambda s: 0.0)
        self.assertEqual(fast.status, dij.status)
        if fast.status == "ok":
            self.assertTrue(math.isclose(fast.J, dij.J, rel_tol=1e-9, abs_tol=1e-6), (fast.J, dij.J))


if __name__ == "__main__":
    unittest.main()
