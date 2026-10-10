"""D57 라우터 서포트 근사 시험 (7.6단계)."""
import math
import unittest

import numpy as np

from pipe_routing import astar_fast
from pipe_routing.layered import LayeredGraph, _route, clear_cache, layered_router_a
from pipe_routing.multi import independent_planner
from pipe_routing.pipeline import run
from pipe_routing.router_astar import astar_route, heuristic_factory
from pipe_routing.scenario import from_dict, load
from pipe_routing.verifier import PipeRoute, verify
from tests.test_astar_fast import MANUAL


def pipe(pid, size, y, z):
    return {"id": pid, "type_id": "T1", "nominal_size": size, "gravity_pipe": False, "min_slope": None,
            "insulation_thickness": 50,
            "start": {"pos": [0, y, z], "kind": "boundary", "dir": [1, 0, 0]},
            "end": {"pos": [40000, y, z], "kind": "boundary", "dir": [1, 0, 0]},
            "valve_positions": [], "branch_points": []}


# 장비 하나 (x 5~35m, y 15~25m, z 0~2.5m). 배관(15A) 단자는 장비 바로 위 높이 z = 3m.
# 장비 위를 곧장 지나면 바닥 지지선이 장비에 막혀 천장(7m)까지 앵글이 길다.
EQUIP = {"block": {"width": 40000, "length": 40000, "height": 10000, "unit": "mm"},
         "obstacles": [{"id": "EQ", "type": "tank", "min": [5000, 15000, 0], "max": [35000, 25000, 2500]}],
         "pipes": [pipe("P1", "15A", 20000, 3000)]}


def supports_kg(sc, p, wps):
    rep = verify(sc, [PipeRoute(p.id, wps)], modules=["support"])
    return sum(s["kg"] for s in rep["pipes"][p.id].supports)


@unittest.skipUnless(astar_fast.HAVE_NUMBA, "numba 없음")
class SupportCostTest(unittest.TestCase):
    def test_route_avoids_long_supports_over_equipment(self):
        """추정 서포트를 넣으면 장비 위를 곧장 지나는 경로 대신 구조면 가까이 도는 경로를 고르고, 검증기 J 가 준다."""
        sc = from_dict(EQUIP)
        p = sc.pipes[0]
        clear_cache()
        r0 = _route(LayeredGraph(sc, p, support_cost=False), p, 60)
        r1 = _route(LayeredGraph(sc, p, support_cost=True), p, 60)
        self.assertEqual(len(r0.waypoints), 2)                    # 서포트를 모르면 장비 위를 곧장
        self.assertGreater(len(r1.waypoints), 2)                  # 알면 돌아간다
        s0, s1 = supports_kg(sc, p, r0.waypoints), supports_kg(sc, p, r1.waypoints)
        self.assertLess(s1, s0 / 2)
        self.assertLess(r1.J_pipe + r1.J_elbow + s1, r0.J_pipe + r0.J_elbow + s0)   # 검증기 기준 J (직관+엘보+서포트)
        self.assertGreater(r1.J_support_est, 0)

    def test_support_cost_nonnegative_and_heuristic_admissible(self):
        sc = load(MANUAL)
        clear_cache()
        for p in sc.pipes:
            g = LayeredGraph(sc, p)
            self.assertTrue(np.all(g.scost >= 0))
            self.assertTrue(np.all(g.scost[g.step < 0] == 0))
            r = _route(g, p, 60)
            # 최적 경로 위 모든 상태에서 h ≤ 남은 실제 비용 (D41 휴리스틱을 그대로 써도 과대추정 없음)
            h = heuristic_factory(g, p)
            costs = [g.cost(a, b) for a, b in zip(r.states, r.states[1:])]
            for i, s in enumerate(r.states):
                self.assertLessEqual(h(s), sum(costs[i:]) + 1e-6)

    def test_matches_dijkstra(self):
        """새 비용에서도 컴파일 A* J = 가지치기 없는 파이썬 Dijkstra J (최적성, D40①)."""
        sc = load(MANUAL)
        clear_cache()
        for p in sc.pipes:
            g = LayeredGraph(sc, p)
            fast = _route(g, p, 60)
            dij = astar_route(g, p, 120, heuristic=lambda s: 0.0)       # 파이썬 구현, 휴리스틱 0
            self.assertEqual(fast.status, "ok")
            self.assertTrue(math.isclose(fast.J, dij.J, rel_tol=1e-9, abs_tol=1e-6), (p.id, fast.J, dij.J))

    def test_pipeline_default_is_layered_a_with_support(self):
        sc = load(MANUAL)
        out = run(sc, planner=independent_planner)
        self.assertTrue(all(x["router"]["name"] == layered_router_a.__name__ for x in out["routes"]))
        est = out["global"]["support_estimate"]
        self.assertGreater(est["est_total"], 0)
        self.assertGreater(est["verifier_total"], 0)


if __name__ == "__main__":
    unittest.main()
