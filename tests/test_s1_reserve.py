"""S1-1 시험: 단자 예약 (D59) · 놓인 배관 꺾임점 엘보 호 판정 (D60)."""
import math
import unittest
from pathlib import Path

import numpy as np

from pipe_routing import astar_fast
from pipe_routing.layered import (LayeredGraph, _route, clear_cache, layered_router_a, make_layered_router,
                                  terminal_reservations)
from pipe_routing.multi import _pieces, sequential_ripup_planner
from pipe_routing.router_astar import astar_route
from pipe_routing.scenario import from_dict, load
from pipe_routing.verifier import PipeRoute, verify
from pipe_routing.verifier.geom import segment_segment_distance

ROOT = Path(__file__).resolve().parent.parent


def bpipe(pid, size, start, sdir, end, edir):
    return {"id": pid, "type_id": "T1", "nominal_size": size, "gravity_pipe": False, "min_slope": None,
            "insulation_thickness": 50,
            "start": {"pos": start, "kind": "boundary", "dir": sdir},
            "end": {"pos": end, "kind": "boundary", "dir": edir},
            "valve_positions": [], "branch_points": []}


# A: x 방향 직선, 바닥 가까이 (z = 300). B: 바닥(z = 0) 에서 위로 올라가는 경계 단자 — 예약 구간이 A 의 직선을 가로지른다.
# 장애물 두 개는 우회용 격자선(팽창 면 좌표)을 만들기 위한 것 — 장애물이 없으면 고정 격자에 돌아갈 선이 없다 (M19)
SC = {"block": {"width": 40000, "length": 40000, "height": 10000, "unit": "mm"},
      "obstacles": [{"id": "O1", "type": "box", "min": [10000, 30000, 8000], "max": [11000, 31000, 9000]},
                    {"id": "O2", "type": "box", "min": [28000, 30000, 8000], "max": [29000, 31000, 9000]}],
      "pipes": [bpipe("A", "100A", [0, 20000, 300], [1, 0, 0], [40000, 20000, 300], [1, 0, 0]),
                bpipe("B", "50A", [20000, 20000, 0], [0, 0, 1], [20000, 20000, 10000], [0, 0, 1])]}


def min_clearance(p1, w1, p2, w2):
    A1, B1, S1 = _pieces(p1, w1)
    A2, B2, S2 = _pieces(p2, w2)
    d, _, _ = segment_segment_distance(A1[:, None], B1[:, None], A2[None], B2[None])
    return float((d - S1[:, None] - S2[None]).min())


@unittest.skipUnless(astar_fast.HAVE_NUMBA, "numba 없음")
class ReservationTest(unittest.TestCase):
    def setUp(self):
        clear_cache()
        self.sc = from_dict(SC)
        self.A, self.B = self.sc.pipes

    def test_reservation_geometry(self):
        """D59: 노즐 = max(§3.3, t₉₀), 경계 = r + t₉₀, start 는 dir 쪽, end 는 dir 반대쪽."""
        from pipe_routing.constants import elbow_tangent
        res = {(p.id, tuple(w[0])): w for p, w in terminal_reservations(self.sc)}
        w = res[("B", (20000.0, 20000.0, 0.0))]
        L = self.B.radius + elbow_tangent("50A", 90)
        self.assertTrue(np.allclose(w[1], [20000, 20000, L]))
        w = res[("B", (20000.0, 20000.0, 10000.0))]
        self.assertTrue(np.allclose(w[1], [20000, 20000, 10000 - L]))
        self.assertEqual({p.id for p, _ in terminal_reservations(self.sc, exclude={"B"})}, {"A"})

    def test_route_avoids_reserved_stub(self):
        """예약이 없으면 A 는 B 의 단자 구간을 곧장 지나고, 예약이 있으면 r_A + r_B 이상 비켜 간다."""
        stub = [list(self.B.start.pos), list(np.array(self.B.start.pos) + [0, 0, 300])]
        free = make_layered_router(reserve=False)(self.sc, self.A, 60, [])
        self.assertEqual(len(free.waypoints), 2)
        self.assertLess(min_clearance(self.A, free.waypoints, self.B, stub), self.A.radius + self.B.radius)
        r = layered_router_a(self.sc, self.A, 60, [])
        self.assertEqual(r.status, "ok")
        for p, w in terminal_reservations(self.sc, exclude={"A"}):
            self.assertGreaterEqual(min_clearance(self.A, r.waypoints, p, w), self.A.radius + p.radius - 1e-6)

    def test_reservation_replaced_by_route_when_placed(self):
        """B 가 놓이면 예약 대신 실제 경로를 피한다 — 놓인 B 의 경로 옆을 지나는 A 는 다시 직선이 가능해질 수 있다."""
        b = layered_router_a(self.sc, self.B, 60, [])
        self.assertEqual(b.status, "ok")
        reserved_now = terminal_reservations(self.sc, exclude={"A", "B"})
        self.assertEqual(reserved_now, [])        # B 가 놓였으므로 B 예약 없음, A 는 자기 자신
        a = layered_router_a(self.sc, self.A, 60, [(self.B, b.waypoints)])
        self.assertEqual(a.status, "ok")
        self.assertGreaterEqual(min_clearance(self.A, a.waypoints, self.B, b.waypoints),
                                self.A.radius + self.B.radius - 1e-6)

    def test_ripup_restores_reservation(self):
        """rip-up 으로 들어낸 배관은 다시 예약된다: 예약은 '놓이지 않은 배관' 에서 매번 새로 계산한다."""
        placed = [(self.B, [[20000, 20000, 0], [20000, 20000, 10000]])]
        self.assertEqual({p.id for p, _ in terminal_reservations(self.sc, exclude={"A"} | {p.id for p, _ in placed})},
                         set())
        self.assertEqual({p.id for p, _ in terminal_reservations(self.sc, exclude={"A"})}, {"B"})   # 들어낸 뒤

    def test_planner_with_reservations(self):
        res = sequential_ripup_planner(self.sc, router=layered_router_a)
        self.assertTrue(all(r.status == "ok" for r in res.routes.values()))
        rep = verify(self.sc, [PipeRoute(k, r.waypoints) for k, r in res.routes.items()], modules=["collision", "bend"])
        self.assertEqual([v.message for x in rep["pipes"].values() for v in x.violations], [])


# proc_009 P002 실제 경로 (7.6단계 측정) — P005 의 꺾임점이 r₁ + r₂ 안 (252 < 259.75 mm) 인데 엘보 호는 비켜 간다 (M20)
P002 = [[14860, 30270, 10000], [14860, 30270, 8830], [21310, 30270, 8830], [23260, 32220, 8830],
        [24490, 32220, 8830], [24490, 32220, 7880]]


@unittest.skipUnless(astar_fast.HAVE_NUMBA, "numba 없음")
class CornerArcTest(unittest.TestCase):
    def test_m20_case_routes_and_verifies(self):
        """D60: 놓인 배관 근처 꺾임점은 엘보 호 기준 — M20 사례(proc_009 P005)가 경로를 찾고 검증기 위반 0."""
        clear_cache()
        sc = load(ROOT / "scenarios" / "procedural" / "proc_009.json")
        pipes = {p.id: p for p in sc.pipes}
        p5, p2 = pipes["P005"], pipes["P002"]
        g = LayeredGraph(sc, p5, [(p2, P002)])
        r = _route(g, p5, 60)
        self.assertEqual(r.status, "ok")
        corner = np.array(r.waypoints[1])
        W = np.array(P002, float)
        d_corner = min(np.linalg.norm(np.cross(W[i + 1] - W[i], W[i] - corner)) / np.linalg.norm(W[i + 1] - W[i])
                       for i in range(len(W) - 1))
        self.assertLess(d_corner, p5.radius + p2.radius)          # 꺾임점은 이격 안 (예전 규칙이면 노드 무효)
        rep = verify(sc, [PipeRoute("P002", P002), PipeRoute("P005", r.waypoints)], modules=["collision", "bend"])
        self.assertEqual([v.message for v in rep["pipes"]["P005"].violations], [])

    def test_dijkstra_with_masks_and_reservations(self):
        """D60 마스크·D59 예약이 들어간 그래프에서도 컴파일 A* J = 가지치기 없는 파이썬 Dijkstra J (D40①)."""
        clear_cache()
        sc = load(ROOT / "scenarios" / "manual" / "manual_01.json")
        p1 = sc.pipes[0]
        placed = [(p1, layered_router_a(sc, p1, 60, []).waypoints)]
        for p in sc.pipes[1:]:
            rv = terminal_reservations(sc, exclude={p.id, p1.id})
            g = LayeredGraph(sc, p, placed, reserved=rv)
            fast = _route(g, p, 60)
            dij = astar_route(g, p, 300, heuristic=lambda s: 0.0)
            self.assertEqual(fast.status, dij.status, p.id)
            if fast.status == "ok":
                self.assertTrue(math.isclose(fast.J, dij.J, rel_tol=1e-9, abs_tol=1e-6), (p.id, fast.J, dij.J))


if __name__ == "__main__":
    unittest.main()
