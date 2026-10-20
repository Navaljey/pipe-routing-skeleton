"""D90 (a1) 서포트 부재 예약 — 나중 배관은 놓인 배관의 서포트 부재(검증기 Session 값)를 피한다, rip-up 시 해제."""
import unittest

import numpy as np

from pipe_routing.layered import SupportMembers, make_layered_router
from pipe_routing.verifier import PipeRoute, verify
from pipe_routing.verifier.core import centerlines_of
from pipe_routing.verifier.geom import segment_segment_distance
from tests.test_verifier import pipe, scenario

A_WPS = [[0, 20000, 1500], [40000, 20000, 1500]]


def setup():
    a = pipe("A", start=A_WPS[0], end=A_WPS[1])
    # A 의 바닥 C 다리 (x = 3 m 배수 위치) 를 z = 300 으로 가로지르는 배관
    b = pipe("B", start=[12000, 0, 300], end=[12000, 40000, 300], sdir=(0, 1, 0), edir=(0, 1, 0))
    sc = scenario([a, b])
    return sc, {p.id: p for p in sc.pipes}


def member_clearance(sc, P, wps, members):
    cl = centerlines_of(P, PipeRoute(P.id, wps))[0]
    A, B, S, _ = cl.arrays()
    best = np.inf
    for op, (m0, m1) in members:
        d = segment_segment_distance(A, B, np.array(m0, float)[None], np.array(m1, float)[None])[0]
        best = min(best, float(np.min(d - S)) - op.radius)
    return best


class MemberReserveTest(unittest.TestCase):
    def test_later_pipe_avoids_members(self):
        sc, P = setup()
        placed = [(P["A"], A_WPS)]
        members = SupportMembers()(sc, placed)
        self.assertTrue(members)
        off = make_layered_router(member_reserve=False)(sc, P["B"], 60, placed)
        self.assertEqual(off.status, "ok")
        self.assertLess(member_clearance(sc, P["B"], off.waypoints, members), P["B"].radius)   # 곧장 가면 다리를 친다
        on = make_layered_router()(sc, P["B"], 60, placed)
        self.assertEqual(on.status, "ok")
        self.assertGreaterEqual(member_clearance(sc, P["B"], on.waypoints, members), P["B"].radius - 1e-6)
        self.assertEqual(on.graph_stats["support_members"], len(members))
        rep = verify(sc, [PipeRoute("A", A_WPS), PipeRoute("B", on.waypoints)])["pipes"]
        self.assertTrue(rep["A"].layer0["support"] and rep["A"].layer0["collision"])

    def test_sync_release(self):
        """놓인 배관 집합에 맞춰 Session 을 동기화 — 들어낸 배관의 부재는 사라지고, 빈 집합은 Session 을 건드리지 않는다."""
        sc, P = setup()
        sm = SupportMembers()
        self.assertTrue(sm(sc, [(P["A"], A_WPS)]))
        self.assertEqual(sm(sc, []), [])
        self.assertEqual(set(sm.routes), {"A"})          # 빈 호출(단독 재라우팅)은 상태 유지
        b_wps = [[12000, 0, 300], [12000, 40000, 300]]
        both = sm(sc, [(P["A"], A_WPS), (P["B"], b_wps)])
        only_b = sm(sc, [(P["B"], b_wps)])
        self.assertEqual(set(sm.routes), {"B"})
        self.assertTrue(all(op.id == "B" for op, _ in only_b))
        self.assertTrue(any(op.id == "A" for op, _ in both))


if __name__ == "__main__":
    unittest.main()
