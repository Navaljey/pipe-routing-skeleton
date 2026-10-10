"""M18 대안 표현 (layered.py) 시험: 경로가 검증기 기준 장애물·놓인 배관과 충돌하지 않는다."""
import unittest

from pipe_routing import astar_fast
from pipe_routing.layered import LayeredGraph, clear_cache, layered_router_a, layered_router_b
from pipe_routing.multi import sequential_ripup_planner
from pipe_routing.scenario import load
from pipe_routing.verifier import PipeRoute, verify
from tests.test_astar_fast import MANUAL


@unittest.skipUnless(astar_fast.HAVE_NUMBA, "numba 없음")
class LayeredTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sc = load(MANUAL)

    def test_variants_route_without_collision(self):
        for router in (layered_router_a, layered_router_b):
            clear_cache()
            res = sequential_ripup_planner(self.sc, router=router)
            ok = {pid: r for pid, r in res.routes.items() if r.status == "ok"}
            # 압력관은 전부 경로 (중력관은 계단식 하향 규칙 D61 — 고정 격자로 못 갈 수 있다)
            self.assertTrue(all(p.id in ok for p in self.sc.pipes if not p.gravity_pipe))
            rep = verify(self.sc, [PipeRoute(pid, r.waypoints) for pid, r in ok.items()], modules=["collision", "bend"])
            self.assertEqual([v.message for x in rep["pipes"].values() for v in x.violations], [], router.__name__)

    def test_fixed_layer_cached_per_size(self):
        clear_cache()
        p = self.sc.pipes[0]
        same = [q for q in self.sc.pipes if q.nominal_size == p.nominal_size]
        g1 = LayeredGraph(self.sc, p)
        g2 = LayeredGraph(self.sc, same[-1])
        self.assertGreater(g1.fixed_sec, 0)
        self.assertEqual(g2.fixed_sec, 0.0)
        self.assertIs(g1.fixed, g2.fixed)

    def test_local_lines_add_patch_nodes(self):
        clear_cache()
        res = sequential_ripup_planner(self.sc, router=layered_router_a, max_ripups=0)
        placed = [(p, res.routes[p.id].waypoints) for p in self.sc.pipes[:2] if res.routes[p.id].status == "ok"]
        tgt = self.sc.pipes[-1]
        a = LayeredGraph(self.sc, tgt, placed)
        b = LayeredGraph(self.sc, tgt, placed, local_lines=True)
        self.assertEqual(a.n_patch_nodes, 0)
        self.assertGreater(b.n_patch_nodes, 0)


if __name__ == "__main__":
    unittest.main()
