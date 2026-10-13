"""S1-5 시험: anytime 기준선 (D68) — 같은 시드·결정적 예산이면 궤적(누적 확장·해·이동 종류)이 같고, 최선 해는 단조 개선,
곡선 집계는 계단 함수 (D72)."""
import unittest
from pathlib import Path

from pipe_routing import astar_fast
from pipe_routing.anytime import _objective, anytime_run
from pipe_routing.curves import relative_value, set_value, value_at
from pipe_routing.layered import make_layered_router
from pipe_routing.scenario import load

ROOT = Path(__file__).resolve().parent.parent
SCEN = ROOT / "scenarios" / "manual" / "manual_01.json"


def sig(tr):
    return [(e, k, tuple((pid, r.status, tuple(map(tuple, r.waypoints or []))) for pid, r in sorted(routes.items())))
            for e, _t, routes, k in tr.points], tr.moves, tr.total_expanded


@unittest.skipUnless(astar_fast.HAVE_NUMBA, "numba 없음")
class AnytimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sc = load(SCEN)
        router = make_layered_router(expand_limit=8 * 2048)   # 작은 예산 — 이동이 여러 번 일어나게
        first = anytime_run(cls.sc, router, seed=0, max_expanded=1)
        cls.budget = first.e0 + 400_000
        cls.a = anytime_run(cls.sc, make_layered_router(expand_limit=8 * 2048), seed=7, max_expanded=cls.budget)
        cls.b = anytime_run(cls.sc, make_layered_router(expand_limit=8 * 2048), seed=7, max_expanded=cls.budget)

    def test_same_seed_same_trajectory(self):
        self.assertGreater(self.a.moves, 1)
        self.assertEqual(sig(self.a), sig(self.b))
        self.assertEqual(self.a.stop, "expansions")

    def test_monotone_best(self):
        objs = [_objective(routes) for _e, _t, routes, _k in self.a.points]
        self.assertEqual(objs, sorted(objs))
        self.assertEqual(self.a.points[0][3], "first")

    def test_curve_step(self):
        c = {"total_sec": 10.0, "total_expanded": 100, "t0_sec": 2.0, "e0": 20, "n_pipes": 4,
             "points": [{"success": 2, "success_rate": 0.5, "J_total": 10.0, "routed": 3, "interference_block": 1,
                         "expanded": 20, "sec": 2.0},
                        {"success": 3, "success_rate": 0.75, "J_total": 12.0, "routed": 4, "interference_block": 0,
                         "expanded": 60, "sec": 6.0}]}
        self.assertIsNone(value_at(c, 1.0)[0])
        self.assertEqual(value_at(c, 5.9)[0]["success"], 2)
        self.assertEqual(value_at(c, 6.0)[0]["success"], 3)
        self.assertTrue(value_at(c, 11.0)[1])                 # 멈춘 뒤 예산 = censored
        self.assertEqual(value_at(c, 59, "expanded")[0]["success"], 2)
        self.assertEqual(set_value([c, c], 7.0)["success"], 6)
        self.assertEqual(relative_value([c], 4, "sec")["success"], 3)   # t0 × 4 = 8초
        self.assertEqual(relative_value([c], 2, "expanded")["success"], 2)   # e0 × 2 = 40


if __name__ == "__main__":
    unittest.main()
