"""S1-5 시험: anytime 기준선 (D68·D75) — 같은 시드·결정적 예산이면 궤적(누적 확장·해·이동 종류·검증 점수)이 같고,
최선 해의 검증 점수는 단조 개선 (통과 수 같으면 검증 J 비증가), 곡선 집계는 계단 함수 (D72)."""
import unittest
from pathlib import Path

from pipe_routing import astar_fast
from pipe_routing.anytime import _objective, anytime_run, better, score
from pipe_routing.curves import relative_value, set_value, value_at
from pipe_routing.layered import make_layered_router
from pipe_routing.scenario import load

ROOT = Path(__file__).resolve().parent.parent
SCEN = ROOT / "scenarios" / "manual" / "manual_01.json"


def sig(tr):
    return [(e, k, sc, tuple((pid, r.status, tuple(map(tuple, r.waypoints or []))) for pid, r in sorted(routes.items())))
            for e, _t, routes, k, sc in tr.points], tr.moves, tr.total_expanded, tr.verified, tr.rejected


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
        """D75: 받아들인 해는 라우터 추정(1차)도, 검증 점수도 직전 최선보다 좋다 — 통과 수가 같으면 검증 J 는 줄기만 한다."""
        objs = [_objective(routes) for _e, _t, routes, _k, _s in self.a.points]
        self.assertEqual(objs, sorted(objs))
        self.assertEqual(self.a.points[0][3], "first")
        scores = [s for *_x, s in self.a.points]
        for prev, cur in zip(scores, scores[1:]):
            self.assertTrue(better(cur, prev))
            self.assertGreaterEqual(cur[0], prev[0])
            if cur[0] == prev[0]:
                self.assertLess(cur[1], prev[1])
        for _e, _t, routes, _k, s in self.a.points:      # 저장한 점수 = 그 해를 다시 채점한 값
            self.assertEqual(score(self.sc, routes), s)
        self.assertGreaterEqual(self.a.verified, len(self.a.points) - 1)
        self.assertEqual(self.a.verified - self.a.rejected, len(self.a.points) - 1)

    def test_better(self):
        self.assertTrue(better((5, 100.0), (4, 50.0)))       # 통과 수 우선
        self.assertTrue(better((4, 49.0), (4, 50.0)))
        self.assertFalse(better((4, 50.0), (4, 50.0)))
        self.assertFalse(better((4, 51.0), (4, 50.0)))       # 라우터 추정은 좋아도 검증 J 가 늘면 버린다
        self.assertFalse(better((3, 10.0), (4, 50.0)))

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


@unittest.skipUnless(astar_fast.HAVE_NUMBA, "numba 없음")
class FirstSolutionOutputTest(unittest.TestCase):
    def test_first_output_equals_pipeline_sequential(self):
        """D110 ②: anytime 첫 해 출력 = pipeline 순차 출력 (deterministic_view 바이트 동일) — 개선 이동이 여러 번 일어난 뒤
        (라우터 캐시·부재 Session 이 움직인 뒤) 예산 밖에서 분류해도 같다. 작은 예산이라 실패 분류(D74 진단)도 일어난다."""
        import json

        from pipe_routing.pipeline import deterministic_view, run
        sc = load(SCEN)
        ref = run(sc, router=make_layered_router(expand_limit=8 * 2048))
        r = make_layered_router(expand_limit=8 * 2048)
        e0 = anytime_run(sc, make_layered_router(expand_limit=8 * 2048), seed=0, max_expanded=1).e0
        tr = anytime_run(sc, r, seed=7, max_expanded=e0 + 400_000)
        self.assertGreater(tr.moves, 1)
        got = run(sc, router=r, planner=lambda sc_, r_, tl: tr.first)
        self.assertTrue(ref["global"]["fail_class"] and any(ref["global"]["fail_class"].values()))
        dump = lambda o: json.dumps(deterministic_view(o), ensure_ascii=False, sort_keys=True)   # noqa: E731
        self.assertEqual(dump(got), dump(ref))
