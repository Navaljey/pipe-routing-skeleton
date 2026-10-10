"""8단계 간섭 사례 3D (caseview) 시험."""
import unittest

from pipe_routing import astar_fast
from pipe_routing.caseview import case_figure
from pipe_routing.multi import sequential_ripup_planner
from pipe_routing.pipeline import run
from tests.test_multi import HOLE_WALL, boundary_pipe, scenario


@unittest.skipUnless(astar_fast.HAVE_NUMBA, "numba 없음")
class CaseViewTest(unittest.TestCase):
    def test_blocker_found_and_drawn(self):
        """구멍 하나를 다투는 두 배관: 실패 배관의 단독 경로를 막는 배관과 최근접 거리 < 필요 이격을 그린다."""
        sc = scenario([boundary_pipe("SMALL", "25A", 30000), boundary_pipe("BIG", "100A", 10000)], HOLE_WALL)
        out = run(sc, planner=lambda s, r, t: sequential_ripup_planner(s, r, t, max_ripups=0))
        self.assertEqual(out["routes"][0]["fail_class"], "interference_block")
        for close in (False, True):
            fig, blockers = case_figure(sc, out, "SMALL", close=close)
            self.assertEqual(blockers, ["BIG"])
            names = [t.name for t in fig.data]
            self.assertTrue(any(n and n.startswith("최근접") for n in names))


if __name__ == "__main__":
    unittest.main()
