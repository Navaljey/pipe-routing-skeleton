"""M10 가속의 결과 불변 시험 (D40①): 가속 전후로 그래프·경로·J·확장 수가 같다."""
import random
import unittest
from pathlib import Path

import numpy as np

from pipe_routing import astar_fast
from pipe_routing.escape_graph import EscapeGraph
from pipe_routing.multi import astar_router
from pipe_routing.router_astar import astar_route
from pipe_routing.scenario import load

ROOT = Path(__file__).resolve().parent.parent
MANUAL = ROOT / "scenarios" / "manual" / "manual_01.json"


def placed_for(sc, upto):
    """manual_01 의 앞 배관들을 순서대로 깔아 놓인 배관 목록을 만든다 (가속 전 구현으로)."""
    placed = []
    for p in sc.pipes[:upto]:
        r = astar_route(EscapeGraph(sc, p, others=placed, fast_build=False), p, impl="python")
        if r.status == "ok":
            placed.append((p, r.waypoints))
    return placed


def same_graph(a: EscapeGraph, b: EscapeGraph) -> bool:
    thr = a.r + a.tangent[135] + 1e-5   # 거리 값은 판정 문턱까지만 의미가 있다 (EscapeGraph._point_clearance)
    return (all(np.array_equal(x, y) for x, y in zip(a.axis_ok, b.axis_ok))
            and all(np.array_equal(a.diag_ok[d], b.diag_ok[d]) for d in a.diag_ok)
            and np.array_equal(a.node_ok, b.node_ok)
            and np.array_equal(np.minimum(a.clearance, thr), np.minimum(b.clearance, thr))
            and np.array_equal(a.pipe_margin, b.pipe_margin))


def same_result(a, b) -> bool:
    return ((a.status, a.J, a.J_pipe, a.J_elbow, a.waypoints, a.bends, a.expanded, a.generated, a.states)
            == (b.status, b.J, b.J_pipe, b.J_elbow, b.waypoints, b.bends, b.expanded, b.generated, b.states))


@unittest.skipUnless(astar_fast.HAVE_NUMBA, "numba 없음 — 파이썬 구현만 쓰므로 비교 대상 없음")
class FastSearchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sc = load(MANUAL)
        cls.placed = placed_for(cls.sc, 3)

    def test_round6_matches_python(self):
        """직진 길이 반올림이 파이썬 round(x, 6) 과 비트 단위로 같다."""
        rnd = random.Random(0)
        vals = [rnd.uniform(0, 5000) for _ in range(20000)]
        vals += [k * 10 * 2 ** 0.5 + j * 10 for k in range(300) for j in range(0, 300, 7)]   # 실제 직진 길이 형태
        vals += [k / 1e6 + 0.5e-6 for k in range(0, 5000000, 9973)]                      # 반올림 경계 근처
        for x in vals:
            self.assertEqual(astar_fast._round6(x), round(x, 6), x)

    def test_graph_identical(self):
        """그래프 구성 가속(인덱스 범위 제한) 전후 노드·엣지 판정이 같다 — 놓인 배관이 있을 때 포함."""
        for p in self.sc.pipes:
            for others in ((), self.placed):
                others = [o for o in others if o[0].id != p.id]
                a = EscapeGraph(self.sc, p, others=others, fast_build=False)
                b = EscapeGraph(self.sc, p, others=others)
                self.assertTrue(same_graph(a, b), (p.id, len(others)))

    def test_search_identical(self):
        """컴파일 A* 와 파이썬 A* 의 경로·J·확장 수·상태 사슬이 같다."""
        for p in self.sc.pipes:
            for others in ((), self.placed):
                others = [o for o in others if o[0].id != p.id]
                a = astar_route(EscapeGraph(self.sc, p, others=others, fast_build=False), p, impl="python")
                b = astar_route(EscapeGraph(self.sc, p, others=others), p, impl="fast")
                self.assertTrue(same_result(a, b), (p.id, len(others), a.summary(), b.summary()))

    def test_procedural_identical(self):
        """procedural 배관 (그래프 수만~십만 노드) 에서도 같다."""
        sc = load(ROOT / "scenarios" / "procedural" / "proc_000.json")
        for p in sc.pipes[1:4]:
            a = astar_route(EscapeGraph(sc, p, fast_build=False), p, impl="python")
            b = astar_route(EscapeGraph(sc, p), p, impl="fast")
            self.assertTrue(same_result(a, b), p.id)

    def test_default_router_uses_fast(self):
        p = self.sc.pipes[1]
        r = astar_router(self.sc, p, 60, self.placed[:1])
        ref = astar_route(EscapeGraph(self.sc, p, others=self.placed[:1], fast_build=False), p, impl="python")
        self.assertTrue(same_result(r, ref))


if __name__ == "__main__":
    unittest.main()
