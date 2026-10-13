"""S1-4 시험: D67 증분 배치층 표시 — 전체 재계산과 비트 동일, 경로·J·확장 수 동일."""
import unittest
from pathlib import Path

import numpy as np

from pipe_routing import astar_fast
from pipe_routing.layered import LayeredGraph, clear_cache, make_layered_router, terminal_reservations
from pipe_routing.multi import sequential_ripup_planner
from pipe_routing.scenario import load

ROOT = Path(__file__).resolve().parent.parent
ARRAYS = ("step", "elen", "scost", "emask", "pipe_margin", "node_valid", "clearance", "hdist")


def same_graph(a, b):
    return all(np.array_equal(getattr(a, k), getattr(b, k)) for k in ARRAYS) and a.n_edges == b.n_edges


@unittest.skipUnless(astar_fast.HAVE_NUMBA, "numba 없음")
class IncrementalMarkingTest(unittest.TestCase):
    def test_bit_identical_through_place_and_remove(self):
        """배치(추가) → 제거(rip-up) → 예약 해제가 섞인 순서에서도 매 단계 표시가 전체 재계산과 같다."""
        clear_cache()
        sc = load(ROOT / "scenarios" / "manual" / "manual_01.json")
        pipes = {p.id: p for p in sc.pipes}
        router = make_layered_router()
        placed = []
        for pid in ("P002", "P001", "P004"):
            r = router(sc, pipes[pid], 60, placed)
            self.assertEqual(r.status, "ok")
            placed.append((pipes[pid], r.waypoints))
        target = pipes["P003"]
        steps = [placed[:1], placed[:2], placed, placed[1:], [placed[0], placed[2]], []]   # 추가·제거 섞기
        for pl in steps:
            rv = terminal_reservations(sc, exclude={target.id} | {p.id for p, _ in pl})
            inc = LayeredGraph(sc, target, pl, reserved=rv)
            full = LayeredGraph(sc, target, pl, reserved=rv, incremental=False)
            self.assertTrue(inc.incremental)
            self.assertTrue(same_graph(inc, full), [p.id for p, _ in pl])

    def test_procedural_late_pipe(self):
        """procedural: 앞 배관들이 놓인 뒤 마지막 배관 — 캐시를 두 번 써도(두 번째 생성) 같다."""
        clear_cache()
        sc = load(ROOT / "scenarios" / "procedural" / "proc_003.json")
        res = sequential_ripup_planner(sc, router=make_layered_router())
        placed = [(p, res.routes[p.id].waypoints) for p in sc.pipes if res.routes[p.id].status == "ok"]
        target, others = placed[-1][0], placed[:-1]
        rv = terminal_reservations(sc, exclude={target.id} | {p.id for p, _ in others})
        full = LayeredGraph(sc, target, others, reserved=rv, incremental=False)
        for _ in range(2):
            self.assertTrue(same_graph(LayeredGraph(sc, target, others, reserved=rv), full))

    def test_planner_results_identical(self):
        """순차 배치 전체: 증분 / 전체 재계산 라우터의 경로·J·확장 수가 같다."""
        sc = load(ROOT / "scenarios" / "procedural" / "proc_005.json")
        out = []
        for inc in (True, False):
            clear_cache()
            res = sequential_ripup_planner(sc, router=make_layered_router(incremental=inc))
            out.append({k: (r.status, [list(map(float, w)) for w in r.waypoints], r.J, r.expanded)
                        for k, r in res.routes.items()})
        self.assertEqual(out[0], out[1])


if __name__ == "__main__":
    unittest.main()
