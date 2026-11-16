"""파이썬 A* 경로의 지배 판정에 서포트 상태 su (D80 ⑨) 비교 — 컴파일 경로 (astar_fast) 와 같은 규칙.

su 가 빠지면 지지 없이 더 길게 온 (su 가 큰) 확정 상태가 su 가 작은 상태를 지배해 버려, 서포트 가능 구간이 경로를
가르는 배관에서 파이썬 경로가 더 비싼 경로를 고르거나 (proc_005 P008: +2.47 %) 예산을 소진한다 (proc_003 P008).
기존 Dijkstra 대조 시험은 서포트 가능 띠 안에서 u 가 한 번도 경로를 가르지 않는 작은 시험 시나리오에서 J 만 비교해
이 차이를 볼 수 없었다.
"""
import unittest
from pathlib import Path

from pipe_routing import astar_fast
from pipe_routing.layered import LayeredGraph, _route, clear_cache
from pipe_routing.router_astar import astar_route
from pipe_routing.scenario import load

ROOT = Path(__file__).resolve().parent.parent
CASES = [("proc_005", "P008"), ("proc_003", "P008"), ("proc_006", "P010")]


@unittest.skipUnless(astar_fast.HAVE_NUMBA, "numba 없음")
class PythonCompiledSuTest(unittest.TestCase):
    def test_python_equals_compiled(self):
        for name, pid in CASES:
            with self.subTest(name=name, pid=pid):
                clear_cache()
                sc = load(ROOT / f"scenarios/procedural/{name}.json")
                p = next(q for q in sc.pipes if q.id == pid)
                g = LayeredGraph(sc, p)
                fast = _route(g, p, 60, 400000)
                py = astar_route(g, p, 300, impl="python", expand_limit=400000)
                self.assertEqual(fast.status, "ok")
                self.assertEqual(py.status, "ok")
                self.assertEqual(py.expanded, fast.expanded)
                self.assertAlmostEqual(py.J, fast.J, places=6)
                self.assertEqual([list(map(float, w)) for w in py.waypoints],
                                 [list(map(float, w)) for w in fast.waypoints])


if __name__ == "__main__":
    unittest.main()
