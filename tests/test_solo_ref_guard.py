"""D101 단독 참고치 — 분기 계통 재사용 예산 (D134) · 메모리 초과 (D123 ③)."""
import unittest
from types import SimpleNamespace

from pipe_routing.layered import EXPAND_BUDGET
from pipe_routing.router_astar import RouteResult
from pipe_routing.solo_ref import solo_routes


def _sc(*nbr):
    return SimpleNamespace(block_pair=None, pipes=[SimpleNamespace(id=f"P{i}", branches=[None] * n)
                                                   for i, n in enumerate(nbr)])


class SoloRefGuardTest(unittest.TestCase):
    def test_tree_reuse_within_tree_budget(self):
        """분기 1 개 계통 = 라우팅 예산 2 × 상한 — 그 안에서 끝난 진단 결과는 재사용, 분기 없는 배관은 상한 그대로."""
        sc = _sc(0, 1)
        calls = []

        def router(sc_, p, limit, placed):
            calls.append(p.id)
            return RouteResult(p.id, "unreachable")
        reuse = {pid: RouteResult(pid, "unreachable", expanded=EXPAND_BUDGET + 10) for pid in ("P0", "P1")}
        out, reused = solo_routes(sc, router, reuse)
        self.assertEqual(calls, ["P0"])
        self.assertEqual(reused, 1)
        self.assertIs(out["P1"], reuse["P1"])

    def test_memory_error_is_resource_exceeded(self):
        sc = _sc(1)

        def router(sc_, p, limit, placed):
            raise MemoryError
        out, _ = solo_routes(sc, router, {})
        self.assertEqual(out["P0"].status, "resource_exceeded")


if __name__ == "__main__":
    unittest.main()
