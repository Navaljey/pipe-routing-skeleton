"""D126 L3 (a)~(d) 저장 형식 축소 — 결과 불변 (D53·D67 원칙).

경로·J·확장 수·확장 순서 비트 동일은 E1 첫 해 41 바이트 비교로 확인한다 (보고서). 여기서는 형식과 불변식:
(a) 밀집 서포트 표는 증분 표시 (변형 A) 에서 템플릿을 만든 뒤 고정층 캐시에서 내려놓는다
(b) 지도 step = int32 · emask = uint16 — 값은 전체 재계산 (int64 시절과 같은 함수) 과 같다
(c) 항목이 없으면 합성이 템플릿을 복사하지 않는다
(d) A* 배열 증가 = 1.25 배, 인덱스 int32 — 용량과 무관하게 같은 탐색."""
import unittest

import numpy as np

from pipe_routing import astar_fast
from pipe_routing.layered import LayeredGraph, _and_items, _route, clear_cache, make_layered_router
from pipe_routing.scenario import load

MANUAL = "scenarios/manual/manual_01.json"


def _sig(r):
    return (r.status, r.waypoints, r.J, r.expanded, r.generated,
            [(s.node, s.dir, s.run, s.bend) for s in (r.states or [])])


@unittest.skipUnless(astar_fast.HAVE_NUMBA, "numba 없음")
class L3Test(unittest.TestCase):
    def setUp(self):
        clear_cache()
        self.sc = load(MANUAL)

    def _placed(self):
        r = make_layered_router()
        placed = []
        for p in self.sc.pipes[:3]:
            res = r(self.sc, p, 60, placed)
            if res.status == "ok":
                placed.append((p, res.waypoints))
        return placed

    def test_dtypes_and_support_cache_released(self):
        p = self.sc.pipes[0]
        g = LayeredGraph(self.sc, p)
        self.assertEqual(g.step.dtype, np.int32)
        self.assertEqual(g.emask.dtype, np.uint16)
        self.assertEqual(g.elen.dtype, np.float64)
        self.assertEqual(g.scost.dtype, np.float64)
        self.assertEqual(g.spack.dtype, np.int64)
        F = g.fixed
        self.assertFalse(F.__dict__.get("_support_cache"))          # (a) 템플릿에 옮긴 뒤 내려놓음
        self.assertTrue(F.__dict__.get("_tmpl_cache"))
        g2 = LayeredGraph(self.sc, p)                                # 캐시된 템플릿 — 밀집 표를 다시 만들지 않는다
        self.assertFalse(F.__dict__.get("_support_cache"))
        for name in ("step", "elen", "scost", "spack", "emask"):
            self.assertTrue(np.array_equal(getattr(g, name), getattr(g2, name)), name)

    def test_incremental_equals_full_marking(self):
        """증분 (L3 경로) 과 전체 재계산 (밀집 서포트 표를 쓰는 이전 경로) 의 지도 배열이 같다."""
        placed = self._placed()
        for p in self.sc.pipes:
            others = [(q, w) for q, w in placed if q.id != p.id]
            clear_cache()
            a = LayeredGraph(self.sc, p, others, incremental=True)
            b = LayeredGraph(self.sc, p, others, incremental=False)
            for name in ("step", "elen", "scost", "spack", "emask", "hdist", "pos"):
                x, y = getattr(a, name), getattr(b, name)
                self.assertTrue(np.array_equal(x, y), (p.id, name))
            self.assertEqual(_sig(_route(a, p, 60, 4_194_304)), _sig(_route(b, p, 60, 4_194_304)), p.id)

    def test_and_items_no_copy_without_items(self):
        obs = np.array([0xFFFF, 0x0F0F, 0], dtype=np.uint16)
        self.assertIs(_and_items(obs, []), obs)
        out = _and_items(obs, [(np.array([0, 1]), np.array([0x00FF, 0xFFFF], dtype=np.uint16))])
        self.assertEqual(out.tolist(), [0x00FF, 0x0F0F, 0])
        self.assertEqual(obs.tolist(), [0xFFFF, 0x0F0F, 0])          # 템플릿은 그대로

    def test_growth_capacity_independent(self):
        """초기 용량을 아주 작게 해 증가를 여러 번 일으켜도 같은 탐색."""
        p = self.sc.pipes[1]
        g = LayeredGraph(self.sc, p, self._placed()[:1])
        base = _route(g, p, 60, 4_194_304)
        orig = astar_fast._generic_arrays
        astar_fast._generic_arrays = lambda cap: orig(64)
        try:
            small = _route(g, p, 60, 4_194_304)
        finally:
            astar_fast._generic_arrays = orig
        self.assertEqual(_sig(small), _sig(base))
        H, S, K = orig(8)
        self.assertEqual(S[0].dtype, np.int32)
        self.assertEqual(H[1].dtype, np.int64)                       # 큐 삽입 순번은 int64 그대로
        grown = astar_fast._grow(S, 100)
        self.assertGreaterEqual(len(grown[0]), 100)
        self.assertEqual(grown[0].dtype, np.int32)


if __name__ == "__main__":
    unittest.main()
