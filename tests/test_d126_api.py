"""D126 지도 조회 API — 읽기 전용 공개 배열 · 공식 비용 덧셈 인자 · 공식 휴리스틱 인자.

인자를 쓰지 않으면 (또는 0 이면) 결과는 기준선과 비트 동일, 비용 인자를 넣으면 가지치기 없는 파이썬 Dijkstra 와 같은
비용, 허용 하한을 넣으면 같은 최적 비용 (D40 ①)."""
import heapq
import math
import unittest

import numpy as np

from pipe_routing import astar_fast
from pipe_routing.layered import (PUBLIC_ARRAYS, LayeredGraph, _route, clear_cache, make_layered_router,
                                  route_graph)
from pipe_routing.router_astar import astar_route
from pipe_routing.scenario import load
from tests.test_s1_corner import scen

MANUAL = "scenarios/manual/manual_01.json"


class _Extra:
    """파이썬 Dijkstra 용 — 엣지 비용에 extra[a.node, b.dir] 를 더한 같은 지도."""

    def __init__(self, g, extra):
        self._g, self._x = g, extra

    def __getattr__(self, k):
        return getattr(self._g, k)

    def cost(self, a, b, pipe=None):
        return self._g.cost(a, b, pipe) + float(self._x[a.node, b.dir])


def _sig(r):
    return (r.status, r.waypoints, r.J, r.expanded, r.generated,
            [(s.node, s.dir, s.run, s.bend) for s in (r.states or [])])


def _relaxed_lower(g, pipe):
    """노드별 하한 = 완화 그래프 (방향·직진·꺾임 무시) 위 목표까지 최단거리 — 엣지 무게 = 길이 × kg/m + scost
    (엣지 끝 마스크 전부 막힘은 제외). 실제 엣지 비용 ≥ 무게라 허용 하한 (D125 rg 와 같은 정의)."""
    N = len(g.pos)
    rev = [[] for _ in range(N)]
    for n in range(N):
        for d in range(18):
            m = int(g.step[n, d])
            if m >= 0 and int(g.emask[n, d]) != 0:
                rev[m].append((n, g.elen[n, d] / 1000 * pipe.kg_per_m + g.scost[n, d]))
    dist = np.full(N, np.inf)
    dist[g.end_node] = 0.0
    pq = [(0.0, int(g.end_node))]
    while pq:
        du, u = heapq.heappop(pq)
        if du > dist[u]:
            continue
        for v, w in rev[u]:
            if du + w < dist[v]:
                dist[v] = du + w
                heapq.heappush(pq, (dist[v], v))
    return np.where(np.isfinite(dist), dist * (1 - 1e-9), 0.0)   # 부동소수 여유 (하한 유지)


@unittest.skipUnless(astar_fast.HAVE_NUMBA, "numba 없음")
class MapApiTest(unittest.TestCase):
    def setUp(self):
        clear_cache()
        self.sc = load(MANUAL)

    def test_public_arrays_read_only(self):
        p = self.sc.pipes[0]
        g = LayeredGraph(self.sc, p)
        for name in PUBLIC_ARRAYS:
            a = getattr(g, name)
            self.assertFalse(a.flags.writeable, name)
            with self.assertRaises(ValueError):
                a.flat[0] = a.flat[0]
        self.assertFalse(g.scost.flags.writeable)

    def test_unused_or_zero_args_bit_identical(self):
        for p in self.sc.pipes:
            g = LayeredGraph(self.sc, p)
            base = _route(g, p, 60, 4_194_304)
            self.assertEqual(_sig(route_graph(g, p, expand_limit=4_194_304)), _sig(base))
            z = route_graph(g, p, expand_limit=4_194_304, extra_cost=np.zeros(g.scost.shape),
                            h_lower=np.zeros(len(g.pos)))
            self.assertEqual(_sig(z), _sig(base), p.id)

    def test_router_callbacks_none_identical(self):
        sc = self.sc
        a = make_layered_router()
        b = make_layered_router(extra_cost=lambda g, p, placed: None, h_lower=lambda g, p, placed: None)
        placed = []
        for p in sc.pipes:
            ra, rb = a(sc, p, 60, placed), b(sc, p, 60, placed)
            self.assertEqual(_sig(ra), _sig(rb))
            if ra.status == "ok":
                placed.append((p, ra.waypoints))

    def test_extra_cost_matches_dijkstra(self):
        sc = scen(50)
        p = sc.pipes[0]
        g = LayeredGraph(sc, p)
        rng = np.random.default_rng(7)
        extra = np.where(rng.random(g.scost.shape) < 0.3, rng.random(g.scost.shape) * 40.0, 0.0)
        fast = route_graph(g, p, extra_cost=extra)
        dij = astar_route(_Extra(g, extra), p, 300, heuristic=lambda s: 0.0)
        self.assertEqual(fast.status, "ok")
        got = fast.J + sum(float(extra[a.node, b.dir]) for a, b in zip(fast.states, fast.states[1:]))
        self.assertTrue(math.isclose(got, dij.J, rel_tol=1e-9, abs_tol=1e-6), (got, dij.J))
        self.assertFalse(np.shares_memory(g.scost, extra))
        self.assertEqual(float(g.scost.sum()), float(LayeredGraph(sc, p).scost.sum()))   # 지도는 그대로

    def test_admissible_lower_bound_same_cost_fewer_expansions(self):
        for p in self.sc.pipes:
            g = LayeredGraph(self.sc, p)
            base = route_graph(g, p)
            hl = _relaxed_lower(g, p)
            r = route_graph(g, p, h_lower=hl)
            self.assertEqual(r.status, base.status, p.id)
            if base.status == "ok":
                self.assertTrue(math.isclose(r.J, base.J, rel_tol=1e-9, abs_tol=1e-6), (p.id, r.J, base.J))
                self.assertLessEqual(r.expanded, base.expanded, p.id)

    def test_bad_args_rejected(self):
        p = self.sc.pipes[0]
        g = LayeredGraph(self.sc, p)
        with self.assertRaises(ValueError):
            route_graph(g, p, extra_cost=np.zeros((3, 18)))
        with self.assertRaises(ValueError):
            route_graph(g, p, extra_cost=-np.ones(g.scost.shape))
        with self.assertRaises(ValueError):
            route_graph(g, p, h_lower=np.full(len(g.pos), np.nan))


if __name__ == "__main__":
    unittest.main()
