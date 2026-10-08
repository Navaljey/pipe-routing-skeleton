import random
import unittest
from collections import deque
from pathlib import Path

import numpy as np

from pipe_routing.escape_graph import EscapeGraph, planar_segment_box_distance
from pipe_routing.scenario import load
from pipe_routing.space import ALLOWED_DEFLECTIONS, DEFLECTION, DIR_INDEX, DIRS, deflection_deg

ROOT = Path(__file__).resolve().parent.parent
MANUAL = ROOT / "scenarios" / "manual" / "manual_01.json"


def sampled_clearance(g: EscapeGraph, a, b, n=400) -> float:
    """선분 위 n 점 샘플의 장애물 최소거리 (정확 계산 대조용)."""
    t = np.linspace(0, 1, n)[:, None]
    pts = np.asarray(a) + t * (np.asarray(b) - np.asarray(a))
    return float(g._point_clearance(pts).min())


class DistanceTest(unittest.TestCase):
    def test_planar_distance_matches_sampling(self):
        rng = np.random.default_rng(0)
        for _ in range(300):
            lo = rng.uniform(0, 50, 3)
            hi = lo + rng.uniform(1, 30, 3)
            c = int(rng.integers(3))
            p0 = rng.uniform(-20, 100, 3)
            p1 = p0.copy()
            a, b = [ax for ax in range(3) if ax != c]
            if rng.random() < 0.5:   # 45°
                t = rng.uniform(1, 60)
                p1[a] += t * rng.choice((-1, 1)); p1[b] += t * rng.choice((-1, 1))
            else:                    # 축 방향
                p1[rng.choice((a, b))] += rng.uniform(-60, 60)
            exact = planar_segment_box_distance(p0[None], p1[None], lo, hi, c)[0]
            s = np.linspace(0, 1, 4001)[:, None]
            pts = p0 + s * (p1 - p0)
            g = np.maximum(0, np.maximum(lo - pts, pts - hi))
            sampled = np.sqrt((g * g).sum(1)).min()
            self.assertLessEqual(exact, sampled + 1e-9)
            self.assertAlmostEqual(exact, sampled, delta=np.linalg.norm(p1 - p0) / 4000 + 1e-9)


class GraphTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sc = load(MANUAL)
        cls.graphs = {p.id: EscapeGraph(cls.sc, p) for p in cls.sc.pipes}

    def all_edges(self, g):
        for d in range(len(DIRS)):
            if d < 6 and DIRS[d][d // 2] < 0:
                continue   # 축 엣지는 + 방향으로 한 번만
            mask = g.axis_ok[d // 2] if d < 6 else g.diag_ok[d]
            for idx in zip(*np.nonzero(mask)):
                yield idx, g._step(idx, d), d

    def test_edges_are_clear(self):
        """모든 엣지가 샘플링으로도 유효 반경 이상 떨어져 있다 (D17)."""
        for g in self.graphs.values():
            for n1, n2, d in self.all_edges(g):
                a, b = g.position(n1), g.position(n2)
                self.assertGreaterEqual(sampled_clearance(g, a, b, 50), g.r - 1e-6)
                self.assertTrue(g.is_free(a, b))

    def test_blocked_axis_edges_really_collide(self):
        """양 끝이 노드인데 막힌 축 엣지는 실제로 유효 반경 안을 지난다 (과잉 차단 없음)."""
        g = self.graphs["P002"]
        n = 0
        for ax in range(3):
            both = np.moveaxis(g.node_ok, ax, 0)
            pair = np.zeros(g.shape, dtype=bool)
            np.moveaxis(pair, ax, 0)[:-1] = both[:-1] & both[1:]
            for idx in zip(*np.nonzero(pair & ~g.axis_ok[ax])):
                nxt = tuple(v + 1 if i == ax else v for i, v in enumerate(idx))
                a, b = g.position(idx), g.position(nxt)
                self.assertLess(sampled_clearance(g, a, b, 2000), g.r + g.edge_length(idx, nxt) / 2000 + 1e-6)
                self.assertFalse(g.is_free(a, b))
                n += 1
        self.assertGreater(n, 0)

    def test_diag_symmetric(self):
        for g in self.graphs.values():
            for d in range(6, len(DIRS)):
                rev = DIR_INDEX[tuple(-v for v in DIRS[d])]
                self.assertEqual(int(g.diag_ok[d].sum()), int(g.diag_ok[rev].sum()))
                for idx in list(zip(*np.nonzero(g.diag_ok[d])))[:50]:
                    self.assertEqual(g._step(g._step(idx, d), rev), idx)

    def test_diag_edges_are_45deg(self):
        for g in self.graphs.values():
            for n1, n2, d in self.all_edges(g):
                if d < 6:
                    continue
                delta = [abs(a - b) for a, b in zip(g.position(n1), g.position(n2))]
                nz = sorted(x for x in delta if x)
                self.assertEqual(len(nz), 2)            # D15: 평면 내, √3 대각 없음
                self.assertAlmostEqual(nz[0], nz[1])    # 45°

    def test_neighbors_respect_bend_rules(self):
        random.seed(0)
        for g in self.graphs.values():
            s = g.start_state()
            first = list(g.neighbors(s))
            self.assertTrue(first, f"{g.pipe.id}: start 에서 나갈 수 없음")
            self.assertTrue(all(n.dir == s.dir for n in first))   # §3.3 시작 직진 강제
            seen, q = {s}, deque([s])
            while q and len(seen) < 3000:
                st = q.popleft()
                for nb in g.neighbors(st):
                    defl = DEFLECTION[st.dir][nb.dir]
                    self.assertIn(defl, ALLOWED_DEFLECTIONS)
                    if defl:
                        self.assertGreaterEqual(st.run, g.L)
                    self.assertLessEqual(nb.run, g.L)
                    self.assertGreater(g.cost(st, nb), 0)
                    if nb not in seen:
                        seen.add(nb); q.append(nb)

    def test_manual_pipes_reachable(self):
        """수작업 시나리오는 모든 배관이 그래프 위에서 start→end 도달 가능하다."""
        for pid, g in self.graphs.items():
            s = g.start_state()
            seen, q, ok = {s}, deque([s]), False
            while q:
                st = q.popleft()
                if g.is_goal(st):
                    ok = True
                    break
                for nb in g.neighbors(st):
                    if nb not in seen:
                        seen.add(nb); q.append(nb)
            self.assertTrue(ok, pid)

    def test_deflection_table(self):
        self.assertEqual(deflection_deg((1, 0, 0), (1, 1, 0)), 45)
        self.assertEqual(deflection_deg((1, 0, 0), (-1, 1, 0)), 135)
        self.assertEqual(deflection_deg((1, 1, 0), (1, 0, 1)), 60)     # 금지 (D37)
        self.assertEqual(deflection_deg((1, 0, 0), (-1, 0, 0)), 180)   # 금지

    def test_procedural_terminals_are_nodes(self):
        sc = load(ROOT / "scenarios" / "procedural" / "proc_000.json")
        for p in sc.pipes[:3]:
            g = EscapeGraph(sc, p)
            self.assertTrue(g.node_ok[g.start_node] and g.node_ok[g.end_node])
            self.assertTrue(list(g.neighbors(g.start_state())))
            st = g.stats()
            self.assertEqual(st["nodes"], int(g.node_ok.sum()))
            self.assertGreater(st["edges_45"], 0)


if __name__ == "__main__":
    unittest.main()
