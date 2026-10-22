"""S1-7 블록 쌍 시험대 (D69·D76): 공유 격벽·관통 창 — 라우터 Dijkstra 대조, 검증기 crossing 모듈, 로더·생성기."""
import math
import unittest

import numpy as np

from pipe_routing import astar_fast
from pipe_routing import blockpair as bp
from pipe_routing.generator import PairConfig, generate_block_pair
from pipe_routing.layered import LayeredGraph, _route, clear_cache, make_layered_router
from pipe_routing.router_astar import astar_route
from pipe_routing.scenario import ScenarioError, from_dict, to_dict
from pipe_routing.verifier import PipeRoute, verify

S = 40000
WIN = [18000, 22000, 4000, 6000]


def pdict(pid, s, e, sd, ed, size="100A", win=None, kind=("boundary", "boundary")):
    d = {"id": pid, "type_id": "T1", "nominal_size": size, "gravity_pipe": False, "min_slope": None,
         "insulation_thickness": 50, "start": {"pos": s, "kind": kind[0], "dir": sd},
         "end": {"pos": e, "kind": kind[1], "dir": ed}, "valve_positions": [], "branch_points": []}
    if win:
        d["crossing_window"] = {"y": win[:2], "z": win[2:]}
    return d


def pair(pipes, obstacles=()):
    return from_dict({"block": {"width": 2 * S, "length": 40000, "height": 10000, "unit": "mm"},
                      "block_pair": {"shared_x": S},
                      "obstacles": [{"id": f"B{i}", "type": "box", "min": lo, "max": hi}
                                    for i, (lo, hi) in enumerate(obstacles)],
                      "pipes": pipes})


X = pdict("X", [0, 20000, 5000], [2 * S, 20000, 5000], [1, 0, 0], [1, 0, 0], win=WIN)
# 창 근처 장애물: 창 앞을 반쯤 가린다 (관통 경로가 돌아가야 함)
OBS = [([S - 3000, 15000, 3000], [S - 1500, 20500, 7000]), ([S + 1500, 19500, 3000], [S + 3000, 25000, 7000])]


def crossing_ok(sc, wps, pid="X"):
    rep = verify(sc, [PipeRoute(pid, wps)], modules=["crossing"])["pipes"][pid]
    return rep.layer0["crossing"], [v.message for v in rep.violations]


class LoaderTest(unittest.TestCase):
    def test_window_required_and_roundtrip(self):
        bad = dict(X)
        bad.pop("crossing_window")
        with self.assertRaises(ScenarioError):
            pair([bad])
        one = pdict("A", [0, 20000, 1000], [20000, 0, 1000], [1, 0, 0], [0, -1, 0], win=WIN)
        with self.assertRaises(ScenarioError):   # 한 블록 안 배관에 창
            pair([one])
        sc = pair([X])
        self.assertEqual(to_dict(from_dict(to_dict(sc))), to_dict(sc))
        self.assertTrue(bp.crossing(sc, sc.pipes[0]))

    def test_generator_deterministic(self):
        a = to_dict(generate_block_pair(PairConfig(seed=20265008), "t"))
        b = to_dict(generate_block_pair(PairConfig(seed=20265008), "t"))
        self.assertEqual(a, b)
        sc = from_dict(a)
        cross = [p for p in sc.pipes if bp.crossing(sc, p)]
        self.assertEqual(len(cross), 4)
        self.assertTrue(all(p.crossing_window for p in cross))


class CrossingVerifierTest(unittest.TestCase):
    def setUp(self):
        self.sc = pair([X])

    def test_good_crossing(self):
        ok, msg = crossing_ok(self.sc, [[0, 20000, 5000], [2 * S, 20000, 5000]])
        self.assertTrue(ok, msg)

    def test_outside_window(self):
        sc = pair([pdict("X", [0, 10000, 5000], [2 * S, 10000, 5000], [1, 0, 0], [1, 0, 0], win=WIN)])
        ok, msg = crossing_ok(sc, [[0, 10000, 5000], [2 * S, 10000, 5000]])
        self.assertFalse(ok)
        self.assertTrue(any("격벽" in m for m in msg))

    def test_twice_and_oblique_and_close_bend(self):
        twice = [[0, 20000, 5000], [S + 2000, 20000, 5000], [S + 2000, 20000, 8000], [S - 2000, 20000, 8000],
                 [S - 2000, 20000, 5000], [2 * S, 20000, 5000]]
        ok, _ = crossing_ok(self.sc, twice)
        self.assertFalse(ok)
        # 45° 로 창을 지남
        obl = [[0, 19000, 5000], [S - 1000, 19000, 5000], [S + 1000, 21000, 5000], [2 * S, 21000, 5000]]
        sc = pair([pdict("X", [0, 19000, 5000], [2 * S, 21000, 5000], [1, 0, 0], [1, 0, 0], win=WIN)])
        ok, msg = crossing_ok(sc, obl)
        self.assertFalse(ok)
        # 격벽 바로 앞(창 안)에서 꺾음 — 엘보 호가 창 면에 r 안
        sc = pair([pdict("X", [0, 20000, 5000], [2 * S, 20000, 4900], [1, 0, 0], [1, 0, 0], win=WIN)])
        close = [[0, 20000, 5000], [S + 100, 20000, 5000], [S + 100, 20000, 4900], [2 * S, 20000, 4900]]
        ok, msg = crossing_ok(sc, close)
        self.assertFalse(ok)

    def test_single_block_pipe_must_not_cross(self):
        sc = pair([pdict("A", [0, 20000, 1000], [0, 30000, 1000], [1, 0, 0], [-1, 0, 0])])
        wps = [[0, 20000, 1000], [S + 5000, 20000, 1000], [S + 5000, 30000, 1000], [0, 30000, 1000]]
        ok, msg = crossing_ok(sc, wps, "A")
        self.assertFalse(ok)


@unittest.skipUnless(astar_fast.HAVE_NUMBA, "numba 없음")
class PairRouterTest(unittest.TestCase):
    def test_route_verifies_and_matches_dijkstra(self):
        """공유 벽 + 창 + 창 앞 장애물: 컴파일 A* (창 휴리스틱 포함) J = 가지치기 없는 파이썬 Dijkstra J (D40 ①),
        찾은 경로는 crossing·collision·bend·boundary·support 위반 0."""
        clear_cache()
        sc = pair([X], OBS)
        p = sc.pipes[0]
        g = LayeredGraph(sc, p)
        fast = _route(g, p, 60)
        self.assertEqual(fast.status, "ok")
        dij = astar_route(g, p, 600, heuristic=lambda s: 0.0)
        self.assertEqual(dij.status, "ok")
        self.assertTrue(math.isclose(fast.J, dij.J, rel_tol=1e-9, abs_tol=1e-6), (fast.J, dij.J))
        rep = verify(sc, [PipeRoute("X", fast.waypoints)])["pipes"]["X"]
        self.assertTrue(rep.success, [v.message for v in rep.violations])
        W = np.array(fast.waypoints)
        k = np.nonzero(np.diff(np.sign(W[:, 0] - S)))[0]
        self.assertEqual(len(k), 1)

    def test_single_block_pipe_stays(self):
        """한 블록 안 배관은 막힌 격벽을 넘지 않는다: 블록 A 를 가로막는 칸막이(x 1000 ~ 격벽)가 있어 x < 1000 틈으로만
        돌아갈 수 있다. 격벽 너머(블록 B)로 돌아가는 길은 없다."""
        clear_cache()
        a = pdict("A", [0, 20000, 1000], [0, 30000, 1000], [1, 0, 0], [-1, 0, 0])
        wall = ([1000, 24000, 0], [S, 26000, 10000])
        sc = pair([a], [wall])
        r = make_layered_router()(sc, sc.pipes[0], 60)
        self.assertEqual(r.status, "ok")
        self.assertTrue(all(w[0] < 1000 for w in r.waypoints if 24000 - 200 < w[1] < 26000 + 200))
        self.assertTrue(all(w[0] < S for w in r.waypoints))
        ok, msg = crossing_ok(sc, r.waypoints, "A")
        self.assertTrue(ok, msg)

    def test_shared_wall_is_support_face(self):
        """격벽 바로 옆 수직 배관 (블록 A, 격벽에서 600) 은 격벽에 붙인 벽 형식(A·B)으로 받친다 (D76 ①)."""
        p = pdict("V", [S - 600, 20000, 0], [S - 600, 20000, 10000], [0, 0, 1], [0, 0, 1])
        sc = pair([p])
        rep = verify(sc, [PipeRoute("V", [[S - 600, 20000, 0], [S - 600, 20000, 10000]])])["pipes"]["V"]
        self.assertTrue(rep.layer0["support"], [v.message for v in rep.violations])
        self.assertTrue(rep.supports and all(s["face"] == "wall_x1" for s in rep.supports))
        for s in rep.supports:
            for m in s["members"]:
                self.assertLessEqual(max(m["a"][0], m["b"][0]), S + 1e-6)


T = 20   # 두꺼운 격벽 (D93)
WIN_LOW = [18000, 22000, 200, 2200]


def thick_pair(pipes, obstacles=()):
    """블록 A [0, S] · 격벽 [S, S + T] · B [S + T, 2S + T]."""
    return from_dict({"block": {"width": 2 * S + T, "length": 40000, "height": 10000, "unit": "mm"},
                      "block_pair": {"shared_x": S + T / 2, "wall_thickness": T},
                      "obstacles": [{"id": f"B{i}", "type": "box", "min": lo, "max": hi}
                                    for i, (lo, hi) in enumerate(obstacles)],
                      "pipes": pipes})


class D92BudgetTest(unittest.TestCase):
    def test_per_set_budget(self):
        """D92 ③: 블록 쌍 세트만 2²³ (진단 4배), 단일 블록은 2²² 그대로 (D73)."""
        from pipe_routing.layered import EXPAND_BUDGET, EXPAND_BUDGET_PAIR
        r = make_layered_router()
        sc = pair([X])
        self.assertEqual(EXPAND_BUDGET_PAIR, 2 ** 23)
        self.assertEqual(r.budget_for(sc), {"kind": "expansions", "limit": 2 ** 23, "set": "block_pair",
                                            "diagnosis_limit": 2 ** 25})
        single = from_dict({"block": {"width": 40000, "length": 40000, "height": 10000, "unit": "mm"}, "obstacles": [],
                            "pipes": [pdict("A", [0, 30000, 5000], [0, 35000, 5000], [1, 0, 0], [-1, 0, 0])]})
        self.assertEqual(r.budget_for(single)["limit"], EXPAND_BUDGET)
        self.assertEqual(r.diagnostic.budget_for(sc)["limit"], 2 ** 25)
        self.assertEqual(r.diagnostic.budget_for(single)["limit"], 4 * EXPAND_BUDGET)


class D93LoaderTest(unittest.TestCase):
    def test_obstacle_attributes(self):
        base = {"block": {"width": 40000, "length": 40000, "height": 10000, "unit": "mm"},
                "pipes": [pdict("A", [0, 30000, 5000], [0, 35000, 5000], [1, 0, 0], [-1, 0, 0])]}
        sc = from_dict({**base, "obstacles": [{"id": "O", "type": "plate", "min": [0, 0, 0], "max": [5000, 20, 3000]}]})
        o = sc.obstacles[0]
        self.assertFalse(o.supportable or o.penetrable)
        self.assertEqual(o.pen_axis(), 1)   # 가장 짧은 변 축 (y)
        for attr in ("supportable", "penetrable"):   # S2 예정 — 미지원 (M27)
            with self.assertRaises(ScenarioError) as cm:
                from_dict({**base, "obstacles": [{"id": "O", "type": "plate", "min": [0, 0, 0], "max": [5000, 20, 3000],
                                                  attr: True}]})
            self.assertIn("S2", str(cm.exception))

    def test_wall_thickness(self):
        sc = thick_pair([pdict("X", [0, 20000, 5000], [2 * S + T, 20000, 5000], [1, 0, 0], [1, 0, 0], win=WIN)])
        self.assertEqual(bp.walls(sc), ((S, S + T),))
        self.assertTrue(bp.crossing(sc, sc.pipes[0]))
        self.assertEqual(to_dict(from_dict(to_dict(sc))), to_dict(sc))
        with self.assertRaises(ScenarioError):   # 장애물이 격벽과 겹침
            thick_pair([pdict("A", [0, 30000, 5000], [0, 35000, 5000], [1, 0, 0], [-1, 0, 0])],
                       [([S - 100, 0, 0], [S + 5, 1000, 1000])])

    def test_generator_thickness(self):
        sc = generate_block_pair(PairConfig(seed=20265008), "t")
        (a, b), = bp.walls(sc)
        self.assertEqual((a, b), (S, S + PairConfig.wall_thickness))
        self.assertEqual(sc.block.width, 2 * S + PairConfig.wall_thickness)


@unittest.skipUnless(astar_fast.HAVE_NUMBA, "numba 없음")
class D94RouterTest(unittest.TestCase):
    """D94 관통 규칙 (두꺼운 격벽): 면에서 엘보 접점까지 직관 ≥ max(3D, r), 관통부 = 서포트 고정점, D95 관통 비용."""

    def setUp(self):
        self._d94 = bp.D94
        bp.D94 = True
        clear_cache()

    def tearDown(self):
        bp.D94 = self._d94
        clear_cache()

    def test_dijkstra_and_rules(self):
        x = pdict("X", [0, 20000, 1000], [2 * S + T, 19000, 1200], [1, 0, 0], [1, 0, 0], win=WIN_LOW)
        sc = thick_pair([x], [([S - 3000, 15000, 0], [S - 1500, 19700, 3000])])
        p = sc.pipes[0]
        g = LayeredGraph(sc, p)
        fast = _route(g, p, 60)
        self.assertEqual(fast.status, "ok")
        dij = astar_route(g, p, 600, heuristic=lambda s: 0.0)
        self.assertEqual(dij.status, "ok")
        self.assertTrue(math.isclose(fast.J, dij.J, rel_tol=1e-9, abs_tol=1e-6), (fast.J, dij.J))
        pen = bp.penetration_kg(sc, p)
        self.assertGreater(pen, 0)
        self.assertTrue(math.isclose(fast.J_penetration, pen, rel_tol=1e-9))
        rep = verify(sc, [PipeRoute("X", fast.waypoints)])["pipes"]["X"]
        self.assertTrue(rep.success, [v.message for v in rep.violations])
        # 관통 직관 양쪽 꺾임점은 격벽 면에서 max(3D, r) + t 이상
        W = np.array(fast.waypoints)
        i = int(np.nonzero((W[:-1, 0] < S) & (W[1:, 0] > S + T))[0][0])
        from pipe_routing.constants import elbow_tangent
        need = bp.straight_min(p) + elbow_tangent(p.nominal_size, 45) - 1e-6
        if i > 0:
            self.assertGreaterEqual(S - W[i, 0], need)
        if i + 1 < len(W) - 1:
            self.assertGreaterEqual(W[i + 1, 0] - (S + T), need)

    def test_crossing_is_support_fixed_point(self):
        """바닥 가까이 창을 지나는 직선 — 창 구멍 근처는 부재를 둘 수 없지만 관통부가 고정점이라 support 통과."""
        x = pdict("X", [0, 20000, 1000], [2 * S + T, 20000, 1000], [1, 0, 0], [1, 0, 0], win=WIN_LOW)
        sc = thick_pair([x])
        wps = [[0, 20000, 1000], [2 * S + T, 20000, 1000]]
        rep = verify(sc, [PipeRoute("X", wps)])
        r = rep["pipes"]["X"]
        self.assertTrue(r.layer0["support"], [v.message for v in r.violations])
        self.assertTrue(r.layer0["crossing"], [v.message for v in r.violations])
        s_mm = sorted(s["s_mm"] for s in r.supports)
        fixed = S + T / 2
        self.assertNotIn(round(fixed, 1), s_mm)   # 관통부에는 서포트를 두지 않는다 (무게 0 고정점)
        gaps = np.diff([0.0] + [v for v in s_mm if v < fixed] + [fixed])
        self.assertTrue(np.all(gaps <= 3000 + 1e-6))
        g = LayeredGraph(sc, sc.pipes[0])
        res = _route(g, sc.pipes[0], 60)
        self.assertEqual(res.status, "ok")


if __name__ == "__main__":
    unittest.main()
