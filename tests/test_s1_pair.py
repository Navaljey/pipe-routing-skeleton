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


if __name__ == "__main__":
    unittest.main()
