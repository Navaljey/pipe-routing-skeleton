"""S2-3 시험 — 분기 1개 정확 풀이 (D112 · D134) · 검증기 나무 규칙 · tee_at 힌트.

(a)·(c) 정확 풀이 = 가지치기 없는 파이썬 Dijkstra (주관 + 티 전이, 다리 한 점 전) 의 최적 — 라우터 비용 일치
(b) 경로 있는 계통의 정합 위반 0 (branch·bend·boundary·collision)
(d) 결정성 — 같은 입력 두 번 = 같은 출력
(e) tee_at = 검증기 branch 판정, 힌트 호출 전후 채점 결과 불변
검증기 규칙: 다리 없음 · 끝점이 자기 단자가 아님 · 티가 주관 밖 (다리끼리 연결 = 사이클) · 135° · 티 → 엘보 < 3D ·
자기 간섭 (티 몸체 구역 밖 위반, 안은 통과) · 같은 계통 D 묶음 (보완 5).
"""
import heapq
import itertools
import json
import math
import unittest

import numpy as np

from pipe_routing import astar_fast
from pipe_routing import tree as T
from pipe_routing.constants import elbow_tangent
from pipe_routing.layered import LayeredGraph, clear_cache, make_layered_router
from pipe_routing.scenario import from_dict
from pipe_routing.space import DIRS, State
from pipe_routing.verifier import PipeRoute, verify
from pipe_routing.verifier.hint import SupportHint

BLOCK = {"width": 12000, "length": 12000, "height": 6000, "unit": "mm"}


def system(flow="out", leg_size="50A", obstacles=(), leg_pos=(6000, 12000, 1000), leg_dir=None):
    leg_dir = leg_dir or ([0, 1, 0] if flow == "out" else [0, -1, 0])
    p = {"id": "P", "type_id": "T3", "nominal_size": "100A", "gravity_pipe": False, "min_slope": None,
         "start": {"pos": [0, 4000, 1000], "kind": "boundary", "dir": [1, 0, 0]},
         "end": {"pos": [12000, 4000, 1000], "kind": "boundary", "dir": [1, 0, 0]},
         "branches": [{"id": "B1", "nominal_size": leg_size, "pos": list(leg_pos), "kind": "boundary",
                       "dir": leg_dir, "flow": flow}]}
    return from_dict({"block": BLOCK, "obstacles": list(obstacles), "pipes": [p]})


OBS = [{"id": "BOX", "type": "tank", "min": [4000, 6000, 0], "max": [8000, 9000, 5000]}]


# --------------------------------------------------------------------------- 파이썬 기준 (가지치기 없음)

class _Proxy:
    """LayeredGraph 를 감싸 bend 등급 "티" (-1) 와 거꾸로 걷는 다리의 B 쪽 규칙을 더한다."""

    def __init__(self, g, tee_need=None, rev_tan=None):
        self._g, self._tee, self._rev = g, tee_need, rev_tan
        self.run_cap = dict(g.run_cap)
        if tee_need:
            self.run_cap[-1] = max(max(tee_need.values()), g.run_cap[135])

    def __getattr__(self, k):
        return getattr(self._g, k)

    def turn_need(self, state, defl):
        if state.bend == -1:
            return self._tee[defl]
        if self._rev is not None and state.bend == 0:
            return self._rev[defl]
        return self._g.turn_need(state, defl)


def _dijkstra(start, succ, accept):
    """모든 상태 펼침, 가지치기 없음. accept(s) → 키 또는 None. 반환 {키: 최소 g}."""
    tie = itertools.count()
    best = {start: 0.0}
    heap = [(0.0, next(tie), start)]
    out = {}
    while heap:
        g, _, s = heapq.heappop(heap)
        if g > best[s] + 1e-12:
            continue
        k = accept(s)
        if k is not None and (k not in out or g < out[k]):
            out[k] = g
        for n, c in succ(s):
            ng = g + c
            if n not in best or ng < best[n] - 1e-12:
                best[n] = ng
                heapq.heappush(heap, (ng, next(tie), n))
    return out


def python_tree_optimum(sc):
    from pipe_routing.tree_router import leg_search_pipe
    pipe = sc.pipes[0]
    b = pipe.branches[0]
    lp = leg_search_pipe(pipe, b)
    gm = LayeredGraph(sc, pipe, support_cost=False, partners=(lp,))
    gl = LayeredGraph(sc, lp, support_cost=False, partners=(pipe,))
    tl = {a: elbow_tangent(b.nominal_size, a) for a in (0, 45, 90, 135)}
    rev = {a: tl[a] + (gl.r_bnd if gl.start_boundary else 0.0) for a in (45, 90, 135)} if b.flow == "out" else None
    PL = _Proxy(gl, rev_tan=rev)
    from pipe_routing.layered import LayeredGraph as LG

    def leg_succ(s):
        for n in LG.neighbors(PL, s):
            yield n, gl.cost(s, n, lp)

    def leg_accept(s):
        if s.run >= T.three_d(b.nominal_size) + tl[s.bend] - 1e-6 and (s.ends & 1):
            return (s.node, s.dir)
        return None
    legc = _dijkstra(gl.start_state(lp), leg_succ, leg_accept)
    teec = {}
    for (n, a), c in legc.items():
        try:
            m = gm._id_of(tuple(gl.pos[n]))
        except (AssertionError, IndexError, ValueError):
            continue
        if not np.array_equal(gm.pos[m], gl.pos[n]):
            continue
        lf = np.asarray(DIRS[a], float) * (-1 if b.flow == "out" else 1)
        lf /= np.linalg.norm(lf)
        for cd in range(6):
            ang = math.degrees(math.acos(float(np.clip(np.dot(DIRS[cd], lf), -1, 1))))
            if min(abs(ang - 45), abs(ang - 90)) < 1e-6:
                teec[(m, cd)] = min(teec.get((m, cd), math.inf), c)
    tm = {a: elbow_tangent(pipe.nominal_size, a) for a in (0, 45, 90, 135)}
    D3 = T.three_d(pipe.nominal_size)
    tee_need = {a: max(gm.L, D3 + tm[a]) for a in (45, 90, 135)}
    PM = _Proxy(gm, tee_need=tee_need)

    def pre(s):
        return (gm.r_bnd if gm.start_boundary else 0.0) if s.bend == 0 else D3 + tm[s.bend]

    def main_succ(x):
        s, t = x
        if t == 0 and (s.node, s.dir) in teec and s.run >= pre(s) - 1e-6 and (s.ends & 1):
            yield (State(s.node, s.dir, 0.0, -1, s.ends, s.hz, s.su), 1), teec[(s.node, s.dir)]
        for n in LG.neighbors(PM, s):
            c = gm.cost(State(s.node, s.dir, s.run, 0 if s.bend == -1 else s.bend, s.ends, s.hz, s.su), n, pipe)
            yield (n, t), c

    end_dir = DIRS.index(tuple(pipe.end.dir))

    def main_accept(x):
        s, t = x
        if t != 1 or s.node != gm.end_node or s.dir != end_dir or not (s.ends & 1):
            return None
        need = (gm.r_bnd if gm.end_boundary else 0.0) if s.bend == -1 else \
            tm[s.bend] + (gm.r_bnd if s.bend and gm.end_boundary else 0.0)
        return "goal" if s.run >= need - 1e-6 else None
    res = _dijkstra((gm.start_state(pipe), 0), main_succ, main_accept)
    return res.get("goal")


@unittest.skipUnless(astar_fast.HAVE_NUMBA, "numba 없음")
class ExactTest(unittest.TestCase):
    def _route(self, sc, support=False):
        clear_cache()
        router = make_layered_router(support_cost=support, reserve=False, member_reserve=False, diagnostic=False)
        return router(sc, sc.pipes[0], 60, [])

    def test_exact_equals_python_optimum(self):
        """(a)·(c) 정확 풀이 라우터 비용 = 가지치기 없는 파이썬 Dijkstra (티 후보 전부) 의 최적."""
        for flow in ("out", "in"):
            with self.subTest(flow=flow):
                sc = system(flow, obstacles=OBS)
                r = self._route(sc)
                self.assertEqual(r.status, "ok")
                self.assertEqual(r.tree["method"], "exact")
                clear_cache()
                ref = python_tree_optimum(sc)
                self.assertIsNotNone(ref)
                self.assertTrue(math.isclose(r.J, ref, rel_tol=1e-9, abs_tol=1e-6), (r.J, ref))

    def test_tree_passes_verifier_and_deterministic(self):
        """(b) 정합 위반 0 · (d) 결정성."""
        outs = []
        for _ in range(2):
            sc = system("out", obstacles=OBS)
            r = self._route(sc, support=True)
            rep = verify(sc, [PipeRoute("P", r.waypoints, r.branches)])["pipes"]["P"]
            self.assertEqual([v.message for v in rep.violations], [])
            self.assertTrue(rep.success)
            outs.append(json.dumps([r.waypoints, r.branches, r.tree], sort_keys=True))
        self.assertEqual(outs[0], outs[1])


class BoundedRebuildTest(unittest.TestCase):
    def test_sequential_rebuilds_bounded_leg_table(self):
        """F1 (2026-11-24 사양 검토): 한계 c 까지의 다리 표 (bounded) 에 든 티가 모두 자기 간섭으로 막히고 c 보다 비싼 티만
        가능하면, 순차 근사 전에 다리 표를 끝까지 다시 만들어 다리를 찾는다. 자기 간섭은 _conflicts 를 바꿔 강제한다 —
        bounded 표에 든 다리 끝 (티 위치 + 티에 닿는 방향) 은 모두 간섭."""
        from pipe_routing import tree_router as TR
        orig_table, orig_conf = TR.leg_table, TR._conflicts
        cheap = set()   # (티 위치, 다리 마지막 구간 방향) — 비용 ≤ c

        def key(pos, d):
            v = np.asarray(d, float)
            return tuple(map(float, pos)), tuple(np.round(v / np.linalg.norm(v), 6))

        def table(g, lp, b, budget, bound_keys=None):
            tab = orig_table(g, lp, b, budget, bound_keys)
            if bound_keys is not None and tab["status"] == "bounded":
                cheap.update(key(g.pos[n], DIRS[d]) for n, d in tab["cand"])   # 한계 c 까지 닿은 다리 끝 전부
            return tab

        def conflicts(pipe, main_wps, legs):
            for _, geom in legs:
                if key(geom[0], np.subtract(geom[0], geom[1])) in cheap:
                    return ["강제 간섭"]
            return orig_conf(pipe, main_wps, legs)

        TR.leg_table, TR._conflicts = table, conflicts
        try:
            # 긴 주관 (40 m) — 한계 c 안의 다리 표는 다리 단자 근처 주관만 덮는다
            # 작은 장애물 열 — 격자선이 늘어 다리 표가 한계 c 에서 멈춘다 (bounded)
            obs = [{"id": f"O{k}", "type": "box", "min": [10000 + 3000 * k, 7000, 3000],
                    "max": [11000 + 3000 * k, 8000, 4000]} for k in range(9)]
            sc = from_dict({"block": {"width": 40000, "length": 12000, "height": 6000, "unit": "mm"}, "obstacles": obs,
                            "pipes": [{"id": "P", "type_id": "T3", "nominal_size": "100A", "gravity_pipe": False,
                                       "min_slope": None,
                                       "start": {"pos": [0, 4000, 1000], "kind": "boundary", "dir": [1, 0, 0]},
                                       "end": {"pos": [40000, 4000, 1000], "kind": "boundary", "dir": [1, 0, 0]},
                                       "branches": [{"id": "B1", "nominal_size": "50A", "pos": [6000, 12000, 1000],
                                                     "kind": "boundary", "dir": [0, 1, 0], "flow": "out"}]}]})
            clear_cache()
            r = make_layered_router(support_cost=False, reserve=False, member_reserve=False, diagnostic=False)(
                sc, sc.pipes[0], 60, [])
            ok_after = r
        finally:
            TR.leg_table, TR._conflicts = orig_table, orig_conf
        self.assertTrue(cheap)
        r = ok_after
        self.assertEqual(r.status, "ok")
        self.assertEqual(r.tree["method"], "sequential")
        self.assertEqual(r.tree["leg_search"], ["bounded"])
        self.assertEqual(r.tree["leg_search_final"], ["done"])
        self.assertTrue(r.tree["search"]["leg_rebuild"])
        leg = r.tree["legs"][0]
        self.assertTrue(leg["found"])
        self.assertNotIn(key(leg["waypoints"][0], np.subtract(leg["waypoints"][0], leg["waypoints"][1])), cheap)


# --------------------------------------------------------------------------- 검증기 나무 규칙

MAIN = [[0, 4000, 1000], [12000, 4000, 1000]]
LEG = [[6000, 4000, 1000], [6000, 12000, 1000]]


def check(sc, legs, main=MAIN):
    rep = verify(sc, [PipeRoute("P", main, legs)])["pipes"]["P"]
    return rep


class VerifierTreeTest(unittest.TestCase):
    def test_good_tree(self):
        rep = check(system(), [LEG])
        self.assertTrue(rep.success, [v.message for v in rep.violations])
        self.assertEqual(sum(1 for f in rep.fittings if f["type"] == "tee"), 1)

    def test_missing_leg(self):
        rep = check(system(), [[]])
        self.assertFalse(rep.layer0["branch"])

    def test_leg_end_not_terminal(self):
        rep = check(system(), [[[6000, 4000, 1000], [6000, 11000, 1000]]])
        self.assertFalse(rep.layer0["branch"])

    def test_tee_off_main_is_cycle_or_floating(self):
        """다리 시작점이 주관 위가 아니면 (다른 다리 위 = 사이클 포함) branch 위반 (D134 ④)."""
        rep = check(system(), [[[6000, 4600, 1000], [6000, 12000, 1000]]])
        self.assertFalse(rep.layer0["branch"])

    def test_angle_135_rejected(self):
        """in 다리 흐름이 주관 흐름과 135° — 기하 방향이 +x 성분 45° 면 합류 흐름은 −x 쪽 → 135°."""
        sc = system("in", leg_pos=(9000, 12000, 1000), leg_dir=[0, -1, 0])
        legs = [[[1000, 4000, 1000], [3000, 6000, 1000], [9000, 6000, 1000], [9000, 12000, 1000]]]
        rep = check(sc, legs)
        self.assertFalse(rep.layer0["branch"])
        self.assertTrue(any("135" in v.message or "분기각" in v.message for v in rep.violations))

    def test_tee_too_close_to_elbow(self):
        main = [[0, 4000, 3000], [6000, 4000, 3000], [6000, 4000, 100]]
        sc = from_dict({"block": BLOCK, "obstacles": [], "pipes": [{
            "id": "P", "type_id": "T3", "nominal_size": "100A", "gravity_pipe": False, "min_slope": None,
            "start": {"pos": [0, 4000, 3000], "kind": "boundary", "dir": [1, 0, 0]},
            "end": {"pos": [6000, 4000, 0], "kind": "boundary", "dir": [0, 0, -1]},
            "branches": [{"id": "B1", "nominal_size": "50A", "pos": [5700, 12000, 3000], "kind": "boundary",
                          "dir": [0, 1, 0]}]}]})
        main = [[0, 4000, 3000], [6000, 4000, 3000], [6000, 4000, 0]]
        rep = check(sc, [[[5700, 4000, 3000], [5700, 12000, 3000]]], main)
        self.assertFalse(rep.layer0["branch"])
        self.assertTrue(any("3D_주관" in v.message for v in rep.violations))

    def test_family_clearance_zone(self):
        """자기 간섭: 티 몸체 구역 안 (티 바로 옆) 은 통과, 구역 밖에서 다리가 자기 주관에 다시 붙으면 collision 위반."""
        ok = check(system(), [LEG])
        self.assertTrue(ok.layer0["collision"])
        back = [[6000, 4000, 1000], [6000, 6000, 1000], [9000, 6000, 1000], [9000, 4150, 1000], [11000, 4150, 1000],
                [11000, 12000, 1000]]
        sc = system(leg_pos=(11000, 12000, 1000))
        bad = check(sc, [back])
        self.assertFalse(bad.layer0["collision"])
        self.assertTrue(any("자기 주관" in v.message for v in bad.violations))


class SameSystemGroupTest(unittest.TestCase):
    def test_main_and_own_leg_share_d(self):
        """D134 보완 5: 주관과 나란한 자기 다리 (D82 조건) 는 D 로 묶인다 — 몫 D81."""
        p = {"id": "P", "type_id": "T3", "nominal_size": "100A", "gravity_pipe": False, "min_slope": None,
             "start": {"pos": [0, 4000, 600], "kind": "boundary", "dir": [1, 0, 0]},
             "end": {"pos": [12000, 4000, 600], "kind": "boundary", "dir": [1, 0, 0]},
             "branches": [{"id": "B1", "nominal_size": "100A", "pos": [12000, 4600, 600], "kind": "boundary",
                           "dir": [1, 0, 0]}]}
        sc = from_dict({"block": BLOCK, "obstacles": [], "pipes": [p]})
        main = [[0, 4000, 600], [12000, 4000, 600]]
        leg = [[600, 4000, 600], [600, 4600, 600], [12000, 4600, 600]]
        rep = verify(sc, [PipeRoute("P", main, [leg])])["pipes"]["P"]
        self.assertTrue(rep.success, [v.message for v in rep.violations])
        d = [x for x in rep.supports if x["form"] == "D"]
        self.assertTrue(d)
        self.assertTrue(all(sorted(x["carried"]) == ["P", "P.B1"] for x in d))


class HintTest(unittest.TestCase):
    def test_tee_at_matches_verifier_and_is_readonly(self):
        sc = system()
        before = verify(sc, [PipeRoute("P", MAIN, [LEG])])["pipes"]["P"].to_dict()
        h = SupportHint(sc, [PipeRoute("P", MAIN, [LEG])])
        good = h.tee_at("P", [6000, 4000, 1000], [0, 1, 0], "50A", "out")
        self.assertTrue(good["ok"], good)
        self.assertEqual(good["angle"], 90.0)
        off = h.tee_at("P", [6000, 4500, 1000], [0, 1, 0], "50A", "out")
        self.assertFalse(off["ok"])
        r135 = h.tee_at("P", [6000, 4000, 1000], [1, 1, 0], "50A", "in")   # 합류 흐름 (−1, −1) ∠ 주관 +x = 135°
        self.assertFalse(r135["ok"])
        bad = check(system("in", leg_pos=(9000, 12000, 1000), leg_dir=[0, -1, 0]),
                    [[[1000, 4000, 1000], [3000, 6000, 1000], [9000, 6000, 1000], [9000, 12000, 1000]]])
        self.assertFalse(bad.layer0["branch"])   # 같은 함수 — 검증기도 135° 위반
        after = verify(sc, [PipeRoute("P", MAIN, [LEG])])["pipes"]["P"].to_dict()
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
