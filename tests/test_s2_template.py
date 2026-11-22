"""S2-T 시험 — 템플릿 펼치기 (D124 ③) · 바닥 개구부 (D130) · 케이블 트레이 (D131).

① 펼치기 결정성: 같은 템플릿·시드·배관 수 → 같은 바이트, 다른 시드 → 다른 변형, sha256 기록, 구조 좌표 공유
② 바닥 개구부: 개구부 위에서는 바닥 지지 불가 (검증기), 라우터는 같은 판정으로 비켜 간다 (경로 검증 위반 0,
   Dijkstra 대조), 힌트 form_at 도 같은 답, 입출력 왕복
③ 케이블 트레이: 배관이 r 이상 비켜 가고 서포트 부재도 트레이에 닿지 않는다
④ 개구부·트레이 없는 시나리오는 불변 (구멍 목록 그대로 — E1 41 바이트 비교는 보고서).
"""
import hashlib
import math
import unittest

import numpy as np

from pipe_routing import astar_fast, template
from pipe_routing import blockpair as bp
from pipe_routing.layered import LayeredGraph, _route, clear_cache
from pipe_routing.router_astar import astar_route
from pipe_routing.scenario import from_dict, load, to_dict
from pipe_routing.verifier import PipeRoute, verify
from pipe_routing.verifier.geom import segment_box_distance
from pipe_routing.verifier.hint import SupportHint
from tests.test_s2_internal import BLOCK, pipe

TPL = "templates/er_2nd_generic_a.yaml"
OPEN = {"id": "H1", "min": [12000, 18000], "max": [28000, 22000]}
LINE = [[10000, 20000, 600], [30000, 20000, 600]]


def scen(obstacles=(), openings=(), *pipes):
    d = {"block": BLOCK, "obstacles": list(obstacles), "pipes": list(pipes)}
    if openings:
        d["floor_openings"] = list(openings)
    return from_dict(d)


def low_pipe():
    return pipe(LINE[0], LINE[1], kind=("nozzle", "nozzle"))


class ExpandTest(unittest.TestCase):
    def test_deterministic(self):
        t, sha = template.load_template(TPL)
        a = template.scenario_bytes(template.expand(t, sha, 40000001, 40))
        b = template.scenario_bytes(template.expand(t, sha, 40000001, 40))
        c = template.scenario_bytes(template.expand(t, sha, 40000002, 40))
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(sha, hashlib.sha256(open(TPL, "rb").read()).hexdigest())
        sc = template.expand(t, sha, 40000001, 40)
        self.assertEqual(sc.meta["template"]["sha256"], sha)
        self.assertEqual(len(sc.pipes), 40)
        self.assertEqual(sc.floor_openings, [{"id": "HATCH_1", "min": [27000, 4000], "max": [33000, 9000]}])

    def test_structure_fixed_and_shared(self):
        """구조는 시드와 무관 (variation.structure fixed), 반복 부재는 같은 좌표 열을 공유한다 (D124 ③)."""
        t, sha = template.load_template(TPL)
        s1 = {o.id: o.box for o in template.expand(t, sha, 30000001, 30).obstacles if o.type in
              ("stiffener", "web_frame", "pillar", "bulkhead", "casing")}
        s2 = {o.id: o.box for o in template.expand(t, sha, 30000002, 50).obstacles if o.type in
              ("stiffener", "web_frame", "pillar", "bulkhead", "casing")}
        self.assertEqual(s1, s2)
        cl = [b for k, b in s1.items() if k.startswith("CL_")]
        self.assertEqual({(b.min[0], b.max[0], b.min[2], b.max[2]) for b in cl}, {(0.0, 36000.0, 4950.0, 5200.0)})
        sf = [b for k, b in s1.items() if k.startswith("SF_")]
        self.assertEqual(len({b.min[0] for b in sf}), len(sf) // 2)        # 양현 늑골은 같은 x 열

    def test_tray_and_access_types(self):
        t, sha = template.load_template(TPL)
        sc = template.expand(t, sha, 40000003, 30)
        trays = [o for o in sc.obstacles if o.type == "tray"]
        self.assertTrue(trays)
        self.assertTrue(all(not (o.supportable or o.penetrable or o.access) for o in trays))   # D131
        self.assertTrue(all(o.access for o in sc.obstacles if o.type in ("passage", "stair", "handrail")))


class OpeningTest(unittest.TestCase):
    def test_io_and_holes(self):
        sc = scen((), [OPEN], low_pipe())
        self.assertEqual(to_dict(sc)["floor_openings"], [OPEN])
        self.assertEqual(from_dict(to_dict(sc)).floor_openings, sc.floor_openings)
        self.assertEqual(len(bp.hole_boxes(sc)), 1)
        e1 = load("scenarios/manual/manual_01.json")
        self.assertEqual(bp.hole_boxes(e1), [])
        self.assertNotIn("floor_openings", to_dict(e1))

    def test_no_floor_support_over_opening(self):
        """개구부 위 위치에서는 바닥 C 불가 — 직선은 support 위반, 개구부가 없으면 통과 (D130)."""
        for op, ok in (((), True), ([OPEN], False)):
            sc = scen((), op, low_pipe())
            rep = verify(sc, [PipeRoute("P", LINE)])["pipes"]["P"]
            self.assertEqual(rep.layer0["support"], ok)
            if op:
                lo, hi = bp.hole_boxes(sc)[0]
                for s in rep.supports:
                    if s["face"] == "floor":
                        for m in s["members"]:
                            d, _ = segment_box_distance(np.array([m["a"]], float), np.array([m["b"]], float), lo, hi)
                            self.assertGreater(float(d[0]), 0.0)
                h = SupportHint(sc, [PipeRoute("P", LINE)])
                row = [r for r in h.form_at("P", [20000, 20000, 600], [1, 0, 0])["options"]
                       if r["face"] == "floor" and r["form"] == "C"][0]
                self.assertEqual(row["status"], "blocked")                    # 힌트 = 검증기 판정

    @unittest.skipUnless(astar_fast.HAVE_NUMBA, "numba 없음")
    def test_router_matches_verifier(self):
        J = {}
        for op in ((), [OPEN]):
            clear_cache()
            sc = scen((), op, low_pipe())
            p = sc.pipes[0]
            g = LayeredGraph(sc, p)
            r = _route(g, p, 60, 2 ** 22)
            self.assertEqual(r.status, "ok")
            v = verify(sc, [PipeRoute("P", r.waypoints)])["pipes"]["P"]
            self.assertEqual([x.message for x in v.violations], [])          # 라우터 u = 검증기 판정
            d = astar_route(g, p, 600, heuristic=lambda s: 0.0)
            self.assertTrue(math.isclose(d.J, r.J, rel_tol=1e-9), (d.J, r.J))
            J[bool(op)] = r
        self.assertEqual(len(J[False].waypoints), 2)                          # 개구부 없음 → 직선
        self.assertGreater(J[True].J, J[False].J)                             # 개구부 → 비켜 감


@unittest.skipUnless(astar_fast.HAVE_NUMBA, "numba 없음")
class TrayTest(unittest.TestCase):
    TRAY = {"id": "TRAY", "type": "tray", "min": [18000, 19000, 150], "max": [22000, 21000, 300]}

    def test_router_avoids_and_members_clear(self):
        clear_cache()
        sc = scen([self.TRAY], (), low_pipe())
        rep = verify(sc, [PipeRoute("P", LINE)])["pipes"]["P"]
        lo, hi = np.array(self.TRAY["min"], float), np.array(self.TRAY["max"], float)
        for s in rep.supports:                                                # 부재가 트레이를 피한다 (D131)
            for m in s["members"]:
                d, _ = segment_box_distance(np.array([m["a"]], float), np.array([m["b"]], float), lo, hi)
                self.assertGreater(float(d[0]), 0.0)
        p = sc.pipes[0]
        g = LayeredGraph(sc, p)
        r = _route(g, p, 60, 2 ** 22)
        v = verify(sc, [PipeRoute("P", r.waypoints)])["pipes"]["P"]
        self.assertEqual([x.message for x in v.violations], [])
        self.assertTrue(v.layer0["collision"])


if __name__ == "__main__":
    unittest.main()
