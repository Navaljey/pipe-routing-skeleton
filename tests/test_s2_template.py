"""S2-T 시험 — 템플릿 펼치기 (D124 ③) · 바닥 개구부 (D130) · 케이블 트레이 (D131).

① 펼치기 (형식 2, D136·D137): 결정성 · 기준 배치 · 구조 좌표 공유 · 성격 블록 · 노즐 · 계통·단자 · 본문 규칙 ·
   형식·use 검사 (eval + 학습·검증 시드 거부) · 점검·편집기 자료
② 바닥 개구부: 개구부 위에서는 바닥 지지 불가 (검증기), 라우터는 같은 판정으로 비켜 간다 (경로 검증 위반 0,
   Dijkstra 대조), 힌트 form_at 도 같은 답, 입출력 왕복
③ 케이블 트레이: 배관이 r 이상 비켜 가고 서포트 부재도 트레이에 닿지 않는다
④ 개구부·트레이 없는 시나리오는 불변 (구멍 목록 그대로 — E1 41 바이트 비교는 보고서).
"""
import copy
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

TPL = "tests/data/template_v2_fixture.yaml"
OPEN = {"id": "H1", "min": [12000, 18000], "max": [28000, 22000]}
LINE = [[10000, 20000, 600], [30000, 20000, 600]]


def scen(obstacles=(), openings=(), *pipes):
    d = {"block": BLOCK, "obstacles": list(obstacles), "pipes": list(pipes)}
    if openings:
        d["floor_openings"] = list(openings)
    return from_dict(d)


def low_pipe():
    return pipe(LINE[0], LINE[1], kind=("nozzle", "nozzle"))


def tpl(**edit):
    """시험 고정 사본을 읽고 (사전 경로, 값) 로 고친다 — 값 None = 지우기."""
    t, sha = template.load_template(TPL)
    t = copy.deepcopy(t)
    for path, v in edit.items():
        keys = path.split("__")
        d = t
        for k in keys[:-1]:
            d = d[int(k)] if isinstance(d, list) else d[k]
        if v is None:
            d.pop(keys[-1], None)
        elif isinstance(d, list):
            d[int(keys[-1])] = v
        else:
            d[keys[-1]] = v
    return t, sha


def errors_of(t, sha, seed=30000001, **kw):
    try:
        template.expand(t, sha, seed, **kw)
    except template.TemplateError as e:
        return e.errors
    return []


class ExpandTest(unittest.TestCase):
    def test_deterministic(self):
        t, sha = template.load_template(TPL)
        a = template.scenario_bytes(template.expand(t, sha, 30000001))
        b = template.scenario_bytes(template.expand(t, sha, 30000001))
        c = template.scenario_bytes(template.expand(t, sha, 30000002))
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(sha, hashlib.sha256(open(TPL, "rb").read()).hexdigest())
        m = template.expand(t, sha, 30000001).meta["template"]
        self.assertEqual((m["sha256"], m["format"], m["use"], m["seed_range"]), (sha, 2, "train", "train"))
        sc = template.expand(t, sha, 30000001, n_pipes=40)
        self.assertEqual(sum(1 + len(p.branches) for p in sc.pipes), 40)       # 배관 수 = 주관 + 다리 (T16)
        self.assertEqual(sc.floor_openings, [{"id": "HATCH_1", "min": [27000, 4000], "max": [33000, 9000]}])

    def test_base_is_template_as_written(self):
        """기준 배치: 장비·데크 단자가 템플릿 좌표 그대로, 계통 = 템플릿 목록."""
        t, sha = template.load_template(TPL)
        sc = template.expand(t, sha, None, base=True)
        obs = {o.id: o for o in sc.obstacles}
        for e in t["equipment"]:
            self.assertEqual(obs[e["id"]].box, template._box(*e["box"]))
        self.assertEqual(len(sc.pipes), len(t["pipes"]))
        self.assertEqual(sc.meta["template"]["moved_terminals"], {})
        self.assertEqual(sc.pipes[7].start.pos, (29300.0, 10000.0, 0.0))            # DECK_DN(29300, 10000)
        self.assertEqual(template.scenario_bytes(sc), template.scenario_bytes(template.expand(t, sha, None, base=True)))

    def test_structure_fixed_and_shared(self):
        """구조는 시드와 무관 (variation.structure fixed), 반복 부재는 같은 좌표 열을 공유한다 (D124 ③)."""
        t, sha = template.load_template(TPL)
        kinds = ("stiffener", "web_frame", "pillar", "bulkhead", "casing")
        s1 = {o.id: o.box for o in template.expand(t, sha, 30000001).obstacles if o.type in kinds}
        s2 = {o.id: o.box for o in template.expand(t, sha, 30000002, n_pipes=50).obstacles if o.type in kinds}
        self.assertEqual(s1, s2)
        cl = [b for k, b in s1.items() if k.startswith("CL_")]
        self.assertEqual({(b.min[0], b.max[0], b.min[2], b.max[2]) for b in cl}, {(0.0, 36000.0, 4950.0, 5200.0)})
        sf = [b for k, b in s1.items() if k.startswith("SF_")]
        self.assertEqual(len({b.min[0] for b in sf}), len(sf) // 2)        # 양현 늑골은 같은 x 열

    def test_attributes_one_place(self):
        """D136 ①: 개별 > 그룹 > D93 기본 (모두 false), 펼친 장애물에 최종 값."""
        t, sha = tpl(attribute_overrides={"FO_SETT_TK": {"supportable": True}, "PIL_01": {"supportable": False}},
                     attribute_defaults__handrails=None)
        sc = template.expand(t, sha, None, base=True)
        o = {x.id: x for x in sc.obstacles}
        self.assertTrue(o["FO_SETT_TK"].supportable and not o["FO_SERV_TK"].supportable)
        self.assertFalse(o["PIL_01"].supportable)
        self.assertTrue(o["PIL_02"].supportable)
        self.assertTrue(o["BHD_PUR"].penetrable and o["ER_CASING_S"].penetrable)     # casing_wall 그룹 (v2)
        self.assertTrue(all(not (x.access or x.supportable) for x in sc.obstacles if x.type == "handrail"))   # D93 기본
        self.assertTrue(all(x.access for x in sc.obstacles if x.type in ("passage", "stair")))
        self.assertTrue(all(not (x.supportable or x.penetrable or x.access) for x in sc.obstacles if x.type == "tray"))
        self.assertEqual(sc.meta["template"]["groups"]["WFP_001"], "ceiling_web_frames")
        for bad, msg in (({"attribute_defaults__stairs": {"access": True, "supportable": True}}, "D119"),
                         ({"attribute_defaults__decks": {"supportable": True}}, "모르는 그룹"),
                         ({"attribute_overrides": {"NO_SUCH": {"access": True}}}, "없는 id")):
            t, sha = tpl(**bad)
            self.assertTrue(any(msg in e for e in errors_of(t, sha, base=True)), bad)

    def test_nozzles(self):
        """D136 ②: 노즐 점 + 면 법선 × r (10 mm 올림), dir 바깥/안쪽, host, 변형에서 장비와 함께."""
        t, sha = template.load_template(TPL)
        for seed, base in ((None, True), (30000001, False)):
            sc = template.expand(t, sha, seed, base=base)
            o = {x.id: x for x in sc.obstacles}
            p = sc.pipes[0]                                         # FO_SETT_TK.OUT (east, at [1000, 400]) → FO_PUR_1.IN (west)
            b = o["FO_SETT_TK"].box
            r = template._up(p.radius)
            self.assertEqual(p.start.pos, (b.max[0] + r, b.min[1] + 1000, b.min[2] + 400))
            self.assertEqual((p.start.dir, p.start.host, p.start.kind), ((1, 0, 0), "FO_SETT_TK", "nozzle"))
            b2 = o["FO_PUR_1"].box
            self.assertEqual(p.end.pos, (b2.min[0] - r, b2.min[1] + 750, b2.min[2] + 1200))
            self.assertEqual(p.end.dir, (1, 0, 0))                  # 도착 = 안쪽 (D29)
        top = template.expand(t, sha, None, base=True).pipes[16].end      # FO_SETT_TK.HEAT (top, at [2000, 2000])
        self.assertEqual((top.pos[:2], top.dir), ((4000.0, 4500.0), (0, 0, -1)))
        for bad, msg in (({"pipes__0__from": "FO_SETT_TK.east"}, "노즐 FO_SETT_TK.east 이 없다"),
                         ({"pipes__0__size": "100A", "pipes__0__legs": []}, "구경"),
                         ({"equipment__0__nozzles__OUT": {"face": "east", "at": [5000, 400], "size": "65A"}}, "면 밖"),
                         ({"pipes__2__to": "FO_SETT_TK.OUT"}, "도 쓴다")):
            t2, sha2 = tpl(**bad)
            self.assertTrue(any(msg in e for e in errors_of(t2, sha2, base=True)), bad)

    def test_systems_and_terminals(self):
        """D136 ③④: 계통 다리 → branches (to = out, from = in), 중력 서비스 다리 = from 만, BHD(id, u, z)."""
        t, sha = template.load_template(TPL)
        sc = template.expand(t, sha, None, base=True)
        p1, p2 = sc.pipes[0], sc.pipes[1]
        self.assertEqual((p1.type_id, [(b.id, b.flow, b.terminal.host) for b in p1.branches]),
                         ("T3", [("B1", "out", "FO_PUR_2")]))
        self.assertEqual([(b.flow, b.terminal.dir) for b in p2.branches], [("in", (1, 0, 0))])   # 합류 = 출발 규약
        st = sc.pipes[16]
        self.assertEqual([b.nominal_size for b in st.branches], ["50A", "50A"])
        self.assertEqual(sc.pipes[0].line_no, "FO-65A-001-CS40")                       # D116
        ff = sc.pipes[20]                                                             # FF … → BHD(BHD_WS, 31000, 3000)
        self.assertEqual((ff.end.host, ff.end.pos[0], ff.end.pos[2], ff.end.dir), ("BHD_WS", 31000.0, 3000.0, (0, 1, 0)))
        self.assertEqual(ff.end.pos[1], 22000 - 10 - template._up(ff.radius))         # 격벽 − 쪽 면 (다른 쪽 단자 y 10000)
        for bad, msg in (({"pipes__22__legs": [{"size": "50A", "to": "DECK_DN(20000, 27000)"}]}, "from (합류) 만"),
                         ({"pipes__0__legs": [{"size": "100A", "to": "FO_PUR_2.IN"}]}, "D112"),
                         ({"pipes__20__to": "BHD(BHD_WS)"}, "D136 ③"),
                         ({"pipes__20__to": "BHD(BHD_WS, 24050, 3000)"}, "여유 밖")):
            t2, sha2 = tpl(**bad)
            self.assertTrue(any(msg in e for e in errors_of(t2, sha2, base=True)), bad)

    def test_rules_in_template(self):
        """D136 ⑤: 문은 doors 에, 곁가지 = 목표 중심에서 고리까지, 트레이 내림 = drop_from·at·to_z."""
        t, sha = template.load_template(TPL)
        sc = template.expand(t, sha, None, base=True)
        o = {x.id: x for x in sc.obstacles}
        sp = o["PASS_SPOKE_0"].box                                  # STAIR_P 중심 (14100, 5250) → 고리 남쪽 바깥 y 8300
        self.assertEqual((sp.min[0], sp.max[0], sp.min[1], sp.max[1]), (13650.0, 14550.0, 5250.0, 8300.0))
        dr = o["PASS_SPOKE_2"].box                                  # DOOR_PUR (9600, 9800) → 고리 서쪽 바깥 x 10800
        self.assertEqual((dr.min[0], dr.max[0], dr.min[1], dr.max[1]), (9600.0, 10800.0, 9350.0, 10250.0))
        tr = o["TRAY_DROP_ME"].box                                  # at (17000, 9300), 남쪽 고리 트레이 진행 = x
        self.assertEqual((tr.min, tr.max), ((16850.0, 9240.0, 3200.0), (17150.0, 9360.0, 4150.0)))
        t2, sha2 = tpl(passages__doors=[t["passages"]["doors"][1]])
        self.assertTrue(any("DOOR_PUR" in e and "doors" in e for e in errors_of(t2, sha2, base=True)))
        t3, sha3 = tpl(cable_trays__3={"id": "TRAY_DROP_ME", "drop_from": "TRAY_RING", "at": [17000, 9300]})
        self.assertTrue(any("수직 내림" in e for e in errors_of(t3, sha3, base=True)))

    def test_format_and_use(self):
        """D136 ⑥ 형식 2 만 · D137 use 필수, eval 템플릿 + 학습·검증 시드 거부."""
        for bad, msg in (({"format": None}, "형식 2"), ({"format": 1}, "형식 2"), ({"use": None}, "D137 ①"),
                         ({"use": "test"}, "D137 ①")):
            t, sha = tpl(**bad)
            self.assertTrue(any(msg in e for e in errors_of(t, sha)), bad)
        t, sha = tpl(use="eval")
        for seed in (30000001, 39999999, 40000000, 49999999):
            self.assertTrue(any("D137 ②" in e for e in errors_of(t, sha, seed)), seed)
        self.assertEqual(errors_of(t, sha, 20260001), [])
        self.assertEqual(errors_of(t, sha, None, base=True), [])
        self.assertEqual(template.expand(t, sha, 20260001).meta["template"]["seed_range"], "eval")
        t, sha = tpl(use="train")
        self.assertEqual(errors_of(t, sha, 40000001), [])
        self.assertTrue(any("v1" in e for e in errors_of(*template.load_template("templates/er_2nd_generic_a.yaml"))))

    def test_checks_and_editor_data(self):
        """D136 ⑦: 점검 (겹침·단자 직진·접근 구역) 은 경고, 편집기 자료는 로더 출력 그대로."""
        t, sha = template.load_template(TPL)
        sc = template.expand(t, sha, None, base=True)
        before = template.scenario_bytes(sc)
        chk = template.check(sc)
        self.assertEqual(template.scenario_bytes(sc), before)                   # 점검은 시나리오를 바꾸지 않는다
        self.assertIn(("PIL_04", "PASS_SPOKE_1"), chk["overlaps"])            # 필라가 곁가지 통로 안 (템플릿 초안)
        self.assertNotIn(("CL_001", "WF_001"), chk["overlaps"])               # 의도된 겹침 (구조끼리)
        self.assertFalse(any("BHD_PUR" in a + b and "SPOKE_2" in a + b for a, b in chk["overlaps"]))   # 문 자리
        ed = template.editor_data(t, sc, chk)
        self.assertEqual(len(ed["boxes"]), len(sc.obstacles) + len(sc.floor_openings))
        self.assertTrue(all(b["key"] in template.GROUPS + ("floor_openings",) for b in ed["boxes"]))
        self.assertEqual(sum(len(s["terms"]) for s in ed["systems"]), sum(2 + len(p.branches) for p in sc.pipes))
        self.assertEqual(len(ed["clash"]), len(chk["overlaps"]))
        # 일부러 막은 단자 → 점검 경고
        t2, sha2 = tpl(pipes__7__from="DECK_DN(29300, 11900)")                # SW_PUMP_1 에서 100 mm (< r)
        chk2 = template.check(template.expand(t2, sha2, None, base=True))
        self.assertTrue(any(m.startswith("P008.start") for m in chk2["terminals"]))


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
