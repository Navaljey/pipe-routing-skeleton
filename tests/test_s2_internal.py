"""S2-1 시험 — 내부 구조재 (D113) · 접근 구역 (D119 앞부분).

① 로더 속성 조합 ② 판 관통 (라우터 경로 = 관통 1회, 검증 통과, 관통비 J, 서포트 고정점, Dijkstra 대조)
③ 잘못된 관통 (판 안 꺾임 · 가장자리 r 안) = crossing 위반 ④ 필라 면 지지 (앵커: 검증기 · 라우터 · 힌트 같은 판정)
⑤ 접근 구역 (access 모듈, 라우터가 피함, 모듈 목록은 접근 구역이 있을 때만) ⑥ 생성기 옵션 (기본 출력 불변 · 결정적).
"""
import math
import unittest

from pipe_routing import astar_fast, internal
from pipe_routing import generator as gen
from pipe_routing.layered import LayeredGraph, _route, clear_cache
from pipe_routing.router_astar import astar_route
from pipe_routing.scenario import ScenarioError, from_dict, load, to_dict
from pipe_routing.verifier import MODULE_ORDER, PipeRoute, verify
from pipe_routing.verifier.core import module_order
from pipe_routing.verifier.hint import SupportHint

BLOCK = {"width": 40000, "length": 40000, "height": 10000, "unit": "mm"}
BHD = {"id": "BHD", "type": "bulkhead", "min": [20000, 0, 0], "max": [20020, 40000, 10000],
       "supportable": True, "penetrable": True}


def pipe(start, end, sdir=(1, 0, 0), edir=(1, 0, 0), kind=("boundary", "boundary"), **kw):
    d = {"id": "P", "type_id": "T1", "nominal_size": "100A", "gravity_pipe": False, "min_slope": None,
         "start": {"pos": list(start), "kind": kind[0], "dir": list(sdir)},
         "end": {"pos": list(end), "kind": kind[1], "dir": list(edir)}}
    d.update(kw)
    return d


def scen(obstacles, *pipes):
    return from_dict({"block": BLOCK, "obstacles": list(obstacles), "pipes": list(pipes)})


class LoaderTest(unittest.TestCase):
    def test_combinations(self):
        ok = [BHD, {"id": "PIL", "type": "pillar", "min": [1000, 1000, 0], "max": [1300, 1300, 10000],
                    "supportable": True},
              {"id": "ACC", "type": "passage", "min": [0, 30000, 0], "max": [40000, 30800, 2000], "access": True}]
        sc = scen(ok, pipe([0, 15000, 5000], [40000, 26000, 5000]))
        self.assertEqual([o.access for o in sc.obstacles], [False, False, True])
        self.assertEqual(to_dict(sc)["obstacles"][2]["access"], True)
        for bad in ({"penetrable": True},                                 # (false, true) 조합 없음
                    {"access": True, "supportable": True}, {"access": True, "penetrable": True, "supportable": True}):
            with self.subTest(bad), self.assertRaises(ScenarioError):
                scen([dict({"id": "X", "type": "x", "min": [1000, 1000, 0], "max": [1300, 1300, 10000]}, **bad)],
                     pipe([0, 15000, 5000], [40000, 26000, 5000]))

    def test_e1_module_order_unchanged(self):
        sc = load("scenarios/manual/manual_01.json")
        self.assertEqual(module_order(sc), MODULE_ORDER)
        self.assertEqual(internal.plates(sc), [])
        self.assertFalse(internal.has_internal(sc))


@unittest.skipUnless(astar_fast.HAVE_NUMBA, "numba 없음")
class PlateTest(unittest.TestCase):
    def test_route_penetrates(self):
        clear_cache()
        sc = scen([BHD], pipe([0, 15000, 5000], [40000, 26000, 5000]))
        p = sc.pipes[0]
        g = LayeredGraph(sc, p)
        r = _route(g, p, 60, 2 ** 22)
        self.assertEqual(r.status, "ok")
        cr = internal.crossings(sc, p, r.waypoints)
        self.assertEqual(len(cr), 1)
        rep = verify(sc, [PipeRoute("P", r.waypoints)])["pipes"]["P"]
        self.assertEqual([v.message for v in rep.violations], [])
        self.assertTrue(rep.layer0["crossing"])
        self.assertAlmostEqual(r.J_penetration, p.kg_per_m * 20 / 1000)       # D95 관통비 (라우터 비용)
        from pipe_routing.pipeline import j_breakdown
        J = j_breakdown(p, r.waypoints, rep.fittings, rep.supports, sc)
        self.assertAlmostEqual(J["penetration"], p.kg_per_m * 20 / 1000)     # D95 (J 분해)
        d = astar_route(g, p, 600, heuristic=lambda s: 0.0)
        self.assertTrue(math.isclose(d.J, r.J, rel_tol=1e-9), (d.J, r.J))

    def test_bad_penetrations(self):
        sc = scen([BHD], pipe([0, 15000, 5000], [40000, 26000, 5000]))
        # 판 안에서 꺾임 (관통 직관 밖 조각이 관통 구역 안)
        bend_in = [[0, 15000, 5000], [20010, 15000, 5000], [20010, 26000, 5000], [40000, 26000, 5000]]
        rep = verify(sc, [PipeRoute("P", bend_in)], modules=["crossing"])["pipes"]["P"]
        self.assertFalse(rep.layer0["crossing"])
        # 판 면에서 접점까지 직관 < max(3D, r) (꺾임점이 판 면 바로 앞)
        short = [[0, 15000, 5000], [19700, 15000, 5000], [19700, 26000, 5000], [40000, 26000, 5000]]
        rep = verify(sc, [PipeRoute("P", short)], modules=["crossing"])["pipes"]["P"]
        self.assertFalse(rep.layer0["crossing"])
        # 정상: 판 양쪽 면에서 충분히 떨어져 꺾음
        good = [[0, 15000, 5000], [10000, 15000, 5000], [10000, 26000, 5000], [40000, 26000, 5000]]
        rep = verify(sc, [PipeRoute("P", good)], modules=["crossing", "collision"])["pipes"]["P"]
        self.assertTrue(rep.layer0["crossing"])
        self.assertTrue(rep.layer0["collision"])   # 판은 collision 장애물이 아니다 (관통 규칙)
        # 부분 판의 가장자리 r 안을 지나는 관통 = 위반
        part = dict(BHD, min=[20000, 0, 0], max=[20020, 20000, 10000])
        sc2 = scen([part], pipe([0, 19950, 5000], [40000, 19950, 5000]))
        rep = verify(sc2, [PipeRoute("P", [[0, 19950, 5000], [40000, 19950, 5000]])], modules=["crossing"])
        self.assertFalse(rep["pipes"]["P"].layer0["crossing"])

    def test_support_fixed_point(self):
        """관통부 = 서포트 고정점 (D94): 판을 지나는 직선은 판 중심에서 간격이 0 으로."""
        sc = scen([BHD], pipe([0, 15000, 9000], [40000, 15000, 9000]))
        from pipe_routing.verifier.core import Context, centerlines_of
        from pipe_routing.verifier.modules import support_fixed_points
        from pipe_routing.support_std import load_standard
        pr = PipeRoute("P", [[0, 15000, 9000], [40000, 15000, 9000]])
        p = sc.pipes[0]
        ctx = Context(sc, {"P": p}, {"P": pr}, {"P": centerlines_of(p, pr)}, std=load_standard())
        self.assertEqual(support_fixed_points(ctx, "P"), [20010.0])


@unittest.skipUnless(astar_fast.HAVE_NUMBA, "numba 없음")
class AnchorTest(unittest.TestCase):
    """필라 면 (supportable) 이 지지면 — 블록 면이 표 범위 밖인 가운데 높이 배관."""

    def _sc(self, sup):
        pil = {"id": "PIL", "type": "pillar", "min": [18000, 14000, 0], "max": [22000, 14300, 10000],
               "supportable": sup}
        return scen([pil], pipe([16000, 14800, 5000], [24000, 14800, 5000], kind=("nozzle", "nozzle")))

    def test_verifier_router_hint(self):
        line = [[16000, 14800, 5000], [24000, 14800, 5000]]
        J = {}
        for sup in (False, True):
            clear_cache()
            sc = self._sc(sup)
            p = sc.pipes[0]
            rep = verify(sc, [PipeRoute("P", line)])["pipes"]["P"]
            self.assertEqual(rep.layer0["support"], sup)
            if sup:
                self.assertTrue(all(s["face"] == "wall_y0@PIL" for s in rep.supports))
                h = SupportHint(sc, [PipeRoute("P", line)])
                s0 = rep.supports[0]
                ch = h.form_at("P", s0["pos"], [1, 0, 0])["chosen"]
                self.assertEqual((ch["face"], ch["form"], ch["kg"]), (s0["face"], s0["form"], s0["kg"]))
            g = LayeredGraph(sc, p)
            r = _route(g, p, 60, 2 ** 22)
            v = verify(sc, [PipeRoute("P", r.waypoints)])["pipes"]["P"]
            self.assertEqual([x.message for x in v.violations], [])
            d = astar_route(g, p, 600, heuristic=lambda s: 0.0)
            self.assertTrue(math.isclose(d.J, r.J, rel_tol=1e-9))
            J[sup] = r
        self.assertEqual(len(J[True].waypoints), 2)               # 필라 면으로 지지 → 직선
        self.assertGreater(len(J[False].waypoints), 2)            # 지지면 없음 → 띠로 우회
        self.assertLess(J[True].J, J[False].J)

    def test_off_face_rejected(self):
        """면 사각형 밖에 닿는 부재는 불가 — 필라가 짧으면 (z 범위가 배관 높이를 덮지 않음) 그 면은 못 쓴다."""
        pil = {"id": "PIL", "type": "pillar", "min": [18000, 14000, 0], "max": [22000, 14300, 3000],
               "supportable": True}
        sc = scen([pil], pipe([16000, 14800, 5000], [24000, 14800, 5000], kind=("nozzle", "nozzle")))
        rep = verify(sc, [PipeRoute("P", [[16000, 14800, 5000], [24000, 14800, 5000]])])["pipes"]["P"]
        self.assertFalse(any("@PIL" in s["face"] for s in rep.supports))


@unittest.skipUnless(astar_fast.HAVE_NUMBA, "numba 없음")
class AccessTest(unittest.TestCase):
    ACC = {"id": "ACC", "type": "passage", "min": [15000, 0, 0], "max": [15800, 40000, 2000], "access": True}

    def test_access_module_and_router(self):
        clear_cache()
        sc = scen([self.ACC], pipe([0, 15000, 1000], [40000, 15000, 1000]))
        self.assertIn("access", module_order(sc))
        rep = verify(sc, [PipeRoute("P", [[0, 15000, 1000], [40000, 15000, 1000]])])["pipes"]["P"]
        self.assertFalse(rep.layer0["access"])
        self.assertTrue(rep.layer0["collision"])          # 접근 구역 위반은 access 모듈로만 센다
        p = sc.pipes[0]
        g = LayeredGraph(sc, p)
        r = _route(g, p, 60, 2 ** 22)
        v = verify(sc, [PipeRoute("P", r.waypoints)])["pipes"]["P"]
        self.assertEqual([x.message for x in v.violations], [])   # 통로 위로 넘어간다
        d = astar_route(g, p, 600, heuristic=lambda s: 0.0)
        self.assertTrue(math.isclose(d.J, r.J, rel_tol=1e-9))

    def test_pipeline_output_schema(self):
        """접근 구역이 있는 시나리오의 파이프라인 출력 (layer0.access) 이 출력 스키마를 통과한다."""
        from pipe_routing.pipeline import run, validate_output
        clear_cache()
        sc = scen([self.ACC], pipe([0, 15000, 1000], [40000, 15000, 1000]))
        out = run(sc, solo_ref=False)
        validate_output(out)
        self.assertIn("access", out["routes"][0]["layer0"])
        # D113 장애물 면 서포트 (면@장애물 id) 도 스키마를 통과하고, 모르는 면은 거부한다
        import copy
        import jsonschema
        r0 = out["routes"][0]
        self.assertTrue(r0["supports"])
        ok = copy.deepcopy(out)
        ok["routes"][0]["supports"][0]["face"] = "ceiling@INT_004"
        validate_output(ok)
        bad = copy.deepcopy(out)
        bad["routes"][0]["supports"][0]["face"] = "roof@INT_004"
        with self.assertRaises(jsonschema.ValidationError):
            validate_output(bad)


class GeneratorTest(unittest.TestCase):
    def test_options(self):
        cfg = gen.GeneratorConfig(seed=20266008, services=True, internal=True, access=True, stair_rule=True,
                                  d94_rule=True)
        a, b = gen.generate(cfg), gen.generate(cfg)
        self.assertEqual(to_dict(a), to_dict(b))
        kinds = {o.type for o in a.obstacles}
        self.assertTrue({"bulkhead", "pillar", "stiffener", "passage", "maintenance"} <= kinds)
        self.assertEqual(a.warnings, [])
        hosts = {t.host for p in a.pipes for t in (p.start, p.end) if t.host}
        types = {o.id: o.type for o in a.obstacles}
        self.assertTrue(all(types[h] == "box" for h in hosts))        # 노즐은 장비에만
        from_dict(to_dict(a))
        meta = gen.generate(gen.GeneratorConfig(seed=20266008)).meta["config"]
        self.assertFalse({"internal", "access", "services"} & set(meta))


if __name__ == "__main__":
    unittest.main()
