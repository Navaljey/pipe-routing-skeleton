"""S2-0b 시험 — 유체 서비스·pipe class·배관별 보온 (D116·D117).

① DEFAULT = §3.1·§3.2 상수 (값 그대로), 공개 CS40 = DEFAULT ② 사양 값을 상수 표에서 직접 읽는 곳 없음 (소스 검사)
③ 로더 해석 순서·오류 ④ 배관 이름 ⑤ 보온·class 가 다른 배관의 라우터–검증기 정합, Dijkstra 대조, J 의 kg/m
⑥ 고정층 묶음 = (구경, class, r) ⑦ 생성기 services 옵션 (결정적, 기본 meta 에 키 없음) ⑧ 출력 필드는 사양을 적은 배관만.
"""
import json
import math
import os
import re
import tempfile
import unittest
from pathlib import Path

from pipe_routing import astar_fast, pipe_spec
from pipe_routing import generator as gen
from pipe_routing.constants import FITTINGS, PIPE_SPECS, effective_radius
from pipe_routing.layered import LayeredGraph, _route, clear_cache, fixed_layer, layer_id
from pipe_routing.pipe_spec import line_numbers
from pipe_routing.router_astar import astar_route
from pipe_routing.scenario import ScenarioError, from_dict, to_dict
from pipe_routing.verifier import PipeRoute, verify

ROOT = Path(__file__).resolve().parent.parent


def pipe(pid="P", size="100A", y=20000, z=1500, **kw):
    d = {"id": pid, "type_id": "T1", "nominal_size": size, "gravity_pipe": False, "min_slope": None,
         "start": {"pos": [0, y, z], "kind": "boundary", "dir": [1, 0, 0]},
         "end": {"pos": [40000, y + 8000, z], "kind": "boundary", "dir": [1, 0, 0]}}
    d.update(kw)
    return d


def scen(*pipes, obstacles=()):
    return from_dict({"block": {"width": 40000, "length": 40000, "height": 10000, "unit": "mm"},
                      "obstacles": [{"id": f"B{i}", "type": "box", "min": list(a), "max": list(b)}
                                    for i, (a, b) in enumerate(obstacles)],
                      "pipes": list(pipes)})


class TablesTest(unittest.TestCase):
    def test_default_is_constants(self):
        for nd in PIPE_SPECS:
            s = pipe_spec.spec("DEFAULT", nd)
            self.assertEqual((s.od, s.kg_per_m), (PIPE_SPECS[nd].od, PIPE_SPECS[nd].kg_per_m))
            self.assertEqual((s.elbow90, s.elbow45, s.gate_valve),
                             (FITTINGS[nd].elbow90, FITTINGS[nd].elbow45, FITTINGS[nd].gate_valve))
            self.assertEqual(pipe_spec.spec("CS40", nd), s)   # 공개 CS40 = DEFAULT
            for c in ("CS80", "SUS10S"):
                self.assertEqual(pipe_spec.spec(c, nd).od, s.od)   # JIS OD 같음
        self.assertGreater(pipe_spec.spec("CS80", "100A").kg_per_m, 16.10)
        self.assertLess(pipe_spec.spec("SUS10S", "100A").kg_per_m, 16.10)

    def test_services_table(self):
        sv = pipe_spec.services()
        self.assertEqual(len(sv), 12)
        self.assertEqual({c for c, s in sv.items() if s.gravity}, {"SD", "DD"})
        self.assertEqual((sv["ST"].pipe_class, sv["ST"].insulation_mm), ("CS80", 75.0))
        self.assertTrue(all(s.pipe_class in pipe_spec.class_names() for s in sv.values()))

    def test_public_pipe_spec(self):   # 공개 API pipe_spec(pipe) — Pipe 속성이 모두 이 값 (D117)
        sc = scen(pipe("A"), pipe("B", service="ST", y=5000))
        for p in sc.pipes:
            s = pipe_spec.pipe_spec(p)
            self.assertEqual((s.od, s.kg_per_m, s.r, s.insulation_mm), (p.od, p.kg_per_m, p.radius,
                                                                        p.insulation_thickness))
            self.assertEqual((s.elbow90, s.elbow45, s.reducer), (p.elbow_kg(90), p.elbow_kg(45), None))
        a, b = (pipe_spec.pipe_spec(p) for p in sc.pipes)
        self.assertEqual((a.pipe_class, a.source, a.r), ("DEFAULT", "builtin", effective_radius("100A")))
        self.assertEqual((b.pipe_class, b.insulation_mm, b.source), ("CS80", 75.0, "public_approx"))

    def test_reducer_kg_fallback(self):   # D118: 표에 없으면 큰 쪽 45° 엘보
        self.assertEqual(pipe_spec.reducer_kg("DEFAULT", "100A", "50A"), FITTINGS["100A"].elbow45)

    def test_private_class_table(self):   # D117 = D78 방식: 환경 변수 표가 공개 표를 대체·추가, 출처·해시 기록
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "pipe_classes.json"
            f.write_text(json.dumps({"classes": {"XC": {"sizes": {"100A": {"od": 114.3, "kg_per_m": 20.0}},
                                                        "fittings": {"100A": {"elbow90": 4, "elbow45": 2,
                                                                              "gate_valve": 21, "reducer": 3}}}}}))
            os.environ[pipe_spec.CLASS_ENV] = str(f)
            pipe_spec.reload()
            try:
                self.assertEqual(pipe_spec.standard_info()["source"], "private")
                self.assertEqual(pipe_spec.spec("XC", "100A").kg_per_m, 20.0)
                self.assertEqual(pipe_spec.reducer_kg("XC", "100A", "100A"), 3.0)
                with self.assertRaises(ScenarioError):   # class 표에 없는 구경
                    scen(pipe(size="50A", pipe_class="XC"))
            finally:
                del os.environ[pipe_spec.CLASS_ENV]
                pipe_spec.reload()
        self.assertEqual(pipe_spec.standard_info()["source"], "public_approx")


class NoDirectReadsTest(unittest.TestCase):
    def test_spec_values_only_through_lookup(self):
        """OD·kg/m·관이음 무게는 배관 사양 조회로만 (§1.16 정합 원칙) — 상수 표를 직접 읽는 곳이 없다."""
        allowed = {"constants.py", "pipe_spec.py", "generator.py"}   # 생성기 = 기본 경로의 단자 r (보온 50) 만
        pat = re.compile(r"PIPE_SPECS\[[^\]]+\]\.(od|kg_per_m)|FITTINGS\[|constants import[^\n]*\b(FITTINGS|elbow_kg)\b"
                         r"|effective_radius\(")
        bad = []
        for f in (ROOT / "pipe_routing").rglob("*.py"):
            if f.name in allowed:
                continue
            for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                if pat.search(line) and "E1" not in line:
                    bad.append(f"{f.relative_to(ROOT)}:{i}: {line.strip()}")
        self.assertEqual(bad, [])


class LoaderTest(unittest.TestCase):
    def test_resolution_order(self):
        sc = scen(pipe("A"), pipe("B", service="ST", y=5000), pipe("C", service="ST", pipe_class="CS40",
                                                                  insulation_thickness=10, y=10000),
                  pipe("D", service="SW", y=30000))
        a, b, c, d = sc.pipes
        self.assertEqual((a.class_name, a.insulation_thickness), ("DEFAULT", 50.0))
        self.assertEqual((b.class_name, b.insulation_thickness), ("CS80", 75.0))
        self.assertEqual((c.class_name, c.insulation_thickness), ("CS40", 10.0))
        self.assertEqual((d.class_name, d.insulation_thickness), ("CS40", 0.0))
        self.assertEqual(d.radius, 114.3 / 2 + 0 + 10)                      # 보온 0 허용 (D117)
        self.assertEqual(a.radius, effective_radius("100A"))
        self.assertEqual(b.kg_per_m, pipe_spec.spec("CS80", "100A").kg_per_m)
        self.assertFalse(any("보온" in w for w in sc.warnings))              # D33 경고 삭제
        out = to_dict(sc)["pipes"]
        self.assertNotIn("service", out[0])
        self.assertEqual((out[1]["service"], out[1]["insulation_thickness"]), ("ST", 75))
        self.assertEqual(out[2]["pipe_class"], "CS40")
        self.assertEqual(to_dict(from_dict(to_dict(sc)))["pipes"], out)     # 다시 읽어도 같다

    def test_errors(self):
        for kw in ({"service": "XX"}, {"pipe_class": "NOPE"},
                   {"service": "SD"},                                    # 중력 서비스 ↔ 압력관
                   {"insulation_thickness": -1}):
            with self.subTest(kw), self.assertRaises(ScenarioError):
                scen(pipe(**kw))
        with self.assertRaises(ScenarioError):                            # 압력 서비스 ↔ 중력관
            scen(pipe(type_id="T5", gravity_pipe=True, min_slope=0.01, service="FO",
                      start={"pos": [0, 20000, 3000], "kind": "boundary", "dir": [1, 0, 0]}))

    def test_line_numbers(self):
        sc = scen(pipe("A", service="FO"), pipe("B", service="ST", size="50A", y=5000),
                  pipe("C", service="FO", line_no="MY-LINE", y=10000), pipe("D", service="FO", y=30000),
                  pipe("E", y=1000), pipe("F", line_no="ONLY", y=15000))
        self.assertEqual(line_numbers(sc), {"A": "FO-100A-001-CS40", "B": "ST-50A-001-CS80", "C": "MY-LINE",
                                            "D": "FO-100A-003-CS40", "F": "ONLY"})


@unittest.skipUnless(astar_fast.HAVE_NUMBA, "numba 없음")
class RouterVerifierTest(unittest.TestCase):
    """보온·class 가 다른 배관: 라우터 경로가 검증기를 통과 (정합), Dijkstra J 일치, J 직관 = class kg/m."""
    OBS = [((12000, 18000, 0), (16000, 26000, 4000))]

    def _check(self, **kw):
        clear_cache()
        sc = scen(pipe(**kw), obstacles=self.OBS)
        p = sc.pipes[0]
        g = LayeredGraph(sc, p)
        fast = _route(g, p, 60)
        self.assertEqual(fast.status, "ok")
        dij = astar_route(g, p, 600, heuristic=lambda s: 0.0)
        self.assertTrue(math.isclose(fast.J, dij.J, rel_tol=1e-9, abs_tol=1e-6), (kw, fast.J, dij.J))
        rep = verify(sc, [PipeRoute("P", fast.waypoints)])
        v = rep["pipes"]["P"]
        self.assertEqual([x.message for x in v.violations], [], kw)
        return sc, p, fast, g

    def test_insulation_and_class(self):
        r = {}
        for kw in ({}, {"service": "SW"}, {"service": "ST"}, {"pipe_class": "SUS10S", "insulation_thickness": 25}):
            sc, p, res, g = self._check(**kw)
            self.assertEqual(g.r, p.radius)
            r[json.dumps(kw)] = res
        # 단위중량이 다르면 경로 비용이 다르다 (같은 경로라도 kg/m 비)
        self.assertNotEqual(r['{"service": "ST"}'].J, r["{}"].J)

    def test_pipeline_fields_and_j(self):
        from pipe_routing.pipeline import run
        clear_cache()
        sc = scen(pipe("A", service="ST"), pipe("B", y=5000), obstacles=self.OBS)
        out = run(sc)
        a, b = out["routes"]
        self.assertEqual((a["line_no"], a["service"], a["pipe_class"], a["insulation_thickness"]),
                         ("ST-100A-001-CS80", "ST", "CS80", 75.0))
        self.assertNotIn("line_no", b)
        self.assertNotIn("pipe_class", b)
        self.assertEqual(out["global"]["pipe_class_standard"]["source"], "public_approx")
        L = sum(math.dist(p, q) for p, q in zip(a["waypoints"], a["waypoints"][1:]))
        self.assertAlmostEqual(a["J"]["pipe"], round(L / 1000 * pipe_spec.spec("CS80", "100A").kg_per_m, 4), 3)
        out2 = run(scen(pipe("B", y=5000), obstacles=self.OBS))
        self.assertNotIn("pipe_class_standard", out2["global"])


class LayerGroupTest(unittest.TestCase):
    def test_layer_key(self):
        clear_cache()
        sc = scen(pipe("A"), pipe("B", y=5000), pipe("C", service="ST", y=10000), pipe("D", service="SW", y=30000))
        a, b, c, d = sc.pipes
        self.assertEqual(layer_id(a), layer_id(b))
        self.assertEqual(len({layer_id(p) for p in (a, c, d)}), 3)
        Fa, _ = fixed_layer(sc, a)
        Fb, _ = fixed_layer(sc, b)
        Fc, _ = fixed_layer(sc, c)
        self.assertIs(Fa, Fb)
        self.assertIsNot(Fa, Fc)
        self.assertEqual(Fc.r, c.radius)


class GeneratorServicesTest(unittest.TestCase):
    def test_services_option(self):
        a = gen.generate(gen.GeneratorConfig(seed=30000001, services=True))
        b = gen.generate(gen.GeneratorConfig(seed=30000001, services=True))
        self.assertEqual(to_dict(a), to_dict(b))                       # 결정적
        self.assertTrue(a.meta["config"]["services"])
        self.assertNotIn("services", gen.generate(gen.GeneratorConfig(seed=30000001)).meta["config"])
        sv = pipe_spec.services()
        for p in a.pipes:
            self.assertIsNotNone(p.service)
            self.assertEqual(sv[p.service].gravity, p.gravity_pipe)
            self.assertEqual(p.insulation_thickness, sv[p.service].insulation_mm)
        self.assertEqual(a.warnings, [])
        from_dict(to_dict(a))                                          # 로더 통과
        pr = gen.generate_block_pair(gen.PairConfig(seed=30000002, services=True))
        self.assertTrue(all(p.service for p in pr.pipes))
        self.assertNotIn("services", gen.generate_block_pair(gen.PairConfig(seed=30000002)).meta["config"])


if __name__ == "__main__":
    unittest.main()
