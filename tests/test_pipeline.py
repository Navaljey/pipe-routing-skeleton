import json
import tempfile
import unittest
from pathlib import Path

from pipe_routing.constants import FITTINGS, PIPE_SPECS
from pipe_routing.pipeline import j_breakdown, main, report_md, run, validate_output
from pipe_routing.router_astar import RouteResult
from pipe_routing.scenario import load

ROOT = Path(__file__).resolve().parent.parent
MANUAL = ROOT / "scenarios" / "manual" / "manual_01.json"


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sc = load(MANUAL)
        cls.out = run(cls.sc)

    def test_output_schema(self):
        validate_output(self.out)
        self.assertEqual(len(self.out["routes"]), len(self.sc.pipes))

    def test_every_pipe_has_layer0_and_J(self):
        """§9-2: 경로 있는 배관은 Layer 0 7모듈 판정과 J 분해, 실패 배관은 원인과 함께."""
        for x in self.out["routes"]:
            self.assertEqual(x["success"], not x["fail_causes"])
            if not x["waypoints"]:
                self.assertTrue(x["fail_causes"][0].startswith("router:"))
                self.assertIsNone(x["J"])
                continue
            self.assertEqual(set(x["layer0"]), {"collision", "boundary", "bend", "gravity_slope", "valve",
                                                "branch", "support"})
            self.assertAlmostEqual(x["J"]["total"], sum(x["J"][k] for k in ("pipe", "elbow", "tee", "valve", "support")),
                                   places=3)
        # D61·D63: 라우터가 계단식 하향 규칙을 지키므로 경로가 있는 중력관은 구배 위반이 없다
        gravity = [x for x in self.out["routes"] if x["type_id"] == "T5" and x["waypoints"]]
        self.assertTrue(gravity)
        self.assertTrue(all(x["layer0"]["gravity_slope"] is True for x in gravity))
        self.assertTrue(all(x["gravity_feasibility"]["feasible"] for x in gravity))

    def test_J_matches_router_cost(self):
        """직관 + 엘보 = 라우터 비용의 직관 + 엘보 (D49: 같은 정의). 라우터 비용의 나머지는 추정 서포트 (D57)."""
        from pipe_routing.layered import LayeredGraph, _route, clear_cache
        from pipe_routing.multi import independent_planner
        out = run(self.sc, planner=independent_planner)   # 배관 단독 경로 = 단독 라우팅과 같은 경로
        clear_cache()
        for p, x in zip(self.sc.pipes, out["routes"]):
            if not x["waypoints"]:
                continue
            r = _route(LayeredGraph(self.sc, p), p, 60)
            self.assertAlmostEqual(x["J"]["pipe"] + x["J"]["elbow"], r.J_pipe + r.J_elbow, places=2)
            self.assertAlmostEqual(r.J, r.J_pipe + r.J_elbow + r.J_support_est, places=6)
            self.assertAlmostEqual(x["router"]["support_est"], r.J_support_est, places=2)

    def test_global(self):
        g = self.out["global"]
        ok = [x for x in self.out["routes"] if x["success"]]
        self.assertEqual(g["success_count"], len(ok))
        self.assertAlmostEqual(g["J_total"], sum(x["J"]["total"] for x in ok), places=3)
        self.assertGreater(g["computation_time_sec"], 0)

    def test_router_failure_reported(self):
        """라우터 실패도 출력에 원인과 함께 남는다 (D34, D40②)."""
        def failing(sc, pipe, limit, placed=()):
            return RouteResult(pipe.id, "timeout", expanded=10, search_sec=60.0)
        out = run(self.sc, router=failing)
        validate_output(out)
        for x in out["routes"]:
            self.assertFalse(x["success"])
            self.assertIsNone(x["J"])
            self.assertEqual(x["fail_causes"], ["router:timeout"])
        self.assertEqual(out["global"]["routed_count"], 0)
        self.assertEqual(out["global"]["J_total"], 0)

    def test_j_breakdown_fittings(self):
        p = self.sc.pipes[0]   # 100A
        fit = [{"type": "elbow_90", "pos": [0, 0, 0]}, {"type": "elbow_45", "pos": [0, 0, 0]},
               {"type": "tee", "pos": [0, 0, 0]}, {"type": "gate_valve", "pos": [0, 0, 0]}]
        sup = [{"pos": [0, 0, 0], "face": "floor", "angle_length_mm": 1100, "kg": 4.873}]
        J = j_breakdown(p, [[0, 0, 0], [10000, 0, 0]], fit, sup)
        f = FITTINGS["100A"]
        self.assertAlmostEqual(J["pipe"], 10 * PIPE_SPECS["100A"].kg_per_m)
        self.assertAlmostEqual(J["elbow"], f.elbow90 + f.elbow45)
        self.assertAlmostEqual(J["tee"], f.tee)
        self.assertAlmostEqual(J["valve"], f.gate_valve)
        self.assertAlmostEqual(J["support"], 4.873)

    def test_report_and_cli(self):
        md = report_md(self.out)
        self.assertIn("J_total", md)
        self.assertIn("P001", md)
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(main([str(MANUAL), "-o", d, "--no-viz"]), 0)
            data = json.loads((Path(d) / "manual_01_output.json").read_text(encoding="utf-8"))
            validate_output(data)
            self.assertTrue((Path(d) / "manual_01_report.md").exists())


if __name__ == "__main__":
    unittest.main()
