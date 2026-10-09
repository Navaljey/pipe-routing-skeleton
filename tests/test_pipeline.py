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
        """§9-2: 모든 배관에 Layer 0 판정과 J 분해, 실패 배관은 원인과 함께."""
        for x in self.out["routes"]:
            self.assertEqual(set(x["layer0"]), {"collision", "boundary", "bend", "gravity_slope", "valve",
                                                "branch", "support"})
            self.assertAlmostEqual(x["J"]["total"], sum(x["J"][k] for k in ("pipe", "elbow", "tee", "valve", "support")),
                                   places=3)
            self.assertEqual(x["success"], not x["fail_causes"])
        gravity = [x for x in self.out["routes"] if x["type_id"] == "T5"]
        self.assertTrue(all("gravity_slope" in x["fail_causes"] for x in gravity))   # D27 기준선

    def test_J_matches_router_cost(self):
        """직관 + 엘보 = A* 비용 J (D49: 같은 정의)."""
        from pipe_routing.escape_graph import EscapeGraph
        from pipe_routing.router_astar import astar_route
        for p, x in zip(self.sc.pipes, self.out["routes"]):
            r = astar_route(EscapeGraph(self.sc, p), p)
            self.assertAlmostEqual(x["J"]["pipe"] + x["J"]["elbow"], r.J, places=2)

    def test_global(self):
        g = self.out["global"]
        ok = [x for x in self.out["routes"] if x["success"]]
        self.assertEqual(g["success_count"], len(ok))
        self.assertAlmostEqual(g["J_total"], sum(x["J"]["total"] for x in ok), places=3)
        self.assertGreater(g["computation_time_sec"], 0)

    def test_router_failure_reported(self):
        """라우터 실패도 출력에 원인과 함께 남는다 (D34, D40②)."""
        def failing(sc, pipe, limit):
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
