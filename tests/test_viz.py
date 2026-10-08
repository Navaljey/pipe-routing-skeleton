import tempfile
import unittest
from pathlib import Path

from pipe_routing.scenario import load
from pipe_routing.viz import scenario_figure, write_html

ROOT = Path(__file__).resolve().parent.parent


class VizTest(unittest.TestCase):
    def test_all_scenarios_render(self):
        for path in sorted((ROOT / "scenarios").rglob("*.json")):
            with self.subTest(path=path.name):
                sc = load(path)
                fig = scenario_figure(sc)
                # 블록 1 + 장애물 면 n + 장애물 모서리 1 + 배관당 (단자·직진·화살표)×2 + 연결선 1
                self.assertEqual(len(fig.data), 2 + len(sc.obstacles) + 7 * len(sc.pipes))

    def test_write_html(self):
        sc = load(ROOT / "scenarios" / "manual" / "manual_01.json")
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "m.html"
            write_html(scenario_figure(sc), out)
            html = out.read_text(encoding="utf-8")
            self.assertIn("plotly", html)
            self.assertIn("P001", html)


if __name__ == "__main__":
    unittest.main()
