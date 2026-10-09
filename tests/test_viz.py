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

    def test_graph_slice(self):
        from pipe_routing.escape_graph import EscapeGraph
        from pipe_routing.viz import add_graph
        sc = load(ROOT / "scenarios" / "manual" / "manual_01.json")
        g = EscapeGraph(sc, sc.pipes[1])
        fig = scenario_figure(sc)
        n0 = len(fig.data)
        info = add_graph(fig, g, 2, 1500)
        self.assertEqual(len(fig.data), n0 + 4)   # 노드, 축 엣지, 45° 엣지, 팽창 장애물
        self.assertEqual(info["value"], 1500)
        self.assertEqual(info["nodes"], int(g.node_ok[:, :, g.index[2][1500.0]].sum()))
        self.assertGreater(info["edges_axis"], 0)

    def test_verification_layer(self):
        from pipe_routing.verifier import PipeRoute, verify
        from pipe_routing.viz import add_verification
        sc = load(ROOT / "scenarios" / "manual" / "manual_01.json")
        p = sc.pipes[2]   # P003 중력관 — 수평 직선 경로는 구배 위반
        rep = verify(sc, [PipeRoute(p.id, [list(p.start.pos), [20000, 34000, 4500], [20000, 34000, 500],
                                            list(p.end.pos)])])
        fig = scenario_figure(sc)
        n0 = len(fig.data)
        add_verification(fig, rep)
        names = [t.name for t in fig.data[n0:]]
        self.assertTrue(any(n.startswith("위반 gravity_slope") for n in names), names)


if __name__ == "__main__":
    unittest.main()
