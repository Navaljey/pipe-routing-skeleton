"""D105 첫 꺾임 격자선 각도별 · D106 실현 가능성 분류 · D107 생성기 (2026-10-22)."""
import unittest

from pipe_routing import blockpair as bp
from pipe_routing import layered
from pipe_routing.constants import elbow_tangent
from pipe_routing.escape_graph import _snap_up
from pipe_routing.generator import PairConfig, d94_stub_len, generate_block_pair
from pipe_routing.scenario import from_dict, terminal_straight_blocked, terminal_straight_length


def pdict(pid, s, e, sd, ed, size="100A", kind=("boundary", "boundary")):
    return {"id": pid, "type_id": "T1", "nominal_size": size, "gravity_pipe": False, "min_slope": None,
            "insulation_thickness": 50, "start": {"pos": s, "kind": kind[0], "dir": sd},
            "end": {"pos": e, "kind": kind[1], "dir": ed}, "valve_positions": [], "branch_points": []}


def scen(pipes, obstacles=()):
    return from_dict({"block": {"width": 40000, "length": 40000, "height": 10000, "unit": "mm"},
                      "obstacles": [{"id": f"B{i}", "type": "box", "min": lo, "max": hi}
                                    for i, (lo, hi) in enumerate(obstacles)], "pipes": pipes})


class D105Test(unittest.TestCase):
    def test_first_bend_lines_each_angle(self):
        sc = scen([pdict("A", [0, 20000, 5000], [40000, 20000, 5000], [1, 0, 0], [1, 0, 0])])
        p = sc.pipes[0]
        xs = set(layered.support_extra_coords(sc, "100A")[0])
        for th in (45, 90, 135):   # 경계 단자: max(3D, r) + t_θ (D96·D105)
            need = max(p.min_straight, bp.straight_min(p) + elbow_tangent("100A", th))
            self.assertIn(_snap_up(need), xs)
            self.assertIn(40000 - _snap_up(need), xs)

    def test_nozzle_lines(self):
        sc = scen([pdict("N", [5000, 20000, 1000], [30000, 20000, 1000], [0, 1, 0], [0, 1, 0], kind=("nozzle", "nozzle"))],
                  [([4000, 18000, 0], [6000, 19870, 2000]), ([29000, 20120, 0], [31000, 22000, 2000])])
        ys = set(layered.support_extra_coords(sc, "100A")[1])
        p = sc.pipes[0]
        for th in (45, 90, 135):
            self.assertIn(20000 + _snap_up(max(p.min_straight, elbow_tangent("100A", th))), ys)


class D106Test(unittest.TestCase):
    def test_terminal_straight_blocked(self):
        # 경계 start (x=0, +x) 앞 600 mm 에 장애물: 강제 직관 max(3D, r) + t₉₀ = 300 + 150 = 450 → 막히지 않음,
        # 장애물을 400 mm 로 당기면 막힘 (r = 117.15)
        p = pdict("A", [0, 20000, 5000], [40000, 20000, 5000], [1, 0, 0], [1, 0, 0])
        L = terminal_straight_length(scen([p]).pipes[0], "start")
        self.assertAlmostEqual(L, 300 + elbow_tangent("100A", 90))
        ok = scen([p], [([700, 19000, 4000], [2000, 21000, 6000])])
        self.assertTrue(terminal_straight_blocked(ok.pipes[0], ok.obstacles)["feasible"])
        bad = scen([p], [([500, 19000, 4000], [2000, 21000, 6000])])
        tb = terminal_straight_blocked(bad.pipes[0], bad.obstacles)
        self.assertFalse(tb["feasible"])
        self.assertEqual(tb["blocked"][0]["terminal"], "start")


class D107Test(unittest.TestCase):
    def test_pair_generator_stub(self):
        sc = generate_block_pair(PairConfig(seed=20265008), "t")
        self.assertTrue(PairConfig.d94_rule)
        for p in sc.pipes:
            self.assertTrue(terminal_straight_blocked(p, sc.obstacles)["feasible"], p.id)
        self.assertGreaterEqual(d94_stub_len(sc.pipes[0]), bp.straight_min(sc.pipes[0]))


if __name__ == "__main__":
    unittest.main()


def test_d109_support_scan_runs(capsys, monkeypatch):
    """D109 진단 도구: 경계 단자 배관에서 끝까지 실행 (공개 임시 표, 결과 불변 — 읽기만)."""
    from pipe_routing import support_scan
    monkeypatch.delenv("PIPE_SUPPORT_STANDARD", raising=False)
    support_scan.main(["scenarios/procedural/proc_000.json", "P002"])
    out = capsys.readouterr().out
    assert "public_approx" in out and "[start]" in out and "꺾은 뒤" in out
