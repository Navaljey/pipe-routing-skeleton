"""D135 (D106 (b) 정정, 2026-11-24): "규칙상 불가능 — 단자 직관 막힘" 은 반드시 곧은 구간만으로 판정한다.

(a) val_stair10_000 P009 (계단 생성기, seed 40264008) — 정정 후 분모에서 빠지지 않고, 기준선 경로가 Layer 0 통과
(b) 경계 단자 강제 직관 max(3D, r) 안에 r 보다 가까운 장애물은 여전히 막힘
(c) E1 41 시나리오의 D106 (b) 제외 = 0 (정정 전 15)
(d) 생성기 기본 경로 비트 동일 — tests/test_s2_compat.py 그대로 (생성기는 바꾸지 않음, D135 ③)
"""
import json
import unittest
from pathlib import Path

from pipe_routing import blockpair as bp
from pipe_routing.constants import elbow_tangent
from pipe_routing.scenario import (EPS, from_dict, load, obstacle_clearance, terminal_stub,
                                   terminal_straight_blocked, terminal_straight_length)

ROOT = Path(__file__).resolve().parents[1]


def pdict(pid, s, e, sd, ed, size="100A", kind=("boundary", "boundary")):
    return {"id": pid, "type_id": "T1", "nominal_size": size, "gravity_pipe": False, "min_slope": None,
            "insulation_thickness": 50, "start": {"pos": s, "kind": kind[0], "dir": sd},
            "end": {"pos": e, "kind": kind[1], "dir": ed}, "valve_positions": [], "branch_points": []}


def scen(pipes, obstacles=()):
    return from_dict({"block": {"width": 40000, "length": 40000, "height": 10000, "unit": "mm"},
                      "obstacles": [{"id": f"B{i}", "type": "box", "min": lo, "max": hi}
                                    for i, (lo, hi) in enumerate(obstacles)], "pipes": pipes})


def old_blocked(p, obstacles) -> bool:
    """정정 전 D106 (b) 길이 (경계 max(3D, r) + t₉₀, 노즐 max(§3.3, t₉₀)) — 비교용."""
    t90 = elbow_tangent(p.nominal_size, 90)
    for which in ("start", "end"):
        t = p.start if which == "start" else p.end
        L = bp.boundary_straight(p) + t90 if t.kind == "boundary" else max(p.min_straight, t90)
        if obstacle_clearance(terminal_stub(p, which, L), obstacles)[0] < p.radius - EPS:
            return True
    return False


class LengthTest(unittest.TestCase):
    def test_lengths(self):
        sc = scen([pdict("B", [0, 20000, 5000], [40000, 20000, 5000], [1, 0, 0], [1, 0, 0]),
                   pdict("N", [5000, 20000, 1000], [30000, 20000, 1000], [0, 1, 0], [0, 1, 0],
                         size="15A", kind=("nozzle", "nozzle"))])
        b, n = sc.pipes
        self.assertAlmostEqual(terminal_straight_length(b, "start"), bp.straight_min(b))   # max(3D, r) = 300
        want = min(max(n.min_straight - elbow_tangent("15A", th), 0.0) for th in (45, 90, 135))
        self.assertAlmostEqual(terminal_straight_length(n, "start"), want)
        for size in ("50A", "100A", "300A", "500A"):   # 대부분 구경은 노즐 곧은 구간 0 → 판정 안 함
            p = scen([pdict("N", [5000, 20000, 1000], [30000, 20000, 1000], [0, 1, 0], [0, 1, 0],
                            size=size, kind=("nozzle", "nozzle"))]).pipes[0]
            self.assertEqual(terminal_straight_length(p, "start"), 0.0)


class BoundaryStillBlocksTest(unittest.TestCase):
    def test_b(self):
        # 경계 start (x=0, +x) — 강제 직관 300 (100A, r = 117.15). 장애물 x ≥ 400: 이격 100 < r → 막힘.
        # x ≥ 500 (이격 200 ≥ r) 은 정정 전 (300 + t₉₀ = 450) 에는 막힘, 정정 후에는 막히지 않음.
        p = pdict("A", [0, 20000, 5000], [40000, 20000, 5000], [1, 0, 0], [1, 0, 0])
        bad = scen([p], [([400, 19000, 4000], [2000, 21000, 6000])])
        tb = terminal_straight_blocked(bad.pipes[0], bad.obstacles)
        self.assertFalse(tb["feasible"])
        self.assertEqual(tb["blocked"][0]["terminal"], "start")
        self.assertAlmostEqual(tb["blocked"][0]["length_mm"], 300.0)
        ok = scen([p], [([500, 19000, 4000], [2000, 21000, 6000])])
        self.assertTrue(terminal_straight_blocked(ok.pipes[0], ok.obstacles)["feasible"])
        self.assertTrue(old_blocked(ok.pipes[0], ok.obstacles))


class ValStairP009Test(unittest.TestCase):
    def test_a(self):
        from pipe_routing.generator import GeneratorConfig, generate
        from pipe_routing.layered import clear_cache, make_layered_router
        from pipe_routing.pipeline import required_pipe_ids
        from pipe_routing.verifier import PipeRoute, verify
        sc = generate(GeneratorConfig(seed=40264008, n_pipes=10, stair_rule=True), "val_stair10_000")
        p = next(q for q in sc.pipes if q.id == "P009")
        self.assertEqual(p.nominal_size, "300A")
        self.assertTrue(old_blocked(p, sc.obstacles))                          # 정정 전: 분모 제외
        self.assertTrue(terminal_straight_blocked(p, sc.obstacles)["feasible"])
        req, exc = required_pipe_ids(sc)
        self.assertIn("P009", req)
        clear_cache()
        r = make_layered_router(reserve=False, diagnostic=False)(sc, p, 60.0, [])
        self.assertEqual(r.status, "ok")
        rep = verify(sc, [PipeRoute(p.id, r.waypoints)])["pipes"][p.id]
        self.assertTrue(rep.success, [v.message for v in rep.violations])


class E1CountTest(unittest.TestCase):
    def test_c(self):
        H = json.loads((ROOT / "docs/reports/s1_8_scenario_sha256.json").read_text())
        files = [f for fs in H.values() for f in fs]
        self.assertEqual(len(files), 41)
        old = new = 0
        for f in files:
            sc = load(ROOT / f)
            for p in sc.pipes:
                old += old_blocked(p, sc.obstacles)
                new += not terminal_straight_blocked(p, sc.obstacles)["feasible"]
        self.assertEqual(old, 15)
        self.assertEqual(new, 0)


class E11BaselineTest(unittest.TestCase):
    def test_reclassify(self):
        """D135 ②: E1 기준선 JSON 의 분류만 다시 — 분류 밖 필드 바이트 동일, ①·② 그대로, 필수 730 → 744."""
        from pipe_routing.reclassify import dumps, reclassify_baseline, strip_class
        base = json.loads((ROOT / "docs/eval_set_E1_baseline.json").read_text())
        new = reclassify_baseline(base)
        self.assertEqual(dumps(strip_class(new)), dumps(strip_class(base)))
        self.assertEqual(dumps(new), dumps(json.loads((ROOT / "docs/eval_set_E1.1_baseline.json").read_text())))
        for key, passed in (("first", 611), ("budget_end", 615)):
            recs = [r[key] for sets in (base, new) for s in sets["sets"].values() for r in s.values()]
            old, cur = recs[:41], recs[41:]
            self.assertEqual(sum(r["required_pipes"] for r in old), 730)
            self.assertEqual(sum(r["required_pipes"] for r in cur), 744)
            self.assertEqual(sum(r["full_success"] for r in old), sum(r["full_success"] for r in cur))
            self.assertEqual(sum(r["full_success"] for r in cur), 7)
            self.assertEqual(sum(r["required_passed"] for r in cur), passed)
            self.assertEqual(sum(r["required_passed"] for r in old), passed)


if __name__ == "__main__":
    unittest.main()
