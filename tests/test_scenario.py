import copy
import json
import unittest
from pathlib import Path

from pipe_routing.constants import effective_radius
from pipe_routing.generator import GeneratorConfig, generate
from pipe_routing.geometry import Box, box_distance, union_volume
from pipe_routing.scenario import ScenarioError, from_dict, load, to_dict

ROOT = Path(__file__).resolve().parent.parent
MANUAL = ROOT / "scenarios" / "manual" / "manual_01.json"


def manual() -> dict:
    return json.loads(MANUAL.read_text(encoding="utf-8"))


class GeometryTest(unittest.TestCase):
    def test_box_distance(self):
        a = Box((0, 0, 0), (1, 1, 1))
        self.assertEqual(box_distance(a, Box((2, 0, 0), (3, 1, 1))), 1)
        self.assertAlmostEqual(box_distance(a, Box((4, 5, 0), (5, 6, 1))), 5)
        self.assertEqual(box_distance(a, Box((0.5, 0.5, 0.5), (2, 2, 2))), 0)

    def test_union_volume_overlap(self):
        a = Box((0, 0, 0), (2, 2, 2))
        b = Box((1, 1, 1), (3, 3, 3))
        self.assertAlmostEqual(union_volume([a, b]), 8 + 8 - 1)
        self.assertAlmostEqual(union_volume([a, a]), 8)

    def test_effective_radius(self):
        self.assertAlmostEqual(effective_radius("15A"), 70.85)   # §3.1 표
        self.assertAlmostEqual(effective_radius("500A"), 314.00)


class LoadTest(unittest.TestCase):
    def test_all_committed_scenarios_load(self):
        paths = sorted((ROOT / "scenarios").rglob("*.json"))
        self.assertGreaterEqual(len(paths), 21)
        for p in paths:
            with self.subTest(path=p.name):
                sc = load(p)
                self.assertEqual(sc.warnings, [])
                self.assertTrue(all(q.type_id in ("T1", "T5") for q in sc.pipes))  # D27

    def test_roundtrip(self):
        data = manual()
        self.assertEqual(to_dict(from_dict(data)), data)

    def assertRejected(self, data, needle):
        with self.assertRaises(ScenarioError) as cm:
            from_dict(data)
        self.assertTrue(any(needle in e for e in cm.exception.errors), cm.exception.errors)

    def test_schema_missing_field(self):
        data = manual()
        del data["pipes"][0]["nominal_size"]
        self.assertRejected(data, "nominal_size")

    def test_off_snap(self):
        data = manual()
        data["obstacles"][0]["min"][0] = 14005
        self.assertRejected(data, "10mm")

    def test_terminal_inside_obstacle(self):
        data = manual()
        data["pipes"][0]["start"]["pos"] = [10750, 30750, 1250]   # 펌프 상면에서 50mm — 유효 반경 미달
        self.assertRejected(data, "유효 반경")

    def test_boundary_dir_must_point_inward_at_start(self):
        data = manual()
        data["pipes"][3]["start"]["dir"] = [1, 0, 0]
        self.assertRejected(data, "D29")

    def test_boundary_not_on_face(self):
        data = manual()
        data["pipes"][1]["end"]["pos"] = [5000, 39000, 6000]
        self.assertRejected(data, "블록 면")

    def test_gravity_flag_mismatch(self):
        data = manual()
        data["pipes"][2]["gravity_pipe"] = False
        self.assertRejected(data, "D27")

    def test_gravity_drop_warning(self):
        data = manual()
        data["pipes"][2]["end"]["pos"] = [0, 34000, 4400]   # 낙차 100mm < 0.01 × 29850
        sc = from_dict(data)
        self.assertTrue(any("구배" in w for w in sc.warnings), sc.warnings)


class GeneratorTest(unittest.TestCase):
    def test_deterministic(self):
        a = to_dict(generate(GeneratorConfig(seed=7)))
        b = to_dict(generate(GeneratorConfig(seed=7)))
        self.assertEqual(a, b)

    def test_generated_scenario_valid(self):
        for seed in range(5):
            sc = generate(GeneratorConfig(seed=seed, n_pipes=15))
            reloaded = from_dict(copy.deepcopy(to_dict(sc)))
            self.assertEqual(reloaded.warnings, [])
            self.assertGreaterEqual(reloaded.summary()["obstacle_fill"], 0.12)

    def test_deck_penetrations(self):
        """D35 — 데크 관통 단자 포함, 중력관은 하부 데크 start·상부 데크 end 금지."""
        decks = 0
        for seed in range(10):
            sc = generate(GeneratorConfig(seed=seed, gravity_ratio=0.5))
            for p in sc.pipes:
                for which in ("start", "end"):
                    t = getattr(p, which)
                    if t.kind != "boundary" or t.pos[2] not in (0, sc.block.height):
                        continue
                    decks += 1
                    top = t.pos[2] == sc.block.height
                    want = (-1 if top else 1) if which == "start" else (1 if top else -1)
                    self.assertEqual(t.dir, (0, 0, want))   # D29·D35 dir 규약
                    if p.gravity_pipe:
                        self.assertNotEqual((which, top), ("start", False))
                        self.assertNotEqual((which, top), ("end", True))
        self.assertGreater(decks, 0)


if __name__ == "__main__":
    unittest.main()
