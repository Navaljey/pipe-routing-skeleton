"""S2-0 상위 호환 시험 (D111·D112·D115, S2 설계안 §7 Q1·Q8).

① 로더가 평가셋 E1 41 시나리오를 그대로 읽는다 — 파일 sha256 = 동결값, 다시 쓰면 같은 내용
   (생성 파일 40개는 바이트까지 같음).
② 생성기 기본값 경로 비트 동일 (D115 보완 2026-10-31) — 같은 시드 → 같은 출력 (난수 소비 순서 포함).
   E1 의 시드 시나리오 40개를 meta.config 로 다시 만들어 파일과 비교하고, 학습용·검증용 시드 범위의
   기본 설정 출력은 S2 착수 전 main (46eeca4) 에서 잰 sha256 으로 고정한다.
③ 새 입력 필드 valves·branches, 이전 필드 valve_positions·branch_points (빈 배열만, 값이 있으면 오류).
④ 출력 스키마 tree (D112) — 선택 필드, 기존 출력 그대로 통과.
"""
import copy
import dataclasses
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import jsonschema

from pipe_routing import generator as g
from pipe_routing.scenario import ScenarioError, from_dict, load, save, to_dict

ROOT = Path(__file__).resolve().parent.parent
E1_HASHES = json.loads((ROOT / "docs/reports/s1_8_scenario_sha256.json").read_text())
E1_FILES = [p for files in E1_HASHES.values() for p in files]

# S2 착수 전 main (46eeca4) 출력의 sha256 (to_dict, sort_keys) — 학습용 3xxxxxxx · 검증용 4xxxxxxx 시드
GOLDEN = {
    ("gen", 30000000): "0a12b51bb223e60512f28e180596d6c4c77cf3f2d3ec2026582e6caa1fc7ad28",
    ("gen", 31234567): "7e7ceb725ef410614abaa7b368fbec5ed78cc8e37a16e6cf28c92cacc6e1199c",
    ("gen", 40000000): "91bf6ed9b427dc227f3d04e5bc66b8f1a9600fea208ef75eed310e9029df2a72",
    ("stair", 30000000): "5c3b5aa443888f4e2d18eb8e3e7bd479618e483a9163083cca79e0d54dc082e3",
    ("stair", 40000000): "5c6d6613f913daf28851753bc550b8562185bb9a4d2470230d6f5899974e4701",
    ("pair", 30000000): "ad8e920077ebd19b62030208ae9c7371d39a763b41d5e12687ff4c00f3a96e9f",
    ("pair", 40000000): "ceb13ac758d07d92185ff203d4693630846ae4f1f7dc9894fd2b02c7d0745f49",
}


def digest(sc) -> str:
    return hashlib.sha256(json.dumps(to_dict(sc), ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def norm(x):
    return json.loads(json.dumps(x))


class E1LoaderTest(unittest.TestCase):
    def test_e1_41_load_unchanged(self):
        self.assertEqual(len(E1_FILES), 41)
        for rel in E1_FILES:
            with self.subTest(rel):
                raw = (ROOT / rel).read_bytes()
                want = [h for files in E1_HASHES.values() for p, h in files.items() if p == rel][0]
                self.assertEqual(hashlib.sha256(raw).hexdigest(), want)
                sc = load(ROOT / rel)
                self.assertEqual(to_dict(sc), json.loads(raw))
                self.assertTrue(all(not p.valves and not p.branches for p in sc.pipes))
                if "config" in sc.meta:   # 생성 파일 → 다시 써도 바이트 같음
                    with tempfile.TemporaryDirectory() as d:
                        save(sc, Path(d) / "x.json")
                        self.assertEqual((Path(d) / "x.json").read_bytes(), raw)


class GeneratorBitIdenticalTest(unittest.TestCase):
    def test_e1_seeded_regenerate(self):
        n = 0
        for rel in E1_FILES:
            data = json.loads((ROOT / rel).read_text())
            meta = data.get("meta", {})
            if "config" not in meta:
                continue
            pair = meta["generator"].endswith("generate_block_pair")
            cls = g.PairConfig if pair else g.GeneratorConfig
            names = {f.name for f in dataclasses.fields(cls)}
            cfg = cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in meta["config"].items() if k in names})
            sc = (g.generate_block_pair if pair else g.generate)(cfg, meta["name"])
            out = to_dict(sc)
            with self.subTest(rel):
                self.assertEqual({k: v for k, v in out.items() if k != "meta"},
                                 {k: v for k, v in data.items() if k != "meta"})
                # 동결 뒤 추가된 설정 키 (기본값) 만 다르다
                self.assertEqual({k: v for k, v in norm(out["meta"]["config"]).items() if k in meta["config"]},
                                 meta["config"])
            n += 1
        self.assertEqual(n, 40)

    def test_golden_seed_ranges(self):
        for (kind, seed), want in GOLDEN.items():
            with self.subTest(kind=kind, seed=seed):
                if kind == "pair":
                    sc = g.generate_block_pair(g.PairConfig(seed=seed))
                else:
                    sc = g.generate(g.GeneratorConfig(seed=seed, stair_rule=(kind == "stair")))
                self.assertEqual(digest(sc), want)


def base(**kw):
    d = {"id": "P", "type_id": "T1", "nominal_size": "100A", "gravity_pipe": False, "min_slope": None,
         "insulation_thickness": 50,
         "start": {"pos": [0, 5000, 1000], "kind": "boundary", "dir": [1, 0, 0]},
         "end": {"pos": [40000, 5000, 1000], "kind": "boundary", "dir": [1, 0, 0]}}
    d.update(kw)
    return d


def scen(*pipes):
    return {"block": {"width": 40000, "length": 40000, "height": 10000, "unit": "mm"},
            "obstacles": [], "pipes": list(pipes)}


LEG = {"id": "B1", "nominal_size": "50A", "pos": [20000, 40000, 1000], "kind": "boundary", "dir": [0, 1, 0]}


class InputFieldsTest(unittest.TestCase):
    def test_legacy_fields_optional_and_empty(self):
        self.assertEqual(from_dict(scen(base())).pipes[0].valve_positions, [])
        from_dict(scen(base(valve_positions=[], branch_points=[])))

    def test_legacy_values_error(self):   # §7 Q8
        for k in ("valve_positions", "branch_points"):
            with self.subTest(k), self.assertRaises(ScenarioError) as cm:
                from_dict(scen(base(**{k: [[20000, 5000, 1000]]})))
            self.assertIn(k, str(cm.exception))

    def test_valves(self):
        sc = from_dict(scen(base(type_id="T2", valves=[{"type": "gate"}])))
        self.assertEqual(sc.pipes[0].valves, ["gate"])
        self.assertEqual(sc.warnings, [w for w in sc.warnings if "valves" not in w])
        self.assertEqual(to_dict(sc)["pipes"][0]["valves"], [{"type": "gate"}])
        for bad in ([{"type": "check"}], [{"type": "gate"}, {"type": "gate"}]):
            with self.assertRaises(ScenarioError):
                from_dict(scen(base(valves=bad)))
        w = from_dict(scen(base(type_id="T2"))).warnings   # 타입 ↔ valves 불일치 = 경고
        self.assertTrue(any("valves" in x for x in w))

    def test_branches(self):
        sc = from_dict(scen(base(type_id="T3", branches=[LEG])))
        b = sc.pipes[0].branches[0]
        self.assertEqual((b.id, b.nominal_size, b.flow, b.which), ("B1", "50A", "out", "end"))
        self.assertFalse(any("branches" in w for w in sc.warnings))
        self.assertEqual(norm(to_dict(sc)["pipes"][0]["branches"]), [LEG])   # 기본 flow 는 쓰지 않음
        merge = dict(LEG, dir=[0, -1, 0], flow="in")
        self.assertEqual(to_dict(from_dict(scen(base(type_id="T3", branches=[merge]))))["pipes"][0]["branches"][0]["flow"],
                         "in")

    def test_branch_errors(self):
        cases = {
            "구경": dict(LEG, nominal_size="150A"),                      # 다리 > 주관 (D112)
            "dir": dict(LEG, dir=[0, -1, 0]),                            # out = 블록 바깥쪽 도착 (D29)
            "스냅": dict(LEG, pos=[20005, 40000, 1000]),
            "같은 위치": dict(LEG, pos=[40000, 5000, 1000], dir=[1, 0, 0]),
        }
        for name, leg in cases.items():
            with self.subTest(name), self.assertRaises(ScenarioError):
                from_dict(scen(base(type_id="T3", branches=[leg])))
        with self.assertRaises(ScenarioError):   # 다리 id 중복
            from_dict(scen(base(type_id="T3", branches=[LEG, dict(LEG, pos=[30000, 40000, 1000])])))
        with self.assertRaises(ScenarioError):   # 6개 (최대 5)
            from_dict(scen(base(type_id="T3", branches=[dict(LEG, id=f"B{i}", pos=[5000 + 5000 * i, 40000, 1000])
                                                        for i in range(6)])))

    def test_gravity_branch_flow_in_only(self):   # §7 Q1
        grav = base(type_id="T7", gravity_pipe=True, min_slope=0.01,
                    start={"pos": [0, 5000, 3000], "kind": "boundary", "dir": [1, 0, 0]},
                    end={"pos": [40000, 5000, 1000], "kind": "boundary", "dir": [1, 0, 0]})
        leg_in = dict(LEG, pos=[20000, 40000, 4000], dir=[0, -1, 0], flow="in")
        from_dict(scen(dict(grav, branches=[leg_in])))
        with self.assertRaises(ScenarioError):
            from_dict(scen(dict(grav, branches=[dict(leg_in, dir=[0, 1, 0], flow="out")])))


class OutputSchemaTest(unittest.TestCase):
    def test_tree_optional(self):
        schema = json.loads((ROOT / "schemas/scenario_output.schema.json").read_text())
        tree = dict(schema["properties"]["routes"]["items"]["properties"]["tree"], **{"$defs": schema["$defs"]})
        jsonschema.validate({"main": [[0, 0, 0], [1, 0, 0]], "method": "exact",
                             "legs": [{"leg_id": "B1", "nominal_size": "50A", "flow": "out", "found": True,
                                       "waypoints": [[1, 0, 0], [1, 1, 0]], "tee": {"pos": [1, 0, 0], "angle": 90}}]},
                            tree)
        for bad in ({"main": [], "legs": [], "method": "greedy"},
                    {"main": [], "method": "exact",
                     "legs": [{"leg_id": "B", "nominal_size": "50A", "waypoints": [], "tee": {"pos": [0, 0, 0], "angle": 135}}]}):
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate(bad, tree)
        self.assertNotIn("tree", schema["properties"]["routes"]["items"]["required"])


if __name__ == "__main__":
    unittest.main()
