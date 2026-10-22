"""S1-4b 시험: D72 결정성 — 같은 시나리오를 병렬 1 과 병렬 4(부하 있음)로 돌려 판정 결과가 바이트 동일."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from pipe_routing import astar_fast
from pipe_routing.pipeline import deterministic_view

ROOT = Path(__file__).resolve().parent.parent
SCEN = ROOT / "scenarios" / "procedural" / "proc_005.json"
LIMIT = 8 * 2048   # 작은 예산 — 예산 소진(timeout)이 실제로 일어나게 해서 그 판정도 결정적인지 본다
PAIR = ROOT / "scenarios" / "pair" / "pair_000.json"


def run(out_dir, hashseed, scen=SCEN):
    env = dict(os.environ, PYTHONHASHSEED=str(hashseed))
    return subprocess.Popen([sys.executable, "-m", "pipe_routing.pipeline", str(scen), "-o", out_dir, "--no-viz",
                             "--expand-limit", str(LIMIT)], cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def view(out_dir, scen=SCEN):
    out = json.loads((Path(out_dir) / f"{scen.stem}_output.json").read_text(encoding="utf-8"))
    return json.dumps(deterministic_view(out), sort_keys=True, ensure_ascii=False).encode()


@unittest.skipUnless(astar_fast.HAVE_NUMBA, "numba 없음")
class DeterminismTest(unittest.TestCase):
    def test_parallel_1_vs_4_with_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            solo = os.path.join(tmp, "solo")
            p = run(solo, 1)
            self.assertEqual(p.wait(), 0, p.stderr.read())
            # 병렬 4 + CPU 부하 프로세스 2개 (경합)
            burners = [subprocess.Popen([sys.executable, "-c", "while True: pass"]) for _ in range(2)]
            try:
                procs = [(os.path.join(tmp, f"par{i}"), run(os.path.join(tmp, f"par{i}"), 100 + i)) for i in range(4)]
                for d, q in procs:
                    self.assertEqual(q.wait(), 0, q.stderr.read())
            finally:
                for b in burners:
                    b.kill()
            ref = view(solo)
            out = json.loads((Path(solo) / "proc_005_output.json").read_text(encoding="utf-8"))
            self.assertGreater(out["global"]["timeout_count"], 0)            # 예산 소진이 실제로 있었다
            self.assertEqual(out["global"]["search_budget"], {"kind": "expansions", "limit": LIMIT, "diagnosis_limit": 4 * LIMIT})
            for d, _ in procs:
                self.assertEqual(view(d), ref, d)

    @unittest.skipUnless(os.environ.get("PIPE_SLOW_TESTS"), "블록 쌍 결정성 — PIPE_SLOW_TESTS=1 일 때만 (수 분, 메모리 수 GB)")
    def test_pair_parallel_1_vs_3_with_load(self):
        """S1-7: 블록 쌍(관통 배관·합성 고정층 LRU·관통 피스) 도 병렬 수와 무관하게 바이트 동일."""
        with tempfile.TemporaryDirectory() as tmp:
            solo = os.path.join(tmp, "solo")
            p = run(solo, 1, PAIR)
            self.assertEqual(p.wait(), 0, p.stderr.read())
            burners = [subprocess.Popen([sys.executable, "-c", "while True: pass"]) for _ in range(2)]
            try:
                procs = [(os.path.join(tmp, f"par{i}"), run(os.path.join(tmp, f"par{i}"), 100 + i, PAIR)) for i in range(3)]
                for d, q in procs:
                    self.assertEqual(q.wait(), 0, q.stderr.read())
            finally:
                for b in burners:
                    b.kill()
            ref = view(solo, PAIR)
            out = json.loads((Path(solo) / "pair_000_output.json").read_text(encoding="utf-8"))
            self.assertGreater(out["global"]["timeout_count"], 0)
            self.assertIn("penetration", json.dumps(out))
            for d, _ in procs:
                self.assertEqual(view(d, PAIR), ref, d)


if __name__ == "__main__":
    unittest.main()
