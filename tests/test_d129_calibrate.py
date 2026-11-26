"""D129 보정 묶음 — 목록·선정 기준·동결 t₀ 합이 v3_contract §e3 과 같다 (선수가 이 목록으로 r 을 잰다)."""
import json
import unittest
from pathlib import Path

from pipe_routing import calibrate

REPO = Path(__file__).resolve().parent.parent


class CalibrateBundleTest(unittest.TestCase):
    def test_frozen_sum(self):
        self.assertAlmostEqual(round(sum(calibrate.frozen_t0().values()), 1), 5331.0)

    def test_bundle_is_median_closest(self):
        d = json.loads((REPO / "docs/eval_set_E1_baseline.json").read_text(encoding="utf-8"))
        for s, p in calibrate.BUNDLE.items():
            t = {k: v["curve"]["t0_sec"] for k, v in d["sets"][s].items()}
            v = sorted(t.values())
            n = len(v)
            med = v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2
            # 같은 거리면 큰 쪽 (기본 세트 proc_013 · proc_012)
            best = min(t, key=lambda k: (round(abs(t[k] - med), 6), -t[k]))
            self.assertEqual(best, Path(p).stem, s)
            self.assertTrue((REPO / p).exists())


if __name__ == "__main__":
    unittest.main()
