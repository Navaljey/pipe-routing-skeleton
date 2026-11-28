"""D129 보완 `--no-classify` — 결과 불변 시험.

분류 (D51·D74 진단 재라우팅·예약 확인 재라우팅) 와 D101 만 건너뛴다. 분류 필드를 뺀 pipeline 출력·anytime 곡선
(벽시계 제외 결정적 부분) 이 켠 경우와 바이트 동일하고, D77 지표가 같다.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from pipe_routing import anytime, astar_fast, calibrate
from pipe_routing.pipeline import deterministic_view, run
from pipe_routing.layered import make_layered_router
from pipe_routing.scenario import load

ROOT = Path(__file__).resolve().parent.parent
SCEN = ROOT / "scenarios" / "manual" / "manual_01.json"
ROUTE_CLASS = ("fail_class", "fail_detail", "solo")
GLOBAL_CLASS = ("fail_class", "fail_class_status", "solo_reference", "classify")
CURVE_SKIP = ("sec", "total_sec", "t0_sec", "machine", "classify", "interference_block")


def strip(out: dict) -> str:
    o = deterministic_view(out)
    o = json.loads(json.dumps(o))
    for r in o["routes"]:
        for k in ROUTE_CLASS:
            r.pop(k, None)
    for k in GLOBAL_CLASS:
        o["global"].pop(k, None)
    return json.dumps(o, ensure_ascii=False, sort_keys=True)


def strip_curve(c: dict) -> str:
    def walk(x):
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items() if k not in CURVE_SKIP}
        if isinstance(x, list):
            return [walk(v) for v in x]
        return x
    return json.dumps(walk(c), ensure_ascii=False, sort_keys=True)


D77 = ("full_success", "required_pipes", "required_passed", "excluded_infeasible", "success_count", "J_total")


@unittest.skipUnless(astar_fast.HAVE_NUMBA, "numba 없음")
class NoClassifyTest(unittest.TestCase):
    def test_pipeline(self):
        sc = load(SCEN)
        on = run(sc, router=make_layered_router())
        off = run(load(SCEN), router=make_layered_router(), classify=False)
        self.assertTrue(any(on["global"]["fail_class"].values()))          # 켜면 분류가 실제로 있다 (시험이 뜻을 가진다)
        self.assertFalse(any(off["global"]["fail_class"].values()))
        self.assertIs(off["global"]["classify"], False)
        self.assertNotIn("classify", on["global"])                         # 켠 출력은 이전과 같은 모양
        self.assertNotIn("solo_reference", off["global"])
        self.assertEqual(strip(on), strip(off))
        self.assertEqual({k: on["global"][k] for k in D77}, {k: off["global"][k] for k in D77})

    def test_anytime_curve(self):
        """anytime CLI 곡선·최선 해·첫 해 출력: 분류 필드와 벽시계를 뺀 결정적 부분이 같다."""
        e0 = anytime.anytime_run(load(SCEN), make_layered_router(), seed=0, max_expanded=1).e0
        res = {}
        with tempfile.TemporaryDirectory() as td:
            for tag, extra in (("on", []), ("off", ["--no-classify"])):
                od = Path(td) / tag
                anytime.main([str(SCEN), "-o", str(od), "--max-expanded", str(e0 + 300_000), "--seed", "7"] + extra)
                res[tag] = {k: json.loads((od / f"manual_01_{k}.json").read_text(encoding="utf-8"))
                            for k in ("curve", "output", "first_output")}
        self.assertEqual(strip_curve(res["on"]["curve"]), strip_curve(res["off"]["curve"]))
        self.assertIs(res["off"]["curve"]["classify"], False)
        self.assertTrue(all(p["interference_block"] is None for p in res["off"]["curve"]["points"]))
        for k in ("output", "first_output"):
            self.assertEqual(strip(res["on"][k]), strip(res["off"][k]), k)

    def test_calibrate_runner(self):
        """보정 실행기 (--no-classify) 의 e0 = anytime 첫 해 e0 (같은 탐색), 서포트 표 source 기록."""
        e0 = anytime.anytime_run(load(SCEN), make_layered_router(), seed=0, max_expanded=1).e0
        with tempfile.TemporaryDirectory() as td:
            r = subprocess.run([sys.executable, "-c", calibrate.RUNNER, str(SCEN), td], cwd=ROOT,
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            d = json.loads((Path(td) / "manual_01_t0.json").read_text())
        self.assertEqual(d["e0"], e0)
        self.assertGreater(d["t0_sec"], 0)
        self.assertIn(d["support_source"], ("private", "public_approx"))


if __name__ == "__main__":
    unittest.main()
