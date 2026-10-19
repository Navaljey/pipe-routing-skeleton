"""S1-6 서포트 표준 (D78·D79·D80) — 형식 선택·무게·보강형·천장 선호·D 묶음·부재 간섭·로더.

사내 표(비공개, D78) 값은 쓰지 않는다. 시험용 합성 표(SYN)로 규칙만 확인한다.
"""
import math
import unittest

import numpy as np

from pipe_routing import support_std as ss
from pipe_routing.verifier import PipeRoute, verify
from tests.test_verifier import pipe, scenario

EXT = [40000.0, 40000.0, 10000.0]


def synthetic() -> ss.Standard:
    """합성 표: A 0~800 (40×40×5), B 0~1000 (50) · 1000~2500 (65), 보강형 1500~, C 0~1500 (50) · 1500~3000 (75),
    보강형 2000~, D1 0~3000 (65), D2 0~2100 (75)."""
    R = ss.Row
    rows = {}
    for nd in ss.SIZES:
        rows[("A", nd)] = R(np.array([0.0, 800.0]), ("40x40x5",), math.inf)
        rows[("B", nd)] = R(np.array([0.0, 1000.0, 2500.0]), ("50x50x6", "65x65x6"), 1500.0)
        rows[("C", nd)] = R(np.array([0.0, 1500.0, 3000.0]), ("50x50x6", "75x75x6"), 2000.0)
        rows[("D1", nd)] = R(np.array([0.0, 3000.0]), ("65x65x6",), math.inf)
        rows[("D2", nd)] = R(np.array([0.0, 2100.0]), ("75x75x6",), math.inf)
    kg = {"40x40x5": 2.95, "50x50x6": 4.43, "65x65x6": 5.91, "75x75x6": 6.85}
    return ss.Standard("synthetic", "0" * 64, rows, kg)


SYN = synthetic()


def run(pipes, routes, obstacles=(), std=SYN):
    sc = scenario(pipes, obstacles)
    return verify(sc, [PipeRoute(r[0], r[1]) for r in routes], std=std)["pipes"]


class WeightTest(unittest.TestCase):
    def test_form_weights(self):
        od = 114.3
        kg, _, rf = ss.form_weight(SYN, "A", "100A", od, np.array([500.0]))
        self.assertAlmostEqual(kg[0], 2.95 * 0.6)
        kg, _, rf = ss.form_weight(SYN, "B", "100A", od, np.array([2000.0]))   # 보강형: + L/4
        self.assertTrue(rf[0])
        self.assertAlmostEqual(kg[0], 5.91 * (2100 + math.sqrt(2000 ** 2 + 1000 ** 2) + 500) / 1000)
        kg, _, rf = ss.form_weight(SYN, "C", "100A", od, np.array([2500.0]))   # 보강형: + (OD + 200)
        self.assertTrue(rf[0])
        self.assertAlmostEqual(kg[0], 6.85 * (5000 + 2 * (od + 200)) / 1000)
        kg, k, _ = ss.form_weight(SYN, "C", "100A", od, np.array([3000.0]))     # 표 끝 = 불가 (D80 ④)
        self.assertTrue(np.isinf(kg[0]) and k[0] == -1)

    def test_d_weight_bigger_angle(self):
        kg, ang = ss.d_weight(SYN, "100A", 1000.0, 800.0)
        self.assertEqual(ang, "75x75x6")                  # L1 표 65, L2 표 75 → 큰 규격
        self.assertAlmostEqual(kg, 6.85 * (2 * 800 + 1000) / 1000)
        self.assertIsNone(ss.d_weight(SYN, "100A", 3200.0, 800.0))

    def test_public_fallback_and_hash(self):
        s = ss.load_standard("/nonexistent")
        self.assertEqual(s.source, "public_approx")
        self.assertEqual(len(s.sha256), 64)


class SelectTest(unittest.TestCase):
    def test_lightest_and_out_of_table(self):
        X = np.array([[20000, 600, 5000], [20000, 2000, 5000], [20000, 20000, 5000]], float)
        U = np.tile([1.0, 0, 0], (3, 1))
        ev = ss.evaluate(SYN, EXT, [], [], "100A", 114.3, X, U)
        got = [ss.OPTIONS[o] if o >= 0 else None for o in ev["opt"]]
        self.assertEqual(got, [("wall_y0", "A"), ("wall_y0", "B"), None])   # 600 → A, 2000 → B, 중앙 → 불가

    def test_ceiling_preference(self):
        """D79 ④: 천장 무게 ≤ 바닥 × 1.05 이면 천장 (높이 5 m 블록, 바닥 2490 · 천장 2510)."""
        ext = [40000.0, 40000.0, 5000.0]
        X = np.array([[20000, 20000, 2490.0]])
        ev = ss.evaluate(SYN, ext, [], [], "100A", 114.3, X, np.array([[1.0, 0, 0]]))
        self.assertEqual(ss.OPTIONS[ev["opt"][0]], ("ceiling", "C"))
        ev = ss.evaluate(SYN, ext, [], [], "100A", 114.3, X, np.array([[1.0, 0, 0]]), prefer_ceiling=False)
        self.assertEqual(ss.OPTIONS[ev["opt"][0]], ("floor", "C"))

    def test_member_interference_with_obstacle(self):
        """C 다리 하나가 장비에 닿으면 그 조합은 불가 → 다른 조합 (천장 C 범위 밖, 격벽 범위 밖 → 불가)."""
        X = np.array([[20000, 20000, 1000.0]])
        U = np.array([[1.0, 0, 0]])
        leg_y = 20000 + (114.3 + 200) / 2
        ev = ss.evaluate(SYN, EXT, [[19900, leg_y + 10, 0]], [[20100, leg_y + 100, 400]], "100A", 114.3, X, U)
        self.assertEqual(ev["opt"][0], -1)      # 다리 반폭 25 > 간격 10
        ev = ss.evaluate(SYN, EXT, [[19900, leg_y + 40, 0]], [[20100, leg_y + 100, 400]], "100A", 114.3, X, U)
        self.assertEqual(ss.OPTIONS[ev["opt"][0]], ("floor", "C"))


class VerifierTest(unittest.TestCase):
    def test_floor_vs_wall_lightest(self):
        """바닥 C (1500) 와 격벽 B (2000) 중 가벼운 조합 (D80 ③)."""
        p = pipe(start=[0, 2000, 1500], end=[40000, 2000, 1500])
        reps = run([p], [("P", [[0, 2000, 1500], [40000, 2000, 1500]])])
        forms = {s["form"] for s in reps["P"].supports}
        self.assertTrue(reps["P"].success)
        self.assertIn("C", forms)               # 바닥 C (1500) vs 격벽 B (2000): 가벼운 쪽

    def test_support_violation_mid_height(self):
        p = pipe(start=[0, 20000, 5000], end=[40000, 20000, 5000])
        reps = run([p], [("P", [[0, 20000, 5000], [40000, 20000, 5000]])])
        self.assertFalse(reps["P"].layer0["support"])
        self.assertTrue(reps["P"].support_gaps)

    def test_d_grouping(self):
        """나란한 두 배관 (같은 높이, 가로 간격 600) — 같은 축 위치의 C 두 개를 D 하나로 묶는다 (D80 ⑥)."""
        a = pipe("A", start=[0, 20000, 1000], end=[40000, 20000, 1000])
        b = pipe("B", start=[0, 20600, 1000], end=[40000, 20600, 1000])
        reps = run([a, b], [("A", [[0, 20000, 1000], [40000, 20000, 1000]]),
                            ("B", [[0, 20600, 1000], [40000, 20600, 1000]])])
        sa, sb = reps["A"].supports, reps["B"].supports
        self.assertTrue(all(s["form"] == "D" for s in sa + sb))
        g = sa[0]
        self.assertEqual(g["carried"], ["A", "B"])
        L1 = 600 + 114.3 + 200
        L2 = 1000 - 114.3 / 2
        self.assertAlmostEqual(g["dims"]["L1"], round(L1, 1))
        self.assertAlmostEqual(g["kg_total"], round(6.85 * (2 * L2 + L1) / 1000, 4))
        self.assertAlmostEqual(g["kg"] + sb[0]["kg"], g["kg_total"], places=3)
        indiv = 4.43 * (2 * 1000 + 314.3) / 1000
        self.assertLess(g["kg_total"], 2 * indiv)

    def test_d_share_unit_weight(self):
        """D81: 묶음 무게를 얹힌 배관 단위중량(§3.1 kg/m) 비례로 귀속 (100A 16.10 · 65A 9.11, 같은 간격 3 m)."""
        a = pipe("A", start=[0, 20000, 1000], end=[40000, 20000, 1000])
        b = pipe("B", size="65A", start=[0, 20600, 980], end=[40000, 20600, 980])
        reps = run([a, b], [("A", [[0, 20000, 1000], [40000, 20000, 1000]]),
                            ("B", [[0, 20600, 980], [40000, 20600, 980]])])
        sa, sb = reps["A"].supports[0], reps["B"].supports[0]
        self.assertEqual((sa["form"], sb["form"], sa["group"]), ("D", "D", sb["group"]))
        self.assertAlmostEqual(sa["kg_share"], sa["kg_total"] * 16.10 / (16.10 + 9.11), places=3)
        self.assertAlmostEqual(sb["kg_share"], sb["kg_total"] * 9.11 / (16.10 + 9.11), places=3)
        self.assertEqual(sa["kg"], sa["kg_share"])

    def test_no_grouping_when_surfaces_differ(self):
        a = pipe("A", start=[0, 20000, 1000], end=[40000, 20000, 1000])
        b = pipe("B", start=[0, 20600, 1100], end=[40000, 20600, 1100])
        reps = run([a, b], [("A", [[0, 20000, 1000], [40000, 20000, 1000]]),
                            ("B", [[0, 20600, 1100], [40000, 20600, 1100]])])
        self.assertTrue(all(s["form"] == "C" for s in reps["A"].supports + reps["B"].supports))

    def test_member_avoids_other_pipe(self):
        """얹히지 않은 배관이 C 다리 자리를 지나가면 그 위치는 피한다 (D80 ⑧)."""
        a = pipe("A", start=[0, 20000, 1500], end=[40000, 20000, 1500])
        # 바닥 근처를 가로지르는 배관 (A 의 다리 높이 대 z=300)
        c = pipe("C", start=[20000, 0, 300], end=[20000, 40000, 300], sdir=(0, 1, 0), edir=(0, 1, 0))
        reps = run([a, c], [("A", [[0, 20000, 1500], [40000, 20000, 1500]]),
                            ("C", [[20000, 0, 300], [20000, 40000, 300]])])
        r_c = 117.15
        for s in reps["A"].supports:
            for m in s["members"]:
                if m["role"] == "leg":
                    self.assertGreater(abs(m["a"][0] - 20000), r_c + 25 - 1e-6)


class TogetherTest(unittest.TestCase):
    """D89 막히면 함께 받친다 · D82 보완 간격 재확인 (2026-10-19)."""

    def test_d89_mutual_pair(self):
        """가로 250 mm 로 붙은 두 100A 배관 (z = 1000): 서로 상대 C 다리 자리를 막고 천장·벽은 표 밖 → A·B·C 불가.
        서로 함께 얹는 D 하나씩 (같은 위치라 실물 하나), D81 몫, support 통과."""
        a = pipe("A", start=[0, 20000, 1000], end=[40000, 20000, 1000])
        b = pipe("B", start=[0, 20250, 1000], end=[40000, 20250, 1000])
        routes = [("A", [[0, 20000, 1000], [40000, 20000, 1000]]), ("B", [[0, 20250, 1000], [40000, 20250, 1000]])]
        alone = run([a], routes[:1])
        self.assertTrue(all(s["form"] == "C" for s in alone["A"].supports))
        reps = run([a, b], routes)
        self.assertTrue(reps["A"].layer0["support"] and reps["B"].layer0["support"])
        sa, sb = reps["A"].supports, reps["B"].supports
        self.assertTrue(all(s.get("origin") == "D89" and s["carried"] == ["A", "B"] for s in sa + sb))
        real = [s for s in sa + sb if not s.get("attached")]
        self.assertTrue(real and all(s["members"] for s in real))
        by_gid = {}
        for s in sa + sb:
            by_gid.setdefault(s["group"], []).append(s)
        for gid, lst in by_gid.items():
            self.assertEqual(len(lst), 2)                         # 실물 1 + 사본 1
            self.assertEqual(sum(1 for s in lst if not s.get("attached")), 1)
            self.assertAlmostEqual(sum(s["kg"] for s in lst), lst[0]["kg_total"], places=3)
            share_a = next(s["kg"] for s in lst if s in sa)
            self.assertAlmostEqual(share_a, lst[0]["kg_total"] / 2, places=3)   # 같은 구경 → 반반 (D81)
        # 서로 막는 쌍이라 양쪽 탐욕 위치가 같다 → B 의 서포트도 모두 B 자기 위치 (s_mm 유지)
        self.assertTrue(all("s_mm" in s for s in sb))

    def test_d82_spacing_recheck(self):
        """B 가 400 mm 늦게 시작 — 탐욕 위치가 서로 400 mm 어긋나 묶으면 평균 위치로 200 mm 씩 옮겨지고,
        최대 간격(100A 3.0 m) 끝에 놓인 서포트가 간격을 넘는다 → 묶지 않는다. 재확인을 끄면 묶인다."""
        from pipe_routing.verifier import support_group as sg
        a = pipe("A", start=[0, 20000, 1000], end=[40000, 20000, 1000])
        b = pipe("B", end=[40000, 20600, 1000])
        b["start"] = {"pos": [400, 20600, 1000], "kind": "nozzle", "dir": [1, 0, 0]}
        routes = [("A", [[0, 20000, 1000], [40000, 20000, 1000]]), ("B", [[400, 20600, 1000], [40000, 20600, 1000]])]
        reps = run([a, b], routes)
        self.assertTrue(all(s["form"] == "C" for s in reps["A"].supports + reps["B"].supports))
        old = sg._group_spacing_ok
        sg._group_spacing_ok = lambda *a_, **k_: True
        try:
            reps = run([a, b], routes)
        finally:
            sg._group_spacing_ok = old
        self.assertTrue(any(s["form"] == "D" for s in reps["A"].supports))

    def test_spacing_ok(self):
        from pipe_routing.verifier import support_group as sg
        from pipe_routing.verifier.core import Context, centerlines_of
        a = pipe("A", start=[0, 20000, 1000])
        a["end"] = {"pos": [10000, 20000, 1000], "kind": "nozzle", "dir": [1, 0, 0]}
        sc = scenario([a])
        r = PipeRoute("A", [[0, 20000, 1000], [10000, 20000, 1000]])
        P = sc.pipes[0]
        ctx = Context(sc, {"A": P}, {"A": r}, {"A": centerlines_of(P, r)}, std=SYN)
        s_list = [3000.0, 6000.0, 8500.0]
        self.assertTrue(sg.spacing_ok(ctx, "A", s_list, 1, 5900.0))
        self.assertFalse(sg.spacing_ok(ctx, "A", s_list, 0, 3100.0))   # 단자(0) → 3.1 m > 3.0
        self.assertFalse(sg.spacing_ok(ctx, "A", s_list, 1, 6100.0))   # 뒤 8500 까지 2.4 지만 앞 3000 → 3.1
        self.assertFalse(sg.spacing_ok(ctx, "A", s_list, 1, 9100.0))   # 이웃을 넘음


if __name__ == "__main__":
    unittest.main()
