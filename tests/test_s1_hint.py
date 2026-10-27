"""S1-7b 서포트 힌트 (D100) — 힌트 답 = 검증기가 실제로 놓는 것, 결과 불변.

합성 표(SYN) 로 규칙만 확인한다 (D78: 사내 표 값은 쓰지 않는다).
"""
import json
import unittest

from pipe_routing.verifier import PipeRoute, Session, verify
from pipe_routing.verifier.hint import SupportHint
from tests.test_s1_support import SYN
from tests.test_verifier import pipe, scenario


def _dump(reps):
    return json.dumps({k: (v.success, v.layer0, v.supports, [x.message for x in v.violations])
                       for k, v in reps.items()}, sort_keys=True, default=str)


def _same_choice(test, s, c):
    test.assertIsNotNone(c)
    for k in ("form", "face", "variant", "section", "kg", "L", "members"):
        test.assertEqual(c[k], s[k], k)


class FormAtTest(unittest.TestCase):
    def test_form_at_matches_verifier(self):
        """여러 배관이 놓인 상태에서 검증기가 놓은 A·B·C 서포트마다 form_at 의 선택이 같다 (형식·면·무게·부재)."""
        pipes = [pipe("A", start=[0, 20000, 1500], end=[40000, 20000, 1500]),
                 pipe("C", start=[20000, 0, 300], end=[20000, 40000, 300], sdir=(0, 1, 0), edir=(0, 1, 0)),
                 pipe("W", start=[0, 600, 5000], end=[40000, 600, 5000])]
        routes = [PipeRoute("A", [[0, 20000, 1500], [40000, 20000, 1500]]),
                  PipeRoute("C", [[20000, 0, 300], [20000, 40000, 300]]),
                  PipeRoute("W", [[0, 600, 5000], [40000, 600, 5000]])]
        sc = scenario(pipes)
        reps = verify(sc, routes, std=SYN)["pipes"]
        before = _dump(reps)
        h = SupportHint(sc, routes, std=SYN)
        n = 0
        forms = set()
        for pid, rep in reps.items():
            for s in rep.supports:
                if s["form"] == "D":
                    continue
                r = h.form_at(pid, s["pos"], s["axis"])
                _same_choice(self, s, r["chosen"])
                ok = [o for o in r["options"] if o["status"] == "ok"]
                self.assertIn((s["face"], s["form"]), {(o["face"], o["form"]) for o in ok})
                forms.add(s["form"])
                n += 1
        self.assertGreater(n, 20)
        self.assertGreaterEqual(len(forms), 2)
        self.assertEqual(_dump(verify(sc, routes, std=SYN)["pipes"]), before)   # 결과 불변

    def test_form_at_blocked_reason(self):
        """다른 배관이 C 다리 자리를 지나가는 위치: 그 (바닥, C) 는 blocked 로 답한다."""
        pipes = [pipe("A", start=[0, 20000, 1500], end=[40000, 20000, 1500]),
                 pipe("C", start=[20000, 0, 300], end=[20000, 40000, 300], sdir=(0, 1, 0), edir=(0, 1, 0))]
        sc = scenario(pipes)
        h = SupportHint(sc, {"C": [[20000, 0, 300], [20000, 40000, 300]]}, std=SYN)
        r = h.form_at("A", [20000, 20000, 1500], [1, 0, 0])
        st = {(o["face"], o["form"]): o["status"] for o in r["options"]}
        self.assertEqual(st[("floor", "C")], "blocked")
        self.assertEqual(st[("wall_x0", "A")], "parallel")


class GroupableTest(unittest.TestCase):
    def _pair(self, zb):
        a = pipe("A", start=[0, 20000, 1000], end=[40000, 20000, 1000])
        b = pipe("B", start=[0, 20600, zb], end=[40000, 20600, zb])
        routes = [PipeRoute("A", [[0, 20000, 1000], [40000, 20000, 1000]]),
                  PipeRoute("B", [[0, 20600, zb], [40000, 20600, zb]])]
        sc = scenario([a, b])
        return sc, routes, verify(sc, routes, std=SYN)["pipes"]

    def test_groupable_matches_verifier_groups(self):
        """검증기가 만든 D 묶음마다 groupable 이 같은 면·같은 묶음 무게로 ok."""
        sc, routes, reps = self._pair(1000)
        h = SupportHint(sc, routes, std=SYN)
        groups = {}
        for pid, rep in reps.items():
            for s in rep.supports:
                if s["form"] == "D" and s.get("origin") != "D89":
                    groups.setdefault(s["group"], []).append((pid, s))
        self.assertTrue(groups)
        for gid, lst in groups.items():
            (pa, sa), (pb, sb) = lst
            g = h.groupable((pa, sa["pos"], sa["axis"]), (pb, sb["pos"], sb["axis"]), face=sa["face"])
            self.assertTrue(g["ok"], g)
            self.assertAlmostEqual(g["kg"], sa["kg_total"], places=4)
            self.assertEqual(g["section"], sa["section"])
            self.assertLess(g["kg"], g["kg_individual"])

    def test_not_groupable_surface(self):
        """표면 높이 차 100 mm (> 50) — 검증기도 묶지 않고 힌트도 그 이유로 아니라고 답한다."""
        sc, routes, reps = self._pair(1100)
        self.assertTrue(all(s["form"] == "C" for r in reps.values() for s in r.supports))
        h = SupportHint(sc, routes, std=SYN)
        sa, sb = reps["A"].supports[0], reps["B"].supports[0]
        g = h.groupable(("A", sa["pos"], sa["axis"]), ("B", sb["pos"], sb["axis"]))
        self.assertFalse(g["ok"])
        self.assertIn("표면 높이", g["reason"])


class TogetherAtTest(unittest.TestCase):
    def test_together_matches_verifier_d89(self):
        """서로 C 다리 자리를 막는 100A 쌍 — 검증기의 D89 실물마다 form_at 은 None, together_at 은 같은 D (id·무게·부재)."""
        a = pipe("A", start=[0, 20000, 1000], end=[40000, 20000, 1000])
        b = pipe("B", start=[0, 20250, 1000], end=[40000, 20250, 1000])
        routes = [PipeRoute("A", [[0, 20000, 1000], [40000, 20000, 1000]]),
                  PipeRoute("B", [[0, 20250, 1000], [40000, 20250, 1000]])]
        sc = scenario([a, b])
        reps = verify(sc, routes, std=SYN)["pipes"]
        sess = Session(sc, std=SYN)
        for r in routes:
            sess.add(r.pipe_id, r.waypoints)
        before = _dump(sess.reports())
        h = SupportHint.from_session(sess)
        n = 0
        for pid, rep in reps.items():
            for s in rep.supports:
                if s.get("origin") != "D89" or s.get("attached"):
                    continue
                self.assertIsNone(h.form_at(pid, s["pos"], s["axis"])["chosen"])
                t = h.together_at(pid, s["pos"], s["axis"])
                for k in ("group", "face", "section", "kg_total", "carried", "members", "shares"):
                    self.assertEqual(t[k], s[k], k)
                n += 1
        self.assertGreater(n, 5)
        self.assertEqual(_dump(sess.reports()), before)   # Session 상태 그대로
        self.assertEqual(_dump(sess.reports()), _dump(reps))


if __name__ == "__main__":
    unittest.main()
