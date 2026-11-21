"""D128 서포트 막힘 조회 support_blockers — 조회 전용·결과 불변, 막는 배관 = 그 배관을 빼면 그 (면, 형식) 이 ok.

합성 표(SYN) 로 규칙만 확인한다 (D78)."""
import json
import unittest

from pipe_routing.verifier import PipeRoute, Session, verify
from pipe_routing.verifier.hint import SupportHint
from tests.test_s1_support import SYN
from tests.test_verifier import pipe, scenario


def _dump(reps):
    return json.dumps({k: (v.success, v.layer0, v.supports, [x.message for x in v.violations])
                       for k, v in reps.items()}, sort_keys=True, default=str)


def _crossing_case(step=1000):
    """A (x 방향, z 1500) 아래로 y 방향 배관들이 step 간격으로 지나간다 — A 의 바닥 C 다리 자리를 막는다."""
    pipes = [pipe("A", start=[0, 20000, 1500], end=[40000, 20000, 1500])]
    routes = [PipeRoute("A", [[0, 20000, 1500], [40000, 20000, 1500]])]
    for i, x in enumerate(range(4000, 36001, step)):
        pid = f"C{i:02d}"
        pipes.append(pipe(pid, start=[x, 0, 300], end=[x, 40000, 300], sdir=(0, 1, 0), edir=(0, 1, 0)))
        routes.append(PipeRoute(pid, [[x, 0, 300], [x, 40000, 300]]))
    return scenario(pipes), routes


class SupportBlockersTest(unittest.TestCase):
    def test_blockers_found_and_result_invariant(self):
        sc, routes = _crossing_case(step=300)
        before = _dump(verify(sc, routes, std=SYN)["pipes"])
        h = SupportHint(sc, routes, std=SYN)
        rows = h.support_blockers("A")
        self.assertTrue(rows)
        self.assertTrue(any(r["kind"] == "gap" for r in rows))   # A 는 support 위반 (R7 모양)
        self.assertFalse(verify(sc, routes, std=SYN)["pipes"]["A"].layer0["support"])
        for r in rows:
            self.assertEqual((r["face"], r["form"]), ("floor", "C"))
            self.assertTrue(r["blocker"].startswith("C"))
            self.assertIn("leg", r["members"])   # C 다리가 아래 배관에 닿는다
            # 막는 배관을 빼면 그 위치의 (바닥, C) 가 막히지 않는다 (같은 판정 함수)
            rest = [x for x in routes if x.pipe_id != r["blocker"]]
            st = {(o["face"], o["form"]): o["status"]
                  for o in SupportHint(sc, rest, std=SYN).form_at("A", r["pos"], r["axis"])["options"]}
            self.assertNotEqual(st[("floor", "C")], "blocked")
        self.assertEqual(h.support_blockers("A"), rows)                       # 캐시 — 같은 답
        self.assertEqual(_dump(verify(sc, routes, std=SYN)["pipes"]), before)   # 결과 불변

    def test_displaced(self):
        """막힘 사이에 다른 위치로 받쳐지면 support 통과, 막힘은 displaced."""
        sc, routes = _crossing_case(step=1000)
        self.assertTrue(verify(sc, routes, std=SYN)["pipes"]["A"].layer0["support"])
        rows = SupportHint(sc, routes, std=SYN).support_blockers("A")
        self.assertTrue(rows)
        self.assertEqual({r["kind"] for r in rows}, {"displaced"})

    def test_no_blockers(self):
        sc, routes = _crossing_case()
        h = SupportHint(sc, routes[:1], std=SYN)          # A 혼자
        self.assertEqual(h.support_blockers("A"), [])
        with self.assertRaises(ValueError):
            SupportHint(sc, routes[1:], std=SYN).support_blockers("A")   # 경로 없는 배관

    def test_from_session_invariant(self):
        sc, routes = _crossing_case(step=2000)
        s = Session(sc, std=SYN)
        for r in routes:
            s.add(r.pipe_id, r.waypoints)
        snap = json.dumps(s.result(), sort_keys=True, default=str)
        rows = SupportHint.from_session(s).support_blockers("A")
        self.assertTrue(rows)
        self.assertEqual(json.dumps(s.result(), sort_keys=True, default=str), snap)


if __name__ == "__main__":
    unittest.main()
