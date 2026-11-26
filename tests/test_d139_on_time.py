"""D139 보완 (나) 기준선 anytime 정시 멈춤 — 결과 불변 시험.

판정 해 (가) = 예산 끝 이전 시각이 찍힌 마지막 해. 정시 멈춤 판 (on_time=True) 의 최종 해가 이전 판 (on_time=False,
같은 예산) 의 "예산 끝 이전 마지막 해" 와 `deterministic_view` 바이트 동일해야 한다.

결정적으로 비교하려고 벽시계 대신 누적 확장 수에 비례하는 가상 시계 (vclock) 를 쓰고, 탐색 안 정시 멈춤 경로
(`astar_fast.budget_over` → ABORT → Deadline) 를 그대로 지나는 모의 라우터를 쓴다 — 확장을 조금씩 세며 budget_over 를
부른다 (컴파일 A* 와 같은 확인 지점). 모의 라우터는 첫 해에서 한 배관을 막고, 순서 무작위 재시작에서 둘 다 경로가
있는 해를 낸다 (개선 1 회, D75 수용).
"""
import json
import unittest

from pipe_routing import astar_fast
from pipe_routing.anytime import anytime_run
from pipe_routing.multi import PlanResult, default_order
from pipe_routing.pipeline import deterministic_view, run
from pipe_routing.router_astar import RouteResult
from tests.test_multi import boundary_pipe, scenario

RATE = 1e3                  # 가상 시계: 확장 1천 = 1 초
CALL_EXP = 4000             # 라우팅 1 회 확장 수
STEP = 500                  # budget_over 확인 간격 (컴파일 A* 의 CHECK_EVERY 처럼)
Z = 1000


def vclock(e):
    return e / RATE


def line(y):
    return [[0.0, float(y), float(Z)], [40000.0, float(y), float(Z)]]


class Mock:
    """BIG 은 SMALL 이 먼저 놓였으면 y 12000, 아니면 y 10000. SMALL 은 BIG 이 y 10000 에 있으면 실패, 아니면 y 30000."""

    def __call__(self, sc, pipe, time_limit, placed=()):
        for e in range(0, CALL_EXP, STEP):          # 탐색 흉내 — 확인 지점마다 정시 멈춤을 본다
            astar_fast.budget_over(0.0, None, 10 ** 12, e)
        wps = {p.id: w for p, w in placed}
        if pipe.id == "BIG":
            w = line(12000 if "SMALL" in wps else 10000)
        else:
            w = None if wps.get("BIG") == line(10000) else line(30000)
        if w is None:
            return RouteResult(pipe.id, "unreachable", expanded=CALL_EXP, search_sec=0.0)
        return RouteResult(pipe.id, "ok", J=pipe.kg_per_m * 40.0, waypoints=w, expanded=CALL_EXP, search_sec=0.0)


def sc_two():
    def p(pid, size, y):
        d = boundary_pipe(pid, size, y, z=Z)
        return d
    return scenario([p("SMALL", "25A", 30000), p("BIG", "100A", 10000)])


def sig(routes):
    return tuple((pid, r.status, tuple(map(tuple, r.waypoints or []))) for pid, r in sorted(routes.items()))


class OnTimeStopTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sc = sc_two()
        cls.full = anytime_run(cls.sc, Mock(), seed=0, max_sec=200.0, on_time=False, vclock=vclock)

    def last_before(self, tr, deadline):
        return [p for p in tr.points if p[1] <= deadline][-1]

    def test_fixture_improves(self):
        pts = self.full.points
        self.assertEqual(len(pts), 2)
        self.assertEqual(pts[1][3], "restart")
        self.assertEqual({k: r.status for k, r in pts[1][2].items()}, {"BIG": "ok", "SMALL": "ok"})
        self.assertEqual({k: r.status for k, r in pts[0][2].items()}, {"BIG": "ok", "SMALL": "unreachable"})

    def test_same_judged_solution(self):
        pts = self.full.points
        t1 = pts[1][1]
        # 예산 끝 = 개선 이동의 라우팅 한가운데 · 개선 직후 · 첫 해와 개선 사이 (다른 이동 안) · 끝 뒤
        ds = [t1 - 2.5, t1 + 0.001, (pts[0][1] + t1) / 2, pts[-1][1] + 7.3]
        inside = 0
        for d in ds:
            with self.subTest(deadline=d):
                old = anytime_run(self.sc, Mock(), seed=0, max_sec=d, on_time=False, vclock=vclock)
                new = anytime_run(self.sc, Mock(), seed=0, max_sec=d, on_time=True, vclock=vclock)
                self.assertIsNone(astar_fast.ABORT)           # 끝나면 해제 (뒤 분류·출력 라우팅에 영향 없음)
                want, got = self.last_before(old, d), new.points[-1]
                self.assertLessEqual(got[1], d)
                self.assertEqual((sig(got[2]), got[3], got[4]), (sig(want[2]), want[3], want[4]))
                self.assertEqual([(p[0], p[1], p[3], p[4]) for p in new.points],
                                 [(p[0], p[1], p[3], p[4]) for p in old.points if p[1] <= d])
                self.assertEqual(new.stop, "deadline")
                self.assertLessEqual(new.total_expanded, d * RATE)          # 예산 끝 뒤 확장은 세지 않는다
                inside += new.total_expanded % CALL_EXP == 0 and old.total_expanded > new.total_expanded
        self.assertGreater(inside, 0)                          # 이동 중간 (라우팅 안) 에서 끊긴 경우가 있다

    def test_deterministic_view_identical(self):
        """판정 해의 pipeline 출력 (`deterministic_view`) 바이트 동일."""
        d = self.full.points[1][1] + 0.001
        old = anytime_run(self.sc, Mock(), seed=0, max_sec=d, on_time=False, vclock=vclock)
        new = anytime_run(self.sc, Mock(), seed=0, max_sec=d, on_time=True, vclock=vclock)
        order = [p.id for p in default_order(self.sc)]
        outs = []
        for routes in (self.last_before(old, d)[2], new.points[-1][2]):
            plan = PlanResult(routes=dict(routes), order=order, planner="anytime_best")
            out = run(self.sc, router=Mock(), planner=lambda sc_, r_, tl: plan, solo_ref=False)
            outs.append(json.dumps(deterministic_view(out), ensure_ascii=False, sort_keys=True))
        self.assertEqual(outs[0], outs[1])

    def test_off_by_default_router(self):
        """정시 멈춤을 끄면 (on_time=False) 이전과 같다 — ABORT 를 걸지 않는다."""
        tr = anytime_run(self.sc, Mock(), seed=0, max_sec=self.full.points[1][1] - 2.5, on_time=False, vclock=vclock)
        self.assertNotEqual(tr.stop, "deadline")


if __name__ == "__main__":
    unittest.main()
