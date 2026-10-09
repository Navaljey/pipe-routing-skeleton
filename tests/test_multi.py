"""7단계 다중 배관 (D50) 시험."""
import unittest

from pipe_routing.multi import default_order, independent_planner, routes_conflict, sequential_ripup_planner
from pipe_routing.router_astar import RouteResult
from pipe_routing.scenario import from_dict
from pipe_routing.verifier import PipeRoute, verify


def boundary_pipe(pid, size, y, z=5000):
    return {"id": pid, "type_id": "T1", "nominal_size": size, "gravity_pipe": False, "min_slope": None,
            "insulation_thickness": 50,
            "start": {"pos": [0, y, z], "kind": "boundary", "dir": [1, 0, 0]},
            "end": {"pos": [40000, y, z], "kind": "boundary", "dir": [1, 0, 0]},
            "valve_positions": [], "branch_points": []}


def scenario(pipes, obstacles=()):
    return from_dict({"block": {"width": 40000, "length": 40000, "height": 10000, "unit": "mm"},
                      "obstacles": [{"id": f"OBS_{i}", "type": "wall", "min": lo, "max": hi}
                                    for i, (lo, hi) in enumerate(obstacles)],
                      "pipes": pipes})


# x = 19000~21000 벽, 구멍 y 19800~20100 · z 4850~5150 (300 × 300) — 100A 와 25A 가 동시에 지나갈 수 없다
HOLE_WALL = [([19000, 0, 0], [21000, 40000, 4850]), ([19000, 0, 5150], [21000, 40000, 10000]),
             ([19000, 0, 4850], [21000, 19800, 5150]), ([19000, 20100, 4850], [21000, 40000, 5150])]


class OrderTest(unittest.TestCase):
    """두 배관이 같은 통로(구멍 하나)를 다툰다 — 순서에 따라 결과가 바뀐다 (실제 A*)."""

    @classmethod
    def setUpClass(cls):
        cls.sc = scenario([boundary_pipe("SMALL", "25A", 30000), boundary_pipe("BIG", "100A", 10000)], HOLE_WALL)

    def test_default_order(self):
        self.assertEqual([p.id for p in default_order(self.sc)], ["BIG", "SMALL"])   # 호칭경 내림차순 (D50①)

    def test_order_decides_who_passes(self):
        big_first = sequential_ripup_planner(self.sc, max_ripups=0)
        self.assertEqual(big_first.routes["BIG"].status, "ok")
        self.assertEqual(big_first.routes["SMALL"].status, "unreachable")
        small_first = sequential_ripup_planner(self.sc, max_ripups=0, order_fn=lambda sc: sc.pipes)
        self.assertEqual(small_first.routes["SMALL"].status, "ok")
        self.assertEqual(small_first.routes["BIG"].status, "unreachable")
        # 놓인 배관을 피해 라우팅했으므로 검증기에서도 배관 간 충돌이 없다 (라우터–검증기 정합)
        rep = verify(self.sc, [PipeRoute(pid, r.waypoints) for pid, r in big_first.routes.items()],
                     modules=["collision", "bend"])
        self.assertEqual([v.message for x in rep["pipes"].values() for v in x.violations], [])

    def test_failure_classified_as_interference(self):
        """최종 실패 배관이 단독으로는 성공 → "간섭" (D50⑥)."""
        res = sequential_ripup_planner(self.sc, max_ripups=0)
        self.assertEqual(res.fail_class, {"SMALL": "interference"})

    def test_ripup_cap(self):
        """구멍 하나를 두고 자리를 바꿔도 성공 수는 그대로 — rip-up 은 최대 3회에서 멈춘다 (D50④)."""
        res = sequential_ripup_planner(self.sc)
        self.assertEqual(res.ripups, 3)
        self.assertEqual(res.reverted, 0)
        self.assertEqual(sum(r.status == "ok" for r in res.routes.values()), 1)
        self.assertEqual(len(res.fail_class), 1)


class MockRouter:
    """플래너 논리 시험용 라우터. 놓인 배관에 따라 미리 정한 경로를 돌려준다."""

    def __init__(self, table):
        self.table = table    # pid → fn(placed_ids, placed_wps) → waypoints | None

    def __call__(self, sc, pipe, time_limit, placed=()):
        ids = {p.id for p, _ in placed}
        wps = {p.id: w for p, w in placed}
        w = self.table[pipe.id](ids, wps)
        if w is None:
            return RouteResult(pipe.id, "unreachable", search_sec=0.0)
        return RouteResult(pipe.id, "ok", J=1.0, waypoints=w, search_sec=0.0)


def line(y, z=5000):
    return [[0, y, z], [40000, y, z]]


SHORT_A, LONG_A = line(20000), line(30000)


class RipupTest(unittest.TestCase):
    def test_ripup_increases_success(self):
        """A(큰 배관)가 먼저 짧은 경로를 차지해 B 가 실패 → B 단독 경로와 충돌하는 A 를 걷어내고
        B 먼저, A 다시 → A 는 긴 경로로 우회해 둘 다 성공."""
        sc = scenario([boundary_pipe("A", "100A", 20000), boundary_pipe("B", "50A", 20100)])
        router = MockRouter({
            "A": lambda ids, w: LONG_A if "B" in ids else SHORT_A,
            "B": lambda ids, w: None if w.get("A") == SHORT_A else line(20100),
        })
        res = sequential_ripup_planner(sc, router=router)
        self.assertEqual(res.ripups, 1)
        self.assertEqual(res.reverted, 0)
        self.assertEqual(res.routes["A"].waypoints, LONG_A)
        self.assertEqual(res.routes["B"].status, "ok")
        self.assertEqual(res.fail_class, {})
        self.assertTrue(routes_conflict(sc.pipes[0], SHORT_A, sc.pipes[1], line(20100)))
        self.assertFalse(routes_conflict(sc.pipes[0], LONG_A, sc.pipes[1], line(20100)))

    def test_ripup_reverted_when_worse(self):
        """B 를 위해 A·C 를 걷어냈더니 A·C 둘 다 다시 못 깔림 (성공 2 → 1) → 직전 상태로 되돌린다."""
        # A(300A, r 219) y=20000, C(200A, r 168) y=20400 — 서로 400 ≥ 387 로 공존. B(50A, r 90) y=20200 은 둘 다와 충돌
        sc = scenario([boundary_pipe("A", "300A", 20000), boundary_pipe("C", "200A", 20400),
                       boundary_pipe("B", "50A", 20200)])
        router = MockRouter({
            "A": lambda ids, w: None if "B" in ids else SHORT_A,
            "C": lambda ids, w: None if "B" in ids else line(20400),
            "B": lambda ids, w: None if ("A" in ids or "C" in ids) else line(20200),
        })
        res = sequential_ripup_planner(sc, router=router)
        self.assertEqual(res.order, ["A", "C", "B"])
        self.assertEqual(res.ripups, 1)
        self.assertEqual(res.reverted, 1)
        self.assertEqual({k for k, r in res.routes.items() if r.status == "ok"}, {"A", "C"})
        self.assertEqual(res.fail_class, {"B": "interference"})

    def test_individual_failure_not_ripped(self):
        """단독으로도 실패하는 배관은 rip-up 하지 않고 "개별 경로"로 분류."""
        sc = scenario([boundary_pipe("A", "100A", 20000), boundary_pipe("B", "50A", 25000)])
        router = MockRouter({"A": lambda ids, w: SHORT_A, "B": lambda ids, w: None})
        res = sequential_ripup_planner(sc, router=router)
        self.assertEqual(res.ripups, 0)
        self.assertEqual(res.fail_class, {"B": "individual"})

    def test_independent_planner(self):
        sc = scenario([boundary_pipe("A", "100A", 20000), boundary_pipe("B", "50A", 20100)])
        router = MockRouter({"A": lambda ids, w: SHORT_A, "B": lambda ids, w: line(20100)})
        res = independent_planner(sc, router=router)
        self.assertTrue(all(r.status == "ok" for r in res.routes.values()))   # 서로를 보지 않는다
        self.assertTrue(all(a["n_obstacle_pipes"] == 0 for t in res.attempts.values() for a in t))


if __name__ == "__main__":
    unittest.main()
