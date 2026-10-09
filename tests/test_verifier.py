"""검증기 채점 정확성 — 수작업 정답/오답 시험 경로 (D27: 분기·밸브는 이 방식으로만 검증).

기본 배관: 100A (r = 117.15, 최소 직진 300, 엘보 R = 150), x=0 경계 → x=40000 경계, y=5000, z=1000.
"""
import copy
import unittest

from pipe_routing.scenario import from_dict
from pipe_routing.verifier import PipeRoute, verify

START, END = [0, 5000, 1000], [40000, 5000, 1000]
STRAIGHT = [START, END]


def pipe(pid="P", type_id="T1", size="100A", start=START, end=END, sdir=(1, 0, 0), edir=(1, 0, 0), **kw):
    gravity = type_id in ("T5", "T6", "T7", "T8")
    d = {"id": pid, "type_id": type_id, "nominal_size": size, "gravity_pipe": gravity,
         "min_slope": 0.01 if gravity else None, "insulation_thickness": 50,
         "start": {"pos": list(start), "kind": "boundary", "dir": list(sdir)},
         "end": {"pos": list(end), "kind": "boundary", "dir": list(edir)},
         "valve_positions": [], "branch_points": []}
    d.update(kw)
    return d


def scenario(pipes, obstacles=()):
    return from_dict({"block": {"width": 40000, "length": 40000, "height": 10000, "unit": "mm"},
                      "obstacles": [{"id": f"OBS_{i}", "type": "box", "min": list(lo), "max": list(hi)}
                                    for i, (lo, hi) in enumerate(obstacles)],
                      "pipes": copy.deepcopy(pipes)})


def run(pipes, routes, obstacles=(), pid="P"):
    sc = scenario(pipes, obstacles)
    rep = verify(sc, [PipeRoute(r[0], r[1], r[2] if len(r) > 2 else []) for r in routes])
    return rep["pipes"][pid]


class VerifierFixtureTest(unittest.TestCase):
    def assertLayer0(self, rep, **expect):
        for module, want in expect.items():
            self.assertEqual(rep.layer0[module], want, f"{module}: {[v.message for v in rep.violations]}")

    # -------- 정답 경로
    def test_straight_passes_all(self):
        rep = run([pipe()], [("P", STRAIGHT)])
        self.assertTrue(rep.success, [v.message for v in rep.violations])
        self.assertLayer0(rep, gravity_slope=None, valve=None, branch=None)
        self.assertEqual(len(rep.supports), 13)   # 40m / 3m 간격 → 내부 서포트 13개
        self.assertTrue(all(s["face"] == "floor" for s in rep.supports))
        self.assertAlmostEqual(rep.supports[0]["kg"], 4.43 * 1.1)   # 거리 1000 + 100

    # -------- collision
    def test_collision_obstacle(self):
        route = [START, [5000, 5000, 1000], [5000, 11000, 1000], [20000, 11000, 1000], [20000, 5000, 1000], END]
        rep = run([pipe()], [("P", route)], obstacles=[([10000, 10000, 0], [12000, 12000, 3000])])
        self.assertLayer0(rep, collision=False, bend=True)
        v = [v for v in rep.violations if v.module == "collision"][0]
        self.assertEqual(v.other, "OBS_0")
        self.assertTrue(10000 <= v.pos[0] <= 12000)   # 위반 위치 = 장애물 안 구간

    def test_collision_between_pipes(self):
        q = pipe("Q", start=[0, 5200, 1000], end=[40000, 5200, 1000])   # 중심 간 200 < 117.15 × 2
        rep = run([pipe(), q], [("P", STRAIGHT), ("Q", [[0, 5200, 1000], [40000, 5200, 1000]])])
        self.assertLayer0(rep, collision=False)
        self.assertEqual([v.other for v in rep.violations if v.module == "collision"], ["Q"])

    # -------- boundary
    def test_boundary_margin(self):
        route = [START, [5000, 5000, 1000], [5000, 5000, 100], [10000, 5000, 100], [10000, 5000, 1000], END]
        rep = run([pipe()], [("P", route)])
        self.assertLayer0(rep, boundary=False)

    def test_boundary_terminal_mismatch(self):
        rep = run([pipe()], [("P", [START, [39000, 5000, 1000]])])
        self.assertLayer0(rep, boundary=False)

    def test_boundary_direction_mismatch(self):
        """end 노즐 dir 은 +x 인데 경로는 +y 로 들어온다 (D29)."""
        p = pipe(end=[30000, 6000, 1000])
        p["end"]["kind"] = "nozzle"
        rep = run([p], [("P", [START, [30000, 5000, 1000], [30000, 6000, 1000]])])
        self.assertLayer0(rep, boundary=False)
        self.assertTrue(any("90.0° 어긋남" in v.message for v in rep.violations))

    # -------- bend
    def test_bend_angle_60(self):
        route = [START, [10000, 5000, 1000], [15000, 5000 + 8660, 1000], [20000, 5000 + 8660, 1000],
                 [20000, 5000, 1000], END]
        rep = run([pipe()], [("P", route)])
        self.assertLayer0(rep, bend=False)
        self.assertTrue(any("60.0°" in v.message for v in rep.violations))

    def test_bend_min_straight(self):
        route = [START, [10000, 5000, 1000], [10000, 5200, 1000], [20000, 5200, 1000], [20000, 5000, 1000], END]
        rep = run([pipe()], [("P", route)])
        self.assertLayer0(rep, bend=False)
        self.assertTrue(any("꺾임 사이 직관 200mm < 필요 300mm" in v.message for v in rep.violations))

    def test_bend_last_segment_exempt(self):
        """마지막 구간은 §3.3 최소 직진 면제, 엘보 접선(100A 90° = 150)만 있으면 된다 (D45 ③)."""
        p = pipe(end=[20000, 5200, 1000], edir=(0, 1, 0))
        p["end"]["kind"] = "nozzle"
        rep = run([p], [("P", [START, [20000, 5000, 1000], [20000, 5200, 1000]])])
        self.assertLayer0(rep, bend=True)

    def test_bend_elbow_overlap(self):
        """135° 엘보 둘(접선 각 R·tan 67.5° = 362mm) 사이 직관 566mm → 엘보가 들어가지 않는다."""
        p = pipe(end=[12000, 5400, 1000])
        p["end"]["kind"] = "nozzle"
        rep = run([p], [("P", [START, [10000, 5000, 1000], [9600, 5400, 1000], [12000, 5400, 1000]])])
        self.assertLayer0(rep, bend=False)
        self.assertTrue(any("꺾임 사이 직관 566mm < 필요 724mm" in v.message for v in rep.violations))

    # -------- D45 경계 사례
    def test_d45_65a_90_90(self):
        """65A 90°-90° 연속: 접선 합 195, §3.3 200 → 필요 200. 195 는 실패(옛 150 규칙이면 통과), 200 은 통과."""
        for gap, want in ((195, False), (200, True)):
            p = pipe(size="65A", end=[30000, 9000, 1000], edir=(0, 1, 0))
            p["end"]["kind"] = "nozzle"
            route = [START, [20000, 5000, 1000], [20000, 5000 + gap, 1000], [30000, 5000 + gap, 1000],
                     [30000, 9000, 1000]]
            rep = run([p], [("P", route)])
            self.assertLayer0(rep, bend=want)
            if not want:
                self.assertTrue(any("꺾임 사이 직관 195mm < 필요 200mm" in v.message for v in rep.violations))

    def test_d45_after_135(self):
        """100A 135° 직후 90°: 필요 max(300, 362.1 + 150) = 512.1. 사이 직관 438 실패, 523 통과."""
        for d, want in ((310, False), (370, True)):     # 대각 길이 d·√2
            a = [20000, 5000, 1000]                     # +x → (-1, 1): 135°
            b = [20000 - d, 5000 + d, 1000]             # (-1, 1) → (1, 1): 90°
            c = [20000 - d + 1000, 5000 + d + 1000, 1000]   # (1, 1) → +y: 45°
            e = [20000 - d + 1000, 9000, 1000]
            p = pipe(end=e, edir=(0, 1, 0))
            p["end"]["kind"] = "nozzle"
            rep = run([p], [("P", [START, a, b, c, e])])
            self.assertLayer0(rep, bend=want)

    def test_d45_turn_just_before_end(self):
        """마지막 꺾임 → end 단자 ≥ 엘보 접선 (100A 90° = 150). 100 실패, 150 통과 (노즐 종단 예외는 추가 직관만 면제)."""
        for tail, want in ((100, False), (150, True)):
            p = pipe(end=[20000, 5000 + tail, 1000], edir=(0, 1, 0))
            p["end"]["kind"] = "nozzle"
            rep = run([p], [("P", [START, [20000, 5000, 1000], [20000, 5000 + tail, 1000]])])
            self.assertLayer0(rep, bend=want)
            if not want:
                self.assertTrue(any("마지막 꺾임→end 직관 100mm < 필요 150mm" in v.message for v in rep.violations))

    def test_bend_elbow_body_collision(self):
        """직관은 이격을 지키지만 엘보 호가 안쪽 모서리 장애물에 닿는다 (엘보 점유 공간)."""
        route = [START, [20000, 5000, 1000], [20000, 20000, 1000], [30000, 20000, 1000], [30000, 5000, 1000], END]
        obs = [([15000, 5120, 0], [19880, 9000, 2000])]
        rep = run([pipe()], [("P", route)], obstacles=obs)
        self.assertLayer0(rep, collision=True, bend=False)
        self.assertTrue(any("엘보 ↔ 장애물" in v.message for v in rep.violations))

    # -------- gravity_slope
    def test_gravity_pass_and_fail(self):
        p = pipe(type_id="T5", start=[0, 5000, 2000], end=[40000, 5000, 1600])
        ok = run([p], [("P", [[0, 5000, 2000], [40000, 5000, 1600]])])
        self.assertLayer0(ok, gravity_slope=True, boundary=True)
        flat = run([p], [("P", [[0, 5000, 2000], [20000, 5000, 2000], [20000, 5000, 1600], [40000, 5000, 1600]])])
        self.assertLayer0(flat, gravity_slope=False)
        up = run([p], [("P", [[0, 5000, 2000], [10000, 5000, 1900], [10000, 5000, 2500], [20000, 5000, 2400],
                              [20000, 5000, 1700], [40000, 5000, 1500], [40000, 5000, 1600]])])
        self.assertLayer0(up, gravity_slope=False)

    def test_gravity_not_applicable_to_pressure(self):
        self.assertLayer0(run([pipe()], [("P", STRAIGHT)]), gravity_slope=None)

    # -------- valve (D27: 시험 경로로만)
    def test_valve_pass(self):
        rep = run([pipe(type_id="T2", valve_positions=[[20000, 5000, 1000]])], [("P", STRAIGHT)])
        self.assertLayer0(rep, valve=True)
        self.assertIn({"type": "gate_valve", "pos": [20000.0, 5000.0, 1000.0]}, rep.fittings)

    def test_valve_height(self):
        p = pipe(type_id="T2", start=[0, 5000, 2000], end=[40000, 5000, 2000], valve_positions=[[20000, 5000, 2000]])
        rep = run([p], [("P", [[0, 5000, 2000], [40000, 5000, 2000]])])
        self.assertLayer0(rep, valve=False)
        self.assertTrue(any("밸브 높이 2000" in v.message for v in rep.violations))

    def test_valve_front_blocked_both_sides(self):
        """배관 축 x → 전면 후보 ±y. 양쪽을 막으면 실패, 한쪽만 막으면 통과 (D43)."""
        p = pipe(type_id="T2", valve_positions=[[20000, 5000, 1000]])
        side_a = ([19500, 5200, 0], [20500, 6500, 2000])
        side_b = ([19500, 3500, 0], [20500, 4800, 2000])
        self.assertLayer0(run([p], [("P", STRAIGHT)], obstacles=[side_a]), valve=True)
        rep = run([p], [("P", STRAIGHT)], obstacles=[side_a, side_b])
        self.assertLayer0(rep, valve=False)
        self.assertTrue(any("전면" in v.message for v in rep.violations))

    def test_valve_front_blocked_by_other_pipe(self):
        p = pipe(type_id="T2", valve_positions=[[20000, 5000, 1000]])
        q1 = pipe("Q1", start=[20300, 0, 1000], end=[20300, 40000, 1000], sdir=(0, 1, 0), edir=(0, 1, 0))
        q2 = pipe("Q2", start=[19700, 0, 1000], end=[19700, 40000, 1000], sdir=(0, 1, 0), edir=(0, 1, 0))
        rep = run([p, q1, q2], [("P", STRAIGHT), ("Q1", [[20300, 0, 1000], [20300, 40000, 1000]]),
                               ("Q2", [[19700, 0, 1000], [19700, 40000, 1000]])])
        self.assertLayer0(rep, valve=False)

    def test_valve_not_on_route(self):
        rep = run([pipe(type_id="T2", valve_positions=[[20000, 5500, 1000]])], [("P", STRAIGHT)])
        self.assertLayer0(rep, valve=False)

    # -------- branch (D27: 시험 경로로만)
    def test_branch_90_and_45_pass(self):
        bp = [20000, 5000, 1000]
        for leg in ([bp, [20000, 9000, 1000]], [bp, [24000, 9000, 1000]]):
            rep = run([pipe(type_id="T3", branch_points=[bp])], [("P", STRAIGHT, [leg])])
            self.assertLayer0(rep, branch=True)
            self.assertIn({"type": "tee", "pos": [20000.0, 5000.0, 1000.0]}, rep.fittings)

    def test_branch_angle_fail(self):
        bp = [20000, 5000, 1000]
        rep = run([pipe(type_id="T3", branch_points=[bp])], [("P", STRAIGHT, [[bp, [22310, 9000, 1000]]])])   # 60°
        self.assertLayer0(rep, branch=False)
        backward = run([pipe(type_id="T3", branch_points=[bp])], [("P", STRAIGHT, [[bp, [16000, 9000, 1000]]])])  # 135°
        self.assertLayer0(backward, branch=False)

    def test_branch_point_off_main(self):
        bp = [20000, 5500, 1000]
        rep = run([pipe(type_id="T3", branch_points=[bp])], [("P", STRAIGHT, [[bp, [20000, 9000, 1000]]])])
        self.assertLayer0(rep, branch=False)

    def test_branch_point_on_elbow(self):
        """주관 꺾임점(엘보 위)에서 분기하면 직관 위가 아니다."""
        main = [START, [20000, 5000, 1000], [20000, 9000, 1000], [30000, 9000, 1000], [30000, 5000, 1000], END]
        bp = [20000, 5000, 1000]
        rep = run([pipe(type_id="T3", branch_points=[bp])], [("P", main, [[bp, [20000, 1000, 1000]]])])
        self.assertLayer0(rep, branch=False)

    def test_branch_missing_leg(self):
        rep = run([pipe(type_id="T3", branch_points=[[20000, 5000, 1000]])], [("P", STRAIGHT)])
        self.assertLayer0(rep, branch=False)

    # -------- support
    def test_support_falls_back_to_next_face(self):
        """D46: 가장 가까운 면(바닥) 지지선이 장애물에 막히면 다음으로 가까운 면(y=0 격벽, 5000mm)을 쓴다."""
        rep = run([pipe()], [("P", STRAIGHT)], obstacles=[([14000, 4000, 0], [26000, 6000, 500])])
        self.assertLayer0(rep, support=True)
        over = [s for s in rep.supports if 14000 < s["pos"][0] < 26000]
        self.assertTrue(over and all(s["face"] == "wall_y0" for s in over))
        self.assertAlmostEqual(over[0]["kg"], round(4.43 * 5.1, 4))   # 실제 선택된 면까지 거리 + 100

    def test_support_over_equipment_hangs_from_ceiling(self):
        """D46: 장비 위 배관 — 바닥 지지선은 장비에 막히고, 그다음 가까운 천장(5500mm)에 매단다."""
        p = pipe(start=[0, 20000, 4500], end=[40000, 20000, 4500])
        rep = run([p], [("P", [[0, 20000, 4500], [40000, 20000, 4500]])],
                  obstacles=[([10000, 15000, 0], [30000, 25000, 4000])])
        self.assertLayer0(rep, support=True)
        over = [s for s in rep.supports if 10000 < s["pos"][0] < 30000]
        self.assertTrue(over and all(s["face"] == "ceiling" for s in over))
        self.assertAlmostEqual(over[0]["angle_length_mm"], 5600)
        self.assertAlmostEqual(over[0]["kg"], round(4.43 * 5.6, 4))

    def test_support_all_faces_blocked(self):
        """바닥·천장·양쪽 격벽 지지선이 모두 막힌 12m 터널 → 설치 위치 없음 (x 격벽은 배관 축과 평행이라 후보 아님)."""
        tunnel = [([14000, 4000, 0], [26000, 6000, 500]), ([14000, 4000, 1500], [26000, 6000, 10000]),
                  ([14000, 0, 0], [26000, 4800, 10000]), ([14000, 5200, 0], [26000, 40000, 10000])]
        rep = run([pipe()], [("P", STRAIGHT)], obstacles=tunnel)
        self.assertLayer0(rep, support=False)
        bad = [v for v in rep.violations if v.module == "support"]
        self.assertTrue(all(14000 <= v.pos[0] <= 26000 for v in bad))

    def test_support_vertical_spacing(self):
        """수직 구간은 수직 간격(100A 4m)을 쓴다."""
        p = pipe(start=[20000, 5000, 0], end=[20000, 5000, 10000], sdir=(0, 0, 1), edir=(0, 0, 1))
        rep = run([p], [("P", [[20000, 5000, 0], [20000, 5000, 10000]])])
        self.assertLayer0(rep, support=True)
        self.assertEqual(len(rep.supports), 2)   # 10m / 4m

    # -------- 경로 없음
    def test_unrouted_pipe(self):
        rep = run([pipe()], [("P", [])])
        self.assertFalse(rep.routed)
        self.assertFalse(rep.success)
        self.assertEqual(rep.layer0, {})


if __name__ == "__main__":
    unittest.main()
