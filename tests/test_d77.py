"""D77 시험: 다중 배관 판정 규칙 ① 완전 성공 — 분모에서 D64 규칙상 불가능 배관만 뺀다 (단독 도달 불가는 분모에 남음)."""
import unittest

from pipe_routing.pipeline import full_success_summary


def route(pid, success, feasible=None):
    gf = None if feasible is None else {"feasible": feasible}
    return {"pipe_id": pid, "success": success, "gravity_feasibility": gf}


class FullSuccessTest(unittest.TestCase):
    def test_all_required_pass(self):
        g = full_success_summary([route("P1", True), route("P2", True, feasible=True), route("P3", False, feasible=False)])
        self.assertTrue(g["full_success"])                 # P3 는 규칙상 불가능 → 분모에서 제외
        self.assertEqual((g["required_pipes"], g["required_passed"]), (2, 2))
        self.assertEqual(g["excluded_infeasible"], ["P3"])

    def test_unreachable_stays_in_denominator(self):
        """경로가 없는(단독 도달 불가) 배관은 증명된 불가능이 아니므로 분모에 남는다 (D34)."""
        g = full_success_summary([route("P1", True), route("P2", False), route("P3", False, feasible=True)])
        self.assertFalse(g["full_success"])
        self.assertEqual((g["required_pipes"], g["required_passed"]), (3, 1))
        self.assertEqual(g["excluded_infeasible"], [])

    def test_no_required(self):
        g = full_success_summary([route("P1", False, feasible=False)])
        self.assertTrue(g["full_success"])
        self.assertEqual(g["required_pipes"], 0)


if __name__ == "__main__":
    unittest.main()
