"""D127 진단 재라우팅 상한 = min(4 × 라우팅 상한, max(라우팅 상한, 2²⁵))."""
import unittest
from types import SimpleNamespace

from pipe_routing.layered import (EXPAND_BUDGET, EXPAND_BUDGET_PAIR, diag_limit, make_layered_router)


class DiagLimitTest(unittest.TestCase):
    def test_formula(self):
        self.assertEqual(diag_limit(2 ** 22), 2 ** 24)    # 단일 블록 (E1 5 세트) — D74 와 같음
        self.assertEqual(diag_limit(2 ** 23), 2 ** 25)    # 블록 쌍 — D74 와 같음
        self.assertEqual(diag_limit(2 ** 25), 2 ** 25)    # 참조 품질선 — 이전 2²⁷
        self.assertEqual(diag_limit(2 ** 14), 2 ** 16)    # 작은 시험 예산 — × 4 그대로
        self.assertEqual(diag_limit(2 ** 24), 2 ** 25)

    def test_router_budgets(self):
        single, pair = SimpleNamespace(block_pair=None), SimpleNamespace(block_pair={"x": 1})
        r = make_layered_router()
        self.assertEqual(r.budget_for(single), {"kind": "expansions", "limit": EXPAND_BUDGET, "set": "single_block",
                                                "diagnosis_limit": 2 ** 24})
        self.assertEqual(r.budget_for(pair)["diagnosis_limit"], 2 ** 25)
        self.assertEqual(r.diagnostic.budget_for(single)["limit"], 2 ** 24)
        self.assertEqual(r.diagnostic.budget_for(pair)["limit"], 2 ** 25)
        self.assertEqual(r.diagnostic.without_reservations.budget_for(single)["limit"], 2 ** 24)   # D74 예약 충돌 판정도 진단 예산
        self.assertEqual(r.search_budget["diagnosis_limit"], 2 ** 24)
        ref = make_layered_router(expand_limit=2 ** 25)   # 참조 품질선 (D110 ③)
        self.assertEqual(ref.budget_for(single), {"kind": "expansions", "limit": 2 ** 25, "diagnosis_limit": 2 ** 25})
        self.assertEqual(ref.diagnostic.budget_for(single)["limit"], 2 ** 25)
        self.assertEqual(ref.diagnostic.without_reservations.budget_for(pair)["limit"], 2 ** 25)
        self.assertEqual(EXPAND_BUDGET_PAIR, 2 ** 23)


if __name__ == "__main__":
    unittest.main()
