"""증분 검증 세션 (S1-6b, D85) — V3 보상 계산용 공개 API (`docs/verifier_api.md`).

배관 하나를 추가·제거·변경하면 그 배관과 영향 범위 안의 이웃만 다시 판정한다. 결과는 같은 배관 집합을
`verify(scenario, routes)` (routes = 시나리오 배관 순서) 로 전체 검증한 것과 비트 동일하다 (결과 불변, D53·D67 원칙).

영향 범위 (배관 X 가 바뀔 때 다시 판정하는 이웃 Y):
  X 의 이전·새 중심선 조각과 Y 의 중심선 조각 사이 최소거리 ≤ r_X + r_Y + slack + REACH.
  다른 배관을 보는 판정은 모두 이 거리 안에서만 결과가 달라진다:
    collision·bend  — 직관·엘보 호 ↔ 다른 배관 r₁ + r₂
    valve           — 밸브 전면 1 m³ 상자 (밸브 중심에서 r + 1,000 · 옆 ±500)
    support         — 서포트 부재 (배관 중심에서 표 최대 L + 100 · 경사재 L/2 · 가로대 OD/2 + 100) + 앵글 반폭
  REACH = max(밸브 도달, 1.2 × (표 최대 L + 100) + 최대 OD/2 + 100 + 최대 앵글 반폭 + 100) — 보수적 상위집합.
  boundary·gravity_slope·branch 는 자기 배관만 본다.
D 묶음(D80 ⑥, 좌표 순서 탐욕 — 연쇄 영향 가능)은 매 호출 전체를 다시 한다 (서포트 목록만 다루므로 비용 작음).
"""
import copy
import time

import numpy as np

from ..constants import PIPE_SPECS, VALVE_FRONT_MM
from ..scenario import Scenario
from .core import MODULE_ORDER, Context, PipeRoute, centerlines_of, pipe_report
from .geom import segment_segment_distance


def _reach(std) -> float:
    from .. import support_std as ss
    lmax = max(float(r.bounds[-1]) for r in std.rows.values())
    od = max(s.od for s in PIPE_SPECS.values())
    sup = 1.2 * (lmax + 100) + od / 2 + ss.C_SIDE_MM + max(ss.ANGLE_HALF.values()) + 100
    valve = float(np.hypot(VALVE_FRONT_MM + 100, VALVE_FRONT_MM)) + 100
    return max(sup, valve)


def _segs(cls):
    A, B, S = [], [], []
    for cl in cls:
        a, b, s, _ = cl.arrays()
        A.append(a); B.append(b); S.append(s)
    return np.concatenate(A), np.concatenate(B), np.concatenate(S)


class Session:
    """verifier.Session(scenario) — add / remove / replace(pipe_id, waypoints) → 즉시 결과.

    반환(각 호출·result()) = {"pipes": {pipe_id: {"success", "layer0", "J", "supports"(kg_share 포함), ...}},
                             "J_total": Layer 0 통과 배관 J 합 (D49), "recomputed": 다시 판정한 배관 id, "sec"}
    reports() = verify() 와 같은 형식의 PipeReport 사전 (비트 동일 비교용).
    """

    def __init__(self, scenario: Scenario, std=None):
        from . import modules as _  # noqa: F401  (모듈 등록)
        from ..support_std import load_standard
        self.scenario = scenario
        self.std = std or load_standard()
        self.pipes = {p.id: p for p in scenario.pipes}
        self.order = [p.id for p in scenario.pipes]
        self.routes: dict = {}     # pipe_id → PipeRoute
        self.cls: dict = {}        # pipe_id → [Centerline]
        self.segs: dict = {}       # pipe_id → (A, B, slack)
        self.raw: dict = {}        # pipe_id → PipeReport (D 묶음 전)
        self.reach = _reach(self.std)
        self._final = None

    # ---------------------------------------------------------------- 공개 API
    def add(self, pipe_id: str, waypoints, branches=()) -> dict:
        if pipe_id in self.routes:
            raise KeyError(f"{pipe_id} 는 이미 놓여 있다 (replace 사용)")
        return self._change(pipe_id, waypoints, branches)

    def replace(self, pipe_id: str, waypoints, branches=()) -> dict:
        if pipe_id not in self.routes:
            raise KeyError(f"{pipe_id} 는 놓여 있지 않다 (add 사용)")
        return self._change(pipe_id, waypoints, branches)

    def remove(self, pipe_id: str) -> dict:
        if pipe_id not in self.routes:
            raise KeyError(f"{pipe_id} 는 놓여 있지 않다")
        return self._change(pipe_id, None, ())

    def reports(self) -> dict:
        """verify(scenario, routes)["pipes"] 와 같은 형식 (D 묶음 반영)."""
        if self._final is None:
            self._final = self._group()
        return self._final

    def result(self, recomputed=(), sec=0.0) -> dict:
        from ..pipeline import j_breakdown
        out, jt = {}, 0.0
        for pid, rep in self.reports().items():
            J = j_breakdown(self.pipes[pid], self.routes[pid].waypoints, rep.fittings, rep.supports) \
                if rep.routed else None
            out[pid] = {"routed": rep.routed, "success": rep.success, "layer0": dict(rep.layer0), "J": J,
                        "supports": rep.supports, "violations": len(rep.violations)}
            if rep.success:
                jt += J["total"]
        return {"pipes": out, "J_total": round(jt, 4), "recomputed": sorted(recomputed), "sec": sec}

    # ---------------------------------------------------------------- 내부
    def _ctx(self) -> Context:
        rmap = {pid: self.routes[pid] for pid in self.order if pid in self.routes}
        cls = {pid: self.cls[pid] for pid in rmap}
        return Context(self.scenario, self.pipes, rmap, cls, std=self.std)

    def _neighbors(self, pid: str, segs_list) -> set:
        r0 = self.pipes[pid].radius
        out = set()
        for q, (B0, B1, BS) in self.segs.items():
            if q == pid:
                continue
            r1 = self.pipes[q].radius
            for (A0, A1, AS) in segs_list:
                lim = r0 + r1 + self.reach
                lo, hi = np.minimum(A0, A1).min(0), np.maximum(A0, A1).max(0)
                blo, bhi = np.minimum(B0, B1).min(0), np.maximum(B0, B1).max(0)
                pad = lim + AS.max() + BS.max()
                if np.any(lo - pad > bhi) or np.any(blo - pad > hi):
                    continue
                d, _, _ = segment_segment_distance(A0[:, None], A1[:, None], B0[None], B1[None])
                if np.any(d - AS[:, None] - BS[None] <= lim):
                    out.add(q)
                    break
        return out

    def _change(self, pid: str, waypoints, branches) -> dict:
        t0 = time.perf_counter()
        affected_by = [self.segs[pid]] if pid in self.segs else []
        if waypoints:
            r = PipeRoute(pid, [list(map(float, w)) for w in waypoints], [list(b) for b in branches])
            self.routes[pid] = r
            self.cls[pid] = centerlines_of(self.pipes[pid], r)
            self.segs[pid] = _segs(self.cls[pid])
            affected_by.append(self.segs[pid])
        else:
            for d in (self.routes, self.cls, self.segs, self.raw):
                d.pop(pid, None)
        redo = self._neighbors(pid, affected_by) if affected_by else set()
        if pid in self.routes:
            redo.add(pid)
        ctx = self._ctx()
        for q in redo:
            self.raw[q] = pipe_report(ctx, self.pipes[q])
        self._final = self._group()
        return self.result(redo, time.perf_counter() - t0)

    def _group(self) -> dict:
        from .core import PipeReport
        from .support_group import group_d
        ctx = self._ctx()
        out = {}
        for pid in self.order:
            if pid in self.raw:
                rep = copy.copy(self.raw[pid])
                rep.supports = copy.deepcopy(self.raw[pid].supports)
                out[pid] = rep
            else:
                out[pid] = PipeReport(pid, False)
        group_d(ctx, out)
        return out


__all__ = ["Session", "MODULE_ORDER"]
