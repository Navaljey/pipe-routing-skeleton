"""증분 검증 세션 (S1-6b, D85 · D87 ⑤ 모듈별 영향 범위) — V3 보상 계산용 공개 API (`docs/verifier_api.md`).

배관 하나를 추가·제거·변경하면 그 배관은 전 모듈을, 이웃 배관은 **영향받을 수 있는 모듈만** 다시 판정한다.
결과는 같은 배관 집합을 `verify(scenario, routes)` (routes = 시나리오 배관 순서) 로 전체 검증한 것과 비트 동일하다
(결과 불변, D53·D67 원칙).

이웃 Y 의 모듈별 재판정 조건 (X = 바뀐 배관의 이전·새 중심선, 거리 = 조각 최소거리 − slack):
  collision·bend  — 거리 ≤ r_X + r_Y + 1 mm. 그보다 먼 조각 쌍은 위반이 될 수 없고, 보고하는 최소 위반 쌍은
                    다른 배관별로 따로 고르므로 X 가 끼어들 수 없다 (modules._pipe_hits).
  valve           — Y 에 밸브가 있고 거리 ≤ r_X + r_Y + 1,250 mm (밸브 전면 상자 도달 상한).
  support         — X 가 **Y 의 서포트 배치에서 다른 배관과 판정한 부재 영역**(AABB, support_std.evaluate trace)에
                    r_X + slack + 반폭 + 1 mm 안으로 닿을 때. 닿지 않으면 판정한 후보마다 막힘 결과가 같으므로
                    탐욕 배치가 같은 후보를 같은 순서로 보고 같은 결과를 낸다.
  boundary·gravity_slope·branch — 자기 배관만 본다.
후보 거르기: Y 와 X 의 거리 ≤ r_X + r_Y + REACH (서포트 부재 최대 도달, `추정:` 보수적) 인 Y 만 위 조건을 본다.
D 묶음(D80 ⑥, 좌표 순서 탐욕 — 연쇄 영향 가능)은 매 호출 전체를 다시 한다 (서포트 목록만 다루므로 비용 작음).
"""
import copy
import time

import numpy as np

from ..constants import PIPE_SPECS, VALVE_FRONT_MM
from ..scenario import Scenario
from .core import MODULE_ORDER, MODULES, Context, PipeRoute, centerlines_of
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


VALVE_REACH_MM = 1250.0     # 밸브 중심 → 전면 상자 먼 모서리 상한 (r + 1,000 깊이 · ±500 폭·높이, r 은 따로 더함)


class Session:
    """verifier.Session(scenario) — add / remove / replace(pipe_id, waypoints) → 즉시 결과.

    반환(각 호출·result()) = {"pipes": {pipe_id: {"success", "layer0", "J", "supports"(kg_share 포함), ...}},
                             "J_total": Layer 0 통과 배관 J 합 (D49), "recomputed": 다시 판정한 배관 id,
                             "recomputed_modules": {pipe_id: [모듈]}, "sec"}
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
        self.mod: dict = {}        # pipe_id → {모듈: 결과 (None | [Violation])}
        self.sup: dict = {}        # pipe_id → (supports, gaps, 판정 부재 영역 [(lo, hi, 반폭)])
        self.reach = _reach(self.std)
        self._final = None
        self._last = {}

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

    def result(self, recomputed=None, sec=0.0) -> dict:
        from ..pipeline import j_breakdown
        recomputed = recomputed or {}
        out, jt = {}, 0.0
        for pid, rep in self.reports().items():
            J = j_breakdown(self.pipes[pid], self.routes[pid].waypoints, rep.fittings, rep.supports) \
                if rep.routed else None
            out[pid] = {"routed": rep.routed, "success": rep.success, "layer0": dict(rep.layer0), "J": J,
                        "supports": rep.supports, "violations": len(rep.violations)}
            if rep.success:
                jt += J["total"]
        return {"pipes": out, "J_total": round(jt, 4), "recomputed": sorted(recomputed),
                "recomputed_modules": {k: list(v) for k, v in sorted(recomputed.items())}, "sec": sec}

    # ---------------------------------------------------------------- 내부
    def _ctx(self) -> Context:
        rmap = {pid: self.routes[pid] for pid in self.order if pid in self.routes}
        cls = {pid: self.cls[pid] for pid in rmap}
        return Context(self.scenario, self.pipes, rmap, cls, std=self.std)

    def _gaps_to(self, pid: str, segs_list) -> dict:
        """후보 이웃 q → X(이전·새) 와의 최소 거리 (slack 뺀 값). REACH 밖은 빼고, 조각 쌍 AABB 로 먼저 거른다."""
        r0 = self.pipes[pid].radius
        out = {}
        for q, (B0, B1, BS) in self.segs.items():
            if q == pid:
                continue
            lim = r0 + self.pipes[q].radius + self.reach
            best = np.inf
            for (A0, A1, AS) in segs_list:
                lo, hi = np.minimum(A0, A1).min(0), np.maximum(A0, A1).max(0)
                blo, bhi = np.minimum(B0, B1).min(0), np.maximum(B0, B1).max(0)
                pad = lim + AS.max() + BS.max()
                if np.any(lo - pad > bhi) or np.any(blo - pad > hi):
                    continue
                d, _, _ = segment_segment_distance(A0[:, None], A1[:, None], B0[None], B1[None])
                best = min(best, float((d - AS[:, None] - BS[None]).min()))
            if best <= lim:
                out[q] = best
        return out

    def _touches_support(self, q: str, pid: str, segs_list) -> bool:
        region = self.sup[q][2] if q in self.sup else []
        if not region:
            return False
        r = self.pipes[pid].radius
        lo = np.array([x[0] for x in region]); hi = np.array([x[1] for x in region])
        half = np.array([x[2] for x in region])
        for (A0, A1, AS) in segs_list:
            alo, ahi = np.minimum(A0, A1), np.maximum(A0, A1)
            pad = (r + AS[:, None] + half[None] + 1.0)[..., None]
            sep = (alo[:, None] - pad > hi[None]) | (lo[None] - pad > ahi[:, None])
            if np.any(~sep.any(-1)):
                return True
        return False

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
            for d in (self.routes, self.cls, self.segs, self.mod, self.sup):
                d.pop(pid, None)
        rx = self.pipes[pid].radius
        redo = {}
        for q, g in (self._gaps_to(pid, affected_by).items() if affected_by else ()):
            ry = self.pipes[q].radius
            mods = []
            if g <= rx + ry + 1.0:
                mods += ["collision", "bend"]
            if self.pipes[q].valve_positions and g <= rx + ry + VALVE_REACH_MM:
                mods.append("valve")
            if self._touches_support(q, pid, affected_by):
                mods.append("support")
            if mods:
                redo[q] = [m for m in MODULE_ORDER if m in mods]
        if pid in self.routes:
            redo[pid] = list(MODULE_ORDER)
        ctx = self._ctx()
        for q, mods in redo.items():
            res = self.mod.setdefault(q, {})
            for name in mods:
                res[name] = MODULES[name](ctx, ctx.routes[q])
            if "support" in mods:
                self.sup[q] = (ctx.cache.get(("supports", q), []), ctx.cache.get(("support_gaps", q), []),
                               ctx.cache.get(("support_trace", q), []))
        self._final = self._group()
        return self.result(redo, time.perf_counter() - t0)

    def _group(self) -> dict:
        from .core import PipeReport, fittings_of
        from .support_group import group_d
        ctx = self._ctx()
        out = {}
        for pid in self.order:
            rep = PipeReport(pid, pid in self.routes)
            if rep.routed:
                for name in MODULE_ORDER:
                    res = self.mod[pid][name]
                    rep.layer0[name] = None if res is None else not res
                    rep.violations += res or []
                rep.fittings = fittings_of(self.cls[pid], self.pipes[pid])
                rep.supports = copy.deepcopy(self.sup[pid][0])
                rep.support_gaps = self.sup[pid][1]
            out[pid] = rep
        group_d(ctx, out)
        return out


__all__ = ["Session", "MODULE_ORDER"]
