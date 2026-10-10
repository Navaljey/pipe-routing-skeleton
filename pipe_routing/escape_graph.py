"""S0 표현: Escape Graph (CLAUDE.md §4.2, D10, D15, D37~D39).

배관 1개마다 만든다 (유효 반경이 구경마다 다르므로, D11 "배관마다 그래프 재생성").
  1) 장애물을 유효 반경만큼 팽창한 면 좌표(10mm 바깥쪽 스냅) + 단자 좌표 + 영역 경계 → 축별 좌표
  2) 좌표 곱 = 격자점. 노드 = 장애물과 유효 반경 이상 떨어진 영역 내 격자점 (정확 판정)
  3) 엣지 = 인접 격자점 사이 축 방향 구간 + 수평면·수직면 45° 구간 (D15, D38)
그래프는 numpy 마스크로 암묵 표현하고, neighbors() 가 상태 (노드, 진입방향, 직진길이) 를 펼친다.
"""
import argparse
import math
import time
from typing import Iterator, Optional

import numpy as np

from .constants import ANGLE_TOL_DEG, PIPE_SPECS, SNAP_MM, elbow_kg, elbow_radius, elbow_tangent
from .geometry import Vec3
from .scenario import Pipe, Scenario, boundary_face, load
from .verifier.geom import build_centerline, elbow_arc_chords, segment_box_distance, segment_segment_distance
from .space import ALLOWED_DEFLECTIONS, AXIS_DIRS, DEFLECTION, DIR_INDEX, DIRS, State

EPS = 1e-6


def _snap_down(v: float) -> float:
    return math.floor(v / SNAP_MM + 1e-9) * SNAP_MM


def _snap_up(v: float) -> float:
    return math.ceil(v / SNAP_MM - 1e-9) * SNAP_MM


def _gap(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """1D 점–구간 거리."""
    return np.maximum(0.0, np.maximum(lo - v, v - hi))


def planar_segment_box_distance(p0: np.ndarray, p1: np.ndarray, lo, hi, const_axis: int) -> np.ndarray:
    """좌표평면 위 선분(N개) ↔ 박스 하나 사이 유클리드 최소거리. 정확 계산.

    선분은 const_axis 좌표가 일정하다 (축 방향 또는 D15 의 평면 내 45°).
    평면 거리 = 교차하면 0, 아니면 min(끝점↔사각형, 사각형 꼭짓점↔선분) — 볼록 도형 사이 최소거리의 정확한 후보 집합.
    """
    a, b = [ax for ax in range(3) if ax != const_axis]
    d3 = _gap(p0[:, const_axis], lo[const_axis], hi[const_axis])
    p = p0[:, [a, b]]
    q = p1[:, [a, b]]
    rl = np.array([lo[a], lo[b]], dtype=float)
    rh = np.array([hi[a], hi[b]], dtype=float)
    d = q - p
    # Liang–Barsky 로 닫힌 사각형과 교차 판정
    tmin = np.zeros(len(p))
    tmax = np.ones(len(p))
    for ax in range(2):
        flat = d[:, ax] == 0
        inside = (p[:, ax] >= rl[ax]) & (p[:, ax] <= rh[ax])
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (rl[ax] - p[:, ax]) / d[:, ax]
            t2 = (rh[ax] - p[:, ax]) / d[:, ax]
        tmin = np.where(flat, np.where(inside, tmin, np.inf), np.maximum(tmin, np.minimum(t1, t2)))
        tmax = np.where(flat, tmax, np.minimum(tmax, np.maximum(t1, t2)))
    hit = tmin <= tmax

    def point_rect(pt):
        return np.hypot(_gap(pt[:, 0], rl[0], rh[0]), _gap(pt[:, 1], rl[1], rh[1]))

    best = np.minimum(point_rect(p), point_rect(q))
    dd = np.einsum("ij,ij->i", d, d)
    for cx in (rl[0], rh[0]):
        for cy in (rl[1], rh[1]):
            c = np.array([cx, cy])
            t = np.clip(((c - p) * d).sum(1) / np.where(dd > 0, dd, 1), 0, 1)
            proj = p + t[:, None] * d
            best = np.minimum(best, np.hypot(proj[:, 0] - cx, proj[:, 1] - cy))
    d2 = np.where(hit, 0.0, best)
    return np.hypot(d2, d3)


class EscapeGraph:
    """SpaceRepresentation 구현 (§4.1). 노드 키 = 격자 인덱스 (i, j, k)."""

    def __init__(self, scenario: Scenario, pipe: Pipe, allow_45: bool = True, others=(), fast_build: bool = True,
                 extra_terminals=(), pipe_grid_lines: bool = True):
        """allow_45=False 는 45° 엣지를 만들지 않는다 — M9 비교 실험 전용 (D15 기본은 True).

        others: 이미 놓인 배관 [(Pipe, waypoints), ...] — 직관 + 엘보 호를 장애물로 본다. 이격 r + r_other (D50②).
        기하는 검증기와 같은 함수(build_centerline, segment_segment_distance)를 쓴다.
        fast_build=False 는 가속(M10) 전 구성 방식 — 그래프 동일성 시험용. 결과는 같다.
        extra_terminals / pipe_grid_lines=False: M18 대안 표현 실험(layered.py) 전용 — 같은 구경 배관들의 단자를 한 그래프에
        넣어 고정 레이어로 공유하고, 놓인 배관이 격자선을 만들지 않게 한다. 기본값은 기존 동작 그대로.
        """
        t0 = time.perf_counter()
        self.fast_build = fast_build
        self.pipe = pipe
        self.r = pipe.radius
        self.L = pipe.min_straight
        # D45: 편향각별 엘보 접선 길이, 직진 길이 상한(이후 어떤 꺾임·도착 판정에도 충분한 값)
        self.tangent = {a: elbow_tangent(pipe.nominal_size, a) for a in (0, 45, 90, 135)}
        # D47: 경계 단자 쪽 첫·끝 직관 ≥ r + t (엘보가 면의 유효 반경 띠 밖에 놓이도록)
        self.start_boundary = pipe.start.kind == "boundary"
        self.end_boundary = pipe.end.kind == "boundary"
        t135 = self.tangent[135]
        self.run_cap = {a: max(self.L, self.tangent[a] + t135,
                               self.r + self.tangent[a] if self.end_boundary else 0.0,
                               self.r + t135 if (a == 0 and self.start_boundary) else 0.0)
                        for a in self.tangent}
        self._arc_ok: dict[tuple, bool] = {}
        self._arc_tpl: dict[tuple, tuple] = {}   # D48 캐시: (노드, 들어오는 방향, 나가는 방향) → 엘보 호 이격 OK
        self.kg_per_m = PIPE_SPECS[pipe.nominal_size].kg_per_m
        ext = scenario.block.extent
        m = _snap_up(self.r)
        # 영역: 블록 안쪽 유효 반경 (10mm 올림). 경계 단자만 예외 (D39)
        self.dom_lo = np.array([m, m, m], dtype=float)
        self.dom_hi = np.array([e - m for e in ext], dtype=float)
        self.box_lo = np.array([o.box.min for o in scenario.obstacles], dtype=float).reshape(-1, 3)
        self.box_hi = np.array([o.box.max for o in scenario.obstacles], dtype=float).reshape(-1, 3)
        self.pipe_groups = self._pipe_groups(others)   # D50②: [(A, B, slack, r_other, lo, hi)]

        # 1) 축별 좌표
        coords = [set() for _ in range(3)]
        for ax in range(3):
            coords[ax].update((self.dom_lo[ax], self.dom_hi[ax]))
            for lo, hi in zip(self.box_lo[:, ax], self.box_hi[:, ax]):
                for v in (_snap_down(lo - self.r), _snap_up(hi + self.r)):
                    if self.dom_lo[ax] <= v <= self.dom_hi[ax]:
                        coords[ax].add(v)
            for t in (pipe.start, pipe.end, *extra_terminals):
                coords[ax].add(float(t.pos[ax]))
            for lo, hi, ro in (self.pipe_segments if pipe_grid_lines else ()):   # 놓인 배관의 꺾임점 사이 구간 AABB 를 r + r_other 만큼 팽창한 면 (D38 과 같은 방식)
                for v in (_snap_down(lo[ax] - self.r - ro), _snap_up(hi[ax] + self.r + ro)):
                    if self.dom_lo[ax] <= v <= self.dom_hi[ax]:
                        coords[ax].add(v)
        self.axes = [np.array(sorted(c)) for c in coords]
        self.shape = tuple(len(a) for a in self.axes)
        self._axes_f = [[float(v) for v in a] for a in self.axes]   # position() 용 (numpy 스칼라 변환 비용 회피)
        self.index = [{v: i for i, v in enumerate(a)} for a in self.axes]

        self.start_node = self.node_of(pipe.start.pos)
        self.end_node = self.node_of(pipe.end.pos)
        self.terminal_nodes = {self.start_node, self.end_node} | {self.node_of(t.pos) for t in extra_terminals}

        # 2) 노드
        X, Y, Z = np.meshgrid(*self.axes, indexing="ij")
        pts = np.stack([X, Y, Z], -1).reshape(-1, 3)
        self.in_domain = np.all((pts >= self.dom_lo - EPS) & (pts <= self.dom_hi + EPS), 1).reshape(self.shape)
        self.clearance = self._grid_clearance(pts).reshape(self.shape)   # 격자점 ↔ 장애물 최소거리 (D48 거르기용)
        clear = self.clearance >= self.r - EPS
        self.node_ok = self.in_domain & clear
        for n in self.terminal_nodes:   # 경계 단자는 영역 밖이지만 노드로 둔다 (D39)
            self.node_ok[n] = clear[n]
        # 놓인 배관까지 여유 (거리 − slack − r_other). 엘보 빠른 통과 판정용으로 r + t_135 범위까지 계산
        self.pipe_margin = np.full(self.shape, np.inf)
        if self.pipe_groups:
            self._pipe_node_margin()
            self.node_ok &= self.pipe_margin >= self.r - EPS

        # 3) 축 방향 엣지: axis_ok[ax][idx] = idx → idx+e_ax 엣지 사용 가능
        self.axis_ok = [self._axis_edges(ax) for ax in range(3)]
        if self.pipe_groups:
            for ax in range(3):
                self._pipe_block_axis(ax)
        # 4) 45° 엣지 (D38): diag_ok[d][idx], diag_to[d] = (평면 첫 축 목표 인덱스, 둘째 축 목표 인덱스) 표
        self.diag_ok: dict[int, np.ndarray] = {}
        self.diag_to: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for d in range(6, len(DIRS)):
            if not allow_45:
                self.diag_ok[d] = np.zeros(self.shape, dtype=bool)
            elif d not in self.diag_ok:
                self._diag_edges(d)
        if self.pipe_groups and allow_45:
            for d in range(6, len(DIRS)):
                self._pipe_block_diag(d)
        self.build_sec = time.perf_counter() - t0

    # ---------------------------------------------------------------- 구성

    def node_of(self, pos: Vec3) -> tuple[int, int, int]:
        return tuple(self.index[ax][float(pos[ax])] for ax in range(3))

    def position(self, node) -> Vec3:
        X, Y, Z = self._axes_f
        return (X[node[0]], Y[node[1]], Z[node[2]])

    def _grid_clearance(self, pts: np.ndarray) -> np.ndarray:
        """격자점(pts = 격자 순서) ↔ 장애물 최소거리. fast_build 면 박스에서 r + t_135 + 10mm 보다 먼 격자점은 그 박스를
        계산하지 않는다 (M10). 이 값은 판정(≥ r, ≥ r + t)에만 쓰이고 판정 문턱은 모두 그 거리 이하라 판정 결과는 같다 —
        먼 점의 값만 실제 거리 대신 더 큰 값(다른 박스까지 거리 또는 inf)이 된다."""
        if not self.fast_build:
            return self._point_clearance(pts)
        best = np.full(self.shape, np.inf)
        P = pts.reshape(self.shape + (3,))
        reach = self.r + self.tangent[135] + 10.0
        for lo, hi in zip(self.box_lo, self.box_hi):
            sl = tuple(self._span(k, lo[k], hi[k], reach, reach) for k in range(3))
            g = np.maximum(0.0, np.maximum(lo - P[sl], P[sl] - hi))
            best[sl] = np.minimum(best[sl], np.sqrt((g * g).sum(-1)))
        return best.reshape(-1)

    def _point_clearance(self, pts: np.ndarray) -> np.ndarray:
        best = np.full(len(pts), np.inf)
        for lo, hi in zip(self.box_lo, self.box_hi):
            g = np.maximum(0.0, np.maximum(lo - pts, pts - hi))
            best = np.minimum(best, np.sqrt((g * g).sum(1)))
        return best

    # ---------------------------------------------------------------- 놓인 배관 (D50②)

    def _pipe_groups(self, others) -> list:
        """놓인 배관의 실제 중심선(검증기와 같은 함수) → 조각 묶음. 직관 조각 하나, 엘보 호 하나가 각각 한 묶음."""
        groups = []
        self.pipe_segments = []   # 격자선용: 꺾임점 사이 구간 AABB (엘보 호는 격자선을 만들지 않고 판정에만 쓴다)
        for op, wps in others:
            if not wps:
                continue
            W = np.asarray(wps, dtype=float)
            for a, b in zip(W[:-1], W[1:]):
                self.pipe_segments.append((np.minimum(a, b), np.maximum(a, b), op.radius))
            cl = build_centerline(wps, elbow_radius(op.nominal_size), ANGLE_TOL_DEG)
            by = {}
            for piece in cl.pieces:
                by.setdefault((piece.kind, piece.ref), []).append(piece)
            for pieces in by.values():
                A = np.array([q.a for q in pieces])
                B = np.array([q.b for q in pieces])
                S = np.array([q.slack for q in pieces])
                groups.append((A, B, S, op.radius, np.minimum(A, B).min(0), np.maximum(A, B).max(0)))
        if groups:
            self._pg_A = np.concatenate([g[0] for g in groups])
            self._pg_B = np.concatenate([g[1] for g in groups])
            self._pg_S = np.concatenate([g[2] for g in groups])
            self._pg_R = np.concatenate([np.full(len(g[0]), g[3]) for g in groups])
            self._pg_lo = np.minimum(self._pg_A, self._pg_B)
            self._pg_hi = np.maximum(self._pg_A, self._pg_B)
        return groups

    def _index_range(self, ax: int, lo: float, hi: float) -> tuple[int, int]:
        """[lo, hi] 와 겹치는 격자 인덱스 구간 (양쪽 한 칸 여유 — 구간을 걸치는 엣지 포함)."""
        a = self.axes[ax]
        i0 = max(0, int(np.searchsorted(a, lo, "right")) - 1)
        i1 = min(len(a), int(np.searchsorted(a, hi, "left")) + 1)
        return i0, i1

    def _sub(self, lo, hi):
        return tuple(slice(*self._index_range(ax, lo[ax], hi[ax])) for ax in range(3))

    def _pipe_node_margin(self) -> None:
        reach = self.r + self.tangent[135]
        for A, B, S, ro, lo, hi in self.pipe_groups:
            pad = reach + ro + S.max()
            sl = self._sub(lo - pad, hi + pad)
            grids = np.meshgrid(*[self.axes[ax][sl[ax]] for ax in range(3)], indexing="ij")
            P = np.stack(grids, -1).reshape(-1, 3)
            if not len(P):
                continue
            D = B - A
            dd = np.where((D * D).sum(1) > 0, (D * D).sum(1), 1.0)
            t = np.clip(((P[:, None, :] - A[None]) * D[None]).sum(-1) / dd[None], 0, 1)
            dist = np.linalg.norm(P[:, None, :] - (A[None] + t[..., None] * D[None]), axis=-1)
            m = (dist - S[None]).min(1) - ro
            sub = self.pipe_margin[sl]
            self.pipe_margin[sl] = np.minimum(sub, m.reshape(sub.shape))

    def _segments_clear_of_pipes(self, P0: np.ndarray, P1: np.ndarray, slack: float = 0.0) -> np.ndarray:
        """선분들이 놓인 배관 조각과 r + r_other 이상 떨어져 있는가 (검증기 _pipe_hits 와 같은 판정)."""
        free = np.ones(len(P0), dtype=bool)
        if not len(P0) or not self.pipe_groups:
            return free
        slo, shi = np.minimum(P0, P1), np.maximum(P0, P1)
        pad = self.r + slack
        for A, B, S, ro, lo, hi in self.pipe_groups:
            g = pad + ro + S.max()
            cand = np.nonzero(free & np.all((slo < hi + g) & (shi > lo - g), 1))[0]
            self._block_cand(free, cand, P0, P1, A, B, S, ro, slack)
        return free

    def _block_cand(self, free, cand, P0, P1, A, B, S, ro, slack=0.0) -> None:
        for k0 in range(0, len(cand), 4096):
            c = cand[k0:k0 + 4096]
            d, _, _ = segment_segment_distance(P0[c][:, None], P1[c][:, None], A[None], B[None])
            bad = np.any(d - slack - S[None] < self.r + ro - EPS, 1)
            free[c[bad]] = False

    def _span(self, ax: int, lo: float, hi: float, ext_lo: float = 0.0, ext_hi: float = 0.0) -> slice:
        """시작 인덱스 범위 (상위집합): 시작 좌표가 (lo − ext_lo, hi + ext_hi) 와 걸칠 수 있는 격자 인덱스. 한 칸씩 여유.

        M10 가속: 조각마다 전체 엣지를 훑지 않고 이 범위만 본다. 범위 안에서는 원래와 같은 AABB·거리 판정을 하므로
        결과(막히는 엣지 집합)는 같다 (tests/test_astar_fast.py 그래프 동일성 시험).
        """
        a = self.axes[ax]
        i0 = max(0, int(np.searchsorted(a, lo - ext_lo, "left")) - 2)
        i1 = min(len(a), int(np.searchsorted(a, hi + ext_hi, "right")) + 1)
        return slice(i0, i1)

    def _pipe_block_axis(self, ax: int) -> None:
        ok = self.axis_ok[ax]
        if not self.fast_build:
            idx = np.nonzero(ok)
            if not len(idx[0]):
                return
            P0 = np.stack([self.axes[k][idx[k]] for k in range(3)], -1)
            nxt = list(idx)
            nxt[ax] = idx[ax] + 1
            P1 = np.stack([self.axes[k][nxt[k]] for k in range(3)], -1)
            free = self._segments_clear_of_pipes(P0, P1)
            ok[tuple(i[~free] for i in idx)] = False
            return
        for A, B, S, ro, lo, hi in self.pipe_groups:
            g = self.r + 0.0 + ro + S.max()
            sl = tuple(self._span(k, lo[k], hi[k], g, g) for k in range(3))
            sub = np.nonzero(ok[sl])
            if not len(sub[0]):
                continue
            idx = tuple(sub[k] + sl[k].start for k in range(3))
            P0 = np.stack([self.axes[k][idx[k]] for k in range(3)], -1)
            nxt = list(idx)
            nxt[ax] = idx[ax] + 1
            P1 = np.stack([self.axes[k][nxt[k]] for k in range(3)], -1)
            free = np.ones(len(P0), dtype=bool)
            slo, shi = np.minimum(P0, P1), np.maximum(P0, P1)
            cand = np.nonzero(np.all((slo < hi + g) & (shi > lo - g), 1))[0]
            self._block_cand(free, cand, P0, P1, A, B, S, ro)
            ok[tuple(i[~free] for i in idx)] = False

    def _pipe_block_diag(self, d: int) -> None:
        ok = self.diag_ok[d]
        vec = DIRS[d]
        a, b = [ax for ax in range(3) if vec[ax] != 0]
        to_a, to_b = self.diag_to[d]

        def edges(idx):
            dst = list(idx)
            dst[a] = to_a[idx[a], idx[b]]
            dst[b] = to_b[idx[a], idx[b]]
            P0 = np.stack([self.axes[k][idx[k]] for k in range(3)], -1)
            P1 = np.stack([self.axes[k][dst[k]] for k in range(3)], -1)
            return P0, P1

        if not self.fast_build:
            idx = np.nonzero(ok)
            if not len(idx[0]):
                return
            P0, P1 = edges(idx)
            free = self._segments_clear_of_pipes(P0, P1)
            ok[tuple(i[~free] for i in idx)] = False
            return
        # 이 방향 엣지의 최대 이동 길이 (평면 두 축 같음) — 시작 인덱스 범위를 넓히는 데 쓴다
        ia, ib = np.nonzero(to_a >= 0)
        tmax = float(np.abs(self.axes[a][to_a[ia, ib]] - self.axes[a][ia]).max()) if len(ia) else 0.0
        for A, B, S, ro, lo, hi in self.pipe_groups:
            g = self.r + 0.0 + ro + S.max()
            sl = []
            for k in range(3):
                if vec[k] > 0:
                    sl.append(self._span(k, lo[k], hi[k], g + tmax, g))
                elif vec[k] < 0:
                    sl.append(self._span(k, lo[k], hi[k], g, g + tmax))
                else:
                    sl.append(self._span(k, lo[k], hi[k], g, g))
            sl = tuple(sl)
            sub = np.nonzero(ok[sl])
            if not len(sub[0]):
                continue
            idx = tuple(sub[k] + sl[k].start for k in range(3))
            P0, P1 = edges(idx)
            free = np.ones(len(P0), dtype=bool)
            slo, shi = np.minimum(P0, P1), np.maximum(P0, P1)
            cand = np.nonzero(np.all((slo < hi + g) & (shi > lo - g), 1))[0]
            self._block_cand(free, cand, P0, P1, A, B, S, ro)
            ok[tuple(i[~free] for i in idx)] = False

    def _axis_edges(self, ax: int) -> np.ndarray:
        n = self.shape[ax]
        ok = np.zeros(self.shape, dtype=bool)
        if n < 2:
            return ok
        a = np.moveaxis(self.node_ok, ax, 0)
        both = a[:-1] & a[1:]
        # 축 선분 ↔ 박스 거리 = 퇴화 박스 거리 (정확)
        grids = np.meshgrid(*[self.axes[i][:-1] if i == ax else self.axes[i] for i in range(3)], indexing="ij")
        lo_pts = np.stack(grids, -1)
        hi_pts = lo_pts.copy()
        nxt = np.meshgrid(*[self.axes[i][1:] if i == ax else self.axes[i] for i in range(3)], indexing="ij")
        hi_pts[..., ax] = nxt[ax]
        best = np.full(lo_pts.shape[:-1], np.inf)
        for blo, bhi in zip(self.box_lo, self.box_hi):
            if self.fast_build:
                # M10: 박스에서 r 보다 먼 엣지는 판정(≥ r)에 영향이 없다 → 박스 근처 인덱스 범위만 계산 (결과 같음)
                sl = tuple(self._span(k, blo[k], bhi[k], self.r, self.r) for k in range(3))
                g = np.maximum(0.0, np.maximum(blo - hi_pts[sl], lo_pts[sl] - bhi))
                best[sl] = np.minimum(best[sl], np.sqrt((g * g).sum(-1)))
                continue
            g = np.maximum(0.0, np.maximum(blo - hi_pts, lo_pts - bhi))
            best = np.minimum(best, np.sqrt((g * g).sum(-1)))
        edge = np.moveaxis(best >= self.r - EPS, ax, 0) & both
        np.moveaxis(ok, ax, 0)[:-1] = edge
        return ok

    def _diag_edges(self, d: int) -> None:
        vec = DIRS[d]
        a, b = [ax for ax in range(3) if vec[ax] != 0]
        c = 3 - a - b
        sa, sb = vec[a], vec[b]
        A, B = self.axes[a], self.axes[b]
        # (ia, ib) → 대각선 위 가장 가까운 격자점 (좌표 일치하는 경우만, D38)
        to_a = np.full((len(A), len(B)), -1)
        to_b = np.full((len(A), len(B)), -1)
        ia_of, ib_of = self.index[a], self.index[b]
        for ia, xa in enumerate(A):
            off_a = {sa * (v - xa) for v in A if sa * (v - xa) > 0}
            if not off_a:
                continue
            for ib, xb in enumerate(B):
                common = off_a & {sb * (v - xb) for v in B if sb * (v - xb) > 0}
                if common:
                    t = min(common)
                    to_a[ia, ib] = ia_of[xa + sa * t]
                    to_b[ia, ib] = ib_of[xb + sb * t]
        self.diag_to[d] = (to_a, to_b)
        # 반대 방향은 같은 엣지를 거꾸로 — 목표 표와 마스크를 대칭으로 채운다
        rev = DIR_INDEX[tuple(-v for v in vec)]
        ra = np.full_like(to_a, -1)
        rb = np.full_like(to_b, -1)
        ia, ib = np.nonzero(to_a >= 0)
        ra[to_a[ia, ib], to_b[ia, ib]] = ia
        rb[to_a[ia, ib], to_b[ia, ib]] = ib
        self.diag_to[rev] = (ra, rb)
        ok = np.zeros(self.shape, dtype=bool)
        rev_ok = np.zeros(self.shape, dtype=bool)
        if len(ia):
            # 모든 c 층에 대해 펼친다
            ic = np.arange(self.shape[c])
            IA = np.repeat(ia, len(ic)); IB = np.repeat(ib, len(ic)); IC = np.tile(ic, len(ia))
            TA = to_a[IA, IB]; TB = to_b[IA, IB]

            def idx(xa, xb, xc):
                out = [None, None, None]
                out[a], out[b], out[c] = xa, xb, xc
                return tuple(out)

            src, dst = idx(IA, IB, IC), idx(TA, TB, IC)
            # 45° 엣지는 양 끝이 영역 안 노드여야 한다 (경계 단자 예외 없음, D39)
            valid = (self.node_ok[src] & self.node_ok[dst] & self.in_domain[src] & self.in_domain[dst])
            p0 = np.stack([self.axes[k][src[k]] for k in range(3)], -1)[valid]
            p1 = np.stack([self.axes[k][dst[k]] for k in range(3)], -1)[valid]
            free = self._segments_clear(p0, p1, c)
            ok[tuple(s[valid] for s in src)] = free
            rev_ok[tuple(s[valid] for s in dst)] = free
        self.diag_ok[d] = ok
        self.diag_ok[rev] = rev_ok

    def _segments_clear(self, p0: np.ndarray, p1: np.ndarray, const_axis: int) -> np.ndarray:
        """선분들이 모든 장애물과 유효 반경 이상 떨어져 있는가. 바운딩박스로 후보를 거른 뒤 정확 판정."""
        free = np.ones(len(p0), dtype=bool)
        seg_lo = np.minimum(p0, p1) - self.r
        seg_hi = np.maximum(p0, p1) + self.r
        for blo, bhi in zip(self.box_lo, self.box_hi):
            cand = np.nonzero(free & np.all((seg_lo < bhi) & (seg_hi > blo), 1))[0]
            if len(cand):
                dist = planar_segment_box_distance(p0[cand], p1[cand], blo, bhi, const_axis)
                free[cand[dist < self.r - EPS]] = False
        return free

    # ---------------------------------------------------------------- 인터페이스 (§4.1)

    def _step(self, node, d: int) -> Optional[tuple[int, int, int]]:
        if d < 6:
            ax = d // 2
            s = AXIS_DIRS[d][ax]
            src = node if s > 0 else tuple(n - 1 if i == ax else n for i, n in enumerate(node))
            if src[ax] < 0 or not self.axis_ok[ax][src]:
                return None
            return tuple(n + s if i == ax else n for i, n in enumerate(node))
        if not self.diag_ok[d][node]:
            return None
        vec = DIRS[d]
        a, b = [ax for ax in range(3) if vec[ax] != 0]
        to_a, to_b = self.diag_to[d]
        out = list(node)
        out[a], out[b] = int(to_a[node[a], node[b]]), int(to_b[node[a], node[b]])
        return tuple(out)

    def edge_length(self, n1, n2) -> float:
        return math.dist(self.position(n1), self.position(n2))

    def start_state(self, pipe: Pipe = None) -> State:
        return State(self.start_node, DIR_INDEX[self.pipe.start.dir], 0.0, 0)

    def is_goal(self, state: State, pipe: Pipe = None) -> bool:
        # D45 ③: 마지막 꺾임 → end 단자 직관 ≥ 그 엘보 접선 길이. D47: 경계 end 면 ≥ r + t
        if state.node != self.end_node or DIRS[state.dir] != self.pipe.end.dir:
            return False
        need = self.tangent[state.bend]
        if state.bend and self.end_boundary:
            need += self.r
        return state.run >= need - EPS

    def turn_need(self, state: State, defl: int) -> float:
        """꺾기 전에 필요한 직관 길이 (D45 ①②, D47)."""
        t_new = self.tangent[defl]
        need = max(self.L, self.tangent[state.bend] + t_new)
        if state.bend == 0 and self.start_boundary:
            need = max(need, self.r + t_new)
        return need

    def elbow_clear(self, node, d_in: int, d_out: int) -> bool:
        """D48: 이 노드에서 d_in → d_out 으로 꺾을 때 엘보 호(R = 1.5D)가 장애물과 r 이상 떨어져 있는가.

        호의 모든 점은 꺾임점에서 접선 길이 t 이내이므로, 꺾임점 이격 ≥ r + t 이면 정밀 판정 없이 통과.
        """
        defl = DEFLECTION[d_in][d_out]
        t = self.tangent[defl]
        if self.clearance[node] >= self.r + t + EPS and self.pipe_margin[node] >= self.r + t + EPS:
            return True
        key = (node, d_in, d_out)
        hit = self._arc_ok.get(key)
        if hit is None:
            A0, B0, M0, half, sag, plo0, phi0 = self.arc_template(d_in, d_out)
            v = np.array(self.position(node))
            A, B, mid = A0 + v, B0 + v, M0 + v
            # 호를 감싸는 박스(현 끝점들의 AABB)와 장애물 거리가 r 이상이면 정밀 판정 생략
            plo, phi = plo0 + v, phi0 + v
            gap = np.maximum(0.0, np.maximum(self.box_lo - phi, plo - self.box_hi))
            near = np.sqrt((gap * gap).sum(1)) < self.r + sag
            hit = True
            for lo, hi in zip(self.box_lo[near], self.box_hi[near]):
                # 정밀 판정(황금분할) 전에 확실한 경우를 거른다 — 판정 결과는 정밀 판정과 같다
                #   하한: 현 중점 거리 − 반 길이 ≥ r + sag 이면 그 현은 통과
                #   상한: 현 끝점 거리 − sag < r 이면 충돌 확정 (정밀 판정도 끝점을 후보로 본다)
                def pdist(q):
                    g = np.maximum(0.0, np.maximum(lo - q, q - hi))
                    return np.sqrt((g * g).sum(1))
                if np.any(np.minimum(pdist(A), pdist(B)) - sag < self.r - EPS):
                    hit = False
                    break
                amb = pdist(mid) - half - sag < self.r - EPS
                if np.any(amb):
                    dist, _ = segment_box_distance(A[amb], B[amb], lo, hi)
                    if np.any(dist - sag < self.r - EPS):
                        hit = False
                        break
            if hit and self.pipe_groups:   # 엘보 호 ↔ 놓인 배관 (D50②)
                g = self.r + sag + self._pg_R + self._pg_S
                cand = np.all((self._pg_lo < phi + g[:, None]) & (self._pg_hi > plo - g[:, None]), 1)
                if np.any(cand):
                    dist, _, _ = segment_segment_distance(A[:, None], B[:, None], self._pg_A[cand][None],
                                                          self._pg_B[cand][None])
                    if np.any(dist - sag - self._pg_S[cand][None] < self.r + self._pg_R[cand][None] - EPS):
                        hit = False
            self._arc_ok[key] = hit
        return hit

    def arc_template(self, d_in: int, d_out: int) -> tuple:
        """호 모양은 꺾임점 위치와 무관 — 방향 조합별 형판(꺾임점 기준 상대 좌표)을 한 번만 만든다.

        반환: (현 시작점, 현 끝점, 현 중점, 현 반 길이, sagitta, 현 끝점 AABB 하한, 상한)
        """
        tpl = self._arc_tpl.get((d_in, d_out))
        if tpl is None:
            defl = DEFLECTION[d_in][d_out]
            u0 = np.array(DIRS[d_in], dtype=float)
            u1 = np.array(DIRS[d_out], dtype=float)
            A0, B0, sag = elbow_arc_chords(np.zeros(3), u0 / np.linalg.norm(u0), u1 / np.linalg.norm(u1),
                                           self.tangent[defl], defl)
            tpl = (A0, B0, (A0 + B0) / 2, np.linalg.norm(B0 - A0, axis=1) / 2, sag,
                   np.minimum(A0.min(0), B0.min(0)), np.maximum(A0.max(0), B0.max(0)))
            self._arc_tpl[(d_in, d_out)] = tpl
        return tpl

    def neighbors(self, state: State, pipe: Pipe = None) -> Iterator[State]:
        """D45 ①②·D47: 꺾기 전 직관 ≥ turn_need. D48: 꺾을 때 엘보 호 ↔ 장애물 이격 ≥ r."""
        row = DEFLECTION[state.dir]
        for d in range(len(DIRS)):
            defl = row[d]
            if defl not in ALLOWED_DEFLECTIONS:
                continue
            if defl and state.run < self.turn_need(state, defl) - EPS:
                continue
            nxt = self._step(state.node, d)
            if nxt is None:
                continue
            if defl and not self.elbow_clear(state.node, state.dir, d):
                continue
            length = self.edge_length(state.node, nxt)
            if defl:
                run, bend = length, defl
            else:
                run, bend = state.run + length, state.bend
            yield State(nxt, d, round(min(run, self.run_cap[bend]), 6), bend)

    def cost(self, a: State, b: State, pipe: Pipe = None) -> float:
        return (self.edge_length(a.node, b.node) / 1000 * self.kg_per_m
                + elbow_kg(self.pipe.nominal_size, DEFLECTION[a.dir][b.dir]))

    def is_free(self, a: Vec3, b: Vec3, pipe: Pipe = None) -> bool:
        p0 = np.array([a], dtype=float)
        p1 = np.array([b], dtype=float)
        const = [ax for ax in range(3) if a[ax] == b[ax]]
        if not const:
            raise ValueError("S0 is_free 는 좌표평면 위 선분만 판정한다 (D15)")
        for blo, bhi in zip(self.box_lo, self.box_hi):
            if planar_segment_box_distance(p0, p1, blo, bhi, const[0])[0] < self.r - EPS:
                return False
        # 영역: 양 끝이 영역 안이면 선분 전체가 영역 안 (볼록). 경계 단자는 dir 축 방향 구간만 허용 (D39)
        for p, q in ((a, b), (b, a)):
            if not self._in_domain(p):
                t = self._boundary_terminal_at(p)
                if t is None or not self._in_domain(q) or [ax for ax in range(3) if p[ax] != q[ax]] != \
                        [i for i in range(3) if t.dir[i]]:
                    return False
        return True

    def _in_domain(self, p) -> bool:
        return bool(np.all((np.asarray(p) >= self.dom_lo - EPS) & (np.asarray(p) <= self.dom_hi + EPS)))

    def _boundary_terminal_at(self, p):
        for t in (self.pipe.start, self.pipe.end):
            if t.kind == "boundary" and tuple(map(float, t.pos)) == tuple(map(float, p)):
                return t
        return None

    # ---------------------------------------------------------------- 통계

    def stats(self) -> dict:
        axis = int(sum(m.sum() for m in self.axis_ok))
        diag = int(sum(m.sum() for m in self.diag_ok.values())) // 2   # 양방향을 따로 셌다
        return {
            "pipe": self.pipe.id, "size": self.pipe.nominal_size, "r_mm": round(self.r, 2),
            "grid": list(self.shape), "grid_points": int(np.prod(self.shape)),
            "nodes": int(self.node_ok.sum()), "edges_axis": axis, "edges_45": diag,
            "edges": axis + diag, "build_sec": round(self.build_sec, 3),
        }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="escape graph 생성 — 배관별 노드·엣지 수 출력 (3단계)")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--pipe", help="이 배관만")
    args = ap.parse_args(argv)
    for path in args.paths:
        sc = load(path)
        print(f"== {path}")
        for p in sc.pipes:
            if args.pipe and p.id != args.pipe:
                continue
            s = EscapeGraph(sc, p).stats()
            print(f"  {s['pipe']:5s} {s['size']:>5s} 격자 {'×'.join(map(str, s['grid'])):>11s} "
                  f"노드 {s['nodes']:>7d}  엣지 {s['edges']:>8d} (축 {s['edges_axis']}, 45° {s['edges_45']})  "
                  f"{s['build_sec']:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
