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

from .constants import PIPE_SPECS, SNAP_MM, elbow_kg, elbow_tangent
from .geometry import Vec3
from .scenario import Pipe, Scenario, boundary_face, load
from .verifier.geom import elbow_arc_chords, segment_box_distance
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

    def __init__(self, scenario: Scenario, pipe: Pipe, allow_45: bool = True):
        """allow_45=False 는 45° 엣지를 만들지 않는다 — M9 비교 실험 전용 (D15 기본은 True)."""
        t0 = time.perf_counter()
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
        self._arc_ok: dict[tuple, bool] = {}   # D48 캐시: (노드, 들어오는 방향, 나가는 방향) → 엘보 호 이격 OK
        self.kg_per_m = PIPE_SPECS[pipe.nominal_size].kg_per_m
        ext = scenario.block.extent
        m = _snap_up(self.r)
        # 영역: 블록 안쪽 유효 반경 (10mm 올림). 경계 단자만 예외 (D39)
        self.dom_lo = np.array([m, m, m], dtype=float)
        self.dom_hi = np.array([e - m for e in ext], dtype=float)
        self.box_lo = np.array([o.box.min for o in scenario.obstacles], dtype=float).reshape(-1, 3)
        self.box_hi = np.array([o.box.max for o in scenario.obstacles], dtype=float).reshape(-1, 3)

        # 1) 축별 좌표
        coords = [set() for _ in range(3)]
        for ax in range(3):
            coords[ax].update((self.dom_lo[ax], self.dom_hi[ax]))
            for lo, hi in zip(self.box_lo[:, ax], self.box_hi[:, ax]):
                for v in (_snap_down(lo - self.r), _snap_up(hi + self.r)):
                    if self.dom_lo[ax] <= v <= self.dom_hi[ax]:
                        coords[ax].add(v)
            for t in (pipe.start, pipe.end):
                coords[ax].add(float(t.pos[ax]))
        self.axes = [np.array(sorted(c)) for c in coords]
        self.shape = tuple(len(a) for a in self.axes)
        self.index = [{v: i for i, v in enumerate(a)} for a in self.axes]

        self.start_node = self.node_of(pipe.start.pos)
        self.end_node = self.node_of(pipe.end.pos)
        self.terminal_nodes = {self.start_node, self.end_node}

        # 2) 노드
        X, Y, Z = np.meshgrid(*self.axes, indexing="ij")
        pts = np.stack([X, Y, Z], -1).reshape(-1, 3)
        self.in_domain = np.all((pts >= self.dom_lo - EPS) & (pts <= self.dom_hi + EPS), 1).reshape(self.shape)
        self.clearance = self._point_clearance(pts).reshape(self.shape)   # 격자점 ↔ 장애물 최소거리 (D48 거르기용)
        clear = self.clearance >= self.r - EPS
        self.node_ok = self.in_domain & clear
        for n in self.terminal_nodes:   # 경계 단자는 영역 밖이지만 노드로 둔다 (D39)
            self.node_ok[n] = clear[n]

        # 3) 축 방향 엣지: axis_ok[ax][idx] = idx → idx+e_ax 엣지 사용 가능
        self.axis_ok = [self._axis_edges(ax) for ax in range(3)]
        # 4) 45° 엣지 (D38): diag_ok[d][idx], diag_to[d] = (평면 첫 축 목표 인덱스, 둘째 축 목표 인덱스) 표
        self.diag_ok: dict[int, np.ndarray] = {}
        self.diag_to: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for d in range(6, len(DIRS)):
            if not allow_45:
                self.diag_ok[d] = np.zeros(self.shape, dtype=bool)
            elif d not in self.diag_ok:
                self._diag_edges(d)
        self.build_sec = time.perf_counter() - t0

    # ---------------------------------------------------------------- 구성

    def node_of(self, pos: Vec3) -> tuple[int, int, int]:
        return tuple(self.index[ax][float(pos[ax])] for ax in range(3))

    def position(self, node) -> Vec3:
        return tuple(float(self.axes[ax][node[ax]]) for ax in range(3))

    def _point_clearance(self, pts: np.ndarray) -> np.ndarray:
        best = np.full(len(pts), np.inf)
        for lo, hi in zip(self.box_lo, self.box_hi):
            g = np.maximum(0.0, np.maximum(lo - pts, pts - hi))
            best = np.minimum(best, np.sqrt((g * g).sum(1)))
        return best

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
        if self.clearance[node] >= self.r + t + EPS:
            return True
        key = (node, d_in, d_out)
        hit = self._arc_ok.get(key)
        if hit is None:
            u0 = np.array(DIRS[d_in], dtype=float)
            u1 = np.array(DIRS[d_out], dtype=float)
            A, B, sag = elbow_arc_chords(self.position(node), u0 / np.linalg.norm(u0), u1 / np.linalg.norm(u1),
                                         t, defl)
            # 호를 감싸는 박스(현 끝점들의 AABB)와 장애물 거리가 r 이상이면 정밀 판정 생략
            plo = np.minimum(A.min(0), B.min(0))
            phi = np.maximum(A.max(0), B.max(0))
            gap = np.maximum(0.0, np.maximum(self.box_lo - phi, plo - self.box_hi))
            near = np.sqrt((gap * gap).sum(1)) < self.r + sag
            hit = True
            for lo, hi in zip(self.box_lo[near], self.box_hi[near]):
                dist, _ = segment_box_distance(A, B, lo, hi)
                if np.any(dist - sag < self.r - EPS):
                    hit = False
                    break
            self._arc_ok[key] = hit
        return hit

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
