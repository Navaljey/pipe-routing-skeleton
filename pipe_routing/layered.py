"""M18 대안 표현 실험: 2층 구조 그래프 (기본 경로는 바꾸지 않는다 — 실험용 라우터 슬롯, D9·D3).

  고정 레이어: 고정 장애물만으로 만든 escape graph. 구경마다 한 번 만들어 캐시한다
               (같은 구경 배관들의 단자 좌표를 모두 넣어 그 구경의 모든 배관이 공유).
  배치 레이어: 놓인 배관(직관 + 엘보 호)이 막는 노드·엣지를 표시해 쓰지 못하게 한다. 기하 판정은 검증기와 같은 함수.

  변형 A: 표시만. 놓인 배관은 격자선을 만들지 않는다.
  변형 B: A + 놓인 배관 꺾임점 사이 구간마다 국소 이격선 (r + r_other). `추정:` 정의 (CLAUDE.md M18)
          - 구간 AABB 를 r + r_other 만큼 팽창한 6면(현재 방식과 같은 좌표)마다, 그 면 위 국소 사각형 패치만 노드로 둔다
          - 사각형: 면 안의 두 축 각각 [AABB − (r + r_o), AABB + (r + r_o)] 띠. 축 방향 구간이면 그 축(구간 방향)으로는
            구간 끝에서 장애물(팽창 r)·다른 놓인 배관(팽창 r + r_o2)·영역 경계에 닿을 때까지 연장
          - 패치 노드 = 그 면 위에서 (고정 레이어 좌표 + 같은 구간의 다른 면 좌표) 교차점. 띠 안에 고정 좌표가 없으면
            띠 바로 바깥 고정 좌표 하나씩을 넣어 고정 레이어와 이어지게 한다
          - 축 엣지 = 같은 직선 위 연속 노드 (사이의 국소 이격선이 없는 곳은 고정 레이어 엣지와 같다). 45° 엣지는 고정
            레이어 것만 쓴다
그래프는 이웃 표(step[n, d])로 들고, 탐색은 astar_fast.astar_route_generic (상태·비용·휴리스틱·엘보 규칙은 기본과 같다).
"""
import math
import time

import numpy as np

from .constants import PIPE_SPECS, SUPPORT_SPACING, elbow_kg, elbow_tangent, support_kg
from .escape_graph import EPS, EscapeGraph, _snap_down, _snap_up
from .router_astar import RouteResult, _finish_chain, heuristic_factory
from .gravity import gravity_lmax, move_kind
from .gravity import step as gstep
from .space import DEFLECTION, DIRS

ANG_INDEX = {0: 0, 45: 1, 90: 2, 135: 3}
from .verifier.geom import support_faces

_FIXED: dict = {}   # (id(scenario), 구경) → (scenario, 고정 레이어 EscapeGraph)


def fixed_layer(sc, pipe):
    """구경별 고정 레이어 (캐시). 반환: (그래프, 이번 호출에서 쓴 구성 시간)."""
    key = (id(sc), pipe.nominal_size)
    hit = _FIXED.get(key)
    if hit is not None and hit[0] is sc:
        return hit[1], 0.0
    same = [p for p in sc.pipes if p.nominal_size == pipe.nominal_size]
    g = EscapeGraph(sc, same[0], extra_terminals=[t for p in same for t in (p.start, p.end)])
    _FIXED[key] = (sc, g)
    return g, g.build_sec


def clear_cache():
    _FIXED.clear()


def terminal_reservations(sc, exclude=()) -> list:
    """D59 단자 예약 구역: [(배관, [단자 위치, 직진 구간 끝])]. exclude = 예약하지 않을 배관 id (놓인 배관·자기 자신).

    구간 길이: 노즐 = max(§3.3, t₉₀), 경계 = r + t₉₀ (D47). 방향: start 는 dir 쪽, end 는 dir 반대쪽(배관이 들어오는 쪽).
    t₉₀ = 그 배관 90° 엘보 접선 길이 (D45). `추정:` 엘보 각도는 90° 기준 (가장 흔한 꺾임, 135° 보다 짧다).
    """
    out = []
    for p in sc.pipes:
        if p.id in exclude:
            continue
        t90 = elbow_tangent(p.nominal_size, 90)
        for term, sgn in ((p.start, 1.0), (p.end, -1.0)):
            L = (p.radius + t90) if term.kind == "boundary" else max(p.min_straight, t90)
            a = np.array(term.pos, dtype=float)
            b = a + sgn * L * np.array(term.dir, dtype=float)
            out.append((p, [a.tolist(), b.tolist()]))
    return out


def support_edge_cost(ext, box_lo, box_hi, size: str, P0: np.ndarray, P1: np.ndarray) -> np.ndarray:
    """D57 직관 구간의 추정 서포트 비용 (kg) — 구간 길이 ÷ 최대 간격 × 서포트 1개 kg.

    지지면 = 검증기와 같은 규칙·함수(support_faces, D46): 배관 축과 평행하지 않은 법선의 면 중 고정 장애물에 막히지 않는
    가장 가까운 면. 다른 놓인 배관에 의한 막힘은 무시 (D57 근사).
    `추정:` ① 거리·막힘은 구간 중점 한 곳에서 본다 (45° 구간처럼 거리가 변하거나 구간 중간에 막히는 면이 바뀌어도 중점 값)
            ② 모든 면이 막히면 막힘을 무시한 가장 먼 후보 면 거리로 친다 (보수적, 검증기는 이 위치에 서포트를 못 둔다)
            ③ 단자를 고정점으로 보는 효과(간격 계산이 단자에서 끊김)는 넣지 않는다 — 길이에 비례하는 밀도로만 본다
            ④ 수직 구간 = 수직 최대 간격, 그 밖(45° 포함) = 수평 최대 간격 (검증기와 같음)
    """
    if not len(P0):
        return np.zeros(0)
    D = P1 - P0
    L = np.linalg.norm(D, axis=1)
    U = D / np.where(L > 0, L, 1.0)[:, None]
    dist, blocked = support_faces(ext, box_lo, box_hi, (P0 + P1) / 2, U)
    d_ok = np.where(blocked, np.inf, dist)
    dmin = d_ok.min(1)
    far = np.where(np.isfinite(dist), dist, -np.inf).max(1)
    dmin = np.where(np.isfinite(dmin), dmin, far)
    hmax, vmax = SUPPORT_SPACING[size]
    smax = np.where(np.abs(U[:, 2]) >= 1 - 1e-9, vmax, hmax)
    return L / smax * support_kg(dmin)


def _fixed_support(F, sc, size):
    """고정 레이어 엣지의 추정 서포트 비용 (구경별 1회, 고정 레이어에 캐시). axis[ax][idx], diag[d][idx] (idx = 출발 격자점)."""
    hit = getattr(F, "_support_cache", None)
    if hit is not None:
        return hit
    ext = np.array(sc.block.extent, dtype=float)
    axis = []
    for ax in range(3):
        arr = np.zeros(F.shape)
        idx = np.nonzero(F.axis_ok[ax])
        if len(idx[0]):
            P0 = np.stack([F.axes[k][idx[k]] for k in range(3)], -1)
            nxt = list(idx)
            nxt[ax] = idx[ax] + 1
            P1 = np.stack([F.axes[k][nxt[k]] for k in range(3)], -1)
            arr[idx] = support_edge_cost(ext, F.box_lo, F.box_hi, size, P0, P1)
        axis.append(arr)
    diag = {}
    for d in range(6, 18):
        arr = np.zeros(F.shape)
        idx = np.nonzero(F.diag_ok[d])
        if len(idx[0]):
            pa, pb = [k for k in range(3) if DIRS[d][k] != 0]
            to_a, to_b = F.diag_to[d]
            dst = list(idx)
            dst[pa] = to_a[idx[pa], idx[pb]]
            dst[pb] = to_b[idx[pa], idx[pb]]
            P0 = np.stack([F.axes[k][idx[k]] for k in range(3)], -1)
            P1 = np.stack([F.axes[k][dst[k]] for k in range(3)], -1)
            arr[idx] = support_edge_cost(ext, F.box_lo, F.box_hi, size, P0, P1)
        diag[d] = arr
    F._support_cache = (axis, diag)
    return F._support_cache


class LayeredGraph:
    """SpaceRepresentation (§4.1) — 노드 = 정수 id. 엘보·도착·직관 규칙은 EscapeGraph 와 같은 함수를 빌려 쓴다."""

    elbow_clear = EscapeGraph.elbow_clear
    arc_template = EscapeGraph.arc_template
    turn_need = EscapeGraph.turn_need
    start_state = EscapeGraph.start_state
    _pipe_groups = EscapeGraph._pipe_groups
    _point_clearance = EscapeGraph._point_clearance
    _segments_clear = EscapeGraph._segments_clear

    def __init__(self, sc, pipe, others=(), local_lines=False, support_cost=True, reserved=()):
        """support_cost: D57 추정 서포트를 엣지 비용에 넣는다 (기본). False = M4 이전 비용 (비교용).
        reserved: D59 단자 예약 구역 [(배관, [a, b])] — 놓인 배관과 같이 이격 r₁ + r₂ 로 피한다 (격자선은 만들지 않음)."""
        t0 = time.perf_counter()
        F, fixed_sec = fixed_layer(sc, pipe)
        self.fixed = F
        self.pipe = pipe
        self.r = pipe.radius
        self.L = pipe.min_straight
        self.tangent = {a: elbow_tangent(pipe.nominal_size, a) for a in (0, 45, 90, 135)}
        self.start_boundary = pipe.start.kind == "boundary"
        self.end_boundary = pipe.end.kind == "boundary"
        t135 = self.tangent[135]
        self.run_cap = {a: max(self.L, self.tangent[a] + t135,
                               self.r + self.tangent[a] if self.end_boundary else 0.0,
                               self.r + t135 if (a == 0 and self.start_boundary) else 0.0)
                        for a in self.tangent}
        self._arc_ok, self._arc_tpl = {}, {}
        self.kg_per_m = PIPE_SPECS[pipe.nominal_size].kg_per_m
        self.lmax = gravity_lmax(pipe.nominal_size) if pipe.gravity_pipe else 0.0   # D62
        self.box_lo, self.box_hi = F.box_lo, F.box_hi
        self.dom_lo, self.dom_hi = F.dom_lo, F.dom_hi
        self.pipe_groups = self._pipe_groups(list(others) + list(reserved))
        self.n_reserved = len(reserved)
        self.local_lines = local_lines

        # 1) 축 좌표 = 고정 레이어 + (B) 국소 이격선 좌표
        patches = self._patches(F, others) if local_lines else []
        axes = [set(map(float, F.axes[k])) for k in range(3)]
        for v, c, p, cp, q, cq in patches:
            axes[v].add(c)
            axes[p].update(cp)
            axes[q].update(cq)
        self.axes = [np.array(sorted(a)) for a in axes]

        # 2) 노드 = 고정 레이어 노드 + 패치 노드
        bi = np.nonzero(F.node_ok)
        base_pts = np.stack([F.axes[k][bi[k]] for k in range(3)], -1)
        pts = [base_pts]
        for v, c, p, cp, q, cq in patches:
            P, Q = np.meshgrid(cp, cq, indexing="ij")
            X = np.empty(P.shape + (3,))
            X[..., v], X[..., p], X[..., q] = c, P, Q
            pts.append(X.reshape(-1, 3))
        allp = np.concatenate(pts)
        idx = np.stack([np.searchsorted(self.axes[k], allp[:, k]) for k in range(3)], -1)
        shape = tuple(len(a) for a in self.axes)
        flat = np.ravel_multi_index(idx.T, shape)
        flat_u, first = np.unique(flat, return_index=True)
        self.flat = flat_u
        self.idx = idx[first]
        self.pos = allp[first]
        self.shape = shape
        n_base = len(base_pts)
        is_base = first < n_base
        N = len(flat_u)
        # 장애물 이격: 고정 레이어 노드는 고정 레이어 값(판정 문턱까지 정확), 패치 노드는 정확 계산
        clear = np.empty(N)
        clear[is_base] = F.clearance[tuple(bi[k][first[is_base]] for k in range(3))]
        if np.any(~is_base):
            clear[~is_base] = self._point_clearance(self.pos[~is_base])
        in_dom = np.all((self.pos >= self.dom_lo - EPS) & (self.pos <= self.dom_hi + EPS), 1)
        ok = is_base | (in_dom & (clear >= self.r - EPS))
        self.clearance = clear
        # D60: 놓인 배관(·예약)과의 이격은 노드가 아니라 엣지(꺾임 접선만큼 깎은 직관)와 엘보 호로 판정한다.
        # pipe_margin 은 엘보 빠른 통과 판정(이격 ≥ r + t)에만 쓴다
        self.pipe_margin = self._node_margin(self.pos)
        self.node_valid = ok
        self.n_patch_nodes = int(np.sum(~is_base & ok))
        self.start_node = self._id_of(pipe.start.pos)
        self.end_node = self._id_of(pipe.end.pos)

        # 3) 엣지 → 이웃 표
        self.step = np.full((N, 18), -1, dtype=np.int64)
        self.elen = np.zeros((N, 18))
        self.scost = np.zeros((N, 18))
        self.emask = np.zeros((N, 18), dtype=np.int64)   # D60 엣지 꺾임 등급 마스크 (start i × 4 + end j)
        self.use_support_cost = support_cost
        self._ext = np.array(sc.block.extent, dtype=float)
        self._sup = _fixed_support(F, sc, pipe.nominal_size) if support_cost else None
        n_axis = self._axis_edges(F, is_base, bi, first)
        n_diag = self._diag_edges(F)
        self.n_edges = n_axis + n_diag
        goal = np.array(pipe.end.pos, dtype=float)
        self.hdist = np.sqrt(((self.pos - goal) ** 2).sum(1))
        self.build_sec = time.perf_counter() - t0 + fixed_sec
        self.fixed_sec = fixed_sec

    # ---------------------------------------------------------------- 구성

    def _id_of(self, p) -> int:
        i = [int(np.searchsorted(self.axes[k], float(p[k]))) for k in range(3)]
        f = np.ravel_multi_index(i, self.shape)
        j = int(np.searchsorted(self.flat, f))
        assert self.flat[j] == f
        return j

    def _node_margin(self, P: np.ndarray) -> np.ndarray:
        """노드 ↔ 놓인 배관 여유 (거리 − slack − r_other). EscapeGraph._pipe_node_margin 과 같은 식, 점 목록용."""
        m = np.full(len(P), np.inf)
        reach = self.r + self.tangent[135]
        for A, B, S, ro, lo, hi in self.pipe_groups:
            pad = reach + ro + S.max()
            sel = np.nonzero(np.all((P > lo - pad) & (P < hi + pad), 1))[0]
            if not len(sel):
                continue
            X = P[sel]
            D = B - A
            dd = np.where((D * D).sum(1) > 0, (D * D).sum(1), 1.0)
            t = np.clip(((X[:, None, :] - A[None]) * D[None]).sum(-1) / dd[None], 0, 1)
            dist = np.linalg.norm(X[:, None, :] - (A[None] + t[..., None] * D[None]), axis=-1)
            m[sel] = np.minimum(m[sel], (dist - S[None]).min(1) - ro)
        return m

    def _pipes_clear(self, P0: np.ndarray, P1: np.ndarray) -> np.ndarray:
        """선분들 ↔ 놓인 배관 조각 (EscapeGraph._segments_clear_of_pipes 와 같은 판정). x 정렬로 후보 범위를 좁힌다."""
        free = np.ones(len(P0), dtype=bool)
        if not len(P0) or not self.pipe_groups:
            return free
        slo, shi = np.minimum(P0, P1), np.maximum(P0, P1)
        order = np.argsort(slo[:, 0], kind="stable")
        sx = slo[order, 0]
        maxlen = float((shi[:, 0] - slo[:, 0]).max())
        for A, B, S, ro, lo, hi in self.pipe_groups:
            g = self.r + ro + S.max()
            i0 = int(np.searchsorted(sx, lo[0] - g - maxlen, "left"))
            i1 = int(np.searchsorted(sx, hi[0] + g, "right"))
            w = order[i0:i1]
            w = w[free[w] & np.all((slo[w] < hi + g) & (shi[w] > lo - g), 1)]
            EscapeGraph._block_cand(self, free, w, P0, P1, A, B, S, ro)
        return free

    def _pipes_mask(self, P0: np.ndarray, P1: np.ndarray) -> np.ndarray:
        """D60: 엣지마다 16비트 마스크 — 비트 (i × 4 + j) = 시작을 꺾임 등급 i, 끝을 등급 j 의 엘보 접선만큼 깎은
        직관이 놓인 배관(·예약)과 r + r_o 이상 (검증기와 같은 거리 식). 등급 0/45/90/135. 0 = 어떤 경우에도 못 씀.

        엣지 위 점 p(s) 와 조각 사이 거리는 s 에 대해 볼록 → 막힌 부분은 조각마다 한 구간 [b0, b1].
        깎기 (ts, te) 로 허용 ⇔ 모든 구간이 b1 ≤ ts 또는 b0 ≥ L − te. 깎기가 엣지보다 길면 엣지 길이까지만
        (그 앞 엣지는 깎지 않고 판정 — 보수적).
        """
        n = len(P0)
        mask = np.full(n, 0xFFFF, dtype=np.int64)
        if not n or not self.pipe_groups:
            return mask
        full = self._pipes_clear(P0, P1)
        bad = np.nonzero(~full)[0]
        if not len(bad):
            return mask
        tr = np.array([0.0, self.tangent[45], self.tangent[90], self.tangent[135]])
        Q0, Q1 = P0[bad], P1[bad]
        D = Q1 - Q0
        L = np.linalg.norm(D, axis=1)
        U = D / np.where(L > 0, L, 1)[:, None]
        m = np.full(len(bad), 0xFFFF, dtype=np.int64)
        slo, shi = np.minimum(Q0, Q1), np.maximum(Q0, Q1)
        from .verifier.geom import segment_segment_distance

        def pdist(X, A, B):   # 점 ↔ 선분 거리 (브로드캐스트)
            AB = B - A
            dd = np.maximum((AB * AB).sum(-1), 1e-12)
            t = np.clip(((X - A) * AB).sum(-1) / dd, 0, 1)
            return np.linalg.norm(X - (A + t[..., None] * AB), axis=-1)

        # (엣지, 조각) 쌍을 모든 묶음에서 모은 뒤 이분 탐색을 한 번에 — 쌍마다 같은 식 (결과 같음)
        E, AA, BB, SS, NN, SST = [], [], [], [], [], []
        for A, B, S, ro, lo, hi in self.pipe_groups:
            g = self.r + ro + S.max()
            near = np.nonzero(np.all((slo < hi + g) & (shi > lo - g), 1))[0]
            if not len(near):
                continue
            need = self.r + ro
            d, c1, _ = segment_segment_distance(Q0[near][:, None], Q1[near][:, None], A[None], B[None])
            hit_e, hit_k = np.nonzero(d - S[None] < need - EPS)
            if not len(hit_e):
                continue
            e = near[hit_e]
            E.append(e); AA.append(A[hit_k]); BB.append(B[hit_k]); SS.append(S[hit_k])
            NN.append(np.full(len(e), need))
            SST.append(np.linalg.norm(c1[hit_e, hit_k] - Q0[e], axis=1))
        if E:
            e = np.concatenate(E); Ak = np.concatenate(AA); Bk = np.concatenate(BB)
            Sk = np.concatenate(SS); need = np.concatenate(NN); s_star = np.concatenate(SST)
            f = lambda sv: pdist(Q0[e] + sv[:, None] * U[e], Ak, Bk) - Sk - need
            lo_s, hi_s = np.zeros(len(e)), s_star.copy()
            for _ in range(40):
                mid = (lo_s + hi_s) / 2
                blocked = f(mid) < -EPS
                hi_s = np.where(blocked, mid, hi_s)
                lo_s = np.where(blocked, lo_s, mid)
            b0 = np.where(f(np.zeros(len(e))) < -EPS, 0.0, lo_s)
            lo_s, hi_s = s_star.copy(), L[e].copy()
            for _ in range(40):
                mid = (lo_s + hi_s) / 2
                blocked = f(mid) < -EPS
                lo_s = np.where(blocked, mid, lo_s)
                hi_s = np.where(blocked, hi_s, mid)
            b1 = np.where(f(L[e]) < -EPS, L[e], hi_s)
            ok = np.zeros(len(e), dtype=np.int64)
            for i in range(4):
                ts = np.minimum(tr[i], L[e])
                for j in range(4):
                    te = np.minimum(tr[j], L[e])
                    good = (b1 <= ts + 1e-6) | (b0 >= L[e] - te - 1e-6)
                    ok |= good.astype(np.int64) << (i * 4 + j)
            np.bitwise_and.at(m, e, ok)
        mask[bad] = m
        return mask

    @staticmethod
    def _swap_mask(m: np.ndarray) -> np.ndarray:
        """반대 방향 엣지 마스크 (시작·끝 등급 맞바꿈)."""
        out = np.zeros_like(m)
        for i in range(4):
            for j in range(4):
                out |= ((m >> (i * 4 + j)) & 1) << (j * 4 + i)
        return out

    def _axis_edges(self, F, is_base, bi, first) -> int:
        n = 0
        Fidx = np.full((len(self.flat), 3), -1)
        Fidx[is_base] = np.stack([bi[k][first[is_base]] for k in range(3)], -1)
        for ax in range(3):
            o1, o2 = [k for k in range(3) if k != ax]
            order = np.lexsort((self.idx[:, ax], self.idx[:, o2], self.idx[:, o1]))
            a, b = order[:-1], order[1:]
            same = (self.idx[a, o1] == self.idx[b, o1]) & (self.idx[a, o2] == self.idx[b, o2])
            a, b = a[same], b[same]
            keep = self.node_valid[a] & self.node_valid[b]
            a, b = a[keep], b[keep]
            P0, P1 = self.pos[a], self.pos[b]
            # 장애물: 고정 레이어에서 인접한 두 노드면 고정 레이어 판정, 아니면 정확 판정
            adj = is_base[a] & is_base[b] & (Fidx[b, ax] == Fidx[a, ax] + 1)
            free = np.zeros(len(a), dtype=bool)
            ia = Fidx[a[adj]]
            free[adj] = F.axis_ok[ax][ia[:, 0], ia[:, 1], ia[:, 2]]
            rest = np.nonzero(~adj)[0]
            if len(rest):
                free[rest] = self._segments_clear(P0[rest], P1[rest], o1)
            sel = np.nonzero(free)[0]
            em = self._pipes_mask(P0[sel], P1[sel])
            sel, em = sel[em != 0], em[em != 0]
            a, b = a[sel], b[sel]
            L = np.abs(self.pos[b, ax] - self.pos[a, ax])
            dp, dm = 2 * ax, 2 * ax + 1    # +축, −축 (AXIS_DIRS 순서)
            self.step[a, dp], self.elen[a, dp] = b, L
            self.step[b, dm], self.elen[b, dm] = a, L
            self.emask[a, dp] = em
            self.emask[b, dm] = self._swap_mask(em)
            if self._sup is not None:      # D57: 고정 레이어 인접 엣지는 캐시, 그 밖(패치)은 같은 함수로 계산
                adj = adj[sel]
                sc_ = np.empty(len(a))
                ia = Fidx[a[adj]]
                sc_[adj] = self._sup[0][ax][ia[:, 0], ia[:, 1], ia[:, 2]]
                rest = np.nonzero(~adj)[0]
                sc_[rest] = support_edge_cost(self._ext, self.box_lo, self.box_hi, self.pipe.nominal_size,
                                              self.pos[a[rest]], self.pos[b[rest]])
                self.scost[a, dp] = sc_
                self.scost[b, dm] = sc_
            n += len(a)
        return n

    def _diag_edges(self, F) -> int:
        n = 0
        for d in range(6, 18):
            vec = DIRS[d]
            pa, pb = [ax for ax in range(3) if vec[ax] != 0]
            src = np.nonzero(F.diag_ok[d])
            if not len(src[0]):
                continue
            to_a, to_b = F.diag_to[d]
            dst = list(src)
            dst[pa] = to_a[src[pa], src[pb]]
            dst[pb] = to_b[src[pa], src[pb]]
            s_id = self._ids_from_fixed(F, src)
            d_id = self._ids_from_fixed(F, dst)
            keep = self.node_valid[s_id] & self.node_valid[d_id]
            s_id, d_id = s_id[keep], d_id[keep]
            src = tuple(i[keep] for i in src)
            em = self._pipes_mask(self.pos[s_id], self.pos[d_id])
            free = em != 0
            s_id, d_id, em = s_id[free], d_id[free], em[free]
            src = tuple(i[free] for i in src)
            self.step[s_id, d] = d_id
            self.emask[s_id, d] = em
            self.elen[s_id, d] = np.linalg.norm(self.pos[d_id] - self.pos[s_id], axis=1)
            if self._sup is not None:
                self.scost[s_id, d] = self._sup[1][d][src]
            n += len(s_id)
        return n // 2

    def _ids_from_fixed(self, F, idx) -> np.ndarray:
        coords = [F.axes[k][idx[k]] for k in range(3)]
        full = [np.searchsorted(self.axes[k], coords[k]) for k in range(3)]
        return np.searchsorted(self.flat, np.ravel_multi_index(full, self.shape))

    def _patches(self, F, others) -> list:
        """변형 B 국소 이격선 패치 목록 [(면 축 v, 면 좌표 c, 축 p, p 좌표들, 축 q, q 좌표들)]."""
        out = []
        segs = []   # (lo, hi, r_other, 배관 번호)
        for i, (op, wps) in enumerate(others):
            W = np.asarray(wps, dtype=float)
            for a, b in zip(W[:-1], W[1:]):
                segs.append((np.minimum(a, b), np.maximum(a, b), op.radius, i))
        for lo, hi, ro, who in segs:
            R = self.r + ro
            moving = [k for k in range(3) if hi[k] > lo[k]]
            s_axis = moving[0] if len(moving) == 1 else None
            band = {}
            for k in range(3):
                band[k] = (_snap_down(lo[k] - R), _snap_up(hi[k] + R))
            for v in range(3):
                for side in (0, 1):
                    c = band[v][side]
                    if not (self.dom_lo[v] <= c <= self.dom_hi[v]):
                        continue
                    p, q = [k for k in range(3) if k != v]
                    cs = []
                    for k in (p, q):
                        k_lo, k_hi = band[k]
                        if k == s_axis:   # 구간 방향: 끝에서 장애물·다른 배관·영역 경계에 닿을 때까지 연장
                            other = 3 - v - k
                            probe = np.empty(3)
                            probe[v], probe[other] = c, (lo[other] + hi[other]) / 2
                            k_lo = min(k_lo, self._ray(probe, k, -1, lo[k], segs, who))
                            k_hi = max(k_hi, self._ray(probe, k, +1, hi[k], segs, who))
                        k_lo, k_hi = max(k_lo, self.dom_lo[k]), min(k_hi, self.dom_hi[k])
                        base = F.axes[k]
                        inside = base[(base >= k_lo) & (base <= k_hi)]
                        i0 = int(np.searchsorted(base, k_lo, "left")) - 1
                        i1 = int(np.searchsorted(base, k_hi, "right"))
                        extra = [base[i0]] if i0 >= 0 else []
                        extra += [base[i1]] if i1 < len(base) else []
                        own = [x for x in band[k] if k_lo <= x <= k_hi]
                        cs.append(np.unique(np.concatenate([inside, extra, own]).astype(float)))
                    out.append((v, float(c), p, cs[0], q, cs[1]))
        return out

    def _ray(self, probe, k, sgn, start, segs, who) -> float:
        """probe 를 지나 k 축 방향(sgn)으로 start 에서 출발한 직선이 팽창 장애물(r)·다른 배관 AABB(r + r_o2)에
        처음 닿는 좌표. 없으면 영역 경계."""
        best = self.dom_hi[k] if sgn > 0 else self.dom_lo[k]
        others = [i for i in range(3) if i != k]
        boxes = [(self.box_lo[j] - self.r, self.box_hi[j] + self.r) for j in range(len(self.box_lo))]
        boxes += [(lo - self.r - ro, hi + self.r + ro) for lo, hi, ro, w in segs if w != who]
        for blo, bhi in boxes:
            if all(blo[i] <= probe[i] <= bhi[i] for i in others):
                if sgn > 0 and bhi[k] > start:
                    best = min(best, max(start, blo[k]))
                elif sgn < 0 and blo[k] < start:
                    best = max(best, min(start, bhi[k]))
        return best

    # ---------------------------------------------------------------- 인터페이스

    def position(self, node):
        p = self.pos[node]
        return (float(p[0]), float(p[1]), float(p[2]))

    def edge_length(self, n1, n2) -> float:
        return math.dist(self.position(n1), self.position(n2))

    def is_goal(self, state, pipe=None) -> bool:
        """EscapeGraph.is_goal + D60: 마지막 엣지가 끝을 깎지 않은 직관으로 도착할 수 있어야 한다."""
        return bool(state.ends & 1) and EscapeGraph.is_goal(self, state, pipe)

    def support_cost(self, a, b) -> float:
        """D57 추정 서포트 (a → b 엣지)."""
        return float(self.scost[a.node, b.dir])

    def cost(self, a, b, pipe=None) -> float:
        """ΔJ_router = 직관 + 엘보 + 추정 서포트 (D57). 컴파일 탐색과 같은 연산 순서."""
        return (self.edge_length(a.node, b.node) / 1000 * self.kg_per_m
                + elbow_kg(self.pipe.nominal_size, DEFLECTION[a.dir][b.dir]) + self.support_cost(a, b))

    def neighbors(self, state, pipe=None):
        """파이썬 구현 (시험·Dijkstra 대조용). EscapeGraph.neighbors 와 같은 규칙, 이웃 표 위에서."""
        from .space import ALLOWED_DEFLECTIONS, State
        row = DEFLECTION[state.dir]
        for d in range(len(DIRS)):
            defl = row[d]
            if defl not in ALLOWED_DEFLECTIONS:
                continue
            if defl and state.run < self.turn_need(state, defl) - EPS:
                continue
            if not ((state.ends >> ANG_INDEX[defl]) & 1):   # D60
                continue
            nxt = int(self.step[state.node, d])
            if nxt < 0:
                continue
            nm = (int(self.emask[state.node, d]) >> (ANG_INDEX[defl] * 4)) & 15
            if nm == 0:
                continue
            if defl and not self.elbow_clear(state.node, state.dir, d):
                continue
            length = float(self.elen[state.node, d])
            hz = 0.0
            if self.pipe.gravity_pipe:   # D61: gravity.step (검증기와 같은 함수, D63)
                h2, ok = gstep(state.hz, move_kind(DIRS[d]), length, self.lmax)
                if not ok:
                    continue
                hz = round(h2, 6)
            if defl:
                run, bend = length, defl
            else:
                run, bend = state.run + length, state.bend
            yield State(nxt, d, round(min(run, self.run_cap[bend]), 6), bend, nm, hz)

    def stats(self) -> dict:
        return {"pipe": self.pipe.id, "size": self.pipe.nominal_size, "grid": list(self.shape),
                "nodes": int(self.node_valid.sum()), "patch_nodes": self.n_patch_nodes, "edges": int(self.n_edges),
                "build_sec": round(self.build_sec, 3), "fixed_sec": round(self.fixed_sec, 3)}


def _route(g: LayeredGraph, pipe, time_limit) -> RouteResult:
    from .astar_fast import astar_route_generic
    t0 = time.perf_counter()
    h = heuristic_factory(g, pipe)
    status, chain, expanded, generated = astar_route_generic(g, pipe, time_limit, h(g.start_state(pipe)))
    if status != "ok":
        return RouteResult(pipe.id, status, expanded=expanded, generated=generated,
                           search_sec=time.perf_counter() - t0)
    return _finish_chain(g, pipe, chain, expanded, generated, time.perf_counter() - t0)


def make_layered_router(local_lines: bool = False, support_cost: bool = True, reserve: bool = True):
    """2층 구조 라우터 슬롯 만들기. local_lines = 변형 B, support_cost = D57 추정 서포트 비용,
    reserve = D59 단자 예약 (아직 놓이지 않은 다른 배관의 단자 직진 구간을 피한다)."""
    def router(sc, pipe, time_limit, placed=()):
        reserved = terminal_reservations(sc, exclude={pipe.id} | {p.id for p, _ in placed}) if reserve else []
        g = LayeredGraph(sc, pipe, placed, local_lines=local_lines, support_cost=support_cost, reserved=reserved)
        r = _route(g, pipe, time_limit)
        r.graph_sec = g.build_sec
        r.graph_stats = g.stats()
        return r
    router.__name__ = (f"layered_router_{'b' if local_lines else 'a'}{'' if support_cost else '_nosup'}"
                       f"{'' if reserve else '_noreserve'}")
    if reserve:   # 실패 분류에서 "예약끼리 충돌" 을 가려내는 데 쓴다 (multi)
        router.without_reservations = make_layered_router(local_lines, support_cost, reserve=False)
    return router


layered_router_a = make_layered_router(False, True)   # 기본 (D55·D57)
layered_router_b = make_layered_router(True, True)    # 비교용 (D55)
