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
from dataclasses import dataclass

import numpy as np

from .constants import PIPE_SPECS, SUPPORT_SPACING, elbow_kg, elbow_tangent
from .escape_graph import EPS, EscapeGraph, _snap_down, _snap_up, swap_mask
from .router_astar import RouteResult, _finish_chain, heuristic_factory
from .gravity import gravity_lmax, move_kind
from .gravity import step as gstep
from .space import DEFLECTION, DIRS

ANG_INDEX = {0: 0, 45: 1, 90: 2, 135: 3}
from .support_router import LineSupport, reverse as reverse_support, support_step
from .support_std import ANGLE_HALF, load_standard

# D72 결정적 탐색 예산 = 배관당 A* 확장 상태 수 상한 (벽시계 60초 대체). 값·근거는 CLAUDE.md D72 (`추정:`).
# CHECK_EVERY(2048) 의 배수라 정확히 이 수에서 멈춘다.
EXPAND_BUDGET = 4_194_304   # = 2²² (D73 — 기준선의 정의, 머신 속도 환산값 아님. D72 의 1,660,928 대체)
DIAG_FACTOR = 4             # D74: 실패 분류용 진단 단독 재라우팅 예산 = 라우팅 예산 × 4 (16,777,216)

_FIXED: dict = {}   # (id(scenario), 구경, 중력관 층 여부, 격자 변형) → (scenario, 고정 레이어 EscapeGraph)

G1_MAX_LEVELS = 20   # D66 G1 `추정:` 중력관 한 개당 높이 보조선 최대 개수


@dataclass(frozen=True)
class GridSpec:
    """D66 고정층 격자 보강 변형. g1 = 중력관 전용 높이·수평 보조선, g2 = 전 배관 일정 간격 보조선 (mm, 0 = 없음)."""
    g1: bool = False
    g2: float = 0.0

    @property
    def name(self) -> str:
        parts = (["g1"] if self.g1 else []) + ([f"g2:{int(self.g2)}"] if self.g2 else [])
        return "+".join(parts) or "g0"


G0 = GridSpec()


def parse_grid(text: str) -> GridSpec:
    """'g0' | 'g1' | 'g2:2000' | 'g1+g2:2000'."""
    g1, g2 = False, 0.0
    for part in text.lower().split("+"):
        if part == "g0":
            continue
        if part == "g1":
            g1 = True
        elif part.startswith("g2:"):
            g2 = float(part[3:])
            if g2 <= 0:
                raise ValueError(text)
        else:
            raise ValueError(f"격자 변형 '{text}' — g0 | g1 | g2:<mm> | g1+g2:<mm>")
    return GridSpec(g1, g2)


def grid_extra_coords(sc, size: str, gravity_layer: bool, spec: GridSpec):
    """D66 보조 격자 좌표 [x, y, z] (영역 밖 값은 EscapeGraph 가 버린다).

    G2: 블록 좌표 0 에서 간격 s 의 배수 (x, y, z 모두).
    G1 (중력관 층만): `추정:` 규칙 —
      높이 보조선 = 그 구경 중력관마다 start 높이에서 아래로 간격 s_z 의 배수 (end 높이 위까지).
                    s_z = max(수직 한 단 최소 낙차 max(§3.3, 2·t₉₀) — 90°-90° 단은 x·y 좌표 우연 일치 없이 어느 노드에서나 가능,
                              낙차 / G1_MAX_LEVELS), 10 mm 올림.
      수평 보조선 = x·y 모두 블록 좌표 0 에서 간격 s_h = L_max / 2 (10 mm 내림)의 배수 — 직진 L_max 안에 하향 기회가
                    두 번 이상 오도록 (꺾임 직관 규칙이 길이를 쓰므로 여유를 둔다).
    """
    xs, ys, zs = [], [], []
    ext = sc.block.extent
    if spec.g2:
        for ax, out in enumerate((xs, ys, zs)):
            out.extend(k * spec.g2 for k in range(1, int(ext[ax] // spec.g2) + 1))
    if spec.g1 and gravity_layer:
        from .gravity import gravity_lmax
        sv = _snap_up(max(PIPE_SPECS[size].min_straight, 2 * elbow_tangent(size, 90)))
        sh = _snap_down(gravity_lmax(size) / 2)
        for ax, out in enumerate((xs, ys)):
            out.extend(k * sh for k in range(1, int(ext[ax] // sh) + 1))
        for p in sc.pipes:
            if p.nominal_size != size or not p.gravity_pipe:
                continue
            z0, z1 = float(p.start.pos[2]), float(p.end.pos[2])
            if z0 <= z1:
                continue
            sz = max(sv, _snap_up((z0 - z1) / G1_MAX_LEVELS))
            k = 1
            while z0 - k * sz > z1:
                zs.append(z0 - k * sz)
                k += 1
    if not (xs or ys or zs):
        return None
    return [xs, ys, zs]


def support_extra_coords(sc, size: str):
    """D80 ⑨ 서포트 보조 격자 좌표 [x, y, z] — 배치와 무관, 규칙에서만 정한다. `추정:`
      ① 서포트 띠 경계: 바닥·천장 형식(C) 표 끝 바로 안쪽 높이 (z = L_max − 10, H − L_max + 10), 격벽 형식(A·B) 표 끝
         바로 안쪽 (x·y = L_max − 10, 끝 − L_max + 10) — 띠 안으로 들어갈 노드
      ② 단자 첫 꺾임 선: 그 구경 배관 단자마다 단자 방향 축으로 첫(마지막) 꺾임 최소 직관만큼 떨어진 좌표
         (max(§3.3, t₁₃₅), 경계 단자는 r + t₁₃₅, 10 mm 올림) — 단자 바로 앞에서 띠 쪽으로 꺾을 수 있게."""
    from .support_std import band_lmax, load_standard
    std = load_standard()
    ext = sc.block.extent
    lm = band_lmax(std, size)
    out = [[], [], []]
    for f, (ax, side) in ss_face_axis().items():
        L = lm[f]
        if math.isfinite(L) and 0 < L < ext[ax]:
            out[ax].append(_snap_down(L - 10) if side == 0 else _snap_up(ext[ax] - L + 10))
    t135 = elbow_tangent(size, 135)
    for p in sc.pipes:
        if p.nominal_size != size:
            continue
        for term, sgn in ((p.start, 1.0), (p.end, -1.0)):
            d = np.asarray(term.dir, dtype=float)
            ax = int(np.argmax(np.abs(d)))
            need = max(p.min_straight, t135) if term.kind != "boundary" else max(p.min_straight, p.radius + t135)
            out[ax].append(float(term.pos[ax]) + sgn * d[ax] * _snap_up(need))
    return out


def ss_face_axis():
    from .support_std import FACE_AXIS
    return FACE_AXIS


def fixed_layer(sc, pipe, grid: GridSpec = G0, support_lines: bool = True):
    """구경별 고정 레이어 (캐시). 반환: (그래프, 이번 호출에서 쓴 구성 시간).
    D66 G1 이면 중력관은 같은 구경 압력관과 따로 중력관 층을 쓴다. support_lines = D80 ⑨ 서포트 보조 격자선."""
    gravity_layer = bool(grid.g1 and pipe.gravity_pipe)
    key = (id(sc), pipe.nominal_size, gravity_layer, grid.g2, support_lines)
    hit = _FIXED.get(key)
    if hit is not None and hit[0] is sc:
        return hit[1], 0.0
    same = [p for p in sc.pipes if p.nominal_size == pipe.nominal_size]
    extra = grid_extra_coords(sc, pipe.nominal_size, gravity_layer, grid)
    if support_lines:
        sl = support_extra_coords(sc, pipe.nominal_size)
        extra = [list(a) + list(b) for a, b in zip(extra or [[], [], []], sl)]
    g = EscapeGraph(sc, same[0], extra_terminals=[t for p in same for t in (p.start, p.end)], corner_arc=True,
                    extra_coords=extra)
    _FIXED[key] = (sc, g)
    return g, g.build_sec


def clear_cache():
    _FIXED.clear()


def fixed_layer_stats(sc) -> list:
    """이 시나리오에서 만든 고정 레이어 목록 (D66 보고용): 구경·중력관 층·G2 간격·격자·노드·엣지·생성 시간."""
    out = []
    for (sid, size, grav, g2, slines), (s, F) in _FIXED.items():
        if s is not sc:
            continue
        edges = int(sum(int(m.sum()) for m in F.axis_ok) + sum(int(m.sum()) for m in F.diag_ok.values()) // 2)
        out.append({"size": size, "gravity_layer": grav, "g2": g2, "support_lines": slines, "grid": list(F.shape),
                    "nodes": int(F.node_ok.sum()), "edges": edges, "build_sec": round(F.build_sec, 3)})
    return sorted(out, key=lambda x: (int(x["size"][:-1]), x["gravity_layer"]))


def terminal_reservations(sc, exclude=()) -> list:
    """D59 단자 예약 구역: [(배관, [단자 위치, 직진 구간 끝])]. exclude = 예약하지 않을 배관 id (놓인 배관·자기 자신).

    구간 길이: 노즐 = max(§3.3, t₉₀), 경계 = r + t₉₀ (D47). 방향: start 는 dir 쪽, end 는 dir 반대쪽(배관이 들어오는 쪽).
    t₉₀ = 그 배관 90° 엘보 접선 길이 (D45). 엘보 각도 90° 기준 — 확정 (D59 보완, 가장 큰 접선 기준).
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


def line_support(F, sc, pipe) -> LineSupport:
    """D80 ⑨ 라우터 서포트 모델 (구경·표준별 1회, 고정 레이어에 캐시)."""
    std = load_standard()
    key = (pipe.nominal_size, std.sha256)
    cache = F.__dict__.setdefault("_line_support", {})
    if key not in cache:
        cache[key] = LineSupport(std, sc.block.extent, F.box_lo, F.box_hi, pipe.nominal_size, pipe.od,
                                 SUPPORT_SPACING[pipe.nominal_size])
    return cache[key]


def support_edge_model(LS: LineSupport, P0: np.ndarray, P1: np.ndarray):
    """엣지들 → (추정 서포트 비용 kg, 서포트 가능 구간 요약 pack) (D80 ⑨). 축 엣지는 격자선 모델, 45° 는 중점."""
    n = len(P0)
    cost = np.zeros(n)
    pk = np.zeros(n, dtype=np.int64)
    if not n:
        return cost, pk
    D = np.abs(P1 - P0)
    nz = (D > 1e-9).sum(1)
    for ax in range(3):
        sel = np.nonzero((nz == 1) & (D[:, ax] > 1e-9))[0]
        if len(sel):
            cost[sel], pk[sel] = LS.axis_edges(ax, P0[sel], P1[sel])
    sel = np.nonzero(nz > 1)[0]
    if len(sel):
        cost[sel], pk[sel] = LS.diag_edges(P0[sel], P1[sel])
    return cost, pk


def _fixed_support(F, sc, pipe):
    """고정 레이어 엣지의 (비용, pack) — 구경·표준별 1회 캐시. axis[ax] = (cost, pack) 배열 (F.shape), diag[d] 도 같다."""
    LS = line_support(F, sc, pipe)
    key = (pipe.nominal_size, LS.std.sha256)
    cache = F.__dict__.setdefault("_support_cache", {})
    if key in cache:
        return cache[key]
    axis = []
    for ax in range(3):
        c_arr = np.zeros(F.shape)
        p_arr = np.zeros(F.shape, dtype=np.int64)
        idx = np.nonzero(F.axis_ok[ax])
        if len(idx[0]):
            P0 = np.stack([F.axes[k][idx[k]] for k in range(3)], -1)
            nxt = list(idx)
            nxt[ax] = idx[ax] + 1
            P1 = np.stack([F.axes[k][nxt[k]] for k in range(3)], -1)
            c_arr[idx], p_arr[idx] = support_edge_model(LS, P0, P1)
        axis.append((c_arr, p_arr))
    diag = {}
    for d in range(6, 18):
        c_arr = np.zeros(F.shape)
        p_arr = np.zeros(F.shape, dtype=np.int64)
        idx = np.nonzero(F.diag_ok[d])
        if len(idx[0]):
            pa, pb = [k for k in range(3) if DIRS[d][k] != 0]
            to_a, to_b = F.diag_to[d]
            dst = list(idx)
            dst[pa] = to_a[idx[pa], idx[pb]]
            dst[pb] = to_b[idx[pa], idx[pb]]
            P0 = np.stack([F.axes[k][idx[k]] for k in range(3)], -1)
            P1 = np.stack([F.axes[k][dst[k]] for k in range(3)], -1)
            c_arr[idx], p_arr[idx] = support_edge_model(LS, P0, P1)
        diag[d] = (c_arr, p_arr)
    cache[key] = (axis, diag)
    return cache[key]


class _Scratch:
    """EscapeGraph._pipe_groups 를 항목 하나에 부를 때 쓰는 빈 객체 (그 함수가 속성을 기록한다)."""


class LayeredGraph:
    """SpaceRepresentation (§4.1) — 노드 = 정수 id. 엘보·도착·직관 규칙은 EscapeGraph 와 같은 함수를 빌려 쓴다."""

    elbow_clear = EscapeGraph.elbow_clear
    arc_template = EscapeGraph.arc_template
    turn_need = EscapeGraph.turn_need
    start_state = EscapeGraph.start_state
    _pipe_groups = EscapeGraph._pipe_groups
    _point_clearance = EscapeGraph._point_clearance
    _segments_clear = EscapeGraph._segments_clear

    def __init__(self, sc, pipe, others=(), local_lines=False, support_cost=True, reserved=(), grid: GridSpec = G0,
                 incremental: bool = True, support_lines: bool = True, members=()):
        """support_cost: D57 추정 서포트를 엣지 비용에 넣는다 (기본). False = M4 이전 비용 (비교용).
        reserved: D59 단자 예약 구역 [(배관, [a, b])] — 놓인 배관과 같이 이격 r₁ + r₂ 로 피한다 (격자선은 만들지 않음).
        grid: D66 고정층 격자 보강 변형 (기본 G0).
        incremental: D67 증분 배치층 표시 (변형 A 만) — 놓인 배관·예약 항목별 표시(엣지 마스크·노드 여유)를 고정 레이어에
        항목 단위로 캐시하고, 지금 항목 집합으로 합성한다 (AND·min — 전체 재계산과 비트 동일). False = 매번 전체 재계산
        (시험 대조용).
        members: D90 (a1) 놓인 배관 서포트 부재 예약 [(부재 op, [a, b])] — op.radius = 앵글 반폭, 이격 r + 반폭 으로
        피한다 (검증기 D80 ⑧ 과 같은 문턱). 예약 구역과 같이 격자선은 만들지 않고 D67 항목 캐시에 들어간다."""
        t0 = time.perf_counter()
        F, fixed_sec = fixed_layer(sc, pipe, grid, support_lines=support_cost and support_lines)
        self.grid = grid
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
        self.pipe_groups = self._pipe_groups(list(others) + list(reserved) + list(members))
        self.n_reserved = len(reserved)
        self.n_members = len(members)
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
        self.use_support_cost = support_cost
        self._ext = np.array(sc.block.extent, dtype=float)
        self._sup = _fixed_support(F, sc, pipe) if support_cost else None
        self._ls = line_support(F, sc, pipe) if support_cost else None
        hm, vm = SUPPORT_SPACING[pipe.nominal_size]
        self.smax_dir = np.array([vm if (DIRS[d][0] == 0 and DIRS[d][1] == 0) else hm for d in range(18)], float)
        # 꺾임 등급(0/45/90/135)별 u 추가 = 2·t_θ / 수평 간격 (support_router.support_step)
        self.bend_u = np.array([0.0] + [2 * elbow_tangent(pipe.nominal_size, a) / hm for a in (45, 90, 135)]) \
            if support_cost else np.zeros(4)
        self.incremental = bool(incremental and not patches)
        self._items = [(("P", op.id) + tuple(map(tuple, np.asarray(w, dtype=float).tolist())), op, w)
                       for op, w in others if w] + \
                      [(("R", op.id) + tuple(map(tuple, np.asarray(w, dtype=float).tolist())), op, w) for op, w in reserved] + \
                      [(("S", op.id, op.radius) + tuple(map(tuple, np.asarray(w, dtype=float).tolist())), op, w)
                       for op, w in members]
        self.pipe_margin = self._item_node_margin(F) if self.incremental else self._node_margin(self.pos)
        self.node_valid = ok
        self.n_patch_nodes = int(np.sum(~is_base & ok))
        self.start_node = self._id_of(pipe.start.pos)
        self.end_node = self._id_of(pipe.end.pos)

        # 3) 엣지 → 이웃 표
        self.step = np.full((N, 18), -1, dtype=np.int64)
        self.elen = np.zeros((N, 18))
        self.scost = np.zeros((N, 18))
        self.spack = np.zeros((N, 18), dtype=np.int64)   # D80 ⑨ 서포트 가능 구간 요약 (0 = 제약 없음)
        self.emask = np.zeros((N, 18), dtype=np.int64)   # D60 엣지 꺾임 등급 마스크 (start i × 4 + end j)
        if self.incremental:
            self.n_edges = self._incremental_edges(F, is_base, bi, first)
        else:
            n_axis = self._axis_edges(F, is_base, bi, first)
            n_diag = self._diag_edges(F)
            self.n_edges = n_axis + n_diag
        goal = np.array(pipe.end.pos, dtype=float)
        self.hdist = np.sqrt(((self.pos - goal) ** 2).sum(1))
        self.build_sec = time.perf_counter() - t0 + fixed_sec
        self.fixed_sec = fixed_sec

    # ---------------------------------------------------------------- D67 증분 배치층 표시

    def _item_contrib(self, F, key, op, w):
        """항목(놓인 배관 또는 예약 구역) 하나의 표시 — 고정 레이어에 캐시 (키 = 유효 반경 + 항목 종류·배관·좌표).
        반환 {"margin": (노드, 값), "axis": [(엣지 번호, 마스크)] × 3, "diag": {d: (엣지 번호, 마스크)}}.
        엣지 번호는 _edge_template 의 순서. 항목에 닿지 않는 엣지(마스크 0xFFFF)·노드(inf)는 저장하지 않는다 (희소)."""
        cache = F.__dict__.setdefault("_item_cache", {})
        ck = (self.r,) + key
        hit = cache.get(ck)
        if hit is not None:
            return hit
        groups = EscapeGraph._pipe_groups(_Scratch(), [(op, w)])
        if not groups:
            hit = {"margin": (np.zeros(0, np.int64), np.zeros(0)), "axis": [None] * 3, "diag": {}}
            cache[ck] = hit
            return hit
        lo = np.min([g[4] for g in groups], 0)
        hi = np.max([g[5] for g in groups], 0)
        gpad = self.r + max(g[3] + g[2].max() for g in groups)
        m = self._node_margin(self.pos, groups)
        sel = np.nonzero(np.isfinite(m))[0]
        out = {"margin": (sel.astype(np.int32), m[sel]), "axis": [], "diag": {}}
        tmpl = self._edge_template(F)
        for kind, part in (("axis", tmpl["axis"]), ("diag", tmpl["diag"])):
            items = enumerate(part) if kind == "axis" else part.items()
            for k, e in items:
                P0, P1 = self.pos[e["a"]], self.pos[e["b"]]
                c = np.nonzero(np.all((np.minimum(P0, P1) < hi + gpad) & (np.maximum(P0, P1) > lo - gpad), 1))[0]
                em = self._pipes_mask(P0[c], P1[c], groups)
                touched = em != 0xFFFF
                entry = (c[touched].astype(np.int32), em[touched].astype(np.uint16))   # 희소·작은 정수형 (메모리)
                if kind == "axis":
                    out["axis"].append(entry)
                else:
                    out["diag"][k] = entry
        cache[ck] = out
        return out

    def _item_node_margin(self, F) -> np.ndarray:
        m = np.full(len(self.pos), np.inf)
        for key, op, w in self._items:
            idx, val = self._item_contrib(F, key, op, w)["margin"]
            m[idx] = np.minimum(m[idx], val)
        return m

    def _edge_template(self, F) -> dict:
        """변형 A 의 고정 레이어 엣지 목록 (놓인 배관과 무관한 부분) — 고정 레이어에 1회 캐시.
        axis[ax]·diag[d] = {a: 시작 노드, b: 끝 노드, obs: 고정 장애물 마스크, L: 길이, sc: 추정 서포트}.
        _axis_edges·_diag_edges 와 같은 식. 좌표는 저장하지 않고 노드 번호로 (메모리)."""
        key = ("tmpl", self.r, self.use_support_cost, self._ls.std.sha256 if self._ls else None,
               self.pipe.nominal_size if self.use_support_cost else None)
        cache = F.__dict__.setdefault("_tmpl_cache", {})
        hit = cache.get(key)
        if hit is not None:
            return hit
        bi = np.nonzero(F.node_ok)
        Fidx = np.stack(bi, -1)   # 변형 A: 노드 = 고정 레이어 노드 (같은 순서)
        axis = []
        for ax in range(3):
            o1, o2 = [k for k in range(3) if k != ax]
            order = np.lexsort((self.idx[:, ax], self.idx[:, o2], self.idx[:, o1]))
            a, b = order[:-1], order[1:]
            same = (self.idx[a, o1] == self.idx[b, o1]) & (self.idx[a, o2] == self.idx[b, o2])
            a, b = a[same], b[same]
            P0, P1 = self.pos[a], self.pos[b]
            adj = Fidx[b, ax] == Fidx[a, ax] + 1
            free = np.zeros(len(a), dtype=bool)
            ia = Fidx[a[adj]]
            free[adj] = F.axis_ok[ax][ia[:, 0], ia[:, 1], ia[:, 2]]
            obs = np.full(len(a), 0xFFFF, dtype=np.int64)
            if F.axis_mask[ax] is not None:
                obs[adj] = F.axis_mask[ax][ia[:, 0], ia[:, 1], ia[:, 2]]
            rest = np.nonzero(~adj)[0]
            if len(rest):
                free[rest] = self._segments_clear(P0[rest], P1[rest], o1)
            sel = np.nonzero(free)[0]
            a, b, adj = a[sel], b[sel], adj[sel]
            L = np.abs(self.pos[b, ax] - self.pos[a, ax])
            sc_ = np.zeros(len(a))
            sp_ = np.zeros(len(a), dtype=np.int64)
            if self._sup is not None:
                ia = Fidx[a[adj]]
                sc_[adj] = self._sup[0][ax][0][ia[:, 0], ia[:, 1], ia[:, 2]]
                sp_[adj] = self._sup[0][ax][1][ia[:, 0], ia[:, 1], ia[:, 2]]
                r_ = np.nonzero(~adj)[0]
                sc_[r_], sp_[r_] = support_edge_model(self._ls, self.pos[a[r_]], self.pos[b[r_]])
            axis.append({"a": a.astype(np.int32), "b": b.astype(np.int32), "obs": obs[sel].astype(np.uint16), "L": L,
                         "sc": sc_, "sp": sp_})
        diag = {}
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
            obs = F.diag_mask[d][src] if d in F.diag_mask else np.full(len(s_id), 0xFFFF, np.uint16)
            diag[d] = {"a": s_id.astype(np.int32), "b": d_id.astype(np.int32), "obs": obs.astype(np.uint16),
                       "L": np.linalg.norm(self.pos[d_id] - self.pos[s_id], axis=1),
                       "sc": self._sup[1][d][0][src] if self._sup is not None else np.zeros(len(s_id)),
                       "sp": self._sup[1][d][1][src] if self._sup is not None else np.zeros(len(s_id), np.int64)}
        cache[key] = {"axis": axis, "diag": diag}
        return cache[key]

    def _incremental_edges(self, F, is_base, bi, first) -> int:
        """D67: 엣지 마스크 = 고정 레이어 마스크 AND (지금 항목들의 캐시된 마스크). _axis_edges·_diag_edges 와 같은 결과."""
        tmpl = self._edge_template(F)
        contribs = [self._item_contrib(F, key, op, w) for key, op, w in self._items]
        n = 0
        for ax, e in enumerate(tmpl["axis"]):
            em = e["obs"].astype(np.int64)
            for c in contribs:
                idx, val = c["axis"][ax]
                em[idx] &= val
            keep = np.nonzero(em != 0)[0]
            a, b, em = e["a"][keep], e["b"][keep], em[keep]
            dp, dm = 2 * ax, 2 * ax + 1
            self.step[a, dp], self.elen[a, dp] = b, e["L"][keep]
            self.step[b, dm], self.elen[b, dm] = a, e["L"][keep]
            self.emask[a, dp] = em
            self.emask[b, dm] = swap_mask(em)
            if self._sup is not None:
                self.scost[a, dp] = e["sc"][keep]
                self.scost[b, dm] = e["sc"][keep]
                self.spack[a, dp] = e["sp"][keep]
                self.spack[b, dm] = reverse_support(e["sp"][keep])
            n += len(a)
        nd = 0
        for d, e in tmpl["diag"].items():
            em = e["obs"].astype(np.int64)
            for c in contribs:
                idx, val = c["diag"][d]
                em[idx] &= val
            keep = np.nonzero(em != 0)[0]
            s_id = e["a"][keep]
            self.step[s_id, d] = e["b"][keep]
            self.emask[s_id, d] = em[keep]
            self.elen[s_id, d] = e["L"][keep]
            if self._sup is not None:
                self.scost[s_id, d] = e["sc"][keep]
                self.spack[s_id, d] = e["sp"][keep]
            nd += len(s_id)
        return n + nd // 2

    # ---------------------------------------------------------------- 구성

    def _id_of(self, p) -> int:
        i = [int(np.searchsorted(self.axes[k], float(p[k]))) for k in range(3)]
        f = np.ravel_multi_index(i, self.shape)
        j = int(np.searchsorted(self.flat, f))
        assert self.flat[j] == f
        return j

    def _node_margin(self, P: np.ndarray, groups=None) -> np.ndarray:
        """노드 ↔ 놓인 배관 여유 (거리 − slack − r_other). EscapeGraph._pipe_node_margin 과 같은 식, 점 목록용.
        groups: 조각 묶음 (기본 = 이 그래프의 놓인 배관·예약 전부, D67 항목 하나만 넘기기도 한다)."""
        groups = self.pipe_groups if groups is None else groups
        m = np.full(len(P), np.inf)
        reach = self.r + self.tangent[135]
        for A, B, S, ro, lo, hi in groups:
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

    def _pipes_clear(self, P0: np.ndarray, P1: np.ndarray, groups=None) -> np.ndarray:
        """선분들 ↔ 놓인 배관 조각 (EscapeGraph._segments_clear_of_pipes 와 같은 판정). x 정렬로 후보 범위를 좁힌다."""
        groups = self.pipe_groups if groups is None else groups
        free = np.ones(len(P0), dtype=bool)
        if not len(P0) or not groups:
            return free
        slo, shi = np.minimum(P0, P1), np.maximum(P0, P1)
        order = np.argsort(slo[:, 0], kind="stable")
        sx = slo[order, 0]
        maxlen = float((shi[:, 0] - slo[:, 0]).max())
        for A, B, S, ro, lo, hi in groups:
            g = self.r + ro + S.max()
            i0 = int(np.searchsorted(sx, lo[0] - g - maxlen, "left"))
            i1 = int(np.searchsorted(sx, hi[0] + g, "right"))
            w = order[i0:i1]
            w = w[free[w] & np.all((slo[w] < hi + g) & (shi[w] > lo - g), 1)]
            EscapeGraph._block_cand(self, free, w, P0, P1, A, B, S, ro)
        return free

    def _pipes_mask(self, P0: np.ndarray, P1: np.ndarray, groups=None) -> np.ndarray:
        """D60: 엣지마다 16비트 마스크 — 비트 (i × 4 + j) = 시작을 꺾임 등급 i, 끝을 등급 j 의 엘보 접선만큼 깎은
        직관이 놓인 배관(·예약)과 r + r_o 이상 (검증기와 같은 거리 식). 등급 0/45/90/135. 0 = 어떤 경우에도 못 씀.

        엣지 위 점 p(s) 와 조각 사이 거리는 s 에 대해 볼록 → 막힌 부분은 조각마다 한 구간 [b0, b1].
        깎기 (ts, te) 로 허용 ⇔ 모든 구간이 b1 ≤ ts 또는 b0 ≥ L − te. 깎기가 엣지보다 길면 엣지 길이까지만
        (그 앞 엣지는 깎지 않고 판정 — 보수적).
        """
        groups = self.pipe_groups if groups is None else groups
        n = len(P0)
        mask = np.full(n, 0xFFFF, dtype=np.int64)
        if not n or not groups:
            return mask
        full = self._pipes_clear(P0, P1, groups)
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
        for A, B, S, ro, lo, hi in groups:
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
            obs = np.full(len(a), 0xFFFF, dtype=np.int64)   # D65: 고정 장애물 꺾임 등급 마스크 (패치 엣지는 전부 r 이상만)
            if F.axis_mask[ax] is not None:
                obs[adj] = F.axis_mask[ax][ia[:, 0], ia[:, 1], ia[:, 2]]
            rest = np.nonzero(~adj)[0]
            if len(rest):
                free[rest] = self._segments_clear(P0[rest], P1[rest], o1)
            sel = np.nonzero(free)[0]
            em = self._pipes_mask(P0[sel], P1[sel]) & obs[sel]
            sel, em = sel[em != 0], em[em != 0]
            a, b = a[sel], b[sel]
            L = np.abs(self.pos[b, ax] - self.pos[a, ax])
            dp, dm = 2 * ax, 2 * ax + 1    # +축, −축 (AXIS_DIRS 순서)
            self.step[a, dp], self.elen[a, dp] = b, L
            self.step[b, dm], self.elen[b, dm] = a, L
            self.emask[a, dp] = em
            self.emask[b, dm] = swap_mask(em)
            if self._sup is not None:      # D57: 고정 레이어 인접 엣지는 캐시, 그 밖(패치)은 같은 함수로 계산
                adj = adj[sel]
                sc_ = np.empty(len(a))
                sp_ = np.zeros(len(a), dtype=np.int64)
                ia = Fidx[a[adj]]
                sc_[adj] = self._sup[0][ax][0][ia[:, 0], ia[:, 1], ia[:, 2]]
                sp_[adj] = self._sup[0][ax][1][ia[:, 0], ia[:, 1], ia[:, 2]]
                rest = np.nonzero(~adj)[0]
                sc_[rest], sp_[rest] = support_edge_model(self._ls, self.pos[a[rest]], self.pos[b[rest]])
                self.scost[a, dp] = sc_
                self.scost[b, dm] = sc_
                self.spack[a, dp] = sp_
                self.spack[b, dm] = reverse_support(sp_)
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
            if d in F.diag_mask:   # D65
                em &= F.diag_mask[d][src].astype(np.int64)
            free = em != 0
            s_id, d_id, em = s_id[free], d_id[free], em[free]
            src = tuple(i[free] for i in src)
            self.step[s_id, d] = d_id
            self.emask[s_id, d] = em
            self.elen[s_id, d] = np.linalg.norm(self.pos[d_id] - self.pos[s_id], axis=1)
            if self._sup is not None:
                self.scost[s_id, d] = self._sup[1][d][0][src]
                self.spack[s_id, d] = self._sup[1][d][1][src]
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
            su, ok = support_step(state.su, int(self.spack[state.node, d]), length, float(self.smax_dir[d]),
                                  float(self.bend_u[ANG_INDEX[defl]]) if self._sup is not None else 0.0)
            if not ok:   # D80 ⑨: 최대 간격 안에 서포트 가능 지점이 없다
                continue
            if defl:
                run, bend = length, defl
            else:
                run, bend = state.run + length, state.bend
            yield State(nxt, d, round(min(run, self.run_cap[bend]), 6), bend, nm, hz, su)

    def stats(self) -> dict:
        return {"pipe": self.pipe.id, "size": self.pipe.nominal_size, "grid": list(self.shape),
                "nodes": int(self.node_valid.sum()), "patch_nodes": self.n_patch_nodes, "edges": int(self.n_edges),
                "support_members": self.n_members,
                "build_sec": round(self.build_sec, 3), "fixed_sec": round(self.fixed_sec, 3),
                "grid_variant": self.grid.name, "fixed_nodes": int(self.fixed.node_ok.sum()),
                "fixed_grid": list(self.fixed.shape)}


def _route(g: LayeredGraph, pipe, time_limit, expand_limit=None) -> RouteResult:
    from .astar_fast import astar_route_generic
    t0 = time.perf_counter()
    h = heuristic_factory(g, pipe)
    status, chain, expanded, generated = astar_route_generic(g, pipe, time_limit, h(g.start_state(pipe)), expand_limit)
    if status != "ok":
        return RouteResult(pipe.id, status, expanded=expanded, generated=generated,
                           search_sec=time.perf_counter() - t0)
    return _finish_chain(g, pipe, chain, expanded, generated, time.perf_counter() - t0)


@dataclass(frozen=True)
class MemberOp:
    """D90 (a1) 서포트 부재 하나를 배치층 항목으로 — 직선 캡슐 (반경 = 앵글 반폭). nominal_size 는 엘보 반경용 자리
    채움 (부재는 두 점 직선이라 엘보가 없다)."""
    id: str
    radius: float
    nominal_size: str = "15A"


class SupportMembers:
    """D90 (a1): 놓인 배관의 서포트 부재 (검증기 Session — 전체 검증과 비트 동일, D85) 를 배치층 항목으로.
    시나리오마다 Session 하나를 놓인 배관 집합에 맞춰 add/remove/replace 로 동기화한다 — rip-up 으로 들어낸 배관의
    부재는 사라진다 (해제). 놓인 배관이 없으면 Session 을 건드리지 않고 빈 목록 (단독 재라우팅·진단)."""

    def __init__(self):
        self.sc = None
        self.sess = None
        self.routes = {}

    def __call__(self, sc, placed) -> list:
        want = {p.id: [list(map(float, w)) for w in wps] for p, wps in placed if wps}
        if not want:
            return []
        from .verifier import Session
        if self.sc is not sc:
            self.sc, self.sess, self.routes = sc, Session(sc), {}
        for pid in [q for q in self.routes if q not in want]:
            self.sess.remove(pid)
            del self.routes[pid]
        for pid, w in want.items():
            if pid not in self.routes:
                self.sess.add(pid, w)
            elif self.routes[pid] != w:
                self.sess.replace(pid, w)
            self.routes[pid] = w
        out = []
        for pid, rep in self.sess.reports().items():
            for s in rep.supports:
                for m in s.get("members") or []:
                    out.append((MemberOp(pid, float(ANGLE_HALF[s["section"]])), [m["a"], m["b"]]))
        return out


def make_layered_router(local_lines: bool = False, support_cost: bool = True, reserve: bool = True,
                        grid: GridSpec = G0, incremental: bool = True, expand_limit="default",
                        diagnostic: bool = True, support_lines: bool = True, member_reserve: bool = True,
                        _members: "SupportMembers | None" = None):
    """2층 구조 라우터 슬롯 만들기. local_lines = 변형 B, support_cost = D57 추정 서포트 비용,
    reserve = D59 단자 예약 (아직 놓이지 않은 다른 배관의 단자 직진 구간을 피한다), grid = D66 격자 보강 변형,
    incremental = D67 증분 배치층 표시 (False = 매번 전체 재계산, 결과 같음 — 시험·시간 비교용),
    expand_limit = D72·D73 배관당 탐색 예산 (확장 상태 수, 기본 EXPAND_BUDGET). None = 예전 벽시계 time_limit (비교용, 비결정적).
    diagnostic = D74 진단용 라우터(예산 × DIAG_FACTOR)를 router.diagnostic 으로 단다 — 실패 분류에만 쓴다.
    member_reserve = D90 (a1) 놓인 배관의 서포트 부재를 피한다 (기본). 하위 라우터(진단·예약 없음)는 같은 Session 을 쓴다."""
    if expand_limit == "default":
        expand_limit = EXPAND_BUDGET
    if member_reserve and _members is None:
        _members = SupportMembers()

    def router(sc, pipe, time_limit, placed=()):
        reserved = terminal_reservations(sc, exclude={pipe.id} | {p.id for p, _ in placed}) if reserve else []
        members = _members(sc, placed) if member_reserve else []
        g = LayeredGraph(sc, pipe, placed, local_lines=local_lines, support_cost=support_cost, reserved=reserved,
                         grid=grid, incremental=incremental, support_lines=support_lines, members=members)
        r = _route(g, pipe, time_limit, expand_limit)
        r.graph_sec = g.build_sec
        r.graph_stats = g.stats()
        return r
    router.__name__ = (f"layered_router_{'b' if local_lines else 'a'}{'' if support_cost else '_nosup'}"
                       f"{'' if reserve else '_noreserve'}{'' if grid == G0 else '_' + grid.name}"
                       f"{'' if incremental else '_full'}{'' if support_lines else '_nosuplines'}"
                       f"{'' if member_reserve else '_nomembers'}")
    router.grid = grid
    router.support_lines = support_lines
    router.search_budget = ({"kind": "expansions", "limit": expand_limit} if expand_limit is not None
                            else {"kind": "wall_clock_sec", "limit": None})
    if reserve:   # 실패 분류에서 "예약끼리 충돌" 을 가려내는 데 쓴다 (multi)
        router.without_reservations = make_layered_router(local_lines, support_cost, reserve=False, grid=grid,
                                                          incremental=incremental, expand_limit=expand_limit,
                                                          diagnostic=False, support_lines=support_lines,
                                                          member_reserve=member_reserve, _members=_members)
    if diagnostic and expand_limit is not None:   # D74: 진단 단독 재라우팅 (분류에만, 결과에는 쓰지 않는다)
        router.diagnostic = make_layered_router(local_lines, support_cost, reserve=reserve, grid=grid,
                                                incremental=incremental, expand_limit=expand_limit * DIAG_FACTOR,
                                                diagnostic=False, support_lines=support_lines,
                                                member_reserve=member_reserve, _members=_members)
        router.search_budget["diagnosis_limit"] = expand_limit * DIAG_FACTOR
    return router


layered_router_a = make_layered_router(False, True)   # 기본 (D55·D57)
layered_router_b = make_layered_router(True, True)    # 비교용 (D55)
