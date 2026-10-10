"""M10 가속: escape graph 전용 A* 의 컴파일(numba) 구현 — 결과 불변 (D40①).

router_astar.astar_route 의 파이썬 구현과 **같은 탐색**을 한다:
  같은 상태 (노드, 방향, 직진 길이, 직전 편향각), 같은 비용·휴리스틱 부동소수 연산 순서,
  같은 우선순위 (f, 삽입 순번), 같은 지배 가지치기, 같은 이웃 순서.
따라서 확장 순서·확장 수·반환 경로·J 가 파이썬 구현과 비트 단위로 같다 (tests/test_astar_fast.py).

정확히 같게 만들기 위한 장치
  - 엣지 길이·휴리스틱 거리는 파이썬 math.dist 로 미리 계산한 값을 표로 넘긴다 (numpy/numba sqrt 와 1ulp 차이 방지)
  - 직진 길이 반올림 round(x, 6) 은 파이썬과 같은 "정확한 십진 반올림(half-even)" 을 오차 없는 곱(Dekker)으로 구현
  - D48 엘보 호 정밀 판정은 그래프(EscapeGraph.elbow_clear)를 그대로 쓴다 — 탐색이 처음 만나는 (노드, 방향, 방향)만
    objmode 로 파이썬을 불러 계산하고 캐시한다 (지연 계산). 빠른 통과 판정(이격 ≥ r + t)과 호 AABB 거르기는
    elbow_clear 와 같은 부동소수 연산으로 컴파일 쪽에서 한다
  - 시간 확인은 파이썬과 같이 확장 2048회마다 (D40② timeout 판정)
"""
import math
import time

import numpy as np

try:
    import numba
    from numba import njit, types
    from numba.typed import Dict
    HAVE_NUMBA = True
except ImportError:   # numba 가 없으면 파이썬 구현을 쓴다 (결과는 같다)
    HAVE_NUMBA = False

from .constants import PIPE_SPECS, FITTINGS, elbow_kg
from .space import ALLOWED_DEFLECTIONS, DEFLECTION, DIR_INDEX, DIRS, State

EPS = 1e-6
CHECK_EVERY = 2048           # router_astar._CHECK_EVERY 와 같다
ANG = (0, 45, 90, 135)       # 편향각 인덱스

# 반환 코드
_FOUND, _EMPTY, _CHECK_TIME = 1, 0, 3

if HAVE_NUMBA:
    @njit(cache=True)
    def _round6(x):
        """파이썬 round(x, 6) 과 같은 값 (x ≥ 0, 유한, x·1e6 < 2^52).

        정확한 값 x·10^6 = p + err (Dekker 오차 없는 곱) 을 가장 가까운 정수(동률이면 짝수)로 반올림한 뒤 10^6 으로
        나눈다 — CPython 은 x 를 정확한 십진 반올림(half-even)한 문자열을 다시 double 로 읽으므로 같은 값이다.
        """
        m = 1e6
        p = x * m
        split = 134217729.0
        c = split * x
        xh = c - (c - x)
        xl = x - xh
        c = split * m
        mh = c - (c - m)
        ml = m - mh
        err = ((xh * mh - p) + xh * ml + xl * mh) + xl * ml
        r = math.floor(p)
        d = (p - r) - 0.5
        if d > -err:
            n = r + 1.0
        elif d < -err:
            n = r
        else:
            n = r if (r % 2.0 == 0.0) else r + 1.0
        return n / m

    @njit(cache=True)
    def _search(T, C, G, ctr, fctr, H, S, K, open_head, closed_head, arc):
        """A* 본체. 탐색 상태는 호출 사이에 유지된다 (배열 + ctr/fctr). 반환: 코드.

        T = 그래프 표, C = 상수, G = 엘보 호 거르기 표.
        H = 우선순위 큐 (f, 삽입 순번, g, 상태 id) 배열 — (f, 순번) 최소 힙. 순번이 유일하므로 꺼내는 순서는
            파이썬 heapq 와 같다
        S = 상태 배열 (노드 평탄 인덱스, 방향, 직진 길이, 편향각 인덱스, g_best, 부모, 같은 (노드, 방향) 다음 상태)
        K = 확정 목록 (run, bend, g, 다음) — (노드, 방향)별 연결 리스트, 머리는 closed_head
        open_head[(노드, 방향)] = 그 (노드, 방향)의 첫 상태 id — 파이썬 g_best 딕셔너리의 키 (노드, 방향, run, bend) 조회와 같다
        ctr: [expanded, generated, tie, pending_id, heap_size, found_id, n_states, n_closed]
        fctr: [pending_g]
        """
        (axis_ok, diag_ok, diag_to_a, diag_to_b, diag_axes, len_ax, diag_len, axes_x, axes_y, axes_z,
         hdist, clearance, pipe_margin, end_node) = T
        (defl_idx, dir_vec, need_m, run_cap, goal_need_m, thr, ekg, kgpm, kgmm, e45, end_dir, check_every,
         goal_x, goal_y, goal_z) = C
        hf, ht, hg, hid = H
        s_node, s_dir, s_run, s_bend, s_g, s_par, s_next = S
        c_run, c_bend, c_g, c_next = K
        ny = axes_y.shape[0]
        nz = axes_z.shape[0]
        end_flat = (end_node[0] * ny + end_node[1]) * nz + end_node[2]
        while True:
            # 배열 여유: 한 번 펼칠 때 상태·큐는 최대 18개, 확정 목록은 1개 늘어난다
            if ctr[4] + 18 > hf.shape[0] or ctr[6] + 18 > s_node.shape[0] or ctr[7] + 1 > c_run.shape[0]:
                return 4
            if ctr[3] >= 0:
                cur = ctr[3]
                g = fctr[0]
            else:
                if ctr[4] == 0:
                    return 0
                # pop (f, 순번) 최소
                g = hg[0]
                cur = hid[0]
                n = ctr[4] - 1
                ctr[4] = n
                if n > 0:
                    lf, lt, lg, li = hf[n], ht[n], hg[n], hid[n]
                    pos = 0
                    while True:
                        c = 2 * pos + 1
                        if c >= n:
                            break
                        if c + 1 < n and (hf[c + 1] < hf[c] or (hf[c + 1] == hf[c] and ht[c + 1] < ht[c])):
                            c += 1
                        if hf[c] < lf or (hf[c] == lf and ht[c] < lt):
                            hf[pos], ht[pos], hg[pos], hid[pos] = hf[c], ht[c], hg[c], hid[c]
                            pos = c
                        else:
                            break
                    hf[pos], ht[pos], hg[pos], hid[pos] = lf, lt, lg, li
                if g > s_g[cur] + 1e-9:
                    continue
                cflat = s_node[cur]
                cd = s_dir[cur]
                crun, cb = s_run[cur], s_bend[cur]
                ckey = cflat * 18 + cd
                dom = False
                h = closed_head[ckey]
                while h >= 0:
                    if c_run[h] >= crun and c_bend[h] <= cb and c_g[h] <= g + 1e-9:
                        dom = True
                        break
                    h = c_next[h]
                if dom:
                    continue
                m = ctr[7]
                c_run[m], c_bend[m], c_g[m], c_next[m] = crun, cb, g, closed_head[ckey]
                closed_head[ckey] = m
                ctr[7] = m + 1
                ctr[0] += 1
                if cflat == end_flat and cd == end_dir and crun >= goal_need_m[cb]:
                    ctr[5] = cur
                    return 1
                if ctr[0] % check_every == 0:
                    ctr[3] = cur
                    fctr[0] = g
                    return 3
            ctr[3] = -1
            cflat = s_node[cur]
            cd = s_dir[cur]
            crun, cb = s_run[cur], s_bend[cur]
            ci = cflat // (ny * nz)
            cj = (cflat // nz) % ny
            ck = cflat % nz
            for d in range(18):
                di = defl_idx[cd, d]
                if di < 0:
                    continue
                if di > 0 and crun < need_m[cb, di]:
                    continue
                ok, ni, nj, nk = _step(axis_ok, diag_ok, diag_to_a, diag_to_b, diag_axes, ci, cj, ck, d)
                if not ok:
                    continue
                if di > 0:
                    if not (clearance[ci, cj, ck] >= thr[di] and pipe_margin[ci, cj, ck] >= thr[di]):
                        k = (cflat * 18 + cd) * 18 + d
                        if k in arc:
                            ok_arc = arc[k]
                        else:
                            ok_arc = _arc_prefilter(G, cd, d, axes_x[ci], axes_y[cj], axes_z[ck])
                            if ok_arc < 0:   # 정밀 판정 필요 → 그래프의 elbow_clear (파이썬) 를 그대로 부른다
                                with numba.objmode(res="int64"):
                                    res = _arc_callback(k)
                                ok_arc = res
                            arc[k] = np.int8(ok_arc)
                        if ok_arc == 0:
                            continue
                if d < 6:
                    ax = d // 2
                    if ax == 0:
                        lo = ci if ni > ci else ni
                    elif ax == 1:
                        lo = cj if nj > cj else nj
                    else:
                        lo = ck if nk > ck else nk
                    length = len_ax[ax, lo]
                else:
                    a0 = diag_axes[d, 0]
                    if a0 == 0:
                        t = abs(axes_x[ni] - axes_x[ci])
                    elif a0 == 1:
                        t = abs(axes_y[nj] - axes_y[cj])
                    else:
                        t = abs(axes_z[nk] - axes_z[ck])
                    length = diag_len[t * 4 + diag_axes[d, 2]]
                if di > 0:
                    run = length
                    nb = di
                else:
                    run = crun + length
                    nb = cb
                cap = run_cap[nb]
                run = _round6(run if run <= cap else cap)
                ng = g + (length / 1000 * kgpm + ekg[di])
                nflat = (ni * ny + nj) * nz + nk
                nkey = nflat * 18 + d
                sid = open_head[nkey]
                while sid >= 0:
                    if s_run[sid] == run and s_bend[sid] == nb:
                        break
                    sid = s_next[sid]
                if sid >= 0 and not (ng + 1e-9 < s_g[sid]):
                    continue
                dom = False
                h = closed_head[nkey]
                while h >= 0:
                    if c_run[h] >= run and c_bend[h] <= nb and c_g[h] <= ng + 1e-9:
                        dom = True
                        break
                    h = c_next[h]
                if dom:
                    continue
                if sid < 0:
                    sid = ctr[6]
                    ctr[6] = sid + 1
                    s_node[sid], s_dir[sid], s_run[sid], s_bend[sid] = nflat, d, run, nb
                    s_next[sid] = open_head[nkey]
                    open_head[nkey] = sid
                s_g[sid] = ng
                s_par[sid] = cur
                # 휴리스틱 (router_astar.heuristic_factory 와 같은 식)
                dist = hdist[ni, nj, nk]
                if d != end_dir:
                    bends = 1
                elif dist == 0:
                    bends = 0
                else:
                    px, py, pz = axes_x[ni], axes_y[nj], axes_z[nk]
                    v0, v1, v2 = goal_x - px, goal_y - py, goal_z - pz
                    e0, e1, e2 = dir_vec[d, 0], dir_vec[d, 1], dir_vec[d, 2]
                    tt = (0 + v0 * e0 + v1 * e1 + v2 * e2) / (e0 * e0 + e1 * e1 + e2 * e2)
                    on_ray = tt > 0 and abs(v0 - tt * e0) < 1e-6 and abs(v1 - tt * e1) < 1e-6 \
                        and abs(v2 - tt * e2) < 1e-6
                    bends = 0 if on_ray else 2
                f = ng + (dist * kgmm + bends * e45)
                # push
                pos = ctr[4]
                ctr[4] = pos + 1
                tie = ctr[2]
                while pos > 0:
                    par = (pos - 1) >> 1
                    if f < hf[par] or (f == hf[par] and tie < ht[par]):
                        hf[pos], ht[pos], hg[pos], hid[pos] = hf[par], ht[par], hg[par], hid[par]
                        pos = par
                    else:
                        break
                hf[pos], ht[pos], hg[pos], hid[pos] = f, tie, ng, sid
                ctr[2] += 1
                ctr[1] += 1

    @njit(cache=True)
    def _arc_prefilter(G, din, dout, vx, vy, vz):
        """EscapeGraph.elbow_clear 의 첫 거르기를 같은 부동소수 연산으로: 호 AABB 가까이에 장애물도 놓인 배관 조각도
        없으면 통과(1). 가까운 것이 있으면 −1 (정밀 판정은 파이썬)."""
        tpl_lo, tpl_hi, tpl_sag, box_lo, box_hi, r, pg_lo, pg_hi, pg_R, pg_S = G
        sag = tpl_sag[din, dout]
        plo0 = vx + tpl_lo[din, dout, 0]
        plo1 = vy + tpl_lo[din, dout, 1]
        plo2 = vz + tpl_lo[din, dout, 2]
        phi0 = vx + tpl_hi[din, dout, 0]
        phi1 = vy + tpl_hi[din, dout, 1]
        phi2 = vz + tpl_hi[din, dout, 2]
        lim = r + sag
        for b in range(box_lo.shape[0]):
            g0 = max(0.0, max(box_lo[b, 0] - phi0, plo0 - box_hi[b, 0]))
            g1 = max(0.0, max(box_lo[b, 1] - phi1, plo1 - box_hi[b, 1]))
            g2 = max(0.0, max(box_lo[b, 2] - phi2, plo2 - box_hi[b, 2]))
            if math.sqrt(g0 * g0 + g1 * g1 + g2 * g2) < lim:
                return -1
        for q in range(pg_lo.shape[0]):
            g = r + sag + pg_R[q] + pg_S[q]
            if (pg_lo[q, 0] < phi0 + g and pg_lo[q, 1] < phi1 + g and pg_lo[q, 2] < phi2 + g
                    and pg_hi[q, 0] > plo0 - g and pg_hi[q, 1] > plo1 - g and pg_hi[q, 2] > plo2 - g):
                return -1
        return 1

    @njit(cache=True, inline="always")
    def _step(axis_ok, diag_ok, diag_to_a, diag_to_b, diag_axes, i, j, k, d):
        """EscapeGraph._step 과 같다. 반환 (가능, i, j, k)."""
        if d < 6:
            ax = d // 2
            s = 1 if d % 2 == 0 else -1
            si, sj, sk = i, j, k
            if s < 0:
                if ax == 0:
                    si -= 1
                elif ax == 1:
                    sj -= 1
                else:
                    sk -= 1
                if (si if ax == 0 else (sj if ax == 1 else sk)) < 0:
                    return False, i, j, k
            if not axis_ok[ax, si, sj, sk]:
                return False, i, j, k
            if ax == 0:
                return True, i + s, j, k
            if ax == 1:
                return True, i, j + s, k
            return True, i, j, k + s
        if not diag_ok[d - 6, i, j, k]:
            return False, i, j, k
        a = diag_axes[d, 0]
        b = diag_axes[d, 1]
        idx = (i, j, k)
        ia = idx[a]
        ib = idx[b]
        ta = diag_to_a[d - 6, ia, ib]
        tb = diag_to_b[d - 6, ia, ib]
        o0, o1, o2 = i, j, k
        if a == 0:
            o0 = ta
        elif a == 1:
            o1 = ta
        else:
            o2 = ta
        if b == 0:
            o0 = tb
        elif b == 1:
            o1 = tb
        else:
            o2 = tb
        return True, o0, o1, o2


# ---------------------------------------------------------------- 파이썬 쪽: 표 만들기 · 재개 루프


def _tables(space, pipe):
    """EscapeGraph → 컴파일 탐색용 표. 거리·길이는 파이썬 math.dist 로 계산해 파이썬 구현과 같은 값을 쓴다."""
    nx, ny, nz = space.shape
    axis_ok = np.stack(space.axis_ok).astype(np.bool_)
    diag_ok = np.stack([space.diag_ok[d] for d in range(6, 18)]).astype(np.bool_)
    m = max(space.shape)
    diag_to_a = np.full((12, m, m), -1, dtype=np.int64)
    diag_to_b = np.full((12, m, m), -1, dtype=np.int64)
    diag_axes = np.zeros((18, 3), dtype=np.int64)
    for d in range(6, 18):
        vec = DIRS[d]
        a, b = [ax for ax in range(3) if vec[ax] != 0]
        diag_axes[d] = (a, b, 3 - a - b)
        if d in space.diag_to:
            ta, tb = space.diag_to[d]
            diag_to_a[d - 6, :ta.shape[0], :ta.shape[1]] = ta
            diag_to_b[d - 6, :tb.shape[0], :tb.shape[1]] = tb
    len_ax = np.zeros((3, m), dtype=np.float64)
    for ax in range(3):
        A = space._axes_f[ax]
        for i in range(len(A) - 1):
            p, q = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
            p[ax], q[ax] = A[i], A[i + 1]
            len_ax[ax, i] = math.dist(p, q)
    diag_len = Dict.empty(types.float64, types.float64)
    for d in range(6, 18):
        a, b, c = diag_axes[d]
        to_a, to_b = space.diag_to.get(d, (np.zeros((0, 0), int), None))
        ia, ib = np.nonzero(to_a >= 0)
        A = space.axes[a]
        for t in np.unique(np.abs(A[to_a[ia, ib]] - A[ia])) if len(ia) else ():
            t = float(t)
            p = [0.0, 0.0, 0.0]
            p[a], p[b] = t, t
            diag_len[t * 4 + int(c)] = math.dist((0.0, 0.0, 0.0), p)
    # 휴리스틱 거리 (노드만)
    goal = tuple(map(float, pipe.end.pos))
    hdist = np.full(space.shape, np.nan)
    X, Y, Z = space._axes_f
    for i, j, k in zip(*np.nonzero(space.node_ok)):
        hdist[i, j, k] = math.dist((X[i], Y[j], Z[k]), goal)
    clearance = np.ascontiguousarray(space.clearance, dtype=np.float64)
    pipe_margin = np.ascontiguousarray(space.pipe_margin, dtype=np.float64)
    T = (axis_ok, diag_ok, diag_to_a, diag_to_b, diag_axes, len_ax, diag_len,
         np.asarray(space.axes[0], dtype=np.float64), np.asarray(space.axes[1], dtype=np.float64),
         np.asarray(space.axes[2], dtype=np.float64), hdist, clearance, pipe_margin,
         np.array(space.end_node, dtype=np.int64))
    _, C = _consts(space, pipe, goal)
    return T, C


_CURRENT = {}


def _arc_callback(k):
    """컴파일 탐색 안에서 objmode 로 부른다: 키 → EscapeGraph.elbow_clear (정밀 판정, 결과 캐시)."""
    space, ny, nz = _CURRENT["space"], _CURRENT["ny"], _CURRENT["nz"]
    d_out = k % 18
    d_in = (k // 18) % 18
    flat = k // 324
    node = (flat // (ny * nz), (flat // nz) % ny, flat % nz)
    return 1 if space.elbow_clear(node, d_in, d_out) else 0


def _arc_tables(space):
    tpl_lo = np.zeros((18, 18, 3))
    tpl_hi = np.zeros((18, 18, 3))
    tpl_sag = np.zeros((18, 18))
    for i in range(18):
        for j in range(18):
            if DEFLECTION[i][j] in (45, 90, 135):
                _, _, _, _, sag, plo0, phi0 = space.arc_template(i, j)
                tpl_lo[i, j], tpl_hi[i, j], tpl_sag[i, j] = plo0, phi0, sag
    if space.pipe_groups:
        pg = (np.ascontiguousarray(space._pg_lo), np.ascontiguousarray(space._pg_hi),
              np.ascontiguousarray(space._pg_R, dtype=np.float64), np.ascontiguousarray(space._pg_S, dtype=np.float64))
    else:
        pg = (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros(0))
    return (tpl_lo, tpl_hi, tpl_sag, np.ascontiguousarray(space.box_lo), np.ascontiguousarray(space.box_hi),
            float(space.r)) + pg


def _consts(space, pipe, goal):
    """탐색 상수 (편향각·직관 길이 문턱·비용 계수). 두 구현이 같이 쓴다."""
    defl_idx = np.full((18, 18), -1, dtype=np.int64)
    for i in range(18):
        for j in range(18):
            if DEFLECTION[i][j] in ALLOWED_DEFLECTIONS:
                defl_idx[i, j] = ANG.index(DEFLECTION[i][j])
    need_m = np.zeros((4, 4))
    for bi, b in enumerate(ANG):
        for di, dfl in enumerate(ANG):
            if dfl:
                need_m[bi, di] = space.turn_need(State(space.start_node, 0, 0.0, b), dfl) - EPS
    run_cap = np.array([space.run_cap[a] for a in ANG], dtype=np.float64)
    goal_need_m = np.zeros(4)
    for bi, b in enumerate(ANG):
        need = space.tangent[b]
        if b and space.end_boundary:
            need += space.r
        goal_need_m[bi] = need - EPS
    thr = np.array([space.r + space.tangent[a] + EPS for a in ANG])
    ekg = np.array([elbow_kg(pipe.nominal_size, a) for a in ANG])
    kgpm = space.kg_per_m
    kgmm = PIPE_SPECS[pipe.nominal_size].kg_per_m / 1000
    e45 = FITTINGS[pipe.nominal_size].elbow45
    C = (defl_idx, np.array(DIRS, dtype=np.int64), need_m, run_cap, goal_need_m, thr, ekg, float(kgpm), float(kgmm),
         float(e45), int(DIR_INDEX[tuple(pipe.end.dir)]), int(CHECK_EVERY), goal[0], goal[1], goal[2])
    return None, C


def _grow(arrs, need):
    """배열 묶음을 need 이상으로 (두 배씩) 늘린다."""
    cap = len(arrs[0])
    while cap < need:
        cap *= 2
    return tuple(np.concatenate([a, np.zeros(cap - len(a), a.dtype)]) for a in arrs)


def astar_route_fast(space, pipe, time_limit, h_start):
    """router_astar.astar_route 와 같은 결과를 내는 컴파일 구현. 반환: (status, goal_state_chain | None, expanded, generated)."""
    t0 = time.perf_counter()
    T, C = _tables(space, pipe)
    G = _arc_tables(space)
    nx, ny, nz = space.shape
    _CURRENT.update(space=space, ny=ny, nz=nz)
    s0 = space.start_state(pipe)
    cap = 1 << 16
    H = (np.zeros(cap), np.zeros(cap, np.int64), np.zeros(cap), np.zeros(cap, np.int64))
    S = (np.zeros(cap, np.int64), np.zeros(cap, np.int64), np.zeros(cap), np.zeros(cap, np.int64), np.zeros(cap),
         np.zeros(cap, np.int64), np.zeros(cap, np.int64))
    K = (np.zeros(cap), np.zeros(cap, np.int64), np.zeros(cap), np.zeros(cap, np.int64))
    open_head = np.full(nx * ny * nz * 18, -1, dtype=np.int32)
    closed_head = np.full(nx * ny * nz * 18, -1, dtype=np.int32)
    arc = Dict.empty(types.int64, types.int8)
    n0 = s0.node
    f0 = (n0[0] * ny + n0[1]) * nz + n0[2]
    S[0][0], S[1][0], S[2][0], S[3][0], S[4][0], S[5][0], S[6][0] = f0, s0.dir, s0.run, ANG.index(s0.bend), 0.0, -1, -1
    open_head[f0 * 18 + s0.dir] = 0
    H[0][0], H[1][0], H[2][0], H[3][0] = h_start, 0, 0.0, 0
    ctr = np.array([0, 0, 1, -1, 1, -1, 1, 0], dtype=np.int64)
    fctr = np.zeros(1)
    while True:
        code = _search(T, C, G, ctr, fctr, H, S, K, open_head, closed_head, arc)
        if code == 4:
            H = _grow(H, ctr[4] + 18)
            S = _grow(S, ctr[6] + 18)
            K = _grow(K, ctr[7] + 1)
            continue
        if code == _CHECK_TIME:
            if time.perf_counter() - t0 > time_limit:
                return "timeout", None, int(ctr[0]), int(ctr[1])
            continue
        if code == _EMPTY:
            return "unreachable", None, int(ctr[0]), int(ctr[1])
        # _FOUND: 상태 사슬 복원
        chain = []
        i = int(ctr[5])
        while i >= 0:
            f = int(S[0][i])
            node = (f // (ny * nz), (f // nz) % ny, f % nz)
            chain.append(State(node, int(S[1][i]), float(S[2][i]), ANG[int(S[3][i])]))
            i = int(S[5][i])
        chain.reverse()
        return "ok", chain, int(ctr[0]), int(ctr[1])


# ---------------------------------------------------------------- 범용 그래프(이웃 표) 탐색 — M18 대안 표현 실험용
#
# 노드 = 정수 id, step[n, d] = 방향 d 로 가는 다음 노드 (−1 = 없음), elen[n, d] = 그 엣지 길이.
# 상태·비용·휴리스틱·지배 가지치기·엘보 규칙은 위 구현과 같다. (대안 표현은 기본 경로와 비트 일치를 요구하지 않는다)

if HAVE_NUMBA:
    @njit(cache=True)
    def _search_generic(T, C, G, ctr, fctr, H, S, K, open_head, closed_head, arc):
        """이웃 표 그래프용 A* 본체 (_search 와 같은 구조, 노드 = 정수 id). 탐색 상태는 호출 사이에 유지된다 (배열 + ctr/fctr). 반환: 코드.

        T = 그래프 표, C = 상수, G = 엘보 호 거르기 표.
        H = 우선순위 큐 (f, 삽입 순번, g, 상태 id) 배열 — (f, 순번) 최소 힙. 순번이 유일하므로 꺼내는 순서는
            파이썬 heapq 와 같다
        S = 상태 배열 (노드 평탄 인덱스, 방향, 직진 길이, 편향각 인덱스, g_best, 부모, 같은 (노드, 방향) 다음 상태)
        K = 확정 목록 (run, bend, g, 다음) — (노드, 방향)별 연결 리스트, 머리는 closed_head
        open_head[(노드, 방향)] = 그 (노드, 방향)의 첫 상태 id — 파이썬 g_best 딕셔너리의 키 (노드, 방향, run, bend) 조회와 같다
        ctr: [expanded, generated, tie, pending_id, heap_size, found_id, n_states, n_closed]
        fctr: [pending_g]
        """
        (step, elen, scost, px, py, pz, hdist, clearance, pipe_margin, end_flat) = T
        (defl_idx, dir_vec, need_m, run_cap, goal_need_m, thr, ekg, kgpm, kgmm, e45, end_dir, check_every,
         goal_x, goal_y, goal_z) = C
        hf, ht, hg, hid = H
        s_node, s_dir, s_run, s_bend, s_g, s_par, s_next = S
        c_run, c_bend, c_g, c_next = K
        while True:
            # 배열 여유: 한 번 펼칠 때 상태·큐는 최대 18개, 확정 목록은 1개 늘어난다
            if ctr[4] + 18 > hf.shape[0] or ctr[6] + 18 > s_node.shape[0] or ctr[7] + 1 > c_run.shape[0]:
                return 4
            if ctr[3] >= 0:
                cur = ctr[3]
                g = fctr[0]
            else:
                if ctr[4] == 0:
                    return 0
                # pop (f, 순번) 최소
                g = hg[0]
                cur = hid[0]
                n = ctr[4] - 1
                ctr[4] = n
                if n > 0:
                    lf, lt, lg, li = hf[n], ht[n], hg[n], hid[n]
                    pos = 0
                    while True:
                        c = 2 * pos + 1
                        if c >= n:
                            break
                        if c + 1 < n and (hf[c + 1] < hf[c] or (hf[c + 1] == hf[c] and ht[c + 1] < ht[c])):
                            c += 1
                        if hf[c] < lf or (hf[c] == lf and ht[c] < lt):
                            hf[pos], ht[pos], hg[pos], hid[pos] = hf[c], ht[c], hg[c], hid[c]
                            pos = c
                        else:
                            break
                    hf[pos], ht[pos], hg[pos], hid[pos] = lf, lt, lg, li
                if g > s_g[cur] + 1e-9:
                    continue
                cflat = s_node[cur]
                cd = s_dir[cur]
                crun, cb = s_run[cur], s_bend[cur]
                ckey = cflat * 18 + cd
                dom = False
                h = closed_head[ckey]
                while h >= 0:
                    if c_run[h] >= crun and c_bend[h] <= cb and c_g[h] <= g + 1e-9:
                        dom = True
                        break
                    h = c_next[h]
                if dom:
                    continue
                m = ctr[7]
                c_run[m], c_bend[m], c_g[m], c_next[m] = crun, cb, g, closed_head[ckey]
                closed_head[ckey] = m
                ctr[7] = m + 1
                ctr[0] += 1
                if cflat == end_flat and cd == end_dir and crun >= goal_need_m[cb]:
                    ctr[5] = cur
                    return 1
                if ctr[0] % check_every == 0:
                    ctr[3] = cur
                    fctr[0] = g
                    return 3
            ctr[3] = -1
            cflat = s_node[cur]
            cd = s_dir[cur]
            crun, cb = s_run[cur], s_bend[cur]
            for d in range(18):
                di = defl_idx[cd, d]
                if di < 0:
                    continue
                if di > 0 and crun < need_m[cb, di]:
                    continue
                nflat = step[cflat, d]
                if nflat < 0:
                    continue
                if di > 0:
                    if not (clearance[cflat] >= thr[di] and pipe_margin[cflat] >= thr[di]):
                        k = (cflat * 18 + cd) * 18 + d
                        if k in arc:
                            ok_arc = arc[k]
                        else:
                            ok_arc = _arc_prefilter(G, cd, d, px[cflat], py[cflat], pz[cflat])
                            if ok_arc < 0:   # 정밀 판정 필요 → 그래프의 elbow_clear (파이썬) 를 그대로 부른다
                                with numba.objmode(res="int64"):
                                    res = _arc_callback_generic(k)
                                ok_arc = res
                            arc[k] = np.int8(ok_arc)
                        if ok_arc == 0:
                            continue
                length = elen[cflat, d]
                if di > 0:
                    run = length
                    nb = di
                else:
                    run = crun + length
                    nb = cb
                cap = run_cap[nb]
                run = _round6(run if run <= cap else cap)
                ng = g + (length / 1000 * kgpm + ekg[di] + scost[cflat, d])   # D57 추정 서포트 포함
                nkey = nflat * 18 + d
                sid = open_head[nkey]
                while sid >= 0:
                    if s_run[sid] == run and s_bend[sid] == nb:
                        break
                    sid = s_next[sid]
                if sid >= 0 and not (ng + 1e-9 < s_g[sid]):
                    continue
                dom = False
                h = closed_head[nkey]
                while h >= 0:
                    if c_run[h] >= run and c_bend[h] <= nb and c_g[h] <= ng + 1e-9:
                        dom = True
                        break
                    h = c_next[h]
                if dom:
                    continue
                if sid < 0:
                    sid = ctr[6]
                    ctr[6] = sid + 1
                    s_node[sid], s_dir[sid], s_run[sid], s_bend[sid] = nflat, d, run, nb
                    s_next[sid] = open_head[nkey]
                    open_head[nkey] = sid
                s_g[sid] = ng
                s_par[sid] = cur
                # 휴리스틱 (router_astar.heuristic_factory 와 같은 식)
                dist = hdist[nflat]
                if d != end_dir:
                    bends = 1
                elif dist == 0:
                    bends = 0
                else:
                    v0, v1, v2 = goal_x - px[nflat], goal_y - py[nflat], goal_z - pz[nflat]
                    e0, e1, e2 = dir_vec[d, 0], dir_vec[d, 1], dir_vec[d, 2]
                    tt = (0 + v0 * e0 + v1 * e1 + v2 * e2) / (e0 * e0 + e1 * e1 + e2 * e2)
                    on_ray = tt > 0 and abs(v0 - tt * e0) < 1e-6 and abs(v1 - tt * e1) < 1e-6 \
                        and abs(v2 - tt * e2) < 1e-6
                    bends = 0 if on_ray else 2
                f = ng + (dist * kgmm + bends * e45)
                # push
                pos = ctr[4]
                ctr[4] = pos + 1
                tie = ctr[2]
                while pos > 0:
                    par = (pos - 1) >> 1
                    if f < hf[par] or (f == hf[par] and tie < ht[par]):
                        hf[pos], ht[pos], hg[pos], hid[pos] = hf[par], ht[par], hg[par], hid[par]
                        pos = par
                    else:
                        break
                hf[pos], ht[pos], hg[pos], hid[pos] = f, tie, ng, sid
                ctr[2] += 1
                ctr[1] += 1


def _arc_callback_generic(k):
    space = _CURRENT["space"]
    return 1 if space.elbow_clear(k // 324, (k // 18) % 18, k % 18) else 0


def astar_route_generic(space, pipe, time_limit, h_start):
    """이웃 표 그래프(space.step, space.elen, space.pos …)용 컴파일 A*. 반환은 astar_route_fast 와 같은 형태."""
    t0 = time.perf_counter()
    goal = tuple(map(float, pipe.end.pos))
    P = space.pos
    T = (space.step, space.elen, space.scost, np.ascontiguousarray(P[:, 0]), np.ascontiguousarray(P[:, 1]),
         np.ascontiguousarray(P[:, 2]), space.hdist, space.clearance, space.pipe_margin, np.int64(space.end_node))
    _, C = _consts(space, pipe, goal)
    G = _arc_tables(space)
    _CURRENT.update(space=space)
    s0 = space.start_state(pipe)
    cap = 1 << 16
    H = (np.zeros(cap), np.zeros(cap, np.int64), np.zeros(cap), np.zeros(cap, np.int64))
    S = (np.zeros(cap, np.int64), np.zeros(cap, np.int64), np.zeros(cap), np.zeros(cap, np.int64), np.zeros(cap),
         np.zeros(cap, np.int64), np.zeros(cap, np.int64))
    K = (np.zeros(cap), np.zeros(cap, np.int64), np.zeros(cap), np.zeros(cap, np.int64))
    N = len(P)
    open_head = np.full(N * 18, -1, dtype=np.int32)
    closed_head = np.full(N * 18, -1, dtype=np.int32)
    arc = Dict.empty(types.int64, types.int8)
    f0 = int(s0.node)
    S[0][0], S[1][0], S[2][0], S[3][0], S[4][0], S[5][0], S[6][0] = f0, s0.dir, s0.run, 0, 0.0, -1, -1
    open_head[f0 * 18 + s0.dir] = 0
    H[0][0], H[1][0], H[2][0], H[3][0] = h_start, 0, 0.0, 0
    ctr = np.array([0, 0, 1, -1, 1, -1, 1, 0], dtype=np.int64)
    fctr = np.zeros(1)
    while True:
        code = _search_generic(T, C, G, ctr, fctr, H, S, K, open_head, closed_head, arc)
        if code == 4:
            H = _grow(H, ctr[4] + 18)
            S = _grow(S, ctr[6] + 18)
            K = _grow(K, ctr[7] + 1)
            continue
        if code == _CHECK_TIME:
            if time.perf_counter() - t0 > time_limit:
                return "timeout", None, int(ctr[0]), int(ctr[1])
            continue
        if code == _EMPTY:
            return "unreachable", None, int(ctr[0]), int(ctr[1])
        chain = []
        i = int(ctr[5])
        while i >= 0:
            chain.append(State(int(S[0][i]), int(S[1][i]), float(S[2][i]), ANG[int(S[3][i])]))
            i = int(S[5][i])
        chain.reverse()
        return "ok", chain, int(ctr[0]), int(ctr[1])
