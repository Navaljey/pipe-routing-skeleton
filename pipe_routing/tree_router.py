"""S2-3 분기 계통 기준선 라우터 (D112 · D134) — 분기 1개 정확 풀이 + 자기 간섭 재시도 + 순차 근사.

정확 풀이 (D112 Q9 — 자기 간섭을 뺀 라우터 비용 최적 나무):
  ① 다리 표: 다리 단자 B 에서 다리 지도 위 한 점 전 탐색 (Dijkstra, 기존 컴파일 커널, 목표 없음). 합류 (in) 다리는
     흐름 그대로 B → 티, 갈래 (out) 다리는 흐름을 거꾸로 걷는다 (B → 티) — 엣지·엘보·서포트 u·부재 판정은 방향과
     무관하고, B 쪽 직관 규칙만 도착 규칙 (노즐 ≥ t, 경계 ≥ r_bnd + t) 으로 바꾼다. 티에 닿는 상태 중 직관 ≥ 3D_분기 + t
     (티 → 다리 첫 엘보 접점 ≥ 3D_분기) 인 것만 후보. teec[(주관 노드, 주관 방향)] = 분기각 ∈ {45, 90} (흐름 기준) 인 다리
     최소 비용.
  ② 주관 탐색: astar_tee._search_tee — 주관 A* 에 티 전이 (teec 를 더한다, 티 앞뒤 3D_주관 규칙은 bend 등급 "티").
  ③ 자기 간섭 (tree.family_conflicts — 검증기 collision 과 같은 함수) 이 있으면 그 (티 노드, 방향) 을 막고 ② 를 다시
     (최대 8 회, method exact_retry). 그래도 안 되면 순차 근사 (주관 단독 → 그 경로 위 티 후보 중 다리 표 비용 순으로
     자기 간섭 없는 첫 티, method sequential).
예산 (D112 Q6): 계통당 확장 상한 = 배관 상한 × (다리 수 + 1), 모든 탐색 (다리 표 · 주관 시도 · 순차 주관) 의 합.
분기 2개 이상 (헤더) 은 S2-4 — 지금은 순차 근사 (다리마다 다리 표, 큰 구경 → 먼 것 순, D112).
"""
import math
import time

import numpy as np

from . import astar_fast as af
from .astar_fast import ANG, HAVE_NUMBA, _arc_tables, _consts, _grow, _lmax, _DKIND, _CURRENT
from .constants import elbow_tangent
from .router_astar import RouteResult, _finish_chain, heuristic_factory
from .scenario import Pipe, Terminal
from .space import DEFLECTION, DIRS, State
from . import tree as T

EPS = 1e-6
MAX_RETRY = 8
INF = 1e308


def leg_search_pipe(pipe: Pipe, b) -> Pipe:
    """다리 표 탐색용 배관: 다리 구경·같은 서비스·class·보온, 출발 = 다리 단자 (in = 단자 dir 그대로, out = 도착 dir 반대).
    끝 = 출발과 같은 점 (목표 없는 탐색 — 쓰지 않는다)."""
    t = b.terminal
    d = tuple(t.dir) if b.flow == "in" else tuple(-c for c in t.dir)
    st = Terminal(t.pos, t.kind, d, t.host)
    p = Pipe(T.leg_id(pipe, b), pipe.type_id, b.nominal_size, pipe.gravity_pipe, pipe.min_slope,
             pipe.insulation_thickness, st, st, service=pipe.service, pipe_class=pipe.pipe_class)
    p.parent, p.leg = pipe.id, b
    return p


# --------------------------------------------------------------------------- ① 다리 표

def _state_arrays(cap, tee=False):
    S = [np.zeros(cap, np.int64), np.zeros(cap, np.int64), np.zeros(cap), np.zeros(cap, np.int64), np.zeros(cap),
         np.zeros(cap, np.int64), np.zeros(cap, np.int64), np.zeros(cap, np.int64), np.zeros(cap), np.zeros(cap)]
    K = [np.zeros(cap), np.zeros(cap, np.int64), np.zeros(cap), np.zeros(cap, np.int64), np.zeros(cap, np.int64),
         np.zeros(cap), np.zeros(cap)]
    if tee:
        S.append(np.zeros(cap, np.int64))
        K.append(np.zeros(cap, np.int64))
    H = (np.zeros(cap), np.zeros(cap, np.int64), np.zeros(cap), np.zeros(cap, np.int64))
    return H, tuple(S), tuple(K)


def leg_table(g, lp: Pipe, b, budget: int, bound_keys=None):
    """다리 단자에서 한 점 전 탐색. 반환 dict: status (done/bounded/budget), expanded, cand = {(노드, 도착 방향): (g, 상태 id)},
    S (상태 배열 — 사슬 복원용).
    bound_keys = 주관 단독 경로 위 티 후보의 다리 지도 키 (노드·18 + 도착 방향) 집합 — 주면 그 키에 닿은 가장 싼 다리 비용
    c 를 찾은 뒤 g > c 인 상태를 꺼낼 차례가 되면 멈춘다 (bounded). 정확: 정확 나무의 다리 비용 ≤ (순차 나무 비용 −
    주관 단독 비용) = c — 티가 든 주관 ≥ 주관 단독 (제약만 더함)."""
    if not HAVE_NUMBA:
        raise RuntimeError("분기 계통 라우터는 numba 가 필요하다")
    P = g.pos
    N = len(P)
    zeros = np.zeros(N)
    Tt = (g.step, g.elen, g.scost, g.emask, np.ascontiguousarray(P[:, 0]), np.ascontiguousarray(P[:, 1]),
          np.ascontiguousarray(P[:, 2]), zeros, g.clearance, g.pipe_margin, np.int64(-1), _DKIND,
          np.array([1.0 if lp.gravity_pipe else 0.0, _lmax(lp), 0.0]), g.spack, g.smax_dir, g.bend_u, np.zeros(1))
    _, C = _consts(g, lp, tuple(map(float, lp.start.pos)))
    C = list(C)
    need_m = C[2].copy()
    tan = {a: elbow_tangent(lp.nominal_size, a) for a in ANG}
    if b.flow == "out":   # 거꾸로 걷기: B 쪽 = 도착 규칙 (D45 ③, D47·D96) — 노즐 ≥ t, 경계 ≥ r_bnd + t
        for di, a in enumerate(ANG):
            if a:
                need_m[0, di] = (tan[a] + (g.r_bnd if g.start_boundary else 0.0)) - EPS
    C[2] = need_m
    need_tee = np.array([T.three_d(lp.nominal_size) + tan[a] for a in ANG])
    C[3] = np.maximum(C[3], need_tee)           # run_cap 이 티 조건에 닿게
    C[9] = 0.0                                   # e45 = 0 (휴리스틱 0 → Dijkstra)
    C[10] = -1                                   # end_dir 없음
    if bound_keys is not None:
        C[11] = 1024                             # 확인 주기만 바꾼다 (탐색 순서 불변)
    C = tuple(C)
    G = _arc_tables(g)
    _CURRENT.update(space=g)
    s0 = g.start_state(lp)
    H, S, K = _state_arrays(1 << 16)
    open_head = np.full(N * 18, -1, dtype=np.int32)
    closed_head = np.full(N * 18, -1, dtype=np.int32)
    arc = af.Dict.empty(af.types.int64, af.types.int8)
    f0 = int(s0.node)
    S[0][0], S[1][0], S[2][0], S[3][0], S[4][0], S[5][0], S[6][0], S[7][0], S[8][0], S[9][0] = (
        f0, s0.dir, s0.run, 0, 0.0, -1, -1, 15, 0.0, 0.0)
    open_head[f0 * 18 + s0.dir] = 0
    H[0][0], H[1][0], H[2][0], H[3][0] = 0.0, 0, 0.0, 0
    ctr = np.array([0, 0, 1, -1, 1, -1, 1, 0], dtype=np.int64)
    fctr = np.zeros(1)
    status = "done"
    keys = None if bound_keys is None else np.fromiter(bound_keys, dtype=np.int64)
    c_hat = INF
    scanned = 0
    while True:
        code = af._search_generic(Tt, C, G, ctr, fctr, H, S, K, open_head, closed_head, arc)
        if code == 4:
            H, S, K = _grow(H, ctr[4] + 18), _grow(S, ctr[6] + 18), _grow(K, ctr[7] + 1)
            continue
        if code == af._CHECK_TIME:
            if keys is not None and len(keys):
                n1 = int(ctr[6])
                sl = slice(scanned, n1)
                kk = S[0][sl] * 18 + S[1][sl]
                hit = np.isin(kk, keys) & (S[2][sl] >= need_tee[S[3][sl]] - EPS) & ((S[7][sl] & 1) == 1)
                if hit.any():
                    c_hat = min(c_hat, float(S[4][sl][hit].min()))
                scanned = n1
                if fctr[0] > c_hat + 1e-9:
                    status = "bounded"
                    break
            if int(ctr[0]) >= budget:
                status = "budget"
                break
            continue
        break   # _EMPTY (목표가 없으므로 다 펼치면 끝)
    n = int(ctr[6])
    node, dirs, run, bend, gg, mask = S[0][:n], S[1][:n], S[2][:n], S[3][:n], S[4][:n], S[7][:n]
    ok = (run >= need_tee[bend] - EPS) & ((mask & 1) == 1)
    cand = {}
    for i in np.nonzero(ok)[0]:
        k = (int(node[i]), int(dirs[i]))
        if k not in cand or gg[i] < cand[k][0]:
            cand[k] = (float(gg[i]), int(i))
    return {"status": status, "expanded": int(ctr[0]), "cand": cand, "S": S, "g": g, "pipe": lp}


def leg_chain(tab, sid) -> list:
    S = tab["S"]
    chain = []
    i = sid
    while i >= 0:
        chain.append(State(int(S[0][i]), int(S[1][i]), float(S[2][i]), ANG[int(S[3][i])], int(S[7][i]),
                           float(S[8][i]), float(S[9][i])))
        i = int(S[5][i])
    chain.reverse()
    return chain


def tee_costs(gm, tab, b) -> tuple:
    """주관 지도 (노드, 방향) 별 다리 최소 비용 teec (N·18) 와 고른 다리 상태 id. 주관 방향 = 축 방향만 (D134 구현
    결정 (나)), 분기각 = 주관 흐름 ∠ 다리 흐름 ∈ {45, 90}. 다리 흐름 = in: 도착 방향 그대로, out: 도착 방향 반대."""
    N = len(gm.pos)
    teec = np.full(N * 18, INF)
    pick = {}
    gl = tab["g"]
    for (n, a), (cost, sid) in tab["cand"].items():
        try:
            m = gm._id_of(tuple(gl.pos[n]))
        except (AssertionError, IndexError, ValueError):
            continue
        if not np.array_equal(gm.pos[m], gl.pos[n]):   # 같은 위치의 노드만 (합집합 층이라도 영역·이격으로 빠질 수 있다)
            continue
        lf = np.asarray(DIRS[a], float)
        if b.flow == "out":
            lf = -lf
        lf = lf / np.linalg.norm(lf)
        for cd in range(6):
            ang = math.degrees(math.acos(float(np.clip(np.dot(DIRS[cd], lf), -1, 1))))
            if min(abs(ang - 45), abs(ang - 90)) > 1e-6:
                continue
            k = m * 18 + cd
            if cost < teec[k]:
                teec[k] = cost
                pick[k] = sid
    return teec, pick


# --------------------------------------------------------------------------- ② 주관 + 티

def _tee_consts(gm, pipe: Pipe):
    _, C = _consts(gm, pipe, tuple(map(float, pipe.end.pos)))
    C = list(C)
    need4, cap4, goal4, pre4 = _class4(gm, pipe)
    need_m = np.vstack([C[2], need4[None]])
    run_cap = np.append(C[3], cap4)
    goal_need = np.append(C[4], goal4)
    tee_pre = pre4
    nb = 5
    ble = np.zeros((nb, nb), dtype=np.bool_)
    for a in range(nb):
        for c in range(nb):
            ble[a, c] = bool(np.all(need_m[a] <= need_m[c] + 1e-12) and goal_need[a] <= goal_need[c] + 1e-12
                             and tee_pre[a] <= tee_pre[c] + 1e-12)
    C[2], C[3], C[4] = need_m, np.maximum(run_cap, np.append(need_m.max(1)[:4], need4.max())), goal_need
    return tuple(C), tee_pre, ble


def _class4(gm, pipe: Pipe):
    """bend 등급 4 = "티 뒤 (아직 꺾지 않음)": 다음 꺾임 앞 직관 ≥ max(§3.3, 3D_주관 + t_새) (티 → 다음 엘보 접점 ≥ 3D),
    끝 단자 도착 = 경계면 r_bnd (관통 직관, `추정:` 보수적), 티 앞 조건 tee_pre[b] = 3D + t_b (b ≥ 1 — 앞 엘보 접점까지 3D),
    b = 0 이면 앞이 단자라 관이음 아님 → 경계 start 면 r_bnd (관통 직관 안에 티 불가), 노즐이면 0. 티는 계통당 1 개 → 4 는 inf."""
    tan = {a: elbow_tangent(pipe.nominal_size, a) for a in ANG}
    D3 = T.three_d(pipe.nominal_size)
    need4 = np.array([0.0] + [max(gm.L, D3 + tan[a]) - EPS for a in ANG[1:]])
    goal4 = (gm.r_bnd if gm.end_boundary else 0.0) - EPS
    cap4 = max(need4.max(), goal4, D3 + tan[135])
    pre = np.array([(gm.r_bnd if gm.start_boundary else 0.0) - EPS] + [D3 + tan[a] - EPS for a in ANG[1:]] + [INF])
    return need4, cap4, goal4, pre


def main_with_tee(gm, pipe: Pipe, teec, budget: int):
    """주관 A* + 티 전이. 반환 (status, chain (티 상태 표시 포함), expanded)."""
    from .astar_tee import _search_tee
    P = gm.pos
    N = len(P)
    C, tee_pre, ble = _tee_consts(gm, pipe)
    Tt = (gm.step, gm.elen, gm.scost, gm.emask, np.ascontiguousarray(P[:, 0]), np.ascontiguousarray(P[:, 1]),
          np.ascontiguousarray(P[:, 2]), gm.hdist, gm.clearance, gm.pipe_margin, np.int64(gm.end_node), _DKIND,
          np.array([1.0 if pipe.gravity_pipe else 0.0, _lmax(pipe), 0.0]), gm.spack, gm.smax_dir, gm.bend_u,
          np.zeros(1), teec, tee_pre, ble)
    G = _arc_tables(gm)
    _CURRENT.update(space=gm)
    s0 = gm.start_state(pipe)
    h0 = heuristic_factory(gm, pipe)(s0)
    H, S, K = _state_arrays(1 << 16, tee=True)
    open_head = np.full(N * 18, -1, dtype=np.int32)
    closed_head = np.full(N * 18, -1, dtype=np.int32)
    arc = af.Dict.empty(af.types.int64, af.types.int8)
    f0 = int(s0.node)
    for k, v in enumerate((f0, s0.dir, s0.run, 0, 0.0, -1, -1, 15, 0.0, 0.0, 0)):
        S[k][0] = v
    open_head[f0 * 18 + s0.dir] = 0
    H[0][0], H[1][0], H[2][0], H[3][0] = h0, 0, 0.0, 0
    ctr = np.array([0, 0, 1, -1, 1, -1, 1, 0], dtype=np.int64)
    fctr = np.zeros(1)
    while True:
        code = _search_tee(Tt, C, G, ctr, fctr, H, S, K, open_head, closed_head, arc)
        if code == 4:
            H, S, K = _grow(H, ctr[4] + 18), _grow(S, ctr[6] + 18), _grow(K, ctr[7] + 1)
            continue
        if code == af._CHECK_TIME:
            if int(ctr[0]) >= budget:
                return "timeout", None, int(ctr[0])
            continue
        if code == af._EMPTY:
            return "unreachable", None, int(ctr[0])
        chain = []
        i = int(ctr[5])
        while i >= 0:
            bi = int(S[3][i])
            chain.append((State(int(S[0][i]), int(S[1][i]), float(S[2][i]), ANG[bi] if bi < 4 else 0, int(S[7][i]),
                                float(S[8][i]), float(S[9][i])), bi == 4, int(S[10][i])))
            i = int(S[5][i])
        chain.reverse()
        return "ok", chain, int(ctr[0])


# --------------------------------------------------------------------------- ③ 조립·자기 간섭·순차 근사

def _geom_leg(tab, sid) -> tuple:
    """다리 사슬 → (기하 순서 꺾임점 (티 → 단자), 다리 RouteResult)."""
    chain = leg_chain(tab, sid)
    rr = _finish_chain(tab["g"], tab["pipe"], chain, 0, 0, 0.0)
    return [list(map(float, w)) for w in rr.waypoints[::-1]], rr


def _conflicts(pipe, main_wps, legs) -> list:
    """legs = [(Branch, 기하 꺾임점)] → 자기 간섭 목록 (tree.family_conflicts)."""
    main_cl = T.centerline_of(pipe.nominal_size, main_wps)
    items = []
    for b, geom in legs:
        lp = T.leg_flow_pipe(pipe, b, geom)
        items.append((lp.id, T.centerline_of(b.nominal_size, geom), b.nominal_size, lp.radius))
    return T.family_conflicts(main_cl, pipe.nominal_size, pipe.radius, items)


def _tee_angle(main_wps, tee, geom, flow) -> float:
    chk = T.tee_check(T.centerline_of("15A", main_wps), "15A", tee, np.subtract(geom[1], geom[0]), flow, "15A")
    if not math.isfinite(chk.angle):
        return None
    return min((45, 90, 135), key=lambda x: abs(chk.angle - x))


def main_tee_candidates(gm, pipe, chain) -> list:
    """주관 단독 사슬 위 티 후보 (노드, 방향) — 축 정렬 직관 위 노드에서 앞뒤 직관 규칙 (_class4 와 같은 식)."""
    tan = {a: elbow_tangent(pipe.nominal_size, a) for a in ANG}
    D3 = T.three_d(pipe.nominal_size)
    out = []
    n = len(chain)
    for k in range(1, n - 1):
        s = chain[k]
        nxt = chain[k + 1]
        if s.dir >= 6 or nxt.dir != s.dir:
            continue
        pre = (gm.r_bnd if gm.start_boundary else 0.0) if s.bend == 0 else D3 + tan[s.bend]
        if s.run < pre - EPS:
            continue
        # 뒤: 다음 꺾임까지 직관 ≥ 3D + t_다음 (끝 단자면 경계 r_bnd)
        dist = 0.0
        ok = None
        for j in range(k + 1, n):
            dist += math.dist(gm.pos[chain[j - 1].node], gm.pos[chain[j].node])
            if j + 1 < n and chain[j + 1].dir != chain[j].dir:
                ok = dist >= D3 + tan[DEFLECTION[chain[j].dir][chain[j + 1].dir]] - EPS
                break
        if ok is None:
            ok = dist >= (gm.r_bnd if gm.end_boundary else 0.0) - EPS
        if ok:
            out.append((s.node, s.dir))
    return out


def route_tree(sc, pipe: Pipe, make_graph, budget: int, sequential_only: bool = False) -> RouteResult:
    """분기 계통 한 개. make_graph(배관, partners) → LayeredGraph. 다리 1 개 = 정확 풀이, 2 개 이상 = 순차 근사 (D112).
    순서: 주관 단독 A* (순차 근사의 주관·다리 표 한계) → 다리 표 (한계까지) → 주관 + 티 (자기 간섭이면 막고 다시)."""
    from .astar_fast import astar_route_generic
    t0 = time.perf_counter()
    legs = list(pipe.branches)
    lps = [leg_search_pipe(pipe, b) for b in legs]
    gm = make_graph(pipe, tuple(lps))
    used = 0

    def fail(status, method):
        r = RouteResult(pipe.id, status, expanded=used, search_sec=time.perf_counter() - t0)
        r.tree = {"method": method, "legs": []}
        return r
    h = heuristic_factory(gm, pipe)
    st, chain0, exp0, _ = astar_route_generic(gm, pipe, None, h(gm.start_state(pipe)), max(1, budget))
    used += exp0
    search = {"main_solo": int(exp0), "leg_tables": [], "tee_search": [], "leg_rebuild": []}   # F4 확장 수 분해
    if st != "ok":
        r = fail(st, "exact" if len(legs) == 1 else "sequential")
        r.tree["search"] = search
        return r
    cands = main_tee_candidates(gm, pipe, chain0)
    tabs = []
    for b, lp in zip(legs, lps):
        gl = make_graph(lp, (pipe,))
        keys = _leg_keys(gm, gl, cands, b) if len(legs) == 1 and not sequential_only else None
        tab = leg_table(gl, lp, b, max(1, budget - used), bound_keys=keys)
        used += tab["expanded"]
        search["leg_tables"].append(int(tab["expanded"]))
        tabs.append(tab)
    method = None
    chosen = None   # (주관 RouteResult, [(Branch, 기하, 다리 RouteResult)])
    if len(legs) == 1 and not sequential_only and tabs[0]["status"] in ("done", "bounded"):
        b, tab = legs[0], tabs[0]
        teec, pick = tee_costs(gm, tab, b)
        for attempt in range(MAX_RETRY + 1):
            if used >= budget:
                break
            st, chain, exp = main_with_tee(gm, pipe, teec, budget - used)
            used += exp
            search["tee_search"].append(int(exp))
            if st != "ok":
                break
            ti = next(i for i, c in enumerate(chain) if c[1])
            tn, td = chain[ti][0].node, chain[ti][0].dir
            states = [c[0] for i, c in enumerate(chain) if i != ti]
            mr = _finish_chain(gm, pipe, states, 0, 0, 0.0)
            geom, lr = _geom_leg(tab, pick[tn * 18 + td])
            geom[0] = list(map(float, gm.pos[tn]))
            if not _conflicts(pipe, mr.waypoints, [(b, geom)]):
                method = "exact" if attempt == 0 else "exact_retry"
                chosen = (mr, [(b, geom, lr)])
                break
            teec = teec.copy()
            teec[tn * 18 + td] = INF   # 그 티 후보를 막고 다시 (Q9)
    leg_search0 = [tb["status"] for tb in tabs]
    if chosen is None:   # 순차 근사 — 주관 단독 경로는 이미 있다
        # F1 (2026-11-24 사양 검토): 다리 표가 bounded (정확 풀이용 한계 c 에서 멈춤) 이면 c 보다 비싼 티 후보가 표에
        # 없다 — 남은 예산으로 한계 없이 끝까지 다시 만든 뒤 순차 근사 (가장 싼 티가 자기 간섭으로 막힌 경우)
        for k, (b, lp) in enumerate(zip(legs, lps)):
            if tabs[k]["status"] != "bounded" or used >= budget:
                continue
            tab = leg_table(make_graph(lp, (pipe,)), lp, b, max(1, budget - used), bound_keys=None)
            used += tab["expanded"]
            search["leg_rebuild"].append(int(tab["expanded"]))
            tabs[k] = tab
        chosen = _sequential(pipe, gm, legs, tabs, chain0, cands)
        method = "sequential"
    mr, legres = chosen
    J = mr.J + sum(lr.J for _, _, lr in legres)
    found = {b.id: (geom, lr) for b, geom, lr in legres}
    tlegs = []
    for b in legs:
        if b.id in found:
            geom, lr = found[b.id]
            tlegs.append({"leg_id": b.id, "nominal_size": b.nominal_size, "flow": b.flow, "found": True,
                          "waypoints": geom, "tee": {"pos": geom[0], "angle": _tee_angle(mr.waypoints, geom[0], geom,
                                                                                         b.flow)},
                          "J_router": lr.J})
        else:
            tlegs.append({"leg_id": b.id, "nominal_size": b.nominal_size, "flow": b.flow, "found": False,
                          "waypoints": [], "tee": None})
    mr.expanded = used
    mr.search_sec = time.perf_counter() - t0
    mr.J_main = mr.J
    mr.J = J
    mr.J_support_est += sum(lr.J_support_est for _, _, lr in legres)
    mr.tree = {"method": method, "legs": tlegs,
               "leg_search": leg_search0, "leg_search_final": [tb["status"] for tb in tabs], "search": search}
    mr.branches = [lg["waypoints"] for lg in tlegs]
    return mr


def _leg_keys(gm, gl, cands, b) -> set:
    """주관 단독 경로 위 티 후보 (주관 노드, 주관 방향) → 다리 지도 키 (노드·18 + 분기각이 맞는 도착 방향)."""
    out = set()
    for m, cd in cands:
        try:
            n = gl._id_of(tuple(gm.pos[m]))
        except (AssertionError, IndexError, ValueError):
            continue
        if not np.array_equal(gl.pos[n], gm.pos[m]):
            continue
        for a in range(18):
            lf = np.asarray(DIRS[a], float)
            if b.flow == "out":
                lf = -lf
            lf = lf / np.linalg.norm(lf)
            ang = math.degrees(math.acos(float(np.clip(np.dot(DIRS[cd], lf), -1, 1))))
            if min(abs(ang - 45), abs(ang - 90)) < 1e-6:
                out.add(n * 18 + a)
    return out


def _sequential(pipe, gm, legs, tabs, chain, cands):
    """D112 순차 근사: 주관 단독 (계통 지도, 이미 찾은 경로) → 다리를 큰 구경 → 주관 start 에서 먼 것 순으로, 주관 경로
    위 티 후보 중 다리 표 비용이 작은 순으로 자기 간섭 없는 첫 티 (다른 다리와 3D·간섭도). 반환 (주관 결과, 놓인 다리)."""
    mr = _finish_chain(gm, pipe, chain, 0, 0, 0.0)
    order = sorted(range(len(legs)), key=lambda k: (-T.nominal_mm(legs[k].nominal_size),
                                                     -sum(abs(a - c) for a, c in zip(legs[k].terminal.pos, pipe.start.pos)),
                                                     k))
    placed = []
    for k in order:
        b, tab = legs[k], tabs[k]
        teec, pick = tee_costs(gm, tab, b)
        opts = sorted(((teec[n * 18 + d], n, d) for n, d in cands if teec[n * 18 + d] < INF), key=lambda x: x[:3])
        for _, n, d in opts:
            geom, lr = _geom_leg(tab, pick[n * 18 + d])
            geom[0] = list(map(float, gm.pos[n]))
            if any(np.linalg.norm(np.subtract(geom[0], g2[0])) < T.three_d(pipe.nominal_size) - EPS
                   for _, g2, _ in placed):
                continue
            if not _conflicts(pipe, mr.waypoints, [(bb, gg) for bb, gg, _ in placed] + [(b, geom)]):
                placed.append((b, geom, lr))
                break
    return mr, placed
