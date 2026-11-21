"""메모리·시간 프로파일 측정 도구 (2026-11-17 소유자 지시 — 측정 전용, 결과 불변).

`python -m pipe_routing.memprof --log PROF.jsonl -- <pipeline 인자>` = pipeline.main 과 같은 실행을, 이 프로세스 안에서만
함수를 감싸 기록한다. 감싼 함수는 원래 함수를 같은 인자로 부르고 반환값을 그대로 돌려준다 — 경로·J·확장 수·출력 JSON 은
감싸지 않은 실행과 바이트 동일 (`tests/test_memprof.py`). A* 는 astar_fast.astar_route_generic 과 같은 코드를 부르고
끝난 뒤 배열만 읽는다.

기록 (한 줄 = JSON 하나, 프로세스가 메모리 상한으로 죽어도 그 전 기록은 남는다):
  layer      고정층 1개 — 키 (구경·class·r·중력관 층·격자), 격자 크기·노드·엣지, 배열별 nbytes, 격자 좌표 출처별 개수, 구성 시간·RSS
  route      라우팅 1회 — 단계 (plan·classify·solo_ref), 놓인 배관 수, 배치층 표시 시간·배열 nbytes, 탐색 시간,
             A* 배열 nbytes, 확장 수, 닫힌 노드 수·읽은 엣지 수 (만진 비율), RSS
  caches     라우팅마다 고정층 캐시 (D67 항목 캐시·엣지 목록) 누적 nbytes
  phase      단계별 시간·최대 RSS (RSS 표본 50 ms)
  done       전체 최대 RSS (getrusage)
"""
import argparse
import json
import resource
import sys
import threading
import time

import numpy as np

_LOG = None
_PHASE = ["main"]
_PEAK = {}
_TIME = {}
_LOCK = threading.Lock()


def rss_mb() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * resource.getpagesize() / 2 ** 20


def _emit(rec: dict) -> None:
    if _LOG is None:
        return
    with _LOCK:
        _LOG.write(json.dumps(rec, ensure_ascii=False) + "\n")
        _LOG.flush()


def _sampler(stop: threading.Event) -> None:
    while not stop.wait(0.05):
        r = rss_mb()
        with _LOCK:
            k = _PHASE[-1]
            if r > _PEAK.get(k, 0.0):
                _PEAK[k] = r


class phase:
    """단계 표시 (중첩 — 시간은 안쪽 단계를 뺀 배타 시간)."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()
        self.child = 0.0
        _STACK.append(self)
        with _LOCK:
            _PHASE.append(self.name)
        return self

    def __exit__(self, *a):
        dt = time.perf_counter() - self.t0
        _STACK.pop()
        if _STACK:
            _STACK[-1].child += dt
        _TIME[self.name] = _TIME.get(self.name, 0.0) + dt - self.child
        with _LOCK:
            _PHASE.pop()
        return False


_STACK = []


def nbytes(obj, seen=None) -> int:
    """numpy 배열 nbytes 합 (dict·list·tuple·set·객체 __dict__ 재귀, 같은 객체 1회)."""
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes) if obj.base is None or id(obj.base) not in seen else 0
    if isinstance(obj, dict):
        return sum(nbytes(v, seen) for v in obj.values()) + sum(nbytes(k, seen) for k in obj
                                                              if isinstance(k, np.ndarray))
    if isinstance(obj, (list, tuple, set, frozenset)):
        return sum(nbytes(v, seen) for v in obj)
    return 0


def attr_bytes(o, skip=()) -> dict:
    out = {}
    for k, v in o.__dict__.items():
        if k in skip:
            continue
        b = nbytes(v)
        if b:
            out[k] = b
    return out


# ---------------------------------------------------------------- 격자 좌표 출처 (고정층 규칙을 출처별로 다시 계산)
def coord_sources(sc, pipe, F, grid, _return_union=False):
    """축별 좌표를 출처별로: 출처가 만드는 (영역 안) 좌표 수와 그 출처만 만드는 좌표 수. 합집합이 고정층 축과 같은지 확인.
    단일 블록 층만 (블록 쌍 층은 None)."""
    from . import internal
    from .blockpair import walls
    from .escape_graph import _snap_down, _snap_up
    from .layered import FIRST_BEND_ANGLES, _same, grid_extra_coords, support_extra_coords, terminal_first_bend
    from .constants import elbow_tangent
    if walls(sc):
        return None
    r = F.r
    lo, hi = F.dom_lo, F.dom_hi
    src = {}

    def add(name, ax, v, anywhere=False):   # 단자는 영역 밖 (경계 단자, D39) 도 좌표 (EscapeGraph 와 같음)
        if anywhere or lo[ax] - 1e-9 <= v <= hi[ax] + 1e-9:
            src.setdefault(name, [set(), set(), set()])[ax].add(float(v))

    def boxes(name, bl):
        for b0, b1 in bl:
            for ax in range(3):
                add(name, ax, _snap_down(b0[ax] - r))
                add(name, ax, _snap_up(b1[ax] + r))
    for ax in range(3):
        add("domain", ax, lo[ax])
        add("domain", ax, hi[ax])
    same = _same(sc, pipe)
    if internal.has_internal(sc):
        for o in internal.solid_obstacles(sc):
            boxes("obstacle:" + ("access" if o.access else o.type), [(o.box.min, o.box.max)])
        _, hard, soft, _ = internal.router_boxes(sc, same[0])
        boxes("plate_edge", hard)
        boxes("plate_zone", soft)
        fb = internal.first_bend_coords(sc, same[0])
        for ax in range(3):
            for v in fb[ax]:
                add("plate_first_bend", ax, v)
    else:
        boxes("obstacle:" + "all", [(o.box.min, o.box.max) for o in sc.obstacles])
    for p in same:
        for t in (p.start, p.end):
            for ax in range(3):
                add("terminal", ax, float(t.pos[ax]), anywhere=True)
    from .layered import G0
    gs = grid if grid is not None else G0
    ex = grid_extra_coords(sc, pipe.nominal_size, bool(gs.g1 and pipe.gravity_pipe), gs, pipes=same)
    for ax in range(3):
        for v in (ex[ax] if ex else ()):
            add("grid_extra", ax, v)
    sl = support_extra_coords(sc, pipe.nominal_size, pipes=same)
    fbl = [[], [], []]
    tangents = [elbow_tangent(pipe.nominal_size, a) for a in FIRST_BEND_ANGLES]
    for p in same:
        for term, sgn in ((p.start, 1.0), (p.end, -1.0)):
            d = np.asarray(term.dir, dtype=float)
            ax = int(np.argmax(np.abs(d)))
            for t in tangents:
                fbl[ax].append(float(term.pos[ax]) + sgn * d[ax] * _snap_up(terminal_first_bend(p, term, t)))
    for ax in range(3):
        rest = list(sl[ax])
        for v in fbl[ax]:
            rest.remove(v)
            add("terminal_first_bend", ax, v)
        for v in rest:
            add("support_band", ax, v)
    union = [set().union(*(s[ax] for s in src.values())) for ax in range(3)]
    if _return_union:
        return union
    axes = [set(map(float, F.axes[ax])) for ax in range(3)]
    out = {"axes": [len(a) for a in axes], "union_matches": all(union[ax] == axes[ax] for ax in range(3)),
           "unattributed": [sorted(axes[ax] - union[ax])[:10] for ax in range(3)],
           "spurious": [sorted(union[ax] - axes[ax])[:10] for ax in range(3)], "sources": {}}
    for name, s in sorted(src.items()):
        others = [set().union(*(t[ax] for n, t in src.items() if n != name)) for ax in range(3)]
        out["sources"][name] = {"coords": [len(s[ax]) for ax in range(3)],
                                "only": [len(s[ax] - others[ax]) for ax in range(3)]}
    return out


def layer_inventory(sc, grid=None) -> list:
    """고정층 지도 목록 (캐시 키 D117 = 구경·class·r·중력관 층·격자) — 지도를 만들지 않고 축 좌표 수·격자점 수만
    (coord_sources 와 같은 규칙). 단일 블록만 (블록 쌍은 블록별 층이라 None)."""
    from types import SimpleNamespace
    from .blockpair import walls
    from .escape_graph import _snap_up
    from .layered import G0, layer_id
    if walls(sc):
        return None
    gs = grid if grid is not None else G0
    out, seen = [], set()
    ext = sc.block.extent
    for p in sc.pipes:
        key = (layer_id(p), bool(gs.g1 and p.gravity_pipe), gs.g2)
        if key in seen:
            continue
        seen.add(key)
        m = _snap_up(p.radius)
        F = SimpleNamespace(r=p.radius, dom_lo=np.array([m, m, m], float),
                            dom_hi=np.array([e - m for e in ext], float), axes=[[], [], []])
        n = [len(x) for x in coord_sources(sc, p, F, gs, _return_union=True)]
        out.append({"size": p.nominal_size, "class": p.class_name, "r": p.radius, "gravity_layer": key[1],
                    "grid": gs.name, "axes": n, "grid_points": int(n[0] * n[1] * n[2]),
                    "pipes": sum(1 for q in sc.pipes if (layer_id(q), bool(gs.g1 and q.gravity_pipe), gs.g2) == key)})
    return out


# ---------------------------------------------------------------- A* (astar_fast.astar_route_generic 과 같은 코드)
def _astar_generic(space, pipe, time_limit, h_start, expand_limit=None):
    from . import astar_fast as af
    from numba import types
    from numba.typed import Dict
    t0 = time.perf_counter()
    goal = tuple(map(float, pipe.end.pos))
    P = space.pos
    T = (space.step, space.elen, space.scost, space.emask, np.ascontiguousarray(P[:, 0]), np.ascontiguousarray(P[:, 1]),
         np.ascontiguousarray(P[:, 2]), space.hdist, space.clearance, space.pipe_margin, np.int64(space.end_node),
         af._DKIND, np.array([1.0 if pipe.gravity_pipe else 0.0, af._lmax(pipe), 0.0]), space.spack, space.smax_dir,
         space.bend_u, np.zeros(1))   # D126 휴리스틱 인자 없음 (astar_route_generic 과 같은 T)
    _, C = af._consts(space, pipe, goal)
    G = af._arc_tables(space)
    af._CURRENT.update(space=space)
    s0 = space.start_state(pipe)
    cap = 1 << 16
    H, S, K = af._generic_arrays(cap)   # astar_route_generic 과 같은 배열 (D126 L3 (d))
    N = len(P)
    open_head = np.full(N * 18, -1, dtype=np.int32)
    closed_head = np.full(N * 18, -1, dtype=np.int32)
    arc = Dict.empty(types.int64, types.int8)
    f0 = int(s0.node)
    S[0][0], S[1][0], S[2][0], S[3][0], S[4][0], S[5][0], S[6][0], S[7][0], S[8][0], S[9][0] = (
        f0, s0.dir, s0.run, 0, 0.0, -1, -1, 15, 0.0, 0.0)
    open_head[f0 * 18 + s0.dir] = 0
    H[0][0], H[1][0], H[2][0], H[3][0] = h_start, 0, 0.0, 0
    ctr = np.array([0, 0, 1, -1, 1, -1, 1, 0], dtype=np.int64)
    fctr = np.zeros(1)
    res = None
    while res is None:
        code = af._search_generic(T, C, G, ctr, fctr, H, S, K, open_head, closed_head, arc)
        if code == 4:
            H = af._grow(H, ctr[4] + 18)
            S = af._grow(S, ctr[6] + 18)
            K = af._grow(K, ctr[7] + 1)
            continue
        if code == af._CHECK_TIME:
            if af.budget_over(t0, time_limit, expand_limit, int(ctr[0])):
                res = ("timeout", None, int(ctr[0]), int(ctr[1]))
            continue
        if code == af._EMPTY:
            res = ("unreachable", None, int(ctr[0]), int(ctr[1]))
            continue
        chain = []
        i = int(ctr[5])
        while i >= 0:
            chain.append(af.State(int(S[0][i]), int(S[1][i]), float(S[2][i]), af.ANG[int(S[3][i])], int(S[7][i]),
                                  float(S[8][i]), float(S[9][i])))
            i = int(S[5][i])
        chain.reverse()
        res = ("ok", chain, int(ctr[0]), int(ctr[1]))
    sec = time.perf_counter() - t0
    ci = np.nonzero(closed_head >= 0)[0]
    oi = np.nonzero(open_head >= 0)[0]
    cn = np.unique(ci // 18)
    tn = np.unique(np.concatenate([cn, oi // 18]))
    total_edges = int(np.count_nonzero(space.step >= 0))
    read_edges = int(np.count_nonzero(space.step[cn] >= 0)) if len(cn) else 0
    _LAST_ASTAR.clear()
    _LAST_ASTAR.update({
        "search_sec": sec, "expanded": res[2], "generated": res[3],
        "closed_states": int(len(ci)), "closed_nodes": int(len(cn)), "touched_nodes": int(len(tn)),
        "nodes": int(N), "edge_entries": total_edges, "edges_read": read_edges,
        "astar_bytes": {"open_head": int(open_head.nbytes), "closed_head": int(closed_head.nbytes),
                        "heap": nbytes(H), "states": nbytes(S), "closed": nbytes(K),
                        "heap_used": int(ctr[4]), "states_used": int(ctr[6]), "closed_used": int(ctr[7])},
        "arc_cache": len(arc)})
    return res


_LAST_ASTAR = {}


# ---------------------------------------------------------------- rg 사전 계산 비용 (D125 — 측정 전용, 탐색에 쓰지 않는다)
_RG_KERNEL = None


def _rg_kernel():
    """역방향 Dijkstra (이진 힙) — numba. 반환 dist (목표까지 완화 그래프 최단거리)."""
    global _RG_KERNEL
    if _RG_KERNEL is None:
        from numba import njit

        @njit(cache=True)
        def run(rptr, rsrc, rw, goal, N):
            dist = np.full(N, np.inf)
            dist[goal] = 0.0
            hk = np.empty(rsrc.size + 1)
            hv = np.empty(rsrc.size + 1, np.int64)
            n = 0
            hk[0] = 0.0
            hv[0] = goal
            n = 1
            while n > 0:
                k, v = hk[0], hv[0]
                n -= 1
                if n > 0:   # 마지막을 뿌리로 내려 보낸다
                    lk, lv = hk[n], hv[n]
                    i = 0
                    while True:
                        c = 2 * i + 1
                        if c >= n:
                            break
                        if c + 1 < n and hk[c + 1] < hk[c]:
                            c += 1
                        if hk[c] < lk:
                            hk[i], hv[i] = hk[c], hv[c]
                            i = c
                        else:
                            break
                    hk[i], hv[i] = lk, lv
                if k > dist[v]:
                    continue
                for e in range(rptr[v], rptr[v + 1]):
                    u = rsrc[e]
                    nd = k + rw[e]
                    if nd < dist[u]:
                        dist[u] = nd
                        i = n
                        n += 1
                        while i > 0:
                            p = (i - 1) // 2
                            if hk[p] > nd:
                                hk[i], hv[i] = hk[p], hv[p]
                                i = p
                            else:
                                break
                        hk[i], hv[i] = nd, u
            return dist
        _RG_KERNEL = run
    return _RG_KERNEL


def rg_measure(space, pipe) -> dict:
    """D125 rg 의 사전 계산 비용: 완화 엣지 무게 (길이 × kg/m + scost, emask == 0·없는 엣지 제외) 로 역방향 인접 (CSR)
    을 만들고 목표에서 Dijkstra. 시간·배열 크기·시작점 하한 (D41 h 대비) 을 기록한다."""
    t0 = time.perf_counter()
    step = space.step
    N = len(step)
    ok = (step >= 0) & (space.emask != 0)
    src, d = np.nonzero(ok)
    dst = step[src, d]
    w = space.elen[src, d] / 1000 * pipe.kg_per_m + space.scost[src, d]
    order = np.argsort(dst, kind="stable")
    rsrc = src[order].astype(np.int64)
    rw = w[order]
    rptr = np.zeros(N + 1, np.int64)
    np.cumsum(np.bincount(dst, minlength=N), out=rptr[1:])
    t1 = time.perf_counter()
    dist = _rg_kernel()(rptr, rsrc, rw, int(space.end_node), N)
    t2 = time.perf_counter()
    s = int(space.start_node)
    h41 = float(space.hdist[s]) / 1000 * pipe.kg_per_m
    return {"rg_build_sec": round(t1 - t0, 4), "rg_dijkstra_sec": round(t2 - t1, 4),
            "rg_bytes": int(rptr.nbytes + rsrc.nbytes + rw.nbytes + dist.nbytes),
            "rg_edges": int(len(rsrc)), "rg_reached": int(np.isfinite(dist).sum()),
            "rg_start": float(dist[s]), "h41_start_dist_term": h41}


# ---------------------------------------------------------------- 감싸기
def install(grid_name: str = "g0") -> None:
    from . import astar_fast, layered, multi, pipeline, solo_ref
    from .layered import LayeredGraph
    seen_layers = set()
    orig_fixed = layered.fixed_layer

    def fixed_layer(sc, pipe, grid=layered.G0, support_lines=True):
        before = rss_mb()
        with phase("fixed_build"):
            F, sec = orig_fixed(sc, pipe, grid, support_lines)
        if id(F) not in seen_layers:
            seen_layers.add(id(F))
            rec = {"kind": "layer", "pipe": pipe.id, "size": pipe.nominal_size, "class": pipe.class_name,
                   "r": pipe.radius, "gravity_layer": bool(grid.g1 and pipe.gravity_pipe), "grid": grid.name,
                   "shape": list(F.shape), "grid_points": int(np.prod(F.shape)), "nodes": int(F.node_ok.sum()),
                   "build_sec": round(sec, 3), "rss_before_mb": round(before, 1), "rss_after_mb": round(rss_mb(), 1),
                   "layer": id(F), "bytes": attr_bytes(F, skip=tuple(k for k in F.__dict__ if (k.endswith("_cache") or k == "_line_support")))}
            try:
                rec["coord_sources"] = coord_sources(sc, pipe, F, grid)
            except Exception as e:   # 측정 보조 — 실패해도 실행은 계속
                rec["coord_sources_error"] = repr(e)
            _emit(rec)
        return F, sec
    layered.fixed_layer = fixed_layer

    orig_init = LayeredGraph.__init__

    def init(self, sc, pipe, others=(), **kw):
        before = rss_mb()
        with phase("marking"):
            orig_init(self, sc, pipe, others, **kw)
        _CUR.update(graph_bytes=attr_bytes(self, skip=("fixed", "_sc", "_sup", "_ls")),
                    marking_sec=self.build_sec - self.fixed_sec, fixed_sec=self.fixed_sec,
                    placed=len(others), reserved=self.n_reserved, members=self.n_members,
                    rss_marking_before_mb=round(before, 1), rss_marking_after_mb=round(rss_mb(), 1),
                    edges=int(self.n_edges), F=self.fixed)
    LayeredGraph.__init__ = init

    def astar(space, pipe, time_limit, h_start, expand_limit=None):
        with phase("search"):
            return _astar_generic(space, pipe, time_limit, h_start, expand_limit)
    astar_fast.astar_route_generic = astar

    orig_route = layered._route

    def route(g, pipe, time_limit, expand_limit=None):
        r = orig_route(g, pipe, time_limit, expand_limit)
        F = _CUR.pop("F", None)
        if RG:
            with phase("rg_measure"):
                _CUR.update(rg_measure(g, pipe))
        rec = {"kind": "route", "phase": _PHASE[-1], "pipe": pipe.id, "size": pipe.nominal_size, "status": r.status,
               "expand_limit": expand_limit, **{k: v for k, v in _CUR.items()}, **_LAST_ASTAR,
               "rss_after_mb": round(rss_mb(), 1)}
        _CUR.clear()
        _emit(rec)
        if F is not None:
            _emit({"kind": "caches", "pipe": pipe.id, "layer": id(F), "layer_size": F.pipe.nominal_size,
                   "item_cache_entries": len(F.__dict__.get("_item_cache", {})),
                   "bytes": {k: nbytes(v) for k, v in F.__dict__.items() if (k.endswith("_cache") or k == "_line_support")}})
        return r
    layered._route = route

    orig_members = layered.SupportMembers.__call__

    def members(self, sc, placed):
        with phase("member_session"):
            return orig_members(self, sc, placed)
    layered.SupportMembers.__call__ = members

    orig_lay = multi.sequential_ripup_planner

    def planner(sc, router=multi.astar_router, time_limit=60.0, **kw):
        with phase("plan"):
            return orig_lay(sc, router, time_limit, **kw)
    pipeline.sequential_ripup_planner = planner
    multi.sequential_ripup_planner = planner

    orig_cls = multi.classify_failures

    def cls(*a, **kw):
        with phase("classify"):
            return orig_cls(*a, **kw)
    multi.classify_failures = cls

    orig_verify = pipeline.verify

    def verify(*a, **kw):
        with phase("verify_final"):
            return orig_verify(*a, **kw)
    pipeline.verify = verify

    orig_attach = solo_ref.attach

    def attach(*a, **kw):
        with phase("solo_ref"):
            return orig_attach(*a, **kw)
    solo_ref.attach = attach


_CUR = {}
RG = True   # D125 rg 사전 계산 비용도 잰다 (측정 전용)


def main(argv=None) -> int:
    global _LOG
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        print("사용: python -m pipe_routing.memprof --log PROF.jsonl -- <pipeline 인자>", file=sys.stderr)
        return 2
    i = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--no-rg", action="store_true", help="D125 rg 사전 계산 비용 측정을 끈다")
    args = ap.parse_args(argv[:i])
    global RG
    RG = not args.no_rg
    _LOG = open(args.log, "w", encoding="utf-8")
    install()
    stop = threading.Event()
    th = threading.Thread(target=_sampler, args=(stop,), daemon=True)
    th.start()
    from . import pipeline
    t0 = time.perf_counter()
    try:
        with phase("total"):
            rc = pipeline.main(argv[i + 1:])
    finally:
        stop.set()
        th.join()
        _emit({"kind": "phase", "time_sec": {k: round(v, 3) for k, v in _TIME.items()},
               "peak_rss_mb": {k: round(v, 1) for k, v in _PEAK.items()}})
        _emit({"kind": "done", "wall_sec": round(time.perf_counter() - t0, 1),
               "maxrss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)})
        _LOG.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
