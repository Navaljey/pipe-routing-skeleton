"""검증기 골격 (§6, D12, D44). 기준별 플러그인 — MODULES 에 등록된 함수가 배관 하나씩 판정한다.

모듈 함수 시그니처: fn(ctx: Context, pr: PipeRoute) -> list[Violation] | None
  None = 해당 없음 (예: 압력관의 gravity_slope), [] = 통과, [...] = 위반
"""
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import numpy as np

from ..constants import ANGLE_TOL_DEG, elbow_radius
from ..scenario import Pipe, Scenario
from .geom import Centerline, build_centerline

Vec3 = tuple[float, float, float]
MODULE_ORDER = ["collision", "boundary", "bend", "gravity_slope", "valve", "branch", "support",
                "crossing"]   # §6.1 + D76 ④ (블록 쌍만, 그 밖은 None)
MODULES: dict[str, Callable] = {}


def module_order(scenario) -> list:
    """시나리오에 쓰는 모듈 순서 — MODULE_ORDER + 그 시나리오에만 있는 모듈 (D119 access: 접근 구역이 있을 때).
    E1 (접근 구역 없음) 은 MODULE_ORDER 그대로 → 출력 layer0 키 불변."""
    out = list(MODULE_ORDER)
    if any(getattr(o, "access", False) for o in scenario.obstacles):
        out.append("access")
    return out


def register(name: str):
    def deco(fn):
        MODULES[name] = fn
        return fn
    return deco


@dataclass
class PipeRoute:
    """검증기 입력 — 라우터와 무관한 순수 기하 (D44).

    waypoints: start 단자 → end 단자 꺾임점 (mm). 분기 배관만 branches = 주관 위 분기점에서 시작하는 꺾임점 목록들
    """
    pipe_id: str
    waypoints: list
    branches: list = field(default_factory=list)


@dataclass
class Violation:
    module: str
    pipe_id: str
    pos: list
    message: str
    value: Optional[float] = None
    other: Optional[str] = None      # 상대 배관·장애물 id


@dataclass
class PipeReport:
    pipe_id: str
    routed: bool
    layer0: dict = field(default_factory=dict)       # 모듈 → True(통과)/False(위반)/None(해당 없음)
    violations: list = field(default_factory=list)
    fittings: list = field(default_factory=list)     # [{"type": "elbow_90", "pos": [...]}] (D25: 135° = 90+45)
    supports: list = field(default_factory=list)     # D80 ⑦ 부재 묶음 [{"pos", "form", "face", "members", "kg", ...}]
    support_gaps: list = field(default_factory=list)  # 서포트 불가 구간 (중심선 점 목록, 3D 표시용)

    @property
    def success(self) -> bool:
        return self.routed and all(v is not False for v in self.layer0.values())

    def to_dict(self) -> dict:
        d = asdict(self)
        d["success"] = self.success
        return d


@dataclass
class Context:
    scenario: Scenario
    pipes: dict                      # pipe_id → Pipe
    routes: dict                     # pipe_id → PipeRoute (경로가 있는 배관만)
    centerlines: dict                # pipe_id → [Centerline] (주관 + 분기)
    angle_tol: float = ANGLE_TOL_DEG
    cache: dict = field(default_factory=dict)
    std: object = None               # 서포트 표준 (D78·D80, support_std.Standard)
    family: dict = field(default_factory=dict)   # D134 분기 계통: 다리 가상 배관 id → 계통 (주관) id
    trees: dict = field(default_factory=dict)    # 계통 id → 입력 PipeRoute (주관 + 다리 기하)

    def root(self, pid: str) -> str:
        return self.family.get(pid, pid)

    def others(self, pipe_id: str, family: bool = False):
        """다른 배관들의 중심선 조각 배열 (A, B, 반경, slack, id). family=False 면 같은 계통 (주관·다리) 도 뺀다 —
        collision·bend 의 자기 계통 이격은 tree.family_conflicts 가 티 구역을 빼고 본다 (D134). family=True (서포트 부재)
        면 자기 자신만 뺀다. 분기 없는 시나리오에서는 둘이 같다."""
        key = ("others", pipe_id, family)
        if key not in self.cache:
            A, B, R, S, ids = [], [], [], [], []
            me = self.root(pipe_id)
            for pid, cls in self.centerlines.items():
                if pid == pipe_id or (not family and self.root(pid) == me):
                    continue
                r = self.pipes[pid].radius
                for cl in cls:
                    a, b, slack, _ = cl.arrays()
                    A.append(a); B.append(b); S.append(slack)
                    R.append(np.full(len(a), r)); ids += [pid] * len(a)
            if A:
                self.cache[key] = (np.concatenate(A), np.concatenate(B), np.concatenate(R), np.concatenate(S), ids)
            else:
                z = np.zeros((0, 3))
                self.cache[key] = (z, z, np.zeros(0), np.zeros(0), [])
        return self.cache[key]


def verify(scenario: Scenario, routes: list, modules=None, std=None) -> dict:
    """routes: PipeRoute 목록 (경로를 못 찾은 배관은 빠져 있거나 waypoints 가 비어 있다).

    반환: {"pipes": {pipe_id: PipeReport}, "sec": 검증 시간}
    """
    from . import modules as _  # noqa: F401  (모듈 등록)
    t0 = time.perf_counter()
    timing = {"centerline": 0.0, **{m: 0.0 for m in (modules or module_order(scenario))}, "group_d": 0.0}   # D84 시간 분해
    pipes, rmap, order, family, trees = expand_trees(scenario, routes)
    cls = {pid: centerlines_of(pipes[pid], r) for pid, r in rmap.items()}
    timing["centerline"] = time.perf_counter() - t0
    from ..support_std import load_standard
    ctx = Context(scenario, pipes, rmap, cls, std=std or load_standard(), family=family, trees=trees)
    out = {pid: pipe_report(ctx, pipes[pid], modules, timing) for pid in order}
    if modules is None or "support" in modules:
        from .support_group import attach_together, group_d
        t1 = time.perf_counter()
        attach_together(out)   # D89: 함께 얹힌 배관에 D81 몫 사본
        group_d(ctx, out)   # D80 ⑥: 배치 후 병합만 (위치 불변)
        timing["group_d"] = time.perf_counter() - t1
    legs = merge_trees(ctx, out)
    res = {"pipes": out, "sec": time.perf_counter() - t0, "support_standard": ctx.std.info(), "timing": timing}
    if legs:
        res["legs"] = legs
    return res


def expand_trees(scenario: Scenario, routes: list):
    """D134 분기 계통 → 주관 + 다리 가상 배관. 반환 (pipes, rmap, 판정 순서, family, trees).
    분기 없는 배관은 그대로 (분기 없는 시나리오 = 이전과 같은 사전·순서). 다리 가상 배관은 자기 주관 바로 뒤 순서
    (`추정:` — 서포트 D89 id·묶음 탐욕 순서에 쓰인다). PipeRoute.branches = pipe.branches 순서의 다리 기하 (티 → 단자),
    없는 다리는 빈 목록. 분기 단자가 없는 배관의 branches (S0 시험 경로, D44) 는 이전처럼 주관 중심선에 붙인다."""
    from ..tree import leg_flow, leg_flow_pipe
    pipes = {p.id: p for p in scenario.pipes}
    rmap, order, family, trees = {}, [], {}, {}
    byid = {r.pipe_id: r for r in routes if r.waypoints}
    for p in scenario.pipes:
        order.append(p.id)
        r = byid.get(p.id)
        if r is None:
            continue
        if not p.branches:
            rmap[p.id] = r
            continue
        trees[p.id] = r
        rmap[p.id] = PipeRoute(p.id, r.waypoints, [])
        geoms = list(r.branches) + [[]] * (len(p.branches) - len(r.branches))
        for b, geom in zip(p.branches, geoms):
            if geom is None or len(geom) < 2:
                continue
            vp = leg_flow_pipe(p, b, geom)
            pipes[vp.id] = vp
            order.append(vp.id)
            rmap[vp.id] = PipeRoute(vp.id, leg_flow(geom, b.flow), [])
            family[vp.id] = p.id
    return pipes, rmap, order, family, trees


def _and(a, b):
    if a is False or b is False:
        return False
    if a is None:
        return b
    if b is None:
        return a
    return True


def merge_trees(ctx: Context, out: dict) -> dict:
    """다리 가상 배관 보고를 계통 (주관) 보고에 합친다 — Layer 0 = 모두 통과해야 통과 (D112 Q10), 위반·관이음·서포트를
    더하고 티 관이음 (무게 0, D112) 을 붙인다. 반환 {계통 id: {다리 id: PipeReport}} (J 분해용)."""
    legs = {}
    for vid, root in ctx.family.items():
        rep = out.pop(vid, None)
        if rep is None:
            continue
        legs.setdefault(root, {})[vid] = rep
        P = out[root]
        if not hasattr(P, "main_fittings"):   # J 분해용 (주관 몫) — 데이터 필드가 아니라 출력 to_dict 에는 없다
            P.main_fittings, P.main_supports = list(P.fittings), list(P.supports)
        for k in P.layer0:
            P.layer0[k] = _and(P.layer0[k], rep.layer0.get(k))
        P.violations += rep.violations
        leg = ctx.pipes[vid].leg
        geom = ctx.trees[root].branches[ctx.pipes[root].branches.index(leg)]
        P.fittings += [{"type": "tee", "pos": [float(v) for v in geom[0]], "leg": leg.id}] + rep.fittings
        P.supports += rep.supports
        P.support_gaps += rep.support_gaps
    return legs


def centerlines_of(pipe: Pipe, r: PipeRoute) -> list:
    """주관 + 분기 실제 중심선 (D44). 분기 계통 (D134) 은 expand_trees 가 다리를 가상 배관으로 떼어 내므로 여기에는
    주관만 온다."""
    R = elbow_radius(pipe.nominal_size)
    return [build_centerline(r.waypoints, R, ANGLE_TOL_DEG)] + \
           [build_centerline(b, R, ANGLE_TOL_DEG) for b in r.branches if len(b) >= 2]


def pipe_report(ctx: Context, p: Pipe, modules=None, timing=None) -> "PipeReport":
    """배관 하나의 Layer 0 판정 (D 묶음 전). 증분 검증(D85)도 이 함수를 쓴다."""
    rep = PipeReport(p.id, p.id in ctx.routes)
    if rep.routed:
        for name in modules or module_order(ctx.scenario):
            t = time.perf_counter()
            res = MODULES[name](ctx, ctx.routes[p.id])
            if timing is not None:
                timing[name] += time.perf_counter() - t
            rep.layer0[name] = None if res is None else not res
            rep.violations += res or []
        rep.fittings = fittings_of(ctx.centerlines[p.id], p)
        rep.supports = ctx.cache.get(("supports", p.id), [])
        rep.support_gaps = ctx.cache.get(("support_gaps", p.id), [])
    return rep


def fittings_of(cls: list, pipe: Pipe) -> list:
    """관이음 청구: 주관 꺾임점별 엘보 (§3.4, D25), 분기마다 티, 밸브. 명목각이 아닌 꺾임은 bend 위반이며 청구하지 않는다."""
    out = []
    for leg in cls[1:]:
        out.append({"type": "tee", "pos": [float(v) for v in leg.points[0]]})
    for b in cls[0].bends:
        pos = [float(v) for v in b.pos]
        if b.nominal == 45:
            out.append({"type": "elbow_45", "pos": pos})
        elif b.nominal == 90:
            out.append({"type": "elbow_90", "pos": pos})
        elif b.nominal == 135:
            out += [{"type": "elbow_90", "pos": pos}, {"type": "elbow_45", "pos": pos}]
    for v in pipe.valve_positions:
        out.append({"type": "gate_valve", "pos": [float(c) for c in v]})
    return out
