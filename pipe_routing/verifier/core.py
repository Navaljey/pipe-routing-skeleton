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
MODULE_ORDER = ["collision", "boundary", "bend", "gravity_slope", "valve", "branch", "support"]   # §6.1
MODULES: dict[str, Callable] = {}


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
    supports: list = field(default_factory=list)     # [{"pos", "face", "angle_length_mm", "kg"}]

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

    def others(self, pipe_id: str):
        """다른 배관들의 중심선 조각 배열 (A, B, 반경, slack, id)."""
        key = ("others", pipe_id)
        if key not in self.cache:
            A, B, R, S, ids = [], [], [], [], []
            for pid, cls in self.centerlines.items():
                if pid == pipe_id:
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


def verify(scenario: Scenario, routes: list, modules=None) -> dict:
    """routes: PipeRoute 목록 (경로를 못 찾은 배관은 빠져 있거나 waypoints 가 비어 있다).

    반환: {"pipes": {pipe_id: PipeReport}, "sec": 검증 시간}
    """
    from . import modules as _  # noqa: F401  (모듈 등록)
    t0 = time.perf_counter()
    pipes = {p.id: p for p in scenario.pipes}
    rmap = {r.pipe_id: r for r in routes if r.waypoints}
    cls = {}
    for pid, r in rmap.items():
        R = elbow_radius(pipes[pid].nominal_size)
        cls[pid] = [build_centerline(r.waypoints, R, ANGLE_TOL_DEG)] + \
                   [build_centerline(b, R, ANGLE_TOL_DEG) for b in r.branches if len(b) >= 2]
    ctx = Context(scenario, pipes, rmap, cls)
    out = {}
    for p in scenario.pipes:
        rep = PipeReport(p.id, p.id in rmap)
        if rep.routed:
            for name in modules or MODULE_ORDER:
                res = MODULES[name](ctx, rmap[p.id])
                rep.layer0[name] = None if res is None else not res
                rep.violations += res or []
            rep.fittings = fittings_of(cls[p.id], p)
            rep.supports = ctx.cache.get(("supports", p.id), [])
        out[p.id] = rep
    return {"pipes": out, "sec": time.perf_counter() - t0}


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
