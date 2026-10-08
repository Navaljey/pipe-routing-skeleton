"""입력 스키마 (CLAUDE.md §7.1) — 로드·검증·저장.

검증은 두 층이다.
  1) 구조: schemas/scenario_input.schema.json (jsonschema)
  2) 의미: 좌표 스냅(D5), 블록 범위, 단자 규약(D28·D29), 이격(D17) 등
오류(error)가 있으면 로드 실패. 경고(warning)는 시나리오가 풀리지 않을 수 있다는 신호일 뿐이다.
"""
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import jsonschema

from .constants import (GRAVITY_TYPES, INSULATION_MM, PIPE_SPECS, S0_ROUTER_TYPES,
                        SNAP_MM, effective_radius)
from .geometry import Box, Vec3, box_distance, union_volume

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "scenario_input.schema.json"
EPS = 1e-6
AXIS_DIRS = {(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)}


class ScenarioError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("시나리오 검증 실패:\n  " + "\n  ".join(errors))
        self.errors = errors


@dataclass
class Block:
    width: float
    length: float
    height: float

    @property
    def extent(self) -> Vec3:
        return (self.width, self.length, self.height)


@dataclass
class Obstacle:
    id: str
    type: str
    box: Box


@dataclass
class Terminal:
    pos: Vec3
    kind: str              # "nozzle" | "boundary" (D6)
    dir: tuple[int, int, int]  # D29: 단자에서의 흐름 방향
    host: Optional[str] = None


@dataclass
class Pipe:
    id: str
    type_id: str
    nominal_size: str
    gravity_pipe: bool
    min_slope: Optional[float]
    insulation_thickness: float
    start: Terminal
    end: Terminal
    valve_positions: list[Vec3] = field(default_factory=list)
    branch_points: list[Vec3] = field(default_factory=list)

    @property
    def radius(self) -> float:
        """유효 반경 (D17)."""
        return effective_radius(self.nominal_size, self.insulation_thickness)

    @property
    def min_straight(self) -> int:
        return PIPE_SPECS[self.nominal_size].min_straight


@dataclass
class Scenario:
    block: Block
    obstacles: list[Obstacle]
    pipes: list[Pipe]
    meta: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        vol = self.block.width * self.block.length * self.block.height
        clipped = [Box(tuple(max(0.0, c) for c in o.box.min),
                       tuple(min(e, c) for e, c in zip(self.block.extent, o.box.max)))
                   for o in self.obstacles]
        types: dict[str, int] = {}
        for p in self.pipes:
            types[p.type_id] = types.get(p.type_id, 0) + 1
        return {
            "name": self.meta.get("name", ""),
            "n_obstacles": len(self.obstacles),
            "obstacle_fill": round(union_volume(clipped) / vol, 4),
            "n_pipes": len(self.pipes),
            "pipe_types": dict(sorted(types.items())),
            "n_warnings": len(self.warnings),
        }


# ---------------------------------------------------------------- 단자 기하

def terminal_stub(pipe: Pipe, which: str, length: Optional[float] = None) -> Box:
    """단자에서 접속 방향으로 뻗는 직진 구간 (퇴화 박스).

    start: pos → pos + dir·L (출발 방향), end: pos − dir·L → pos (도착 방향). D29.
    """
    t = pipe.start if which == "start" else pipe.end
    L = pipe.min_straight if length is None else length
    sign = 1 if which == "start" else -1
    far = tuple(p + sign * d * L for p, d in zip(t.pos, t.dir))
    return Box.of_points(t.pos, far)


def boundary_face(pos: Vec3, extent: Vec3) -> list[tuple[int, int]]:
    """pos 가 놓인 블록 면 목록 [(축, -1|+1)]."""
    faces = []
    for ax in range(3):
        if abs(pos[ax]) < EPS:
            faces.append((ax, -1))
        elif abs(pos[ax] - extent[ax]) < EPS:
            faces.append((ax, +1))
    return faces


def inside_block_with_margin(box: Box, extent: Vec3, r: float, skip_axis: Optional[int] = None) -> bool:
    """박스가 블록 안쪽 r 여유 내에 들어오는가. skip_axis 는 경계 관통 축(검사 제외)."""
    for ax in range(3):
        if ax == skip_axis:
            continue
        if box.min[ax] < r - EPS or box.max[ax] > extent[ax] - r + EPS:
            return False
    return True


def obstacle_clearance(box: Box, obstacles: list[Obstacle]) -> tuple[float, Optional[str]]:
    """박스와 가장 가까운 장애물까지 거리와 그 id."""
    best, who = math.inf, None
    for o in obstacles:
        d = box_distance(box, o.box)
        if d < best:
            best, who = d, o.id
    return best, who


def terminal_errors(pipe: Pipe, which: str, block: Block, obstacles: list[Obstacle]) -> list[str]:
    """단자 하나의 필수 조건 (D28·D29·D17). 생성기와 로더가 같이 쓴다."""
    t = pipe.start if which == "start" else pipe.end
    tag = f"{pipe.id}.{which}"
    r = pipe.radius
    ext = block.extent
    errs = []
    if t.dir not in AXIS_DIRS:
        return [f"{tag}: dir {list(t.dir)} 은 축 방향 단위벡터가 아님 (D29)"]
    ax = next(i for i in range(3) if t.dir[i] != 0)
    point = Box(t.pos, t.pos)
    if t.kind == "boundary":
        faces = boundary_face(t.pos, ext)
        if len(faces) != 1:
            return [f"{tag}: 경계 단자는 블록 면 하나 위에 있어야 함 (모서리·내부 불가)"]
        f_ax, f_side = faces[0]
        # start 는 블록 안으로(면 법선 반대), end 는 블록 밖으로(면 법선)
        want = -f_side if which == "start" else f_side
        if ax != f_ax or t.dir[ax] != want:
            errs.append(f"{tag}: 경계 단자 dir 은 면에 수직이고 "
                        f"{'블록 안쪽' if which == 'start' else '블록 바깥쪽'}을 향해야 함 (D29)")
        if not inside_block_with_margin(point, ext, r, skip_axis=f_ax):
            errs.append(f"{tag}: 경계 단자가 면 가장자리에서 유효 반경 {r:.2f} 이내")
    else:
        if not inside_block_with_margin(point, ext, r):
            errs.append(f"{tag}: 노즐 단자가 블록 경계에서 유효 반경 {r:.2f} 이내")
        if t.host is not None and t.host not in {o.id for o in obstacles}:
            errs.append(f"{tag}: host '{t.host}' 장애물 없음")
    d, who = obstacle_clearance(point, obstacles)
    if d < r - EPS:
        errs.append(f"{tag}: 장애물 {who} 와 거리 {d:.2f} < 유효 반경 {r:.2f} (D17, D28)")
    return errs


def terminal_warnings(pipe: Pipe, which: str, block: Block, obstacles: list[Obstacle]) -> list[str]:
    """단자 직진 구간(§3.3 최소 직진)이 막혀 있으면 경고."""
    t = pipe.start if which == "start" else pipe.end
    stub = terminal_stub(pipe, which)
    skip = boundary_face(t.pos, block.extent)[0][0] if t.kind == "boundary" else None
    out = []
    if not inside_block_with_margin(stub, block.extent, pipe.radius, skip_axis=skip):
        out.append(f"{pipe.id}.{which}: 직진 구간 {pipe.min_straight}mm 가 블록 밖으로 나감")
    d, who = obstacle_clearance(stub, obstacles)
    if d < pipe.radius - EPS:
        out.append(f"{pipe.id}.{which}: 직진 구간 {pipe.min_straight}mm 가 장애물 {who} 와 간섭")
    return out


# ---------------------------------------------------------------- 파싱

def _vec(v) -> Vec3:
    return tuple(float(c) for c in v)


def _terminal(d: dict) -> Terminal:
    return Terminal(_vec(d["pos"]), d["kind"], tuple(int(c) for c in d["dir"]), d.get("host"))


def from_dict(data: dict) -> Scenario:
    """구조 검증 → 객체화 → 의미 검증. 오류가 있으면 ScenarioError."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    struct = [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}"
              for e in jsonschema.Draft202012Validator(schema).iter_errors(data)]
    if struct:
        raise ScenarioError(struct)

    b = data["block"]
    sc = Scenario(
        block=Block(float(b["width"]), float(b["length"]), float(b["height"])),
        obstacles=[Obstacle(o["id"], o["type"], Box(_vec(o["min"]), _vec(o["max"])))
                   for o in data["obstacles"]],
        pipes=[Pipe(p["id"], p["type_id"], p["nominal_size"], p["gravity_pipe"], p["min_slope"],
                    float(p["insulation_thickness"]), _terminal(p["start"]), _terminal(p["end"]),
                    [_vec(v) for v in p["valve_positions"]], [_vec(v) for v in p["branch_points"]])
               for p in data["pipes"]],
        meta=dict(data.get("meta", {})),
    )
    errors, warnings = validate(sc)
    if errors:
        raise ScenarioError(errors)
    sc.warnings = warnings
    return sc


def validate(sc: Scenario) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    ext = sc.block.extent

    def snapped(tag: str, v: Vec3):
        if any(abs(c / SNAP_MM - round(c / SNAP_MM)) > EPS for c in v):
            errors.append(f"{tag}: 좌표 {list(v)} 가 {SNAP_MM}mm 격자에 맞지 않음 (D5)")

    snapped("block", ext)
    ids = set()
    for o in sc.obstacles:
        if o.id in ids:
            errors.append(f"장애물 id 중복: {o.id}")
        ids.add(o.id)
        snapped(f"{o.id}.min", o.box.min)
        snapped(f"{o.id}.max", o.box.max)
        if any(lo >= hi for lo, hi in zip(o.box.min, o.box.max)):
            errors.append(f"{o.id}: min < max 가 아님")
        if any(lo < 0 or hi > e for lo, hi, e in zip(o.box.min, o.box.max, ext)):
            errors.append(f"{o.id}: 블록 범위 밖")

    ids = set()
    for p in sc.pipes:
        if p.id in ids:
            errors.append(f"배관 id 중복: {p.id}")
        ids.add(p.id)
        if p.gravity_pipe != (p.type_id in GRAVITY_TYPES):
            errors.append(f"{p.id}: gravity_pipe={p.gravity_pipe} 가 type_id {p.type_id} 와 불일치 (D27)")
        if p.gravity_pipe != (p.min_slope is not None):
            errors.append(f"{p.id}: min_slope 는 중력관만 지정 (D13)")
        if p.type_id not in S0_ROUTER_TYPES:
            warnings.append(f"{p.id}: {p.type_id} 는 S0 라우터 대상 아님 (D27)")
        if p.insulation_thickness != INSULATION_MM:
            warnings.append(f"{p.id}: 보온재 {p.insulation_thickness}mm ≠ D17 기준 {INSULATION_MM}mm")
        for which in ("start", "end"):
            snapped(f"{p.id}.{which}", getattr(p, which).pos)
            errors += terminal_errors(p, which, sc.block, sc.obstacles)
        for i, v in enumerate(p.valve_positions):
            snapped(f"{p.id}.valve[{i}]", v)
        for i, v in enumerate(p.branch_points):
            snapped(f"{p.id}.branch[{i}]", v)
        if p.start.pos == p.end.pos:
            errors.append(f"{p.id}: start 와 end 가 같은 위치")
    if errors:
        return errors, warnings

    for p in sc.pipes:
        for which in ("start", "end"):
            warnings += terminal_warnings(p, which, sc.block, sc.obstacles)
        if p.gravity_pipe:
            drop = p.start.pos[2] - p.end.pos[2]
            horiz = math.dist(p.start.pos[:2], p.end.pos[:2])
            if drop < p.min_slope * horiz - EPS:
                warnings.append(f"{p.id}: 낙차 {drop:.0f}mm < 최소 구배 {p.min_slope:g} × 수평거리 "
                                f"{horiz:.0f}mm — 구배 충족 불가 (§3.5)")

    # 다른 배관 단자끼리 직진 구간이 겹치면 이격 위반이 불가피하다
    stubs = [(p, w, terminal_stub(p, w)) for p in sc.pipes for w in ("start", "end")]
    for i, (p, w, s) in enumerate(stubs):
        for q, v, t in stubs[i + 1:]:
            if p is q:
                continue
            need = p.radius + q.radius
            if box_distance(s, t) < need - EPS:
                warnings.append(f"{p.id}.{w} ↔ {q.id}.{v}: 단자 직진 구간 거리 < {need:.2f} (D17)")
    return errors, warnings


def to_dict(sc: Scenario) -> dict:
    def num(c: float):
        return int(c) if float(c).is_integer() else c

    def vec(v):
        return [num(c) for c in v]

    def term(t: Terminal) -> dict:
        d = {"pos": vec(t.pos), "kind": t.kind, "dir": list(t.dir)}
        if t.host is not None:
            d["host"] = t.host
        return d

    out = {}
    if sc.meta:
        out["meta"] = sc.meta
    out["block"] = {"width": num(sc.block.width), "length": num(sc.block.length),
                    "height": num(sc.block.height), "unit": "mm"}
    out["obstacles"] = [{"id": o.id, "type": o.type, "min": vec(o.box.min), "max": vec(o.box.max)}
                        for o in sc.obstacles]
    out["pipes"] = [{
        "id": p.id, "type_id": p.type_id, "nominal_size": p.nominal_size,
        "gravity_pipe": p.gravity_pipe, "min_slope": p.min_slope,
        "insulation_thickness": num(p.insulation_thickness),
        "start": term(p.start), "end": term(p.end),
        "valve_positions": [vec(v) for v in p.valve_positions],
        "branch_points": [vec(v) for v in p.branch_points],
    } for p in sc.pipes]
    return out


def load(path) -> Scenario:
    return from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def save(sc: Scenario, path) -> None:
    Path(path).write_text(json.dumps(to_dict(sc), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    """python -m pipe_routing.scenario <json>... — 로드하고 요약을 출력한다."""
    import argparse
    ap = argparse.ArgumentParser(description="시나리오 로드 검사 (1단계 완료 기준: 로드 성공)")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("-v", "--verbose", action="store_true", help="경고 내용 출력")
    args = ap.parse_args(argv)
    failed = 0
    for path in args.paths:
        try:
            sc = load(path)
        except ScenarioError as e:
            failed += 1
            print(f"FAIL {path}\n  " + "\n  ".join(e.errors))
            continue
        print(f"OK   {path} {json.dumps(sc.summary(), ensure_ascii=False)}")
        if args.verbose:
            for w in sc.warnings:
                print(f"     경고: {w}")
    print(f"\n로드 {len(args.paths) - failed}/{len(args.paths)} 성공")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
