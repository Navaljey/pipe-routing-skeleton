"""Procedural 시나리오 생성기 (1단계).

v3 `g3_colab.ipynb` 셀 7 의 gen_obstacle_grid 를 옮긴 것이다. 차이:
  - 격자(20×20×10 셀) 대신 mm 좌표 박스 목록을 출력한다 (D10, D5 10mm 스냅)
  - 박스 크기·채움률은 v3 비율을 블록(D4)에 그대로 대응시킨다 (D30)
  - 배관은 T1·T5 만 (D27), 단자 규약은 D28·D29, 경계 단자는 6면 모두 (D35)
  - 도달성 검증(v3 의 astar_gnarl)은 라우터가 생기는 4단계 이후로 미룬다 (D31)
"""
import argparse
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .constants import (BLOCK_HEIGHT, BLOCK_LENGTH, BLOCK_WIDTH, CLEARANCE_MM, INSULATION_MM, NOMINAL_SIZES,
                        SNAP_MM, default_drain_slope, effective_radius)
from .geometry import Box, box_distance, union_volume
from .scenario import (Block, Obstacle, Pipe, Scenario, ScenarioError, Terminal, save,
                       terminal_errors, terminal_stub, terminal_warnings, validate)


@dataclass
class GeneratorConfig:
    seed: int = 0
    # 장애물 (D30: v3 셀 단위 범위를 블록 크기 비율로 환산)
    obstacle_fill: float = 0.12                  # v3 cfg.obstacle_fill
    box_xy_mm: tuple[int, int] = (4000, 8000)    # v3 2~4셀 / 20셀 × 40m
    box_z_mm: tuple[int, int] = (1000, 4000)     # v3 1~4셀 / 10셀 × 10m
    max_boxes: int = 1000                        # v3 guard
    # 배관
    n_pipes: int = 10
    sizes: tuple[str, ...] = NOMINAL_SIZES       # D18
    gravity_ratio: float = 0.3                   # T5 비율 (나머지 T1)
    nozzle_ratio: float = 0.5                    # 단자가 노즐일 확률 (나머지 경계)
    deck_ratio: float = 0.5                      # 경계 단자 중 데크 관통 비율 (D35, 나머지 측면 격벽)
    min_terminal_dist_mm: int = 5000             # start–end 맨해튼 거리 하한 (v3 min_path_cells 대응)
    max_tries: int = 500                         # 배관 1개 단자 샘플링 시도 횟수
    stair_rule: bool = False                     # D64: 중력관 낙차 ≥ 계단식 하향 규칙(D61) 하한 — 새 세트에만
    d94_rule: bool = False                       # D107: 경계 단자 직진 구간 ≥ max(3D, r) + t₁₃₅ (D94·D96) — 새 세트에만
    exclude_faces: tuple = ()                    # 경계 단자를 두지 않을 면 [(축, -1|+1)] — 블록 쌍 공유 격벽 (D76)
    services: bool = False                       # D116·D117: 배관마다 서비스 (→ class·보온) — 별도 난수열, 기본 꺼짐
    internal: bool = False                       # D113: 내부 격벽·데크·필라·스티프너 — 별도 난수열, 기본 꺼짐
    access: bool = False                         # D119: 주 통로 + 장비 정비 공간 — 별도 난수열, 기본 꺼짐
    stiffeners: tuple = (10, 20)                 # D113 스티프너 개수 범위 (internal 일 때만) — 기능 시험용 작은 세트 (4, 8) (D123 ②)
    maint_ratio: float = 1.0                     # D119 정비 공간을 두는 장비 비율 (access 일 때만) — 작은 세트 0.3 (D123 ②)


# S2 생성 옵션 (D115 보완): 꺼져 있으면 meta.config 에 쓰지 않는다 — 기본 출력 비트 동일
OPTION_KEYS = ("services", "internal", "access")
SUB_OPTIONS = {"stiffeners": "internal", "maint_ratio": "access"}   # 부모 옵션이 꺼져 있으면 meta 에 쓰지 않는다
SERVICE_SALT = 0x5E5   # `추정:` 서비스 난수열 = Random(seed ^ SERVICE_SALT) — 기본 난수열 소비 불변
INTERNAL_SALT = 0x1A7  # `추정:` 내부 구조재 난수열 (D113)
ACCESS_SALT = 0xACC    # `추정:` 접근 구역 난수열 (D119)
PLATE_T = 20           # `추정:` 격벽·데크 두께 mm — s2_plan §2 의 12 mm 강판은 10 mm 격자 (D5) 밖이라 20 으로 올림
PILLAR_MM = 300        # `추정:` 필라 단면
STIFF_W, STIFF_T = 150, 20   # `추정:` 스티프너 평판 150 × 12 → 10 mm 격자에 맞춰 두께 20
PASSAGE_W, PASSAGE_H, MAINT_D = 800, 2000, 800   # D119 `추정:` 주 통로 폭·높이, 정비 공간 깊이


def _overlap(a: Box, b: Box) -> bool:
    return all(a.min[k] < b.max[k] and b.min[k] < a.max[k] for k in range(3))


def gen_passage(arng: random.Random) -> Box:
    """D119 주 통로: 데크 상면 (z 0 ~ 2,000), 폭 800, x 또는 y 방향 블록 관통, 위치 균등 (`추정:`)."""
    ext = (BLOCK_WIDTH, BLOCK_LENGTH, BLOCK_HEIGHT)
    ax = arng.randrange(2)
    o = 1 - ax
    c = _rand_snapped(arng, 2000, ext[o] - 2000 - PASSAGE_W)
    lo, hi = [0.0, 0.0, 0.0], [float(ext[0]), float(ext[1]), float(PASSAGE_H)]
    lo[o], hi[o] = float(c), float(c + PASSAGE_W)
    return Box(tuple(lo), tuple(hi))


def gen_plates(irng: random.Random) -> list:
    """D113 판 (`추정:` s2_plan §2): 격벽 2~3 (x 또는 y 법선, 블록 전폭·전고, 같은 방향끼리 4 m 이상 떨어짐),
    데크 0~1 (z 3,000~7,000 전면). 장비보다 먼저 놓고 장비가 판을 피한다. 반환 [(종류, 박스, 관통 축)]."""
    W, L, H = BLOCK_WIDTH, BLOCK_LENGTH, BLOCK_HEIGHT
    out = []
    for _ in range(irng.randint(2, 3)):
        ax = irng.randrange(2)
        for _try in range(50):
            c = _rand_snapped(irng, 6000, (W if ax == 0 else L) - 6000)
            if all(abs(c - b.min[a]) >= 4000 for k, b, a in out if a == ax):
                lo = [0.0, 0.0, 0.0]; hi = [float(W), float(L), float(H)]
                lo[ax], hi[ax] = float(c), float(c + PLATE_T)
                out.append(("bulkhead", Box(tuple(lo), tuple(hi)), ax)); break
    if irng.random() < 0.5:
        z = _rand_snapped(irng, 3000, 7000)
        out.append(("deck", Box((0.0, 0.0, float(z)), (float(W), float(L), float(z + PLATE_T))), 2))
    return out


def gen_internal(irng: random.Random, plates: list, equip: list, avoid: list, stiff=(10, 20)) -> list:
    """D113 판 + 필라 4~8 (300 × 300, 전고) + 스티프너 10~20 (격벽·블록 벽 면에 붙은 평판 150 × 20, 길이 2~6 m)
    (`추정:` s2_plan §2). 필라·스티프너는 장비·접근 구역·(필라는) 격벽과 겹치지 않게."""
    W, L, H = BLOCK_WIDTH, BLOCK_LENGTH, BLOCK_HEIGHT
    solid = [o.box for o in equip] + list(avoid)
    bulk = [b for k, b, a in plates if k == "bulkhead"]
    out = [(k, b, a) for k, b, a in plates]

    def free(b, against):
        return not any(_overlap(b, a) for a in against)
    for _ in range(irng.randint(4, 8)):
        for _try in range(50):
            x = _rand_snapped(irng, 1000, W - 1000 - PILLAR_MM); y = _rand_snapped(irng, 1000, L - 1000 - PILLAR_MM)
            b = Box((float(x), float(y), 0.0), (float(x + PILLAR_MM), float(y + PILLAR_MM), float(H)))
            if free(b, solid + bulk):
                out.append(("pillar", b, None)); break
    walls = [(0, 0.0, +1), (0, float(W), -1), (1, 0.0, +1), (1, float(L), -1)]
    walls += [(a, b.min[a], -1) for b, a in ((b, a) for k, b, a in plates if k == "bulkhead")]
    walls += [(a, b.max[a], +1) for b, a in ((b, a) for k, b, a in plates if k == "bulkhead")]
    for _ in range(irng.randint(*stiff)):
        for _try in range(50):
            wa, wc, sgn = irng.choice(walls)
            o = 1 - wa
            ln = _snap(irng.uniform(2000, 6000))
            vertical = irng.random() < 0.5
            s0 = _rand_snapped(irng, 500, (L if o == 1 else W) - 500 - (STIFF_W if vertical else ln))
            z0 = _rand_snapped(irng, 500, H - 500 - (ln if vertical else STIFF_W))
            lo = [0.0, 0.0, 0.0]; hi = [0.0, 0.0, 0.0]
            lo[wa], hi[wa] = (wc, wc + STIFF_T) if sgn > 0 else (wc - STIFF_T, wc)
            lo[o], hi[o] = float(s0), float(s0 + (STIFF_W if vertical else ln))
            lo[2], hi[2] = float(z0), float(z0 + (ln if vertical else STIFF_W))
            b = Box(tuple(lo), tuple(hi))
            if free(b, solid):
                out.append(("stiffener", b, None)); break
    obs = []
    for k, (kind, b, ax) in enumerate(out):
        pen = kind in ("bulkhead", "deck")
        obs.append(Obstacle(f"INT_{k + 1:03d}", kind, b, True, pen, ax if pen else None))
    return obs


def gen_maintenance(arng: random.Random, equip: list, solid: list, ratio: float = 1.0) -> list:
    """D119 장비 정비 공간: 장비마다 수평 4면 중 하나 (균등 `추정:`) 앞 깊이 800 × 장비 높이 × 면 폭. 블록 밖은 잘라내고,
    다른 장비·내부 구조재와 겹치면 건너뛴다."""
    out = []
    ext = (BLOCK_WIDTH, BLOCK_LENGTH, BLOCK_HEIGHT)
    for o in equip:
        if ratio < 1.0 and arng.random() >= ratio:   # 비율 1 이면 난수를 쓰지 않는다 (기존 세트 비트 동일)
            continue
        ax = arng.randrange(2)
        side = arng.choice((-1, 1))
        lo, hi = list(o.box.min), list(o.box.max)
        if side > 0:
            lo[ax], hi[ax] = o.box.max[ax], min(float(ext[ax]), o.box.max[ax] + MAINT_D)
        else:
            lo[ax], hi[ax] = max(0.0, o.box.min[ax] - MAINT_D), o.box.min[ax]
        if hi[ax] - lo[ax] < 100:
            continue
        b = Box(tuple(lo), tuple(hi))
        if any(_overlap(b, s) for s in solid if s is not o.box):
            continue
        out.append(Obstacle(f"ACC_{len(out) + 2:03d}", "maintenance", b, access=True))
    return out



def _config_meta(cfg) -> dict:
    d = asdict(cfg)
    for k in OPTION_KEYS:
        if k in d and not d[k]:
            del d[k]
    for k, parent in SUB_OPTIONS.items():
        if parent not in d:
            d.pop(k, None)
        elif k in d and isinstance(d[k], tuple):
            d[k] = list(d[k])
    return d


def _pick_service(svc_rng, size: str, gravity: bool):
    """(서비스 코드, 보온 mm, 유효 반경) — 압력 배관 = 압력 서비스 균등, 중력 배관 = 중력 서비스 균등 (D116, `추정:`)."""
    from . import pipe_spec
    codes = [c for c, s in pipe_spec.services().items() if s.gravity == gravity]
    code = svc_rng.choice(codes)
    cls, ins = pipe_spec.resolve(code, None, None)
    return code, ins, pipe_spec.spec(cls, size).od / 2 + ins + CLEARANCE_MM


def _snap(v: float) -> int:
    return int(round(v / SNAP_MM)) * SNAP_MM


def _snap_up(v: float) -> int:
    return int(math.ceil(v / SNAP_MM - 1e-9)) * SNAP_MM


def _rand_snapped(rng: random.Random, lo: float, hi: float) -> Optional[int]:
    lo_s, hi_s = _snap_up(lo), int(math.floor(hi / SNAP_MM)) * SNAP_MM
    if lo_s > hi_s:
        return None
    return rng.randrange(lo_s, hi_s + 1, SNAP_MM)


def gen_obstacles(rng: random.Random, cfg: GeneratorConfig, avoid=()) -> list[Obstacle]:
    """avoid = 겹치면 안 되는 박스 (D119 주 통로) — 겹친 후보는 버린다 (난수는 같은 순서로 쓴다, avoid 가 없으면 이전과 같음)."""
    ext = (BLOCK_WIDTH, BLOCK_LENGTH, BLOCK_HEIGHT)
    target = cfg.obstacle_fill * math.prod(ext)
    boxes: list[Box] = []
    filled = 0.0
    guard = 0
    while filled < target and guard < cfg.max_boxes:
        guard += 1
        size = (_snap(rng.uniform(*cfg.box_xy_mm)), _snap(rng.uniform(*cfg.box_xy_mm)),
                _snap(rng.uniform(*cfg.box_z_mm)))
        lo = tuple(_rand_snapped(rng, 0, e - s) for e, s in zip(ext, size))
        cand = Box(lo, tuple(l + s for l, s in zip(lo, size)))
        if avoid and any(_overlap(cand, a) for a in avoid):
            continue
        boxes.append(cand)
        filled = union_volume(boxes)
    return [Obstacle(f"OBS_{i + 1:03d}", "box", b) for i, b in enumerate(boxes)]


def _sample_nozzle(rng, which, size, obstacles, r=None) -> Optional[Terminal]:
    """장애물 면 위 한 점에서 법선 방향으로 유효 반경만큼 띄운 위치 (D28). r = 배관 유효 반경 (기본 = 보온 50)."""
    r = effective_radius(size) if r is None else r
    host = rng.choice(obstacles)
    ax = rng.randrange(3)
    side = rng.choice((-1, 1))
    pos = [0.0, 0.0, 0.0]
    for a in range(3):
        if a == ax:
            face = host.box.max[a] if side > 0 else host.box.min[a]
            pos[a] = face + side * _snap_up(r)
        else:
            v = _rand_snapped(rng, host.box.min[a] + r, host.box.max[a] - r)
            if v is None:
                return None
            pos[a] = v
    normal = [0, 0, 0]
    normal[ax] = side
    # start 는 노즐에서 나가는 방향(바깥 법선), end 는 노즐로 들어가는 방향 (D29)
    d = tuple(normal) if which == "start" else tuple(-n for n in normal)
    return Terminal(tuple(float(c) for c in pos), "nozzle", d, host.id)


def _sample_boundary(rng, which, size, cfg, gravity, r=None) -> Optional[Terminal]:
    """블록 6면 중 한 면 위 한 점 (D35). 측면 격벽 : 데크 관통 = 1 − deck_ratio : deck_ratio."""
    r = effective_radius(size) if r is None else r
    ext = (BLOCK_WIDTH, BLOCK_LENGTH, BLOCK_HEIGHT)
    if rng.random() < cfg.deck_ratio:
        ax = 2
        # 중력관은 하부 데크에서 start, 상부 데크에서 end 할 수 없다 (D35, D20)
        if gravity:
            side = 1 if which == "start" else -1
        else:
            side = rng.choice((-1, 1))
    else:
        ax = rng.randrange(2)
        side = rng.choice((-1, 1))
    if (ax, side) in cfg.exclude_faces:
        return None
    pos = [0.0, 0.0, 0.0]
    for a in range(3):
        if a == ax:
            pos[a] = 0.0 if side < 0 else float(ext[a])
        else:
            pos[a] = float(_rand_snapped(rng, r, ext[a] - r))
    d = [0, 0, 0]
    d[ax] = -side if which == "start" else side   # start 는 안쪽, end 는 바깥쪽
    return Terminal(tuple(pos), "boundary", tuple(d))


def _sample_terminal(rng, which, size, cfg, obstacles, gravity, r=None):
    hosts = [o for o in obstacles if not (o.supportable or o.penetrable or o.access)]   # 노즐은 장비에만 (D113·D119)
    if hosts and rng.random() < cfg.nozzle_ratio:
        return _sample_nozzle(rng, which, size, hosts, r)
    return _sample_boundary(rng, which, size, cfg, gravity, r)


def gen_pipe(rng: random.Random, cfg: GeneratorConfig, idx: int, block: Block,
             obstacles: list[Obstacle], placed: list[Pipe], svc_rng=None) -> Optional[Pipe]:
    gravity = rng.random() < cfg.gravity_ratio
    size = rng.choice(cfg.sizes)
    slope = default_drain_slope(size) if gravity else None
    code, ins, r = _pick_service(svc_rng, size, gravity) if svc_rng is not None else (None, INSULATION_MM, None)
    for _ in range(cfg.max_tries):
        s = _sample_terminal(rng, "start", size, cfg, obstacles, gravity, r)
        e = _sample_terminal(rng, "end", size, cfg, obstacles, gravity, r)
        if s is None or e is None:
            continue
        if sum(abs(a - b) for a, b in zip(s.pos, e.pos)) < cfg.min_terminal_dist_mm:
            continue
        pipe = Pipe(f"P{idx + 1:03d}", "T5" if gravity else "T1", size, gravity, slope,
                    ins, s, e, service=code)
        # 생성기는 로더의 오류뿐 아니라 경고 조건까지 모두 피한다
        if any(terminal_errors(pipe, w, block, obstacles) or terminal_warnings(pipe, w, block, obstacles)
               for w in ("start", "end")):
            continue
        if gravity and s.pos[2] - e.pos[2] < slope * sum(abs(a - b) for a, b in zip(s.pos[:2], e.pos[:2])):
            continue  # 수평 맨해튼 거리 기준으로 여유 있게 (D31)
        if gravity and cfg.stair_rule:
            from .gravity import feasibility
            if not feasibility(pipe)["feasible"]:
                continue   # D64: 새 규칙의 낙차 하한 (상향 단자 포함 배제)
        if cfg.d94_rule and not d94_stub_ok(pipe, block, obstacles):
            continue   # D107
        if _clashes(pipe, placed):
            continue
        return pipe
    return None


def d94_stub_len(pipe: Pipe) -> float:
    """D107: 경계 단자 직진 구간 보장 길이 = max(3D, r) + t₁₃₅ (D94·D96, 10 mm 올림)."""
    from .blockpair import straight_min
    from .constants import elbow_tangent
    return math.ceil((straight_min(pipe) + elbow_tangent(pipe.nominal_size, 135)) / 10) * 10


def d94_stub_ok(pipe: Pipe, block: Block, obstacles) -> bool:
    """D107: 경계 단자의 D94 직진 구간이 블록 안(면 가장자리 r)·장애물과 r 이상 (노즐은 기존 D31 그대로)."""
    from .scenario import boundary_face, inside_block_with_margin, obstacle_clearance
    for w in ("start", "end"):
        t = getattr(pipe, w)
        if t.kind != "boundary":
            continue
        stub = terminal_stub(pipe, w, d94_stub_len(pipe))
        if not inside_block_with_margin(stub, block.extent, pipe.radius,
                                        skip_axis=boundary_face(t.pos, block.extent)[0][0]):
            return False
        if obstacle_clearance(stub, obstacles)[0] < pipe.radius - 1e-6:
            return False
    return True


def _clashes(pipe: Pipe, placed: list[Pipe]) -> bool:
    for q in placed:
        need = pipe.radius + q.radius
        for w in ("start", "end"):
            for v in ("start", "end"):
                if box_distance(terminal_stub(pipe, w), terminal_stub(q, v)) < need:
                    return True
    return False


def generate(cfg: GeneratorConfig, name: str = "") -> Scenario:
    rng = random.Random(cfg.seed)
    block = Block(BLOCK_WIDTH, BLOCK_LENGTH, BLOCK_HEIGHT)
    arng = random.Random(cfg.seed ^ ACCESS_SALT) if cfg.access else None
    irng = random.Random(cfg.seed ^ INTERNAL_SALT) if cfg.internal else None
    passage = gen_passage(arng) if arng is not None else None
    plates = gen_plates(irng) if irng is not None else []
    avoid = ([passage] if passage else []) + [b for _k, b, _a in plates]
    obstacles = gen_obstacles(rng, cfg, avoid=tuple(avoid))
    equip = list(obstacles)
    if irng is not None:
        obstacles += gen_internal(irng, plates, equip, [passage] if passage else [], tuple(cfg.stiffeners))
    if arng is not None:
        obstacles += [Obstacle("ACC_001", "passage", passage, access=True)]
        obstacles += gen_maintenance(arng, equip, [o.box for o in obstacles], cfg.maint_ratio)
    svc_rng = random.Random(cfg.seed ^ SERVICE_SALT) if cfg.services else None
    pipes: list[Pipe] = []
    for _ in range(cfg.n_pipes):
        p = gen_pipe(rng, cfg, len(pipes), block, obstacles, pipes, svc_rng)
        if p is not None:
            pipes.append(p)
    if not pipes:
        raise RuntimeError(f"seed {cfg.seed}: 배관을 하나도 배치하지 못함")
    meta = {"name": name or f"proc_seed{cfg.seed}", "source": "procedural",
            "generator": "pipe_routing.generator", "config": _config_meta(cfg),
            "n_pipes_requested": cfg.n_pipes}
    sc = Scenario(block, obstacles, pipes, meta)
    errors, warnings = validate(sc)
    if errors:
        raise ScenarioError(errors)
    sc.warnings = warnings
    return sc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="procedural 시나리오 세트 생성")
    ap.add_argument("--out", default="scenarios/procedural")
    ap.add_argument("--n", type=int, default=20, help="시나리오 수 (M5: 시작 20)")
    ap.add_argument("--seed", type=int, default=20261008, help="세트 기준 시드 (i 번째 = seed + i)")
    ap.add_argument("--n-pipes", type=int, default=GeneratorConfig.n_pipes)
    ap.add_argument("--fill", type=float, default=GeneratorConfig.obstacle_fill)
    ap.add_argument("--prefix", default="proc", help="시나리오 이름 앞부분 (밀집 세트 등, D58)")
    ap.add_argument("--stair-rule", action="store_true",
                    help="중력관 낙차를 계단식 하향 규칙(D61) 하한 이상으로 보장 (D64, 새 세트용)")
    ap.add_argument("--d94-rule", action="store_true",
                    help="경계 단자 직진 구간 ≥ max(3D, r) + t₁₃₅ 보장 (D107, 새 세트용)")
    ap.add_argument("--services", action="store_true", help="배관마다 유체 서비스 (D116·D117, 새 세트용)")
    ap.add_argument("--internal", action="store_true", help="내부 격벽·데크·필라·스티프너 (D113, 새 세트용)")
    ap.add_argument("--access", action="store_true", help="주 통로·장비 정비 공간 (D119, 새 세트용)")
    ap.add_argument("--stiffeners", type=int, nargs=2, default=list(GeneratorConfig.stiffeners),
                    help="스티프너 개수 범위 (D113, 기능 시험용 작은 세트 4 8 — D123 ②)")
    ap.add_argument("--maint-ratio", type=float, default=GeneratorConfig.maint_ratio,
                    help="정비 공간을 두는 장비 비율 (D119, 작은 세트 0.3 — D123 ②)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        cfg = GeneratorConfig(seed=args.seed + i, n_pipes=args.n_pipes, obstacle_fill=args.fill,
                              stair_rule=args.stair_rule, d94_rule=args.d94_rule, services=args.services, internal=args.internal,
                              access=args.access, stiffeners=tuple(args.stiffeners),
                              maint_ratio=args.maint_ratio)
        name = f"{args.prefix}_{i:03d}"
        sc = generate(cfg, name)
        save(sc, out / f"{name}.json")
        print(f"{name}: {json.dumps(sc.summary(), ensure_ascii=False)}")
    return 0




# ---------------------------------------------------------------- S1-7 블록 쌍 (D69·D76)

@dataclass
class PairConfig:
    """`추정:` D76 ⑥ — 블록당 기본 세트 수준 채움 + 블록당 10배관 + 관통 4배관, 창 4 m × 2 m."""
    seed: int = 20265008
    n_pipes_per_block: int = 10
    n_crossing: int = 4
    window_mm: tuple = (4000, 2000)      # (y 폭, z 높이)
    wall_thickness: float = 20.0         # 공유 격벽 두께 mm (D93, `추정:` 생성기 기본값)
    stair_rule: bool = True              # 새 세트 → D64 낙차 하한 보장
    d94_rule: bool = True                # 새 세트 → D107 경계 단자·관통 창 직진 구간 보장
    services: bool = False               # D116·D117 (생성기와 같은 별도 난수열, 기본 꺼짐)


def _shift_terminal(t: Terminal, dx: float) -> Terminal:
    return Terminal((t.pos[0] + dx, t.pos[1], t.pos[2]), t.kind, t.dir, t.host)


def _shift_pipe(p: Pipe, dx: float, pid: str, window=None) -> Pipe:
    return Pipe(pid, p.type_id, p.nominal_size, p.gravity_pipe, p.min_slope, p.insulation_thickness,
                _shift_terminal(p.start, dx), _shift_terminal(p.end, dx), crossing_window=window, service=p.service)


def _windows_overlap(a, b) -> bool:
    return a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]


def _end_only(p: Pipe) -> Pipe:
    """start 를 end 와 같은 노즐로 바꾼 사본 (d94_stub_ok 를 end 쪽만 보게)."""
    return Pipe(p.id, p.type_id, p.nominal_size, p.gravity_pipe, p.min_slope, p.insulation_thickness,
                Terminal(p.end.pos, "nozzle", p.end.dir), p.end)


def _window_zone_ok(p: Pipe, win, local: Block, obsA, obsB) -> bool:
    """D107 `추정:` 관통 창 주변 직진 구간 보장 — 창(가장자리에서 r 안쪽) × 격벽 면에서 max(3D, r) + t₁₃₅ 까지의
    상자가 각 블록 장애물과 r 이상 (블록 지역 좌표: A 는 x ∈ [W − L, W], B 는 [0, L])."""
    from .scenario import obstacle_clearance
    y0, y1, z0, z1 = win
    r = p.radius
    L = d94_stub_len(p)
    W = local.width
    lo_y, hi_y, lo_z, hi_z = y0 + r, y1 - r, z0 + r, z1 - r
    if lo_y > hi_y or lo_z > hi_z:
        return False
    boxA = Box((W - L, lo_y, lo_z), (W, hi_y, hi_z))
    boxB = Box((0.0, lo_y, lo_z), (L, hi_y, hi_z))
    return obstacle_clearance(boxA, obsA)[0] >= r - 1e-6 and obstacle_clearance(boxB, obsB)[0] >= r - 1e-6


def generate_block_pair(pc: PairConfig, name: str = "") -> Scenario:
    """블록 A [0, W] · 공유 격벽 [W, W + T] · B [W + T, 2W + T] (x 방향, T = 격벽 두께 D93). 블록마다 기본 생성기(D30)로 장애물·배관, 공유 격벽 면에는 경계 단자를
    두지 않는다. 관통 배관은 start 를 A, end 를 B 에서 뽑고 배관마다 공유 격벽 위 창(4 m × 2 m)을 둔다 —
    창끼리는 겹치지 않게 (`추정:`). 도달성은 보장하지 않는다 (D31)."""
    rng = random.Random(pc.seed)
    W = BLOCK_WIDTH
    T = float(pc.wall_thickness)
    dB = W + T   # B 블록 x 이동
    local = Block(BLOCK_WIDTH, BLOCK_LENGTH, BLOCK_HEIGHT)
    cfgA = GeneratorConfig(seed=pc.seed, n_pipes=pc.n_pipes_per_block, stair_rule=pc.stair_rule, d94_rule=pc.d94_rule,
                           exclude_faces=((0, 1),))
    cfgB = GeneratorConfig(seed=pc.seed, n_pipes=pc.n_pipes_per_block, stair_rule=pc.stair_rule, d94_rule=pc.d94_rule,
                           exclude_faces=((0, -1),))
    obsA = gen_obstacles(rng, cfgA)
    obsB = gen_obstacles(rng, cfgB)
    svc_rng = random.Random(pc.seed ^ SERVICE_SALT) if pc.services else None
    obstacles = [Obstacle(f"OBS_{i + 1:03d}", "box", o.box) for i, o in enumerate(obsA)]
    obstacles += [Obstacle(f"OBS_{len(obsA) + i + 1:03d}", "box",
                           Box((o.box.min[0] + dB, *o.box.min[1:]), (o.box.max[0] + dB, *o.box.max[1:])))
                  for i, o in enumerate(obsB)]
    idB = {o.id: f"OBS_{len(obsA) + i + 1:03d}" for i, o in enumerate(obsB)}
    pipes: list[Pipe] = []
    localA: list[Pipe] = []
    for _ in range(pc.n_pipes_per_block):
        p = gen_pipe(rng, cfgA, len(localA), local, obsA, localA, svc_rng)
        if p is not None:
            localA.append(p)
    localB: list[Pipe] = []
    for _ in range(pc.n_pipes_per_block):
        p = gen_pipe(rng, cfgB, len(localB), local, obsB, localB, svc_rng)
        if p is not None:
            localB.append(p)
    for p in localA:
        pipes.append(_shift_pipe(p, 0.0, f"P{len(pipes) + 1:03d}"))
    for p in localB:
        q = _shift_pipe(p, dB, f"P{len(pipes) + 1:03d}")
        for t in (q.start, q.end):
            if t.host:
                t.host = idB[t.host]
        pipes.append(q)
    # 관통 배관: start ∈ A, end ∈ B
    windows = []
    wy, wz = pc.window_mm
    for k in range(pc.n_crossing):
        for _ in range(GeneratorConfig.max_tries):
            gravity = rng.random() < cfgA.gravity_ratio
            size = rng.choice(cfgA.sizes)
            slope = default_drain_slope(size) if gravity else None
            code, ins, r = _pick_service(svc_rng, size, gravity) if svc_rng is not None else (None, INSULATION_MM, None)
            s_ = _sample_terminal(rng, "start", size, cfgA, obsA, gravity, r)
            e_ = _sample_terminal(rng, "end", size, cfgB, obsB, gravity, r)
            if s_ is None or e_ is None:
                continue
            y0 = _rand_snapped(rng, 0, BLOCK_LENGTH - wy)
            z0 = _rand_snapped(rng, 0, BLOCK_HEIGHT - wz)
            win = (float(y0), float(y0 + wy), float(z0), float(z0 + wz))
            if any(_windows_overlap(win, w) for w in windows):
                continue
            pa = Pipe("tmp", "T5" if gravity else "T1", size, gravity, slope, ins, s_, e_, service=code)
            if terminal_errors(pa, "start", local, obsA) or terminal_warnings(pa, "start", local, obsA):
                continue
            if terminal_errors(pa, "end", local, obsB) or terminal_warnings(pa, "end", local, obsB):
                continue
            if pc.d94_rule and not (d94_stub_ok(pa, local, obsA) and d94_stub_ok(_end_only(pa), local, obsB)):
                continue   # D107 (start 는 A, end 는 B 의 지역 좌표)
            if pc.d94_rule and not _window_zone_ok(pa, win, local, obsA, obsB):
                continue   # D107: 관통 창 양쪽 직진 구간
            e_g = _shift_terminal(e_, dB)
            if e_g.host:
                e_g.host = idB[e_g.host]
            pipe = Pipe(f"P{len(pipes) + 1:03d}", pa.type_id, size, gravity, slope, ins, s_, e_g,
                        crossing_window=win, service=code)
            if gravity:
                if s_.pos[2] <= e_g.pos[2]:
                    continue
                if pc.stair_rule:
                    from .gravity import feasibility
                    if not feasibility(pipe, (W, W + T))["feasible"]:
                        continue   # D64 + 관통 창 높이 (start → 창 → end, S1-7 보완)
            if _clashes(pipe, pipes):
                continue
            windows.append(win)
            pipes.append(pipe)
            break
    meta = {"name": name or f"pair_seed{pc.seed}", "source": "procedural",
            "generator": "pipe_routing.generator.generate_block_pair", "config": _config_meta(pc),
            "n_pipes_requested": 2 * pc.n_pipes_per_block + pc.n_crossing,
            "n_crossing": sum(1 for p in pipes if p.crossing_window)}
    sc = Scenario(Block(2 * W + T, BLOCK_LENGTH, BLOCK_HEIGHT), obstacles, pipes, meta,
                  block_pair={"shared_x": W + T / 2, "wall_thickness": T})
    errors, warnings = validate(sc)
    if errors:
        raise ScenarioError(errors)
    sc.warnings = warnings
    return sc


def main_pair(argv=None) -> int:
    ap = argparse.ArgumentParser(description="S1-7 블록 쌍 시나리오 세트 생성 (D76)")
    ap.add_argument("--out", default="scenarios/pair")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20265008)
    ap.add_argument("--services", action="store_true", help="배관마다 유체 서비스 (D116·D117)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        name = f"pair_{i:03d}"
        sc = generate_block_pair(PairConfig(seed=args.seed + i, services=args.services), name)
        save(sc, out / f"{name}.json")
        print(f"{name}: {json.dumps(sc.summary(), ensure_ascii=False)} 관통 {sc.meta['n_crossing']}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main_pair(sys.argv[2:]) if sys.argv[1:2] == ["pair"] else main())
