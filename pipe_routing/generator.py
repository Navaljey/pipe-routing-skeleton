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

from .constants import (BLOCK_HEIGHT, BLOCK_LENGTH, BLOCK_WIDTH, INSULATION_MM, NOMINAL_SIZES,
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


def _snap(v: float) -> int:
    return int(round(v / SNAP_MM)) * SNAP_MM


def _snap_up(v: float) -> int:
    return int(math.ceil(v / SNAP_MM - 1e-9)) * SNAP_MM


def _rand_snapped(rng: random.Random, lo: float, hi: float) -> Optional[int]:
    lo_s, hi_s = _snap_up(lo), int(math.floor(hi / SNAP_MM)) * SNAP_MM
    if lo_s > hi_s:
        return None
    return rng.randrange(lo_s, hi_s + 1, SNAP_MM)


def gen_obstacles(rng: random.Random, cfg: GeneratorConfig) -> list[Obstacle]:
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
        boxes.append(Box(lo, tuple(l + s for l, s in zip(lo, size))))
        filled = union_volume(boxes)
    return [Obstacle(f"OBS_{i + 1:03d}", "box", b) for i, b in enumerate(boxes)]


def _sample_nozzle(rng, which, size, obstacles) -> Optional[Terminal]:
    """장애물 면 위 한 점에서 법선 방향으로 유효 반경만큼 띄운 위치 (D28)."""
    r = effective_radius(size)
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


def _sample_boundary(rng, which, size, cfg, gravity) -> Optional[Terminal]:
    """블록 6면 중 한 면 위 한 점 (D35). 측면 격벽 : 데크 관통 = 1 − deck_ratio : deck_ratio."""
    r = effective_radius(size)
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


def _sample_terminal(rng, which, size, cfg, obstacles, gravity):
    if obstacles and rng.random() < cfg.nozzle_ratio:
        return _sample_nozzle(rng, which, size, obstacles)
    return _sample_boundary(rng, which, size, cfg, gravity)


def gen_pipe(rng: random.Random, cfg: GeneratorConfig, idx: int, block: Block,
             obstacles: list[Obstacle], placed: list[Pipe]) -> Optional[Pipe]:
    gravity = rng.random() < cfg.gravity_ratio
    size = rng.choice(cfg.sizes)
    slope = default_drain_slope(size) if gravity else None
    for _ in range(cfg.max_tries):
        s = _sample_terminal(rng, "start", size, cfg, obstacles, gravity)
        e = _sample_terminal(rng, "end", size, cfg, obstacles, gravity)
        if s is None or e is None:
            continue
        if sum(abs(a - b) for a, b in zip(s.pos, e.pos)) < cfg.min_terminal_dist_mm:
            continue
        pipe = Pipe(f"P{idx + 1:03d}", "T5" if gravity else "T1", size, gravity, slope,
                    INSULATION_MM, s, e)
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
    obstacles = gen_obstacles(rng, cfg)
    pipes: list[Pipe] = []
    for _ in range(cfg.n_pipes):
        p = gen_pipe(rng, cfg, len(pipes), block, obstacles, pipes)
        if p is not None:
            pipes.append(p)
    if not pipes:
        raise RuntimeError(f"seed {cfg.seed}: 배관을 하나도 배치하지 못함")
    meta = {"name": name or f"proc_seed{cfg.seed}", "source": "procedural",
            "generator": "pipe_routing.generator", "config": asdict(cfg),
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
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        cfg = GeneratorConfig(seed=args.seed + i, n_pipes=args.n_pipes, obstacle_fill=args.fill,
                              stair_rule=args.stair_rule)
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


def _shift_terminal(t: Terminal, dx: float) -> Terminal:
    return Terminal((t.pos[0] + dx, t.pos[1], t.pos[2]), t.kind, t.dir, t.host)


def _shift_pipe(p: Pipe, dx: float, pid: str, window=None) -> Pipe:
    return Pipe(pid, p.type_id, p.nominal_size, p.gravity_pipe, p.min_slope, p.insulation_thickness,
                _shift_terminal(p.start, dx), _shift_terminal(p.end, dx), crossing_window=window)


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
    obstacles = [Obstacle(f"OBS_{i + 1:03d}", "box", o.box) for i, o in enumerate(obsA)]
    obstacles += [Obstacle(f"OBS_{len(obsA) + i + 1:03d}", "box",
                           Box((o.box.min[0] + dB, *o.box.min[1:]), (o.box.max[0] + dB, *o.box.max[1:])))
                  for i, o in enumerate(obsB)]
    idB = {o.id: f"OBS_{len(obsA) + i + 1:03d}" for i, o in enumerate(obsB)}
    pipes: list[Pipe] = []
    localA: list[Pipe] = []
    for _ in range(pc.n_pipes_per_block):
        p = gen_pipe(rng, cfgA, len(localA), local, obsA, localA)
        if p is not None:
            localA.append(p)
    localB: list[Pipe] = []
    for _ in range(pc.n_pipes_per_block):
        p = gen_pipe(rng, cfgB, len(localB), local, obsB, localB)
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
            s_ = _sample_terminal(rng, "start", size, cfgA, obsA, gravity)
            e_ = _sample_terminal(rng, "end", size, cfgB, obsB, gravity)
            if s_ is None or e_ is None:
                continue
            y0 = _rand_snapped(rng, 0, BLOCK_LENGTH - wy)
            z0 = _rand_snapped(rng, 0, BLOCK_HEIGHT - wz)
            win = (float(y0), float(y0 + wy), float(z0), float(z0 + wz))
            if any(_windows_overlap(win, w) for w in windows):
                continue
            pa = Pipe("tmp", "T5" if gravity else "T1", size, gravity, slope, INSULATION_MM, s_, e_)
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
            pipe = Pipe(f"P{len(pipes) + 1:03d}", pa.type_id, size, gravity, slope, INSULATION_MM, s_, e_g,
                        crossing_window=win)
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
            "generator": "pipe_routing.generator.generate_block_pair", "config": asdict(pc),
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
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        name = f"pair_{i:03d}"
        sc = generate_block_pair(PairConfig(seed=args.seed + i), name)
        save(sc, out / f"{name}.json")
        print(f"{name}: {json.dumps(sc.summary(), ensure_ascii=False)} 관통 {sc.meta['n_crossing']}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main_pair(sys.argv[2:]) if sys.argv[1:2] == ["pair"] else main())
