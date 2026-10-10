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
        if _clashes(pipe, placed):
            continue
        return pipe
    return None


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
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        cfg = GeneratorConfig(seed=args.seed + i, n_pipes=args.n_pipes, obstacle_fill=args.fill)
        name = f"{args.prefix}_{i:03d}"
        sc = generate(cfg, name)
        save(sc, out / f"{name}.json")
        print(f"{name}: {json.dumps(sc.summary(), ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
