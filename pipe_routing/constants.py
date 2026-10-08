"""물리 상수 (CLAUDE.md §3). 규격표 상수이며 탐색·튜닝 대상이 아니다."""
from dataclasses import dataclass

# D4 — 단위 블록 (mm)
BLOCK_WIDTH = 40000
BLOCK_LENGTH = 40000
BLOCK_HEIGHT = 10000

# D5 — 위치 정밀도 (mm)
SNAP_MM = 10

# D17 — 이격 기준
INSULATION_MM = 50
CLEARANCE_MM = 10


@dataclass(frozen=True)
class PipeSpec:
    od: float            # 외경 (mm)
    kg_per_m: float      # 직관 중량 (§3.1)
    min_straight: int    # 최소 직진 길이 (mm, §3.3)


# D18 — 시나리오 구경 10종 (§3.1 + §3.3)
PIPE_SPECS = {
    "15A": PipeSpec(21.7, 1.27, 100),
    "25A": PipeSpec(34.0, 2.57, 100),
    "50A": PipeSpec(60.5, 5.44, 150),
    "65A": PipeSpec(76.3, 9.11, 150),
    "100A": PipeSpec(114.3, 16.10, 300),
    "150A": PipeSpec(165.2, 28.20, 450),
    "200A": PipeSpec(216.3, 42.50, 600),
    "300A": PipeSpec(318.5, 79.70, 900),
    "400A": PipeSpec(406.4, 124.00, 1200),  # 최소 직진 추정: (M2)
    "500A": PipeSpec(508.0, 185.00, 1500),
}
NOMINAL_SIZES = tuple(PIPE_SPECS)

# D27 — 타입 분류. T1~T4 압력, T5~T8 중력 (D27 의 T1=압력·단순, T5=중력·단순)
PIPE_TYPES = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")
GRAVITY_TYPES = frozenset({"T5", "T6", "T7", "T8"})
S0_ROUTER_TYPES = frozenset({"T1", "T5"})


def effective_radius(nominal_size: str, insulation_mm: float = INSULATION_MM) -> float:
    """D17 — 유효 반경 = OD/2 + 보온재 + 10mm."""
    return PIPE_SPECS[nominal_size].od / 2 + insulation_mm + CLEARANCE_MM


def default_drain_slope(nominal_size: str) -> float:
    """§3.5 — 배수관 일반 1/100, 200A 이상 1/200."""
    return 1 / 200 if int(nominal_size[:-1]) >= 200 else 1 / 100
