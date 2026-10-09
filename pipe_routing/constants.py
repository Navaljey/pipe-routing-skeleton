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


@dataclass(frozen=True)
class FittingSpec:
    elbow90: float   # kg (§3.2)
    elbow45: float
    tee: float
    gate_valve: float


# §3.2 관이음 중량 (JIS B2312 LR). 15A·50A·150A·400A 는 D26 보간값 (추정:)
FITTINGS = {
    "15A": FittingSpec(0.11, 0.06, 0.20, 0.82),
    "25A": FittingSpec(0.30, 0.17, 0.54, 2.0),
    "50A": FittingSpec(1.08, 0.60, 1.94, 6.30),
    "65A": FittingSpec(1.80, 0.99, 3.24, 10.0),
    "100A": FittingSpec(3.80, 2.09, 6.84, 20.0),
    "150A": FittingSpec(9.33, 5.13, 16.79, 44.53),
    "200A": FittingSpec(18.0, 9.90, 32.4, 80.0),
    "300A": FittingSpec(51.0, 28.1, 91.8, 210.0),
    "400A": FittingSpec(102.72, 56.47, 184.89, 402.43),
    "500A": FittingSpec(195.0, 107.0, 351.0, 730.0),
}


def elbow_kg(nominal_size: str, deflection_deg: int) -> float:
    """§3.4 엘보 청구. 135° = 90° + 45° (D25)."""
    f = FITTINGS[nominal_size]
    return {0: 0.0, 45: f.elbow45, 90: f.elbow90, 135: f.elbow90 + f.elbow45}[deflection_deg]


# §3.6 서포트 최대 간격 (mm): (수평, 수직)
SUPPORT_SPACING = {
    "15A": (1800, 2500), "25A": (1800, 2500),
    "50A": (2500, 3000),
    "65A": (3000, 4000), "100A": (3000, 4000),
    "150A": (4000, 5000), "200A": (4000, 5000),
    "300A": (5000, 6000), "400A": (5000, 6000), "500A": (5000, 6000),
}
SUPPORT_ANGLE_KG_PER_M = 4.43   # D24 L50×50×6t
SUPPORT_MARGIN_MM = 100         # D23

# §3.7 / D19 밸브
VALVE_Z_RANGE = (700, 1500)
VALVE_FRONT_MM = 1000           # D43 1m × 1m × 1m

# D44 — 검증기 기하
ELBOW_R_FACTOR = 1.5            # LR 엘보 중심선 반경 = 1.5 × 호칭경(mm) (추정: JIS B2312 LR 정의)
ANGLE_TOL_DEG = 2.0             # 편향각·방향 허용오차 (중력관 구배로 생기는 기울기 흡수)


def elbow_radius(nominal_size: str) -> float:
    return ELBOW_R_FACTOR * int(nominal_size[:-1])


def support_kg(dist_mm: float) -> float:
    """§3.6 support_kg = 4.43 × (거리 + 100) / 1000."""
    return SUPPORT_ANGLE_KG_PER_M * (dist_mm + SUPPORT_MARGIN_MM) / 1000
