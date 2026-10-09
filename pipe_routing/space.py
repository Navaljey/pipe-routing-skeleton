"""표현 독립 인터페이스 (CLAUDE.md §4.1, D9, D37).

라우터는 이 인터페이스만 본다. 표현(escape graph 등)은 교체 가능 부품이다.
V3 등 다른 라우터도 같은 인터페이스 위에서 동작한다 (D3).
"""
import math
from typing import Iterable, NamedTuple, Optional, Protocol

from .geometry import Vec3
from .scenario import Pipe

# D15 — 진행 방향 18종: 축 6 + 수평면·수직면 45° 12. 3D 꼭짓점 대각(√3)은 없다
AXIS_DIRS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
DIAG_DIRS = [d for d in ((a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1))
             if sum(map(abs, d)) == 2]
DIRS = AXIS_DIRS + DIAG_DIRS
DIR_INDEX = {d: i for i, d in enumerate(DIRS)}
ALLOWED_DEFLECTIONS = (0, 45, 90, 135)   # D15·D37: 180° 역행, 60°·120°(서로 다른 평면의 45° 사이) 금지


def deflection_deg(d1: tuple[int, int, int], d2: tuple[int, int, int]) -> int:
    """두 진행 방향 사이 편향각 (정수 degree)."""
    dot = sum(a * b for a, b in zip(d1, d2))
    cos = dot / math.sqrt(sum(a * a for a in d1) * sum(b * b for b in d2))
    return int(round(math.degrees(math.acos(max(-1.0, min(1.0, cos))))))


# 18×18 편향각 표 (자주 쓰므로 미리 계산)
DEFLECTION = [[deflection_deg(a, b) for b in DIRS] for a in DIRS]


class State(NamedTuple):
    """라우터 탐색 상태 (§4.2-5, D37).

    node: 표현이 정하는 노드 키 (escape graph 는 격자 인덱스 (i, j, k))
    dir:  진입 방향 인덱스 (DIRS). 시작 상태는 start 단자 dir
    run:  마지막 꺾임(또는 start 단자) 이후 직진 길이 mm. 이후 어떤 판정에도 충분한 값에서 상한
    bend: 마지막 꺾임의 편향각 (0 = 아직 꺾지 않음). 엘보 접선 길이 판정용 (D45)

    지배 관계 (라우터 가지치기용): 같은 node·dir 에서 run 이 크거나 같고 bend 가 작거나 같으면
    이후 허용 이동이 포함관계로 넓다 (접선 길이는 편향각에 대해 단조 증가).
    """
    node: tuple
    dir: int
    run: float
    bend: int = 0


class SpaceRepresentation(Protocol):
    """배관 1개에 대해 만든 탐색 공간. 메서드의 pipe 인자는 §4.1 시그니처를 따른다."""

    def start_state(self, pipe: Pipe) -> State: ...

    def is_goal(self, state: State, pipe: Pipe) -> bool: ...

    def neighbors(self, state: State, pipe: Pipe) -> Iterable[State]:
        """이동 가능한 다음 상태. 편향각(D15)·직관 길이(§3.3, D45)·충돌(D17)을 이미 만족한다."""
        ...

    def cost(self, a: State, b: State, pipe: Pipe) -> float:
        """ΔJ (kg) = 직관 길이 × kg/m + 엘보 중량 (§5, §3.4). 서포트는 제외 (§5)."""
        ...

    def is_free(self, a: Vec3, b: Vec3, pipe: Pipe) -> bool:
        """좌표 a–b 직선 구간이 장애물·블록 경계와 유효 반경(D17) 이상 떨어져 있는가. 정확 판정."""
        ...

    def position(self, node) -> Vec3:
        """노드 키 → 좌표 (mm)."""
        ...


def route(space: SpaceRepresentation, pipe: Pipe) -> Optional[list[State]]:
    """라우터 슬롯 (§4.1). 4단계에서 A* 로 구현한다."""
    raise NotImplementedError("4단계")
