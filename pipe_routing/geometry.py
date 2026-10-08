"""축 정렬 박스(AABB) 기하. 점·축 방향 선분도 퇴화 박스로 다룬다."""
import math
from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class Box:
    min: Vec3
    max: Vec3

    @staticmethod
    def of_points(a: Sequence[float], b: Sequence[float]) -> "Box":
        return Box(tuple(map(min, a, b)), tuple(map(max, a, b)))

    @property
    def volume(self) -> float:
        return math.prod(hi - lo for lo, hi in zip(self.min, self.max))


def box_distance(a: Box, b: Box) -> float:
    """두 박스 사이 유클리드 최소거리 (겹치면 0)."""
    gaps = (max(0.0, b.min[i] - a.max[i], a.min[i] - b.max[i]) for i in range(3))
    return math.sqrt(sum(g * g for g in gaps))


def union_volume(boxes: Iterable[Box]) -> float:
    """박스 합집합 부피 (좌표 압축). 박스 수십 개 규모 전용."""
    boxes = [b for b in boxes if b.volume > 0]
    xs = sorted({c for b in boxes for c in (b.min[0], b.max[0])})
    zs = sorted({c for b in boxes for c in (b.min[2], b.max[2])})
    total = 0.0
    for (x0, x1), (z0, z1) in product(zip(xs, xs[1:]), zip(zs, zs[1:])):
        xc, zc = (x0 + x1) / 2, (z0 + z1) / 2
        spans = sorted((b.min[1], b.max[1]) for b in boxes
                       if b.min[0] < xc < b.max[0] and b.min[2] < zc < b.max[2])
        covered, end = 0.0, -math.inf
        for lo, hi in spans:  # y 방향 구간 합집합 길이
            if hi > end:
                covered += hi - max(lo, end)
                end = hi
        total += covered * (x1 - x0) * (z1 - z0)
    return total
