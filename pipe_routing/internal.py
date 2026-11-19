"""S2-1 블록 내부 구조재 (D113) · 접근 구역 (D119) — 장애물 속성별 기하.

장애물 속성 조합 (D113·D119, 로더가 확인):
  장비            supportable false · penetrable false (기존 — 속성 없음)
  내부 격벽·데크  supportable true  · penetrable true   = 판 (plate): 관통 축 방향 직관으로만 지난다 (D94)
  필라·스티프너   supportable true  · penetrable false
  접근 구역       access true (supportable·penetrable false) — 배관·서포트 부재 진입 금지 (D119)

판 관통 (D94 일반화, 블록 쌍 공유 격벽과 같은 기하 — `blockpair` 의 창 대신 면 전체, §7 Q13):
  - 관통 구역 = 판을 관통 축 방향으로 e = max(3D, r) − r 만큼 늘린 박스 — 관통 축 직관 말고는 장애물 (라우터 soft 박스,
    검증기 crossing 모듈). 따라서 판 면에서 인접 엘보 접점까지 직관 ≥ max(3D, r).
  - 관통 구멍 (유효 반경 r 원) 은 판 면 안 (가장자리에서 r 이상) — 판의 네 옆면을 두께 0 의 "가장자리 박스" 로 두고
    관통 축 직관도 그 박스와 r 이상 (라우터 hard, 검증기 같은 판정).
  - 관통부 = 서포트 고정점 (판 중심면, 간격 0·무게 0), 관통비 = kg/m × 판 두께 (D95).
속성이 없는 시나리오 (E1) 에서는 모든 함수가 빈 목록을 돌려주고 호출 쪽은 기존 경로를 그대로 탄다.
"""
from dataclasses import dataclass

import numpy as np

EPS = 1e-6


def penetration_axis(o) -> int:
    """D93: 입력 관통 축, 없으면 박스의 가장 짧은 변 축 (`추정:`)."""
    if o.penetration_axis is not None:
        return int(o.penetration_axis)
    d = [o.box.max[k] - o.box.min[k] for k in range(3)]
    return int(np.argmin(d))


@dataclass(frozen=True)
class Plate:
    id: str
    lo: np.ndarray
    hi: np.ndarray
    axis: int

    @property
    def thickness(self) -> float:
        return float(self.hi[self.axis] - self.lo[self.axis])

    @property
    def center(self) -> float:
        return float(self.lo[self.axis] + self.hi[self.axis]) / 2

    def zone(self, e: float) -> tuple:
        """관통 구역 (판을 관통 축으로 e 만큼 늘린 박스)."""
        lo, hi = self.lo.copy(), self.hi.copy()
        lo[self.axis] -= e
        hi[self.axis] += e
        return lo, hi

    def edges(self) -> list:
        """가장자리 박스 4개 (판의 옆면, 두께 0) — 관통 구멍이 면 안에 있도록."""
        out = []
        for q in range(3):
            if q == self.axis:
                continue
            for v in (self.lo[q], self.hi[q]):
                lo, hi = self.lo.copy(), self.hi.copy()
                lo[q] = hi[q] = v
                out.append((lo, hi))
        return out

    def in_face(self, c: np.ndarray, r: float) -> bool:
        """관통 축에 수직인 두 좌표가 면 안 (가장자리에서 r 이상)."""
        return all(self.lo[q] + r - EPS <= c[q] <= self.hi[q] - r + EPS for q in range(3) if q != self.axis)


def plates(sc) -> list:
    return [Plate(o.id, np.array(o.box.min, float), np.array(o.box.max, float), penetration_axis(o))
            for o in sc.obstacles if o.penetrable]


def access_boxes(sc) -> list:
    return [(np.array(o.box.min, float), np.array(o.box.max, float)) for o in sc.obstacles if o.access]


def has_internal(sc) -> bool:
    return any(o.penetrable or o.supportable or o.access for o in sc.obstacles)


def solid_obstacles(sc) -> list:
    """라우터·검증기 collision 이 장애물로 보는 박스 — 판 (관통 가능) 을 뺀 전부 (장비·필라·스티프너·접근 구역).
    접근 구역은 라우터에서 장애물과 같다 (D119). 검증기는 접근 구역 위반을 collision 이 아니라 access 모듈로 센다."""
    return [o for o in sc.obstacles if not o.penetrable]


def straight_min(pipe) -> float:
    from .blockpair import straight_min as sm
    return sm(pipe)


def router_boxes(sc, pipe) -> tuple:
    """(장애물 [(lo, hi)], hard 추가 박스 (판 가장자리), soft [(lo, hi)], soft 축) — 고정층 EscapeGraph 인자.
    soft = 판 관통 구역 (그 축 방향 엣지만 지나간다), e = max(3D, r) − r (D94)."""
    obs = [(o.box.min, o.box.max) for o in solid_obstacles(sc)]
    ps = plates(sc)
    e = straight_min(pipe) - pipe.radius
    hard = [b for p in ps for b in p.edges()]
    soft = [p.zone(e) for p in ps]
    return obs, hard, soft, [p.axis for p in ps]


def first_bend_coords(sc, pipe, angles=(45, 90, 135)) -> list:
    """판 양쪽 면에서 관통 첫 꺾임 선 (D105 각도별): 면 ± (max(3D, r) + t_θ), 10 mm 올림 — 축별 [x, y, z]."""
    from .constants import elbow_tangent
    out = [[], [], []]
    s0 = straight_min(pipe)
    for p in plates(sc):
        for th in angles:
            d = float(int(np.ceil((s0 + elbow_tangent(pipe.nominal_size, th)) / 10 - 1e-9)) * 10)
            out[p.axis] += [float(p.lo[p.axis] - d), float(p.hi[p.axis] + d)]
    return out


def crossings(sc, pipe, waypoints) -> list:
    """관통 직관이 판 중심면을 지나는 점 [(구간 인덱스, 점, 판)] — 서포트 고정점·관통비 (D94·D95).
    관통 축과 평행한 직관이 판 면 안에서 중심면을 지날 때만."""
    out = []
    W = np.asarray(waypoints, dtype=float)
    for p in plates(sc):
        ax, c = p.axis, p.center
        for i in range(len(W) - 1):
            a, b = W[i], W[i + 1]
            d = b - a
            if np.count_nonzero(np.abs(d) > 1e-6) != 1 or abs(d[ax]) < 1e-6:
                continue
            if (a[ax] - c) * (b[ax] - c) >= 0:
                continue
            q = a + (c - a[ax]) / d[ax] * d
            if p.in_face(q, 0.0):
                out.append((i, q, p))
    return sorted(out, key=lambda x: (x[0], float(np.dot(x[1] - W[x[0]], W[x[0] + 1] - W[x[0]]))))


def penetration_kg(sc, pipe, waypoints) -> float:
    """D95 관통비 = Σ kg/m × 판 두께 (내부 판)."""
    return sum(pipe.kg_per_m * p.thickness / 1000 for _, _, p in crossings(sc, pipe, waypoints))


def fixed_points(sc, pipe) -> list:
    """LineSupport 고정점 [(축, (c1 범위), (c2 범위), 선 좌표 t)] — 판 면 안 (가장자리 r) 관통 축 선은 판 중심에서 지지."""
    r = pipe.radius
    out = []
    for p in plates(sc):
        o1, o2 = [q for q in range(3) if q != p.axis]
        out.append((p.axis, (p.lo[o1] + r, p.hi[o1] - r), (p.lo[o2] + r, p.hi[o2] - r), p.center))
    return out
