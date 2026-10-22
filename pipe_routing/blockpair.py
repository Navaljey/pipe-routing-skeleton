"""S1-7 블록 쌍 시험대 (D69·D76·D91~D95) — 공유 격벽·관통 창 기하.

합친 맵: 블록 A [0, a] · 공유 격벽 [a, b] · 블록 B [b, W] (x 방향, a·b = 격벽 양쪽 면, y·z 는 D4).
공유 격벽 = supportable·penetrable 박스 1개 (D93), 두께 = block_pair.wall_thickness (없으면 0, 생성기 기본값 `추정:`).
  - 한 블록 안 배관: 격벽 전체가 벽이다 (유효 반경 r 이격, 넘지 못함). 격자는 자기 블록만 (D91).
  - 관통 배관 (start·end 가 다른 블록): 자기 crossing_window 만 뚫린 벽 (D76 ②). 창은 x 축 직관으로만 지난다 (D94).
    관통 구역(창 × 격벽 양쪽 면에서 e = max(3D, r) − r 바깥까지)은 x 축 직관 말고는 장애물이다 — 엘보 호·다른 방향
    직관이 그 구역과 r 이상 떨어져야 하므로 격벽 면에서 인접 엘보 접점까지 직관 ≥ max(3D, r) (D94).
  - 관통부는 서포트 고정점 (D94): 간격 계산을 0 으로, 무게 0.
  - 서포트: 격벽 면은 지지면 (D76 ①). 면 거리는 점이 속한 블록 기준, 부재는 그 블록 밖으로 나가지 못한다.
    `추정:` 모든 관통 창은 구멍 — 부재가 창 사각형(반폭 안)에 닿으면 그 위치·형식은 불가 (창 안에 앵커 불가).
"""
import numpy as np

EPS = 1e-6


def walls(sc) -> tuple:
    """공유 격벽 목록 ((a, b),) — 격벽 양쪽 면 x. 블록 쌍이 아니면 ()."""
    bp = getattr(sc, "block_pair", None)
    if not bp:
        return ()
    s, t = float(bp["shared_x"]), float(bp.get("wall_thickness", 0.0))
    return ((s - t / 2, s + t / 2),)


splits = walls   # 이전 이름 (서포트 함수의 sp 인자)


def center(sc) -> float:
    a, b = walls(sc)[0]
    return (a + b) / 2


def thickness(sc) -> float:
    a, b = walls(sc)[0]
    return b - a


def side(x: float, s: float) -> int:
    return 0 if x < s - EPS else 1


def block_of(sc, x: float) -> int:
    return side(x, center(sc))


def crossing(sc, pipe) -> bool:
    """관통 배관인가 (start·end 가 다른 블록)."""
    if not walls(sc):
        return False
    s = center(sc)
    return side(pipe.start.pos[0], s) != side(pipe.end.pos[0], s)


def nominal_mm(pipe) -> float:
    """호칭경 mm (D94 의 D, R = 1.5D 와 같은 정의)."""
    return float(pipe.nominal_size[:-1])


def straight_min(pipe) -> float:
    """D94·D96: 판(격벽·블록 경계) 면에서 인접 엘보 접점까지 직관 하한 = max(3D, r)."""
    return max(3 * nominal_mm(pipe), pipe.radius)


D94 = True    # 관통 직관 규칙 (D94: max(3D, r) + t, 관통부 서포트 고정점). False = 이전 D76 ④ (r + t) — 비교용
D96 = True    # 경계 단자 직관 규칙 (D96: max(3D, r) + t, 소유자 결정). False = 이전 D47 (r + t) — 영향 측정·비교용


def boundary_straight(pipe) -> float:
    """블록 바깥 경계 단자 면에서 인접 엘보 접점까지 직관 하한 — D47 r, D96 max(3D, r)."""
    return straight_min(pipe) if D96 else pipe.radius


def crossing_straight(pipe) -> float:
    """공유 격벽 면에서 관통 직관의 인접 엘보 접점까지 직관 하한 — D76 ④ r, D94 max(3D, r)."""
    return straight_min(pipe) if D94 else pipe.radius


def penetration_kg(sc, pipe) -> float:
    """D95 관통 피스 무게 = 배관 kg/m × 관통 길이(격벽 두께). `추정:` 정식 표가 오면 계수만 교체."""
    from .constants import PIPE_SPECS
    return PIPE_SPECS[pipe.nominal_size].kg_per_m * thickness(sc) / 1000 if walls(sc) else 0.0


def window_box(sc, window) -> tuple:
    """창 (격벽 두께만큼의 박스 lo, hi) — 서포트 부재용 구멍."""
    a, b = walls(sc)[0]
    y0, y1, z0, z1 = window
    return np.array([a, y0, z0], float), np.array([b, y1, z1], float)


def crossing_zone(sc, pipe) -> tuple:
    """관통 구역: 창 × [a − e, b + e], e = crossing_straight − r (D76 ④ 0, D94 max(3D, r) − r). x 축 직관만 지나갈 수
    있다 — 엘보 호·다른 조각이 이 구역과 r 이상이면 격벽 면에서 접점까지 직관 ≥ crossing_straight (라우터 soft 박스·검증기 공용)."""
    a, b = walls(sc)[0]
    e = crossing_straight(pipe) - pipe.radius
    y0, y1, z0, z1 = pipe.crossing_window
    return np.array([a - e, y0, z0], float), np.array([b + e, y1, z1], float)


def wall_boxes(sc, window=None) -> list:
    """공유 격벽 (창을 뺀) 박스 목록 [(lo, hi)]. window = (y0, y1, z0, z1) 또는 None (막힌 벽)."""
    w = walls(sc)
    if not w:
        return []
    a, b = w[0]
    L, H = float(sc.block.length), float(sc.block.height)
    if window is None:
        return [(np.array([a, 0.0, 0.0]), np.array([b, L, H]))]
    y0, y1, z0, z1 = window
    out = []
    for lo, hi in (((0.0, 0.0), (y0, H)), ((y1, 0.0), (L, H)), ((y0, 0.0), (y1, z0)), ((y0, z1), (y1, H))):
        if hi[0] - lo[0] > EPS and hi[1] - lo[1] > EPS:
            out.append((np.array([a, lo[0], lo[1]]), np.array([b, hi[0], hi[1]])))
    return out


def hole_boxes(sc) -> list:
    """모든 관통 창 (서포트 부재 판정용 구멍)."""
    return [window_box(sc, p.crossing_window) for p in sc.pipes if getattr(p, "crossing_window", None)]


def local_bounds(ext, X, sp=()) -> tuple:
    """점마다 그 점이 속한 블록 범위 (lo, hi) — 블록 쌍이면 x 를 공유 격벽 양쪽 면에서 나눈다."""
    X = np.asarray(X, dtype=float)
    lo = np.zeros((len(X), 3))
    hi = np.broadcast_to(np.asarray(ext, dtype=float), (len(X), 3)).copy()
    for a, b in sp:
        right = X[:, 0] >= (a + b) / 2
        lo[right, 0] = np.maximum(lo[right, 0], b)
        hi[~right, 0] = np.minimum(hi[~right, 0], a)
    return lo, hi


def block_ranges(ext, sp=()) -> list:
    """블록별 x 범위 [(x0, x1)]."""
    xs = [0.0]
    for a, b in sp:
        xs += [a, b]
    xs.append(float(ext[0]))
    return list(zip(xs[0::2], xs[1::2]))


def crossing_points(sc, pipe, waypoints) -> list:
    """관통 직관이 격벽 중심면을 지나는 점들 [(구간 인덱스, 점)] — 서포트 고정점·관통 비용 (D94·D95)."""
    if not walls(sc):
        return []
    s = center(sc)
    W = np.asarray(waypoints, dtype=float)
    out = []
    for i in range(len(W) - 1):
        a, b = W[i][0] - s, W[i + 1][0] - s
        if a * b < 0:
            t = a / (a - b)
            out.append((i, W[i] + t * (W[i + 1] - W[i])))
    return out
