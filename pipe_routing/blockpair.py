"""S1-7 블록 쌍 시험대 (D69·D76) — 공유 격벽·관통 창 기하.

합친 맵: 블록 A [0, s] · 블록 B [s, W] (x 방향, s = block_pair.shared_x, y·z 는 D4). 공유 격벽 = 평면 x = s (두께 0).
  - 한 블록 안 배관: 격벽 전체가 벽이다 (유효 반경 r 이격, 넘지 못함).
  - 관통 배관 (start·end 가 다른 블록): 자기 crossing_window 만 뚫린 벽. 창 안을 x 축 직관으로만 지난다 (D76 ④⑤).
    창 패치(평면 위 창 사각형)는 x 축 직관 말고는 장애물이다 — 엘보 호·다른 방향 직관은 r 이상 떨어져야 하므로
    창을 지나는 직관은 양쪽으로 r + t 이상이다 (D47 과 같은 규칙).
  - 서포트: 격벽은 지지면 (D76 ①). 면 거리는 점이 속한 블록 기준, 부재는 그 블록 밖으로 나가지 못한다.
    `추정:` 모든 관통 창은 구멍 — 부재가 창 사각형(반폭 안)에 닿으면 그 위치·형식은 불가 (창 안에 앵커 불가).
"""
import numpy as np

EPS = 1e-6


def splits(sc) -> tuple:
    bp = getattr(sc, "block_pair", None)
    return (float(bp["shared_x"]),) if bp else ()


def side(x: float, s: float) -> int:
    return 0 if x < s - EPS else 1


def crossing(sc, pipe) -> bool:
    """관통 배관인가 (start·end 가 다른 블록)."""
    sp = splits(sc)
    return bool(sp) and side(pipe.start.pos[0], sp[0]) != side(pipe.end.pos[0], sp[0])


def window_box(sc, window) -> tuple:
    """창 사각형 (퇴화 박스 lo, hi)."""
    s = splits(sc)[0]
    y0, y1, z0, z1 = window
    return np.array([s, y0, z0], float), np.array([s, y1, z1], float)


def wall_boxes(sc, window=None) -> list:
    """공유 격벽 (창을 뺀) 퇴화 박스 목록 [(lo, hi)]. window = (y0, y1, z0, z1) 또는 None (막힌 벽)."""
    sp = splits(sc)
    if not sp:
        return []
    s = sp[0]
    L, H = float(sc.block.length), float(sc.block.height)
    if window is None:
        return [(np.array([s, 0.0, 0.0]), np.array([s, L, H]))]
    y0, y1, z0, z1 = window
    out = []
    for lo, hi in (((0.0, 0.0), (y0, H)), ((y1, 0.0), (L, H)), ((y0, 0.0), (y1, z0)), ((y0, z1), (y1, H))):
        if hi[0] - lo[0] > EPS and hi[1] - lo[1] > EPS:
            out.append((np.array([s, lo[0], lo[1]]), np.array([s, hi[0], hi[1]])))
    return out


def hole_boxes(sc) -> list:
    """모든 관통 창 (서포트 부재 판정용 구멍)."""
    return [window_box(sc, p.crossing_window) for p in sc.pipes if getattr(p, "crossing_window", None)]


def local_bounds(ext, X, sp=()) -> tuple:
    """점마다 그 점이 속한 블록 범위 (lo, hi) — 블록 쌍이면 x 를 공유 격벽에서 나눈다."""
    X = np.asarray(X, dtype=float)
    lo = np.zeros((len(X), 3))
    hi = np.broadcast_to(np.asarray(ext, dtype=float), (len(X), 3)).copy()
    for s in sp:
        right = X[:, 0] >= s - EPS
        lo[right, 0] = np.maximum(lo[right, 0], s)
        hi[~right, 0] = np.minimum(hi[~right, 0], s)
    return lo, hi


def block_ranges(ext, sp=()) -> list:
    """블록별 x 범위 [(x0, x1)]."""
    xs = [0.0, *sp, float(ext[0])]
    return list(zip(xs[:-1], xs[1:]))
