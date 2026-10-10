"""검증기 기하 (D44). 라우터·그래프와 독립 — 경로 꺾임점 목록만 받아 실제 중심선을 만든다.

중심선 = 직관 조각(엘보 접점 사이) + 엘보 호(반경 R = 1.5 × 호칭경). 호는 현(chord) 조각으로 나누되
현과 호의 최대 간격(sagitta) ≤ 0.1mm 이고, 거리 판정 시 그 간격만큼 보수적으로 뺀다.
거리는 모두 유클리드 정확 계산 (선분↔박스: 볼록함수 황금분할 탐색, 선분↔선분: 닫힌 해).
"""
import math
from dataclasses import dataclass, field

import numpy as np

SAGITTA_MM = 0.1
EPS = 1e-6
_GOLDEN = (math.sqrt(5) - 1) / 2


def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def angle_deg(u, v) -> float:
    c = float(np.clip(np.dot(unit(u), unit(v)), -1.0, 1.0))
    return math.degrees(math.acos(c))


@dataclass
class Bend:
    vertex: int                # 꺾임점 인덱스 (정리된 waypoints 기준)
    pos: np.ndarray
    angle: float               # 실제 편향각 (deg)
    nominal: int | None        # 45/90/135 (허용오차 내) 또는 None
    tangent: float             # 접선 길이 R·tan(θ/2) (θ ≥ 180° 이면 inf)


@dataclass
class Piece:
    a: np.ndarray
    b: np.ndarray
    kind: str                  # "straight" | "arc"
    ref: int                   # straight: 구간 인덱스 i (P_i→P_i+1), arc: 꺾임점 인덱스
    slack: float = 0.0         # 거리 판정 시 빼는 값 (호의 sagitta)


@dataclass
class Centerline:
    points: np.ndarray                     # 정리된 waypoints (N,3)
    bends: list[Bend] = field(default_factory=list)
    pieces: list[Piece] = field(default_factory=list)
    overlaps: list[tuple[int, float, float]] = field(default_factory=list)   # (구간, 접선합, 구간길이)

    def arrays(self, kinds=("straight", "arc")):
        sel = [p for p in self.pieces if p.kind in kinds]
        if not sel:
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), []
        return (np.array([p.a for p in sel]), np.array([p.b for p in sel]),
                np.array([p.slack for p in sel]), sel)


def clean_waypoints(wps, angle_tol: float) -> np.ndarray:
    """중복점 제거, 편향각 < 허용오차인 꺾임점(사실상 직선) 병합."""
    pts = [np.asarray(p, dtype=float) for p in wps]
    out = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - out[-1]) > EPS:
            out.append(p)
    i = 1
    while i < len(out) - 1:
        if angle_deg(out[i] - out[i - 1], out[i + 1] - out[i]) < angle_tol:
            out.pop(i)
        else:
            i += 1
    return np.array(out)


def build_centerline(wps, R: float, angle_tol: float, nominal_angles=(45, 90, 135)) -> Centerline:
    pts = clean_waypoints(wps, angle_tol)
    cl = Centerline(points=pts)
    n = len(pts)
    tan = np.zeros(n)
    for i in range(1, n - 1):
        u0, u1 = pts[i] - pts[i - 1], pts[i + 1] - pts[i]
        th = angle_deg(u0, u1)
        nom = next((a for a in nominal_angles if abs(th - a) <= angle_tol), None)
        t = R * math.tan(math.radians(th) / 2) if th < 180 - 1e-9 else math.inf
        tan[i] = t
        cl.bends.append(Bend(i, pts[i], th, nom, t))
    # 직관 조각: 접점 사이. 접선이 겹치면 기록하고 구간 중점에서 자른다
    trim_a = np.zeros(n)
    trim_b = np.zeros(n)
    for i in range(n - 1):
        L = float(np.linalg.norm(pts[i + 1] - pts[i]))
        ta, tb = tan[i], tan[i + 1]
        if ta + tb > L + EPS:
            cl.overlaps.append((i, ta + tb, L))
            if math.isinf(ta) or math.isinf(tb):
                ta, tb = (0.0 if math.isinf(ta) else ta), (0.0 if math.isinf(tb) else tb)
            s = min(1.0, L / (ta + tb)) if ta + tb > 0 else 1.0
            ta, tb = ta * s, tb * s
        trim_a[i], trim_b[i] = ta, tb
        u = unit(pts[i + 1] - pts[i])
        a, b = pts[i] + u * ta, pts[i + 1] - u * tb
        if np.linalg.norm(b - a) > EPS:
            cl.pieces.append(Piece(a, b, "straight", i))
    # 엘보 호
    for bend in cl.bends:
        i = bend.vertex
        if math.isinf(bend.tangent) or bend.angle < angle_tol:
            continue
        u0, u1 = unit(pts[i] - pts[i - 1]), unit(pts[i + 1] - pts[i])
        t = min(trim_b[i - 1], trim_a[i])   # 겹침으로 접선이 줄었으면 반경도 줄인 호 (양쪽 같게)
        if t <= EPS:
            continue
        A, B, sag = elbow_arc_chords(pts[i], u0, u1, t, bend.angle)
        for a, b in zip(A, B):
            cl.pieces.append(Piece(a, b, "arc", i, slack=sag))
    return cl


def elbow_arc_chords(vertex, u0, u1, t: float, angle_deg_: float):
    """꺾임점 vertex 의 엘보 호를 현 조각으로 (D44). 라우터(D48)와 검증기가 같이 쓴다.

    u0·u1 = 들어오는·나가는 단위 방향, t = 접선 길이 (호 반경 = t / tan(θ/2)).
    반환: (시작점 배열 (k,3), 끝점 배열 (k,3), sagitta) — 거리 판정 때 sagitta 를 빼면 보수적이다.
    """
    vertex = np.asarray(vertex, dtype=float)
    u0 = np.asarray(u0, dtype=float)
    u1 = np.asarray(u1, dtype=float)
    T0, T1 = vertex - u0 * t, vertex + u1 * t
    th = math.radians(angle_deg_)
    r_eff = t / math.tan(th / 2)
    center = T0 + unit(u1 - u0 * np.dot(u0, u1)) * r_eff
    v0, v1 = T0 - center, T1 - center
    phi_max = 2 * math.acos(max(-1.0, 1 - SAGITTA_MM / max(r_eff, 1e-9))) if r_eff > SAGITTA_MM else th
    k = max(1, math.ceil(th / max(phi_max, 1e-6)))
    sag = r_eff * (1 - math.cos(th / k / 2))
    s = np.arange(k + 1)[:, None] / k
    pts = center + (np.sin((1 - s) * th) * v0 + np.sin(s * th) * v1) / math.sin(th)
    return pts[:-1], pts[1:], sag


# ---------------------------------------------------------------- 거리

def axis_segment_box_distance(P: np.ndarray, ax: int, target: float, lo, hi) -> np.ndarray:
    """축 방향 선분들 (P → P 의 ax 좌표만 target 으로 바꾼 점) ↔ 박스 하나 사이 유클리드 최소거리. 닫힌 식 (정확).

    서포트 지지선(배관 중심 → 구조면 수직)은 항상 축 방향이므로 검증기(D46)·라우터(D57)가 같이 쓴다.
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    a = np.minimum(P[:, ax], target)
    b = np.maximum(P[:, ax], target)
    g2 = np.zeros(len(P))
    for k in range(3):
        if k == ax:
            g = np.maximum(0.0, np.maximum(lo[k] - b, a - hi[k]))
        else:
            g = np.maximum(0.0, np.maximum(lo[k] - P[:, k], P[:, k] - hi[k]))
        g2 += g * g
    return np.sqrt(g2)


def support_faces(ext, box_lo, box_hi, P: np.ndarray, U: np.ndarray):
    """서포트 지지면 후보 (D46). 반환: (거리 (N, 6), 장애물에 막힘 (N, 6)).

    면 순서 = FACES (floor, ceiling, wall_x0, wall_x1, wall_y0, wall_y1). 배관 축(U)과 평행한 법선의 면은 거리 inf.
    막힘 = 지지선이 장애물 박스와 닿음 (거리 ≤ EPS). 다른 배관에 의한 막힘은 호출자가 더한다 (검증기만, D57).
    """
    dist = np.full((len(P), 6), np.inf)
    blocked = np.zeros((len(P), 6), dtype=bool)
    for f, (ax, side) in enumerate(((2, 0), (2, 1), (0, 0), (0, 1), (1, 0), (1, 1))):
        ok = np.abs(U[:, ax]) < 1 - 1e-9
        dist[ok, f] = np.abs(P[ok, ax] - side * ext[ax])
        for lo, hi in zip(box_lo, box_hi):
            blocked[:, f] |= axis_segment_box_distance(P, ax, side * ext[ax], lo, hi) <= EPS
    return dist, blocked


def segment_box_distance(A: np.ndarray, B: np.ndarray, lo, hi, iters: int = 60):
    """선분 N개 ↔ 박스 1개 최소거리와 최근접점. 박스까지 거리는 t 에 대해 볼록 → 황금분할 탐색."""
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    D = B - A

    def f(t):
        p = A + t[:, None] * D
        g = np.maximum(0.0, np.maximum(lo - p, p - hi))
        return np.sqrt((g * g).sum(1))

    a = np.zeros(len(A))
    b = np.ones(len(A))
    for _ in range(iters):
        c = b - _GOLDEN * (b - a)
        d = a + _GOLDEN * (b - a)
        left = f(c) < f(d)          # 최소점은 [a, d] 안
        b = np.where(left, d, b)
        a = np.where(left, a, c)
    ts = np.stack([np.zeros(len(A)), np.ones(len(A)), (a + b) / 2], 1)
    vals = np.stack([f(ts[:, 0]), f(ts[:, 1]), f(ts[:, 2])], 1)
    k = vals.argmin(1)
    t = ts[np.arange(len(A)), k]
    return vals[np.arange(len(A)), k], A + t[:, None] * D


def segment_segment_distance(P0, P1, Q0, Q1):
    """선분 쌍(브로드캐스트) 최소거리와 양쪽 최근접점 (Ericson, Real-Time Collision Detection 5.1.9)."""
    d1 = P1 - P0
    d2 = Q1 - Q0
    r = P0 - Q0
    a = (d1 * d1).sum(-1)
    e = (d2 * d2).sum(-1)
    f = (d2 * r).sum(-1)
    c = (d1 * r).sum(-1)
    b = (d1 * d2).sum(-1)
    denom = a * e - b * b
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(denom > 1e-12, np.clip((b * f - c * e) / denom, 0, 1), 0.0)
        s = np.where(a > 1e-12, s, 0.0)
        t = np.where(e > 1e-12, (b * s + f) / e, 0.0)
        s = np.where(t < 0, np.where(a > 1e-12, np.clip(-c / a, 0, 1), 0.0), s)
        s = np.where(t > 1, np.where(a > 1e-12, np.clip((b - c) / a, 0, 1), 0.0), s)
        t = np.clip(t, 0, 1)
    p = P0 + s[..., None] * d1
    q = Q0 + t[..., None] * d2
    return np.linalg.norm(p - q, axis=-1), p, q


def boxes_overlap(lo1, hi1, lo2, hi2) -> bool:
    """양의 부피로 겹치는가 (면 접촉은 겹침 아님)."""
    return all(lo1[i] < hi2[i] - EPS and lo2[i] < hi1[i] - EPS for i in range(3))
