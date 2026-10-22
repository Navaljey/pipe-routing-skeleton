"""중력관 계단식 하향 규칙 (D61~D64) — 라우터와 검증기가 같이 쓰는 판정 함수 (D63).

규칙 (§3.5, D61):
  - 위로 가는 이동(+z 성분) 금지
  - 수평 구간은 구배 0 허용
  - 마지막 하향 이동(수직 하향 또는 수직면 안 45° 하향) 이후 누적 수평 길이 ≤ L_max (D62). 하향 이동을 지나면 0
"""
import math

from .constants import ANGLE_TOL_DEG, PIPE_SPECS, elbow_tangent

# D62 L_max (mm) — §3.5 표 (소유자가 준 값)
GRAVITY_LMAX_MM = {"15A": 4000, "25A": 4710, "50A": 6470, "65A": 7530, "100A": 10000, "150A": 15000,
                   "200A": 20000, "300A": 30000, "400A": 40000, "500A": 50000}

Z_TOL = 1e-6   # mm — 좌표는 10 mm 격자라 상하 판정에 충분히 작다

UP, DOWN, FLAT = "up", "down", "flat"


def gravity_lmax(size: str) -> float:
    return float(GRAVITY_LMAX_MM[size])


def move_kind(vec) -> str:
    """이동(구간) 방향 → "up" (+z 성분) · "down" (수직 하향 또는 45° 하향) · "flat" (수평, 또는 완만한 하향).

    하향 이동(D61 ④) = 수평에서 내려가는 각이 45° − 허용오차(2°, D44) 이상. 그보다 완만한 하향은 수평으로 보고
    수평 투영 길이를 누적한다 (라우터는 축·45° 방향만 만들므로 이 경계에 걸리지 않는다).
    """
    dx, dy, dz = (float(v) for v in vec)
    if dz > Z_TOL:
        return UP
    if dz < -Z_TOL and math.degrees(math.atan2(-dz, math.hypot(dx, dy))) >= 45 - ANGLE_TOL_DEG:
        return DOWN
    return FLAT


def horizontal_length(vec) -> float:
    return math.hypot(float(vec[0]), float(vec[1]))


def step(h: float, kind: str, length: float, lmax: float):
    """누적 수평 길이 h 에서 길이 length 의 이동 → (새 h, 허용 여부). 라우터 엣지·검증기 구간이 같이 쓴다 (D63)."""
    if kind == UP:
        return h, False
    if kind == DOWN:
        return 0.0, True
    h2 = h + length
    return h2, h2 <= lmax + 1e-6


def check_waypoints(waypoints, lmax: float) -> list:
    """꺾임점 목록을 start → end 로 따라가며 위반 [(구간 번호, 종류, 값)] — 종류 "up" (값 = dz) | "lmax" (값 = 누적 h)."""
    out = []
    h = 0.0
    for i in range(len(waypoints) - 1):
        a, b = waypoints[i], waypoints[i + 1]
        vec = [b[k] - a[k] for k in range(3)]
        kind = move_kind(vec)
        h, ok = step(h, kind, horizontal_length(vec), lmax)
        if not ok:
            out.append((i, kind if kind == UP else "lmax", (b[2] - a[2]) if kind == UP else h))
    return out


def min_step_drop(size: str) -> float:
    """한 단의 최소 낙차 (D61 ⑤): min(90°-90° 단 = max(§3.3, 2·t90), 45°-45° 단 = max(§3.3, 2·t45) × sin45°)."""
    L = PIPE_SPECS[size].min_straight
    d90 = max(L, 2 * elbow_tangent(size, 90))
    d45 = max(L, 2 * elbow_tangent(size, 45)) * math.sin(math.radians(45))
    return min(d90, d45)


def _stair_drop(size: str, H: float) -> float:
    """수평 거리 H 를 계단식 하향 규칙으로 갈 때 필요한 낙차 하한 (단자 수직 직관 제외)."""
    L = gravity_lmax(size)
    d = min_step_drop(size)
    k0 = max(1, int((H + d) / (L + d)))
    return max(0.0, min(max((k - 1) * d, H - k * L) for k in range(max(1, k0 - 1), k0 + 3)))


def required_drop(pipe) -> tuple:
    """D64: 새 규칙상 필요한 최소 낙차 하한 (mm)과 근거. 하한이므로 실제 높이 차가 이보다 작으면 규칙상 불가능.

    - start dir +z 이거나 end dir +z 면 무한대 (상향 금지)
    - H = start–end 수평 유클리드 거리 (수평 경로 길이의 하한), L = L_max, d = 한 단 최소 낙차 (D61 ⑤)
    - 수평 구간 k 개는 각각 ≤ L, 그 사이 하향 이동 ≥ k − 1 개 (각 낙차 ≥ d). 45° 하향은 낙차만큼 수평도 가므로
      H ≤ k·L + (총 낙차) → 낙차 ≥ max((k − 1)·d, H − k·L). k ≥ 1 에 대한 최솟값이 하한
    - start dir −z: 첫 직관 ≥ §3.3 (D45 ②) 만큼 수직으로 내려간다 / end dir −z: 마지막 직관 ≥ t₉₀ (D45 ③) — 더한다
    """
    size = pipe.nominal_size
    sd, ed = pipe.start.dir, pipe.end.dir
    if sd[2] > 0 or ed[2] > 0:
        return math.inf, "단자 방향이 위 (상향 금지)"
    H = math.dist(pipe.start.pos[:2], pipe.end.pos[:2])
    drop = _stair_drop(size, H)
    extra = _terminal_extra(pipe)
    return drop + extra, (f"수평 {H / 1000:.2f} m, L_max {gravity_lmax(size) / 1000:.2f} m, 단 낙차 ≥ "
                          f"{min_step_drop(size):.0f} mm → 계단 {drop:.0f} + 단자 수직 {extra:.0f} mm")


def _terminal_extra(pipe, start=True, end=True) -> float:
    extra = 0.0
    if start and pipe.start.dir[2] < 0:
        extra += PIPE_SPECS[pipe.nominal_size].min_straight
    if end and pipe.end.dir[2] < 0:
        extra += elbow_tangent(pipe.nominal_size, 90)
    return extra


def crossing_drop(pipe, wall) -> dict:
    """D64 보완 (S1-7, `추정:` — D64 와 같은 하한 원칙): 블록 쌍 관통 중력관은 start → 창 → end 두 구간의 하한.

    관통은 격벽 면에 수직인 수평 직관 (D94) 이므로 창에서는 수평으로 들어가 수평으로 나온다 — 창 점은 수평 방향
    가상 단자. 창 중심선 범위 = 창 안쪽 r (y·z). 구간마다 수평 거리 H 는 창 y 범위 안에서 최소 (구간마다 따로 고르므로
    합은 하한). 낙차 하한 req₁ (start → 창, start 수직 직관 포함), req₂ (창 → end, end 수직 직관 포함).
    가능 ⇔ 창 높이 zw 가 [max(z_end + req₂, 창 아래), min(z_start − req₁, 창 위)] 안에 있다. 장애물은 보지 않는다."""
    a, b = wall
    size, r = pipe.nominal_size, pipe.radius
    y0, y1, z0, z1 = pipe.crossing_window
    ylo, yhi, zlo, zhi = y0 + r, y1 - r, z0 + r, z1 - r
    sx, sy, sz = pipe.start.pos
    ex, ey, ez = pipe.end.pos
    f1, f2 = (a, b) if sx < a else (b, a)        # 진행 순서의 격벽 면

    def h(x, y, f):
        return math.hypot(x - f, y - min(max(y, ylo), yhi))

    if pipe.start.dir[2] > 0 or pipe.end.dir[2] > 0:
        req1 = req2 = math.inf
    else:
        req1 = _stair_drop(size, h(sx, sy, f1)) + _terminal_extra(pipe, end=False)
        req2 = _stair_drop(size, h(ex, ey, f2)) + _terminal_extra(pipe, start=False)
    lo, hi = max(ez + req2, zlo), min(sz - req1, zhi)
    return {"req_start_window_mm": round(req1, 1) if math.isfinite(req1) else None,
            "req_window_end_mm": round(req2, 1) if math.isfinite(req2) else None,
            "window_z_mm": [round(zlo, 1), round(zhi, 1)],
            "window_z_feasible_mm": [round(lo, 1), round(hi, 1)] if lo <= hi + 1e-6 else None,
            "drop_ok": bool(sz - ez >= req1 + req2 - 1e-6),
            "feasible": bool(lo <= hi + 1e-6)}


def feasibility(pipe, wall=None) -> dict:
    """D64 분류: {"feasible", "available_mm", "required_mm", "basis"} — 중력관이 아니면 None.
    wall = 블록 쌍 공유 격벽 (a, b) — 관통 배관이면 창을 지나는 하한 (crossing_drop) 을 더해 판정한다."""
    if not pipe.gravity_pipe:
        return None
    req, basis = required_drop(pipe)
    avail = pipe.start.pos[2] - pipe.end.pos[2]
    out = {"feasible": bool(avail >= req - 1e-6), "available_mm": round(avail, 1),
           "required_mm": round(req, 1) if math.isfinite(req) else None, "basis": basis}
    if wall is not None and getattr(pipe, "crossing_window", None):
        c = crossing_drop(pipe, wall)
        out["crossing"] = c
        if c["req_start_window_mm"] is not None:
            out["required_mm"] = max(out["required_mm"] or 0.0, round(c["req_start_window_mm"] + c["req_window_end_mm"], 1))
        if out["feasible"] and not c["feasible"]:
            out["feasible"] = False
            out["basis"] += (" · 관통 창 높이 제약 (낙차 부족)" if not c["drop_ok"] else " · 관통 창 높이 제약 (창 z 범위)")
    return out
