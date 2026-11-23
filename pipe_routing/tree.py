"""S2-3 분기 계통 (D112 · D134) — 검증기·라우터·힌트가 같이 쓰는 나무 기하.

계통 = 입력 배관 하나 (주관 start → end + `branches` 다리 1~5개). 다리 경로는 **기하 순서 = 티 → 다리 단자**
(D134 ④: 시작점이 자기 주관 직관 위, 끝점이 자기 단자) 로 둔다. 흐름 방향은 다리의 flow 로 정한다:
out (갈래) = 티 → 단자, in (합류) = 단자 → 티 (중력관은 in 만, §7 Q1).

티 규칙 (D112, D134 구현 결정 (나)):
  - 티는 **축 정렬** 주관 직관 위 (엘보 호 위 불가, 1 mm 안)
  - 분기각 = 주관 흐름 벡터 ∠ 다리 흐름 벡터 ∈ {45, 90} (135° 금지, Q2)
  - 티 중심 → 주관 앞·뒤 다음 관이음 (엘보 접점·다른 티) ≥ 3D_주관, 분기 쪽 다음 관이음 (다리 첫 엘보 접점) ≥ 3D_분기
    (D = 호칭경 mm, 기준점 = 티 중심, Q7). 단자는 관이음이 아니다
  - 티 = 주관의 서포트 고정점 아님, 다리 서포트 간격은 티에서 0 부터 (Q3 (b)) — 다리 가상 배관의 단자로 둔다
자기 간섭 (같은 계통 안, D134 보완 8 `추정:`): 예외는 티 몸체 구역만 — family_conflicts. 검증기 collision 과
라우터 정확 풀이의 자기 간섭 판정이 이 함수를 같이 쓴다.
"""
import math
from dataclasses import dataclass

import numpy as np

from .constants import ANGLE_TOL_DEG, elbow_radius
from .scenario import Pipe, Terminal

ON_LINE_MM = 1.0
EPS = 1e-6


def nominal_mm(nd: str) -> float:
    return float(nd.rstrip("A"))


def three_d(nd: str) -> float:
    """티 직관 3D (D112, D = 호칭경 mm)."""
    return 3.0 * nominal_mm(nd)


def leg_id(pipe: Pipe, b) -> str:
    return f"{pipe.id}.{b.id}"


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def leg_flow(geom, flow: str) -> list:
    """기하 순서 (티 → 단자) 꺾임점 → 흐름 순서."""
    g = [list(map(float, w)) for w in geom]
    return g if flow == "out" else g[::-1]


def leg_flow_pipe(pipe: Pipe, b, geom) -> Pipe:
    """다리 가상 배관 (검증기·라우터): 다리 구경·같은 서비스·class·보온, 티 쪽 단자 = 티 점 (노즐형, 서포트 고정점),
    단자 쪽 = 입력 다리 단자. start/end 는 흐름 순서."""
    g = [tuple(map(float, w)) for w in geom]
    tee = g[0]
    first = _unit(np.subtract(g[1], g[0])) if len(g) >= 2 else np.zeros(3)
    if b.flow == "out":
        start = Terminal(tee, "nozzle", tuple(float(c) for c in first))
        end = b.terminal
    else:
        start = b.terminal
        end = Terminal(tee, "nozzle", tuple(float(-c) for c in first))
    p = Pipe(leg_id(pipe, b), pipe.type_id, b.nominal_size, pipe.gravity_pipe, pipe.min_slope,
             pipe.insulation_thickness, start, end, service=pipe.service, pipe_class=pipe.pipe_class)
    p.parent = pipe.id
    p.leg = b
    return p


# --------------------------------------------------------------------------- 티 판정 (검증기 = 라우터 = 힌트)

@dataclass
class TeeCheck:
    ok: bool
    reasons: list            # 위반 사유 (빈 목록 = 통과)
    seg: int = -1            # 주관 구간 인덱스
    s: float = math.nan      # 주관 구간 시작점에서 티까지 거리
    angle: float = math.nan  # 분기각 (흐름 벡터 사이)
    main_flow: tuple = ()
    leg_flow: tuple = ()


def main_segments(main_cl) -> list:
    """주관 직관 조각 [(구간 i, a, b, 방향, 앞 관이음 있음, 뒤 관이음 있음)] — 앞·뒤 관이음 = 그쪽 끝이 엘보 접점."""
    n = len(main_cl.points)
    out = []
    for pc in main_cl.pieces:
        if pc.kind != "straight":
            continue
        i = pc.ref
        u = _unit(np.asarray(pc.b) - np.asarray(pc.a))
        out.append((i, np.asarray(pc.a, float), np.asarray(pc.b, float), u, i > 0, i + 1 < n - 1))
    return out


def tee_check(main_cl, main_nd: str, tee_pos, leg_dir_geom, flow: str, leg_nd: str, other_tees=(),
              leg_first_straight=None, tol=ANGLE_TOL_DEG) -> TeeCheck:
    """티 하나의 규칙 (D112·D134). leg_dir_geom = 티에서 다리 쪽 첫 방향 (기하 순서), leg_first_straight = 티 → 다리
    첫 엘보 접점 직관 길이 (다리에 엘보가 없으면 None). other_tees = 같은 주관의 다른 티 점."""
    T = np.asarray(tee_pos, float)
    reasons = []
    if nominal_mm(leg_nd) > nominal_mm(main_nd):
        reasons.append(f"다리 구경 {leg_nd} > 주관 {main_nd} (D112)")
    best = None
    for i, a, b, u, fa, fb in main_segments(main_cl):
        L = float(np.linalg.norm(b - a))
        s = float(np.dot(T - a, u))
        d = float(np.linalg.norm(T - (a + u * min(max(s, 0.0), L))))
        if d <= ON_LINE_MM + EPS and -EPS <= s <= L + EPS and (best is None or d < best[0]):
            best = (d, i, a, b, u, fa, fb, s, L)
    if best is None:
        return TeeCheck(False, reasons + ["티가 주관 직관 위에 없음 (엘보 위·주관 밖 불가)"])
    _, i, a, b, u, fa, fb, s, L = best
    if int(np.count_nonzero(np.abs(u) > 1e-9)) != 1:
        reasons.append("티는 축 정렬 주관 직관 위에만 (D134 구현 결정 (나))")
    need = three_d(main_nd)
    if fa and s < need - EPS:
        reasons.append(f"티 → 앞 엘보 접점 {s:.1f} < 3D_주관 {need:.0f} (D112)")
    if fb and L - s < need - EPS:
        reasons.append(f"티 → 뒤 엘보 접점 {L - s:.1f} < 3D_주관 {need:.0f} (D112)")
    for q in other_tees:
        dq = float(np.linalg.norm(np.asarray(q, float) - T))
        if dq < need - EPS:
            reasons.append(f"티끼리 거리 {dq:.1f} < 3D_주관 {need:.0f} (D112)")
    g = _unit(leg_dir_geom)
    lf = g if flow == "out" else -g
    ang = math.degrees(math.acos(float(np.clip(np.dot(u, lf), -1.0, 1.0))))
    nom = next((x for x in (45, 90) if abs(ang - x) <= tol), None)
    if nom is None:
        reasons.append(f"분기각 {ang:.1f}° ∉ {{45, 90}} (주관 흐름 ∠ 다리 흐름, 135° 금지 — D112 Q2)")
    if leg_first_straight is not None and leg_first_straight < three_d(leg_nd) - EPS:
        reasons.append(f"티 → 다리 첫 엘보 접점 {leg_first_straight:.1f} < 3D_분기 {three_d(leg_nd):.0f} (D112)")
    return TeeCheck(not reasons, reasons, i, s, ang, tuple(map(float, u)), tuple(map(float, lf)))


def leg_first_straight(leg_cl):
    """다리 (기하 순서) 첫 직관: 티 → 첫 엘보 접점 길이 (엘보 없으면 None)."""
    if len(leg_cl.points) <= 2:
        return None
    pc = next((p for p in leg_cl.pieces if p.kind == "straight" and p.ref == 0), None)
    if pc is None:
        return 0.0
    return float(np.linalg.norm(np.asarray(pc.b) - np.asarray(leg_cl.points[0])))


# --------------------------------------------------------------------------- 자기 간섭 (다리 ↔ 자기 주관·다른 다리)

def ordered_pieces(cl) -> list:
    """중심선 조각을 경로 순서로 (직관 i → 호 i+1 → 직관 i+1 …)."""
    st = {p.ref: p for p in cl.pieces if p.kind == "straight"}
    arcs = {}
    for p in cl.pieces:
        if p.kind == "arc":
            arcs.setdefault(p.ref, []).append(p)
    out = []
    n = len(cl.points)
    for i in range(n - 1):
        if i in arcs:
            out += arcs[i]
        if i in st:
            out.append(st[i])
    return out


def _split_pieces(cl, keep_zone):
    """중심선 조각 → (구역 안 조각, 구역 밖 조각) 배열. keep_zone(p) = (a, b) 구역 안 부분 (없으면 None) 과 그 밖 부분 목록."""
    zin, zout = ([], [], []), ([], [], [])
    for p in cl.pieces:
        parts_in, parts_out = keep_zone(p)
        for (a, b) in parts_in:
            zin[0].append(a); zin[1].append(b); zin[2].append(p.slack)
        for (a, b) in parts_out:
            zout[0].append(a); zout[1].append(b); zout[2].append(p.slack)

    def arr(z):
        if not z[0]:
            e = np.zeros((0, 3))
            return e, e, np.zeros(0)
        return np.array(z[0]), np.array(z[1]), np.array(z[2])
    return arr(zin), arr(zout)


def _cut(a, b, c0, c1, origin, u):
    """조각 a→b 를 직선 origin + t·u 의 t ∈ [c0, c1] 구간 안·밖으로 나눈다 (조각이 그 직선 위 직관일 때만 안이 생긴다)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = float(np.linalg.norm(d))
    on = L > EPS and abs(abs(float(np.dot(d / L, u))) - 1) < 1e-9 and \
        float(np.linalg.norm(np.cross(a - origin, u))) <= ON_LINE_MM and \
        float(np.linalg.norm(np.cross(b - origin, u))) <= ON_LINE_MM
    if not on:
        return [], [(a, b)]
    ta, tb = float(np.dot(a - origin, u)), float(np.dot(b - origin, u))
    lo, hi = max(min(ta, tb), c0), min(max(ta, tb), c1)
    if lo >= hi - EPS:
        return [], [(a, b)]

    def at(t):
        return a + d * ((t - ta) / (tb - ta))
    ins = [(at(lo), at(hi))]
    outs = []
    for x0, x1 in ((min(ta, tb), lo), (hi, max(ta, tb))):
        if x1 - x0 > EPS:
            outs.append((at(x0), at(x1)))
    return ins, outs


def zone_reach(leg_cl, main_cl, r_main: float, r_leg: float) -> float:
    """(r_주관 + r_다리) / sin θ — θ = 다리 첫 직관 ∠ 주관 축 (티가 있는 주관 직관). 이 길이를 지나면 다리가 주관 축에서
    r_주관 + r_다리 이상 떨어진다 (D134 보완 8 수정, 2026-11-23 소유자)."""
    P = leg_cl.points
    if len(P) < 2:
        return 0.0
    T = np.asarray(P[0], float)
    g = _unit(P[1] - P[0])
    for i, a, b, u, fa, fb in main_segments(main_cl):
        if float(np.linalg.norm(np.cross(T - a, u))) <= ON_LINE_MM:
            sin = math.sqrt(max(0.0, 1.0 - float(np.dot(g, u)) ** 2))
            return (r_main + r_leg) / max(sin, 1e-6)
    return 0.0


def family_conflicts(main_cl, main_nd: str, r_main: float, legs: list) -> list:
    """같은 계통 안 간섭 (D134 보완 8, `추정:` — 2026-11-23 소유자 수정): 예외 = **티 몸체 구역**만 — 다리의 첫 직관 중
    티 중심에서 max(3D_분기, R / sin θ) 까지 ↔ 주관 직관 중 티 중심 ±max(3D_주관, R / sin θ) (R = r_주관 + r_다리,
    θ = 다리 첫 직관 ∠ 주관 축), 이 두 조각 사이만 뺀다. 3D 만으로는 r (보온 포함) 이 3D_분기 보다 큰 흔한 티가 늘
    위반이 되어 구역을 이격 밖까지 늘렸다. 그 밖은 다른 배관과 같다 (다리 ↔ 자기 주관, 다리 ↔ 다리 모두
    r₁ + r₂, 엘보 호 포함). legs = [(다리 id, 다리 중심선 (기하 순서, 티 → 단자), 다리 구경, r_다리)].
    반환 [(다리 id, 상대 id ("main" 또는 다리 id), 점, 거리, 필요)] — 다리마다 상대별 가장 나쁜 한 쌍."""
    from .verifier.geom import segment_segment_distance
    out = []
    groups = []
    for lid, cl, nd, r in legs:
        T = np.asarray(cl.points[0], float)
        g = _unit(cl.points[1] - cl.points[0]) if len(cl.points) >= 2 else np.zeros(3)
        zr = zone_reach(cl, main_cl, r_main, r)
        lz = max(three_d(nd), zr)
        leg_in, leg_out = _split_pieces(cl, lambda p: _cut(p.a, p.b, 0.0, lz, T, g) if p.kind == "straight"
                                        else ([], [(np.asarray(p.a, float), np.asarray(p.b, float))]))
        mu = None
        for i, a, b, u, fa, fb in main_segments(main_cl):
            if float(np.linalg.norm(np.cross(T - a, u))) <= ON_LINE_MM and -EPS <= float(np.dot(T - a, u)) <= \
                    float(np.linalg.norm(b - a)) + EPS:
                mu = u
                break
        mz = max(three_d(main_nd), zr)
        if mu is None:
            main_in = (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0))
            mA, mB, mS, _ = main_cl.arrays()
            main_out = (mA, mB, mS)
        else:
            main_in, main_out = _split_pieces(main_cl, lambda p: _cut(p.a, p.b, -mz, mz, T, mu) if p.kind == "straight"
                                              else ([], [(np.asarray(p.a, float), np.asarray(p.b, float))]))
        groups.append((lid, r, leg_in, leg_out, main_in, main_out))

    def cat(*zs):
        zs = [z for z in zs if len(z[0])]
        if not zs:
            e = np.zeros((0, 3))
            return e, e, np.zeros(0)
        return tuple(np.concatenate([z[k] for z in zs]) for k in range(3))

    def worst(A, B, oA, oB, need):
        if not len(A[0]) or not len(oA[0]):
            return None
        d, p, _ = segment_segment_distance(A[0][:, None], A[1][:, None], oA[0][None], oA[1][None])
        gap = d - A[2][:, None] - oA[2][None]
        i, j = np.unravel_index(int(gap.argmin()), gap.shape)
        return (p[i, j], float(gap[i, j])) if gap[i, j] < need - EPS else None

    for k, (lid, r, leg_in, leg_out, main_in, main_out) in enumerate(groups):
        need = r_main + r
        hits = [worst(leg_in, None, main_out, None, need), worst(leg_out, None, cat(main_in, main_out), None, need)]
        hits = [h for h in hits if h is not None]
        if hits:
            p, g = min(hits, key=lambda h: h[1])
            out.append((lid, "main", p, g, need))
        full = cat(leg_in, leg_out)
        for oid, ro, o_in, o_out, _, _ in groups[k + 1:]:
            h = worst(full, None, cat(o_in, o_out), None, r + ro)
            if h is not None:
                out.append((lid, oid, h[0], h[1], r + ro))
    return out


def centerline_of(nd: str, wps):
    from .verifier.geom import build_centerline
    return build_centerline(wps, elbow_radius(nd), ANGLE_TOL_DEG)
