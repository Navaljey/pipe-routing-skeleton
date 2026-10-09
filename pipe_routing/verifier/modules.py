"""Layer 0 판정 모듈 7종 (§6.1, D12, D43, D44). 결과는 실제 기하(중심선 + 엘보 호)로 판정한다."""
import math

import numpy as np

from ..constants import SUPPORT_SPACING, VALVE_FRONT_MM, VALVE_Z_RANGE, support_kg
from .core import Context, PipeRoute, Violation, register
from .geom import EPS, angle_deg, boxes_overlap, segment_box_distance, segment_segment_distance, unit

ON_LINE_MM = 1.0   # 밸브·분기점이 직관 위에 있다고 볼 거리 허용오차


def _pt(p):
    return [round(float(c), 3) for c in p]


def _obstacle_hits(ctx: Context, A, B, slack, r, module, pid, what):
    out = []
    for o in ctx.scenario.obstacles:
        if not len(A):
            break
        d, at = segment_box_distance(A, B, o.box.min, o.box.max)
        d = d - slack
        k = int(d.argmin())
        if d[k] < r - EPS:
            out.append(Violation(module, pid, _pt(at[k]), f"{what} ↔ 장애물 {o.id} 거리 {d[k]:.1f} < 유효 반경 {r:.2f}",
                                 float(d[k]), o.id))
    return out


def _pipe_hits(ctx: Context, A, B, slack, r, module, pid, what):
    oA, oB, oR, oS, ids = ctx.others(pid)
    if not len(A) or not len(oA):
        return []
    d, p, _ = segment_segment_distance(A[:, None], B[:, None], oA[None], oB[None])
    gap = d - slack[:, None] - oS[None] - (r + oR[None])
    out = []
    for other in sorted(set(ids)):
        mask = np.array([i == other for i in ids])
        g = np.where(mask[None], gap, np.inf)
        i, j = np.unravel_index(int(g.argmin()), g.shape)
        if g[i, j] < -EPS:
            out.append(Violation(module, pid, _pt(p[i, j]),
                                 f"{what} ↔ 배관 {other} 이격 부족 {-g[i, j]:.1f}mm (필요 {r + oR[j]:.2f})",
                                 float(d[i, j]), other))
    return out


# ---------------------------------------------------------------- collision

@register("collision")
def collision(ctx: Context, pr: PipeRoute):
    """직관 조각 ↔ 장애물·다른 배관 (유효 반경, D17). 엘보 호는 bend 모듈이 본다."""
    pipe = ctx.pipes[pr.pipe_id]
    out = []
    for cl in ctx.centerlines[pr.pipe_id]:
        A, B, S, _ = cl.arrays(("straight",))
        out += _obstacle_hits(ctx, A, B, S, pipe.radius, "collision", pipe.id, "직관")
        out += _pipe_hits(ctx, A, B, S, pipe.radius, "collision", pipe.id, "직관")
    return out


# ---------------------------------------------------------------- boundary

@register("boundary")
def boundary(ctx: Context, pr: PipeRoute):
    """양 끝 = 단자 좌표, 첫·끝 구간 방향 = 단자 dir (D29), 중심선이 블록 안쪽 유효 반경 이내 (경계 단자 관통 구간 제외)."""
    pipe = ctx.pipes[pr.pipe_id]
    ext = np.array(ctx.scenario.block.extent)
    r = pipe.radius
    main = ctx.centerlines[pipe.id][0]
    P = main.points
    out = []
    for which, idx, seg in (("start", 0, P[1] - P[0]), ("end", -1, P[-1] - P[-2])):
        t = getattr(pipe, which)
        if np.linalg.norm(P[idx] - np.array(t.pos)) > 1e-3:
            out.append(Violation("boundary", pipe.id, _pt(P[idx]), f"경로 {which} 점이 단자 {list(t.pos)} 와 다름"))
        ang = angle_deg(seg, t.dir)
        if ang > ctx.angle_tol:
            out.append(Violation("boundary", pipe.id, _pt(P[idx]),
                                 f"{which} 구간 방향이 단자 dir {list(t.dir)} 와 {ang:.1f}° 어긋남 (D29)", ang))
    n_seg = len(P) - 1
    exempt = {}   # 구간 인덱스 → 검사 제외 축 (경계 단자 관통)
    for which, seg_i in (("start", 0), ("end", n_seg - 1)):
        t = getattr(pipe, which)
        if t.kind == "boundary":
            exempt[seg_i] = next(i for i in range(3) if t.dir[i])
    worst = {}   # (중심선, 조각 종류, ref) → (여유, 점, 축) — 같은 직관·같은 엘보는 가장 나쁜 점 하나만 보고
    for cl_i, cl in enumerate(ctx.centerlines[pipe.id]):
        for piece in cl.pieces:
            skip = exempt.get(piece.ref) if cl_i == 0 and piece.kind == "straight" else None
            key = (cl_i, piece.kind, piece.ref)
            for q in (piece.a, piece.b):
                for ax in range(3):
                    if ax == skip:
                        continue
                    gap = min(q[ax], ext[ax] - q[ax])
                    if gap < r - EPS and gap < worst.get(key, (math.inf,))[0]:
                        worst[key] = (gap, q, ax)
    for (cl_i, kind, ref), (gap, q, ax) in worst.items():
        what = "직관" if kind == "straight" else "엘보"
        out.append(Violation("boundary", pipe.id, _pt(q),
                             f"{what}이 블록 경계까지 {gap:.1f} < 유효 반경 {r:.2f} ({'xyz'[ax]}축)", float(gap)))
    return out


# ---------------------------------------------------------------- bend

@register("bend")
def bend(ctx: Context, pr: PipeRoute):
    """편향각 ∈ {45, 90, 135}, 최소 직진(§3.3, 마지막 구간 면제 = 노즐 종단 예외), 엘보 접선 겹침, 엘보 호 충돌."""
    pipe = ctx.pipes[pr.pipe_id]
    main = ctx.centerlines[pipe.id][0]
    P = main.points
    out = []
    for b in main.bends:
        if b.nominal is None:
            out.append(Violation("bend", pipe.id, _pt(b.pos), f"편향각 {b.angle:.1f}° ∉ {{45, 90, 135}}", b.angle))
    for i in range(len(P) - 2):       # 마지막 구간(end 단자로 들어가는 구간)은 면제
        L = float(np.linalg.norm(P[i + 1] - P[i]))
        if L < pipe.min_straight - EPS:
            out.append(Violation("bend", pipe.id, _pt((P[i] + P[i + 1]) / 2),
                                 f"직진 {L:.0f}mm < 최소 직진 {pipe.min_straight}mm (§3.3)", L))
    for i, need, L in main.overlaps:
        out.append(Violation("bend", pipe.id, _pt((P[i] + P[i + 1]) / 2),
                             f"엘보 접선 합 {need:.0f}mm > 구간 길이 {L:.0f}mm (엘보가 들어가지 않음)", need))
    A, B, S, _ = main.arrays(("arc",))
    out += _obstacle_hits(ctx, A, B, S, pipe.radius, "bend", pipe.id, "엘보")
    out += _pipe_hits(ctx, A, B, S, pipe.radius, "bend", pipe.id, "엘보")
    return out


# ---------------------------------------------------------------- gravity_slope

@register("gravity_slope")
def gravity_slope(ctx: Context, pr: PipeRoute):
    """중력관만: start→end(D20) 모든 구간 하향, 수평 성분이 있으면 낙차/수평거리 ≥ min_slope (§3.5)."""
    pipe = ctx.pipes[pr.pipe_id]
    if not pipe.gravity_pipe:
        return None
    P = ctx.centerlines[pipe.id][0].points
    out = []
    for i in range(len(P) - 1):
        d = P[i + 1] - P[i]
        h = math.hypot(d[0], d[1])
        drop = -d[2]
        mid = _pt((P[i] + P[i + 1]) / 2)
        if h > EPS:
            if drop < pipe.min_slope * h - 1e-9:
                out.append(Violation("gravity_slope", pipe.id, mid,
                                     f"구배 {drop / h:+.4f} < 최소 {pipe.min_slope:g} (구간 {i})", drop / h))
        elif drop <= 0:
            out.append(Violation("gravity_slope", pipe.id, mid, f"수직 상향 구간 (구간 {i})", drop))
    return out


# ---------------------------------------------------------------- valve

def _on_straight(cl, q):
    """q 가 놓인 직관 조각 (없으면 None)."""
    for piece in cl.pieces:
        if piece.kind != "straight":
            continue
        ab = piece.b - piece.a
        t = np.clip(np.dot(q - piece.a, ab) / np.dot(ab, ab), 0, 1)
        if np.linalg.norm(piece.a + t * ab - q) <= ON_LINE_MM:
            return piece
    return None


def valve_front_free(ctx: Context, pid: str, c, axis) -> list:
    """D43 — 밸브 전면 1m³ 가 비어 있는 수평 방향 목록."""
    pipe = ctx.pipes[pid]
    r = pipe.radius
    ext = ctx.scenario.block.extent
    oA, oB, oR, oS, _ = ctx.others(pid)
    free = []
    for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)):
        if abs(np.dot(unit(axis), d)) > 1 - 1e-6:
            continue   # 배관 축과 평행
        a = 0 if d[0] else 1
        b = 1 - a
        s = d[a]
        lo, hi = np.array(c, dtype=float), np.array(c, dtype=float)
        near, far = c[a] + s * r, c[a] + s * (r + VALVE_FRONT_MM)
        lo[a], hi[a] = min(near, far), max(near, far)
        lo[b], hi[b] = c[b] - VALVE_FRONT_MM / 2, c[b] + VALVE_FRONT_MM / 2
        lo[2], hi[2] = c[2] - VALVE_FRONT_MM / 2, c[2] + VALVE_FRONT_MM / 2
        if np.any(lo < -EPS) or np.any(hi > np.array(ext) + EPS):
            continue
        if any(boxes_overlap(lo, hi, o.box.min, o.box.max) for o in ctx.scenario.obstacles):
            continue
        if len(oA):
            dist, _ = segment_box_distance(oA, oB, lo, hi)
            if np.any(dist - oS < oR - EPS):
                continue
        free.append(d)
    return free


@register("valve")
def valve(ctx: Context, pr: PipeRoute):
    """밸브 위치가 직관 위, 높이 ∈ [700, 1500] (D19), 전면 1m³ (D43)."""
    pipe = ctx.pipes[pr.pipe_id]
    if not pipe.valve_positions:
        return None
    main = ctx.centerlines[pipe.id][0]
    out = []
    for v in pipe.valve_positions:
        q = np.array(v, dtype=float)
        piece = _on_straight(main, q)
        if piece is None:
            out.append(Violation("valve", pipe.id, _pt(q), "밸브가 경로의 직관 위에 없음"))
            continue
        if not VALVE_Z_RANGE[0] <= q[2] <= VALVE_Z_RANGE[1]:
            out.append(Violation("valve", pipe.id, _pt(q), f"밸브 높이 {q[2]:.0f} ∉ [700, 1500] (D19)", float(q[2])))
        if not valve_front_free(ctx, pipe.id, q, piece.b - piece.a):
            out.append(Violation("valve", pipe.id, _pt(q), "전면 1m×1m×1m 여유 공간 없음 (D43)"))
    return out


# ---------------------------------------------------------------- branch

@register("branch")
def branch(ctx: Context, pr: PipeRoute):
    """분기점이 주관 직관 위, 분기각(흐름 방향 기준) ∈ {45, 90}, 입력 branch_points 와 일치 (D44)."""
    pipe = ctx.pipes[pr.pipe_id]
    if not pr.branches and not pipe.branch_points:
        return None
    main = ctx.centerlines[pipe.id][0]
    legs = ctx.centerlines[pipe.id][1:]
    out = []
    starts = []
    for cl in legs:
        s = cl.points[0]
        starts.append(s)
        piece = _on_straight(main, s)
        if piece is None:
            out.append(Violation("branch", pipe.id, _pt(s), "분기점이 주관 직관 위에 없음"))
            continue
        ang = angle_deg(piece.b - piece.a, cl.points[1] - cl.points[0])
        if not any(abs(ang - a) <= ctx.angle_tol for a in (45, 90)):
            out.append(Violation("branch", pipe.id, _pt(s), f"분기각 {ang:.1f}° ∉ {{45, 90}}", ang))
    for bp in pipe.branch_points:
        q = np.array(bp, dtype=float)
        if not any(np.linalg.norm(q - s) <= ON_LINE_MM for s in starts):
            out.append(Violation("branch", pipe.id, _pt(q), "입력 분기점에서 시작하는 분기 경로 없음"))
    if pipe.branch_points:
        for s in starts:
            if not any(np.linalg.norm(np.array(bp) - s) <= ON_LINE_MM for bp in pipe.branch_points):
                out.append(Violation("branch", pipe.id, _pt(s), "입력에 없는 위치에서 분기"))
    return out


# ---------------------------------------------------------------- support

FACES = (("floor", 2, 0), ("ceiling", 2, 1), ("wall_x0", 0, 0), ("wall_x1", 0, 1), ("wall_y0", 1, 0), ("wall_y1", 1, 1))
CANDIDATE_STEP_MM = 10   # D5 정밀도
_CHUNK = 64


def support_candidates(ctx: Context, pid: str, pts: np.ndarray, dirs: np.ndarray) -> list:
    """후보 위치마다 서포트 (face, 길이, kg) 또는 None (D21~D23).

    지지면 = 배관 축과 평행하지 않은 법선의 구조면 중 **가장 가까운 면** 하나. 지지선(배관 중심 → 면 수직)이
    장애물과 닿거나 다른 배관 유효 반경 안을 지나면 그 위치는 불가.
    """
    ext = np.array(ctx.scenario.block.extent)
    oA, oB, oR, oS, _ = ctx.others(pid)
    res = []
    for k0 in range(0, len(pts), _CHUNK):
        P = pts[k0:k0 + _CHUNK]
        U = dirs[k0:k0 + _CHUNK]
        best = []
        for p, u in zip(P, U):
            cand = [(abs(p[ax] - side * ext[ax]), name, ax, side) for name, ax, side in FACES
                    if abs(u[ax]) < 1 - 1e-9]
            best.append(min(cand))
        Q = P.copy()
        for i, (_, _, ax, side) in enumerate(best):
            Q[i, ax] = side * ext[ax]
        blocked = np.zeros(len(P), dtype=bool)
        for o in ctx.scenario.obstacles:
            d, _ = segment_box_distance(P, Q, o.box.min, o.box.max, iters=40)
            blocked |= d <= EPS
        if len(oA):
            d, _, _ = segment_segment_distance(P[:, None], Q[:, None], oA[None], oB[None])
            blocked |= np.any(d - oS[None] < oR[None] - EPS, 1)
        for i, (dist, name, _, _) in enumerate(best):
            res.append(None if blocked[i] else (name, float(dist), support_kg(dist)))
    return res


@register("support")
def support(ctx: Context, pr: PipeRoute):
    """§3.6 최대 간격 안에 설치 가능 위치가 있는가. 단자는 고정점(서포트)으로 본다 (D44).

    간격 규칙: 직전 서포트부터 Σ(구간 길이 / 그 구간 최대간격) ≤ 1 (수직 구간 = 수직 간격, 그 밖 = 수평 간격).
    배치 = 탐욕적 최원거리 (도달 가능한 가장 먼 가능 위치). 결과 서포트 목록은 J 서포트 항(6단계)에 쓴다.
    """
    pipe = ctx.pipes[pr.pipe_id]
    P = ctx.centerlines[pipe.id][0].points
    seg = P[1:] - P[:-1]
    lens = np.linalg.norm(seg, axis=1)
    hmax, vmax = SUPPORT_SPACING[pipe.nominal_size]
    is_vert = np.abs(seg[:, 2]) >= lens * (1 - 1e-9)
    smax = np.where(is_vert, vmax, hmax)
    cum = np.concatenate([[0], np.cumsum(lens)])          # 길이
    ucum = np.concatenate([[0], np.cumsum(lens / smax)])  # 정규화 길이
    total = float(cum[-1])

    def locate(s):
        i = int(min(np.searchsorted(cum, s, side="right") - 1, len(lens) - 1))
        return P[i] + (s - cum[i]) / lens[i] * seg[i], i

    def u_at(s):
        i = int(min(np.searchsorted(cum, s, side="right") - 1, len(lens) - 1))
        return ucum[i] + (s - cum[i]) / smax[i]

    def s_at_u(u):
        i = int(min(np.searchsorted(ucum, u, side="right") - 1, len(lens) - 1))
        return min(total, cum[i] + (u - ucum[i]) * smax[i])

    supports, out = [], []
    s_last = 0.0
    while u_at(total) - u_at(s_last) > 1 + 1e-9:
        s_reach = s_at_u(u_at(s_last) + 1)
        grid = np.arange(math.floor(s_reach / CANDIDATE_STEP_MM) * CANDIDATE_STEP_MM, s_last, -CANDIDATE_STEP_MM)
        grid = grid[grid > s_last + EPS]
        found = None
        for k0 in range(0, len(grid), _CHUNK):
            ss = grid[k0:k0 + _CHUNK]
            locs = [locate(s) for s in ss]
            pts = np.array([q for q, _ in locs])
            dirs = np.array([unit(seg[i]) for _, i in locs])
            for s, q, c in zip(ss, pts, support_candidates(ctx, pipe.id, pts, dirs)):
                if c is not None:
                    found = (float(s), q, c)
                    break
            if found:
                break
        if found is None:
            mid, _ = locate((s_last + s_reach) / 2)
            out.append(Violation("support", pipe.id, _pt(mid),
                                 f"{s_last / 1000:.2f}~{s_reach / 1000:.2f} m 구간에 서포트 설치 가능 위치 없음 (§3.6)",
                                 s_reach - s_last))
            s_last = s_reach
            continue
        s, q, (face, dist, kg) = found
        supports.append({"pos": _pt(q), "face": face, "angle_length_mm": round(dist + 100, 1), "kg": round(kg, 4)})
        s_last = s
    ctx.cache[("supports", pipe.id)] = supports
    return out
