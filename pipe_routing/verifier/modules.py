"""Layer 0 판정 모듈 7종 (§6.1, D12, D43, D44). 결과는 실제 기하(중심선 + 엘보 호)로 판정한다."""
import math

import numpy as np

from .. import support_std as ss
from ..constants import SUPPORT_SPACING, VALVE_FRONT_MM, VALVE_Z_RANGE
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
    # 결과 불변 가속 (D84): AABB 간격이 이격 문턱 + 1 mm 보다 먼 조각 쌍은 위반일 수 없다 (선분 거리 ≥ AABB 간격).
    # 행을 64 개씩 묶어 묶음 AABB 로 열을 거른 뒤 그 쌍만 계산한다. 배관별 최소는 (gap, 행, 열) 사전식 —
    # 전체 계산의 argmin (같은 값이면 행 우선 첫 번째) 과 같다.
    olo, ohi = np.minimum(oA, oB), np.maximum(oA, oB)
    lim_o = oS + r + oR + 1.0
    best = {}
    for k0 in range(0, len(A), 64):
        a, b, sl = A[k0:k0 + 64], B[k0:k0 + 64], slack[k0:k0 + 64]
        clo, chi = np.minimum(a, b).min(0), np.maximum(a, b).max(0)
        g = np.maximum(0.0, np.maximum(olo - chi, clo - ohi))
        jj = np.nonzero(np.sqrt((g * g).sum(-1)) <= lim_o + sl.max())[0]
        if not len(jj):
            continue
        d, p, _ = segment_segment_distance(a[:, None], b[:, None], oA[jj][None], oB[jj][None])
        gap = d - sl[:, None] - oS[jj][None] - (r + oR[jj][None])
        idj = np.array([ids[j] for j in jj])
        for other in np.unique(idj):
            gm = np.where((idj == other)[None], gap, np.inf)
            i, j = np.unravel_index(int(gm.argmin()), gm.shape)
            if gm[i, j] < -EPS and (str(other) not in best or gm[i, j] < best[str(other)][0]):
                best[str(other)] = (gm[i, j], p[i, j], float(d[i, j]), float(oR[jj[j]]))
    out = []
    for other in sorted(best):
        g, q, dd, ro = best[other]
        out.append(Violation(module, pid, _pt(q), f"{what} ↔ 배관 {other} 이격 부족 {-g:.1f}mm (필요 {r + ro:.2f})",
                             dd, other))
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
    """편향각 ∈ {45, 90, 135}, 직관 길이(D45), 엘보 호 충돌(엘보 점유 공간).

    직관 길이 (D45, t = 엘보 접선 길이 R·tan(θ/2)):
      start → 첫 꺾임 ≥ max(§3.3, t_첫), 꺾임 사이 ≥ max(§3.3, t_앞 + t_뒤), 마지막 꺾임 → end ≥ t_마지막
    """
    pipe = ctx.pipes[pr.pipe_id]
    main = ctx.centerlines[pipe.id][0]
    P = main.points
    out = []
    for b in main.bends:
        if b.nominal is None:
            out.append(Violation("bend", pipe.id, _pt(b.pos), f"편향각 {b.angle:.1f}° ∉ {{45, 90, 135}}", b.angle))
    n = len(P)
    tan = {b.vertex: b.tangent for b in main.bends}
    Ls = pipe.min_straight
    for i in range(n - 1):
        if n == 2:
            break                     # 꺾임 없음
        ta = tan.get(i, 0.0)
        tb = tan.get(i + 1, 0.0)
        if math.isinf(ta) or math.isinf(tb):
            continue                  # 180° 역행 — 편향각 위반으로 이미 보고
        need = ta if i == n - 2 else max(Ls, ta + tb)
        length = float(np.linalg.norm(P[i + 1] - P[i]))
        if length < need - EPS:
            where = "start→첫 꺾임" if i == 0 else ("마지막 꺾임→end" if i == n - 2 else "꺾임 사이")
            basis = (f"엘보 접선 {ta:.0f}" if i == n - 2 else f"§3.3 {Ls}, 엘보 접선 {ta:.0f}+{tb:.0f}")
            out.append(Violation("bend", pipe.id, _pt((P[i] + P[i + 1]) / 2),
                                 f"{where} 직관 {length:.0f}mm < 필요 {need:.0f}mm ({basis}, D45)", length))
    A, B, S, _ = main.arrays(("arc",))
    out += _obstacle_hits(ctx, A, B, S, pipe.radius, "bend", pipe.id, "엘보")
    out += _pipe_hits(ctx, A, B, S, pipe.radius, "bend", pipe.id, "엘보")
    return out


# ---------------------------------------------------------------- gravity_slope

@register("gravity_slope")
def gravity_slope(ctx: Context, pr: PipeRoute):
    """중력관만: 계단식 하향 규칙 (D61, §3.5) — 상향 이동 금지, 마지막 하향 이후 수평 누적 ≤ L_max (D62).

    라우터와 같은 판정 함수 (`pipe_routing.gravity.check_waypoints`, D63). 구간 = 꺾임점 사이.
    """
    from ..gravity import check_waypoints, gravity_lmax
    pipe = ctx.pipes[pr.pipe_id]
    if not pipe.gravity_pipe:
        return None
    P = ctx.centerlines[pipe.id][0].points
    lmax = gravity_lmax(pipe.nominal_size)
    out = []
    for i, kind, val in check_waypoints([list(map(float, q)) for q in P], lmax):
        mid = _pt((P[i] + P[i + 1]) / 2)
        if kind == "up":
            out.append(Violation("gravity_slope", pipe.id, mid, f"상향 구간 (구간 {i}, dz {val:+.0f} mm)", val))
        else:
            out.append(Violation("gravity_slope", pipe.id, mid,
                                 f"하향 없이 수평 누적 {val / 1000:.2f} m > L_max {lmax / 1000:.2f} m (구간 {i})", val))
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

CANDIDATE_STEP_MM = 10   # D5 정밀도
_CHUNK = 64


def support_candidates(ctx: Context, pid: str, pts: np.ndarray, dirs: np.ndarray) -> list:
    """후보 위치마다 서포트 dict 또는 None (D80 ③④⑧).

    (면, 형식) 조합 중 표 범위 안이고 부재(앵글 단면 막대)가 장애물·얹히지 않은 배관(유효 반경)·블록 밖과 간섭하지 않는
    가장 가벼운 것 (천장 선호 D79 ④). 라우터(D80 ⑨)와 같은 함수 support_std.evaluate 를 쓴다.
    """
    pipe = ctx.pipes[pid]
    std = ctx.std
    ext = np.array(ctx.scenario.block.extent, dtype=float)
    oA, oB, oR, oS, _ = ctx.others(pid)
    box_lo = [o.box.min for o in ctx.scenario.obstacles]
    box_hi = [o.box.max for o in ctx.scenario.obstacles]
    od = pipe.od
    res = []
    for k0 in range(0, len(pts), _CHUNK):
        P = pts[k0:k0 + _CHUNK]
        U = dirs[k0:k0 + _CHUNK]
        ev = ss.evaluate(std, ext, box_lo, box_hi, pipe.nominal_size, od, P, U, others=(oA, oB, oR, oS), exact=True,
                         trace=ctx.cache.setdefault(("support_trace", pid), []))
        for i in range(len(P)):
            o = int(ev["opt"][i])
            if o < 0:
                res.append(None)
                continue
            face, frm = ss.OPTIONS[o]
            L = float(ev["L"][i])
            reinf = bool(ev["reinf"][i])
            ang = std.angle_of(frm, pipe.nominal_size, int(ev["k"][i]))
            mem = ss.members(frm, P[i], U[i], face, L, od, reinf)
            res.append({"form": frm, "variant": "reinforced" if reinf else "standard", "face": face,
                        "section": ang, "L": round(L, 1), "dims": {"L": round(L, 1)},
                        "members": [{"a": _pt(a), "b": _pt(b), "role": role} for a, b, role in mem],
                        "member_length_mm": round(float(ss.member_length(frm, np.array([L]), od,
                                                                         np.array([reinf]))[0]), 1),
                        "kg": round(float(ev["kg"][i]), 4)})
    return res


def _together_scan(ctx, pipe, far, locate, seg):
    """D89 위치 탐색 (먼 쪽부터). 반환 = (s, 점, 방향, 서포트) 또는 None.
    증분 검증(D85) 영향 범위: 본 수평 축 정렬 직관 구간마다 가로 ±(L1 표 끝 + 최대 OD/2 + 100), 높이 전체를 판정 영역으로
    남긴다 — 그 안에 다른 배관이 들어오거나 나가면 함께 얹을 배관·D 부재 막힘이 바뀔 수 있다."""
    from . import support_group as sg
    trace = ctx.cache.setdefault(("support_trace", pipe.id), [])
    H = float(ctx.scenario.block.extent[2])
    lat = sg.l1_max(ctx.std) + max(p.od for p in ctx.pipes.values()) / 2 + ss.C_SIDE_MM
    half = max(ss.ANGLE_HALF.values())
    by_seg, cands = {}, {}
    out = None
    for s in far:
        q, i = locate(s)
        u = unit(seg[i])
        if abs(u[2]) > 1e-9 or abs(abs(u).max() - 1) > 1e-9:
            continue
        by_seg.setdefault(i, [q, q, u])
        by_seg[i][1] = q
        if out is not None:
            continue
        if i not in cands:
            cands[i] = sg.together_cands(ctx, pipe.id, q, u)
        ax = int(np.argmax(np.abs(u)))
        pcs = [c for c in cands[i] if min(c[2][ax], c[3][ax]) - 1e-6 <= q[ax] <= max(c[2][ax], c[3][ax]) + 1e-6]
        if pcs:
            r = sg.together_at(ctx, pipe.id, q, u, float(s), pieces=pcs)
            if r is not None:
                out = (float(s), q, u, r)
    for i, (qa, qb, u) in by_seg.items():
        ax = int(np.argmax(np.abs(u)))
        lo, hi = np.minimum(qa, qb).astype(float), np.maximum(qa, qb).astype(float)
        for k in range(2):
            if k != ax:
                lo[k] -= lat
                hi[k] += lat
        lo[2], hi[2] = 0.0, H
        trace.append((lo, hi, float(half)))
    return out


@register("support")
def support(ctx: Context, pr: PipeRoute):
    """§3.6 최대 간격 안에 설치 가능 위치가 있는가. 단자는 고정점(서포트)으로 본다 (D44).

    간격 규칙: 직전 서포트부터 Σ(구간 길이 / 그 구간 최대간격) ≤ 1 (수직 구간 = 수직 간격, 그 밖 = 수평 간격).
    배치 = 탐욕적 최원거리 (도달 가능한 가장 먼 가능 위치, D80 ⑤). 위치마다 형식 선택 = D80 ③.
    결과 서포트 목록은 J 서포트 항(6단계)에 쓰고, D 묶음(D80 ⑥)은 verify() 끝에서 병합한다.
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

    supports, out, gaps = [], [], []
    ctx.cache[("support_trace", pipe.id)] = []   # 판정한 부재 영역 (증분 검증 D85 영향 범위)
    s_last = 0.0
    while u_at(total) - u_at(s_last) > 1 + 1e-9:
        s_reach = s_at_u(u_at(s_last) + 1)
        grid = np.arange(math.floor(s_reach / CANDIDATE_STEP_MM) * CANDIDATE_STEP_MM, s_last, -CANDIDATE_STEP_MM)
        grid = grid[grid > s_last + EPS]
        found = None
        for k0 in range(0, len(grid), _CHUNK):
            ss_ = grid[k0:k0 + _CHUNK]
            locs = [locate(s) for s in ss_]
            pts = np.array([q for q, _ in locs])
            dirs = np.array([unit(seg[i]) for _, i in locs])
            for s, q, u, c in zip(ss_, pts, dirs, support_candidates(ctx, pipe.id, pts, dirs)):
                if c is not None:
                    found = (float(s), q, u, c)
                    break
            if found:
                break
        # D89 (2026-10-19): A·B·C 가 모두 불가한 위치에서 나란한 배관을 함께 얹는 D. found 보다 먼 위치만 본다
        # (그 위치들은 A·B·C 불가 — 그리드를 먼 쪽부터 봤으므로). 가장 먼 가능 위치 (D80 ⑤ 탐욕 그대로).
        far = grid if found is None else grid[grid > found[0] + EPS]
        if len(far):
            t = _together_scan(ctx, pipe, far, locate, seg)
            if t is not None:
                found = t
        if found is None:
            mid, _ = locate((s_last + s_reach) / 2)
            out.append(Violation("support", pipe.id, _pt(mid),
                                 f"{s_last / 1000:.2f}~{s_reach / 1000:.2f} m 구간에 서포트 설치 가능 위치 없음 (§3.6, D80 ④)",
                                 s_reach - s_last))
            n = max(2, int((s_reach - s_last) / 200) + 1)
            gaps.append([_pt(locate(s)[0]) for s in np.linspace(s_last, s_reach, n)])
            s_last = s_reach
            continue
        s, q, u, c = found
        if c.get("origin") == "D89":
            c.update({"pos": _pt(q), "axis": _pt(u)})
        else:
            c.update({"pos": _pt(q), "axis": _pt(u), "carried": [pipe.id], "basis_nd": pipe.nominal_size,
                      "angle_length_mm": c["member_length_mm"], "kg_total": c["kg"], "kg_share": c["kg"]})
        c["s_mm"] = round(float(s), 1)   # 호 위치 (D82 보완 간격 재확인)
        supports.append(c)
        s_last = s
    ctx.cache[("supports", pipe.id)] = supports
    ctx.cache[("support_gaps", pipe.id)] = gaps
    return out
