"""D 형식 묶음 (D80 ⑥) — 배치가 끝난 C 서포트들을 병합만 한다 (위치 불변, 결정적).

`추정:` 묶음 조건 (D80 ⑥):
  같은 바닥·천장 면, 수평으로 평행한 배관(축 방향이 같은 수평 축), 서포트 위치를 배관 축에 투영해 ±500 mm 안 (D82 ③, 2026-10-18),
  지지면 쪽 표면 높이(바닥 = 아랫면 z − OD/2, 천장 = 윗면 z + OD/2) 차이 ≤ 50 mm, 가로대 폭 L1·다리 높이 L2 가
  D 표(L1 = D1 표, L2 = D2 표, 기준 구경 = 묶음 최대 구경) 범위 안, 가로대·다리가 장애물·묶이지 않은 배관과 간섭 없음,
  묶음 무게 < 개별 합일 때만. 좌표 순서 탐욕 (같은 축 위치의 서포트를 가로 방향 순서로 하나씩 붙여 본다).
D82 보완 (2026-10-19): 묶음 가로대는 축 위치 평균에 놓이므로, 묶인 각 배관에서 그 위치로 옮긴 서포트의 앞뒤 간격(§3.6,
단자 고정점 포함)을 다시 확인하고 어느 배관이든 넘으면 그 묶음은 만들지 않는다.
배관별 kg_share = 묶음 무게를 얹힌 배관들의 직관 단위중량(§3.1 kg/m) 비례로 나눈 값 (D81 J 귀속 규칙 — 물리적 유일해가
없는 회계 규약), J 서포트 항은 이 값을 쓴다.
"""
import numpy as np

from .. import support_std as ss
from ..constants import PIPE_SPECS, SUPPORT_SPACING

ATTRIBUTION_RULE = "unit_weight_proportional"   # D81
WINDOW_MM = 500.0   # D82 ③ (2026-10-18 소유자: ±250 → ±500)
SURFACE_TOL_MM = 50.0


def _cands(ctx, reports):
    out = []
    for pid, rep in reports.items():
        for i, s in enumerate(rep.supports):
            if s.get("form") != "C":
                continue
            u = np.array(s["axis"], dtype=float)
            if abs(u[2]) > 1e-9:
                continue
            ax = int(np.argmax(np.abs(u)))
            if abs(abs(u[ax]) - 1) > 1e-9:      # 45° 수평 배관은 묶지 않는다 (축 정렬 배관만, `추정:`)
                continue
            p = np.array(s["pos"], dtype=float)
            od = ctx.pipes[pid].od
            surf = p[2] - od / 2 if s["face"] == "floor" else p[2] + od / 2
            out.append({"pid": pid, "i": i, "face": s["face"], "ax": ax, "p": p, "od": od, "surf": surf,
                        "nd": ctx.pipes[pid].nominal_size, "kg": s["kg"]})
    return out


def _group_geom(ctx, grp, face):
    """묶음 부재 (가로대 + 다리 2) 와 L1·L2·기준 구경."""
    ax = grp[0]["ax"]
    sx = 1 - ax if ax < 2 else 0                       # 수평 가로 방향 축 (x↔y)
    a0 = float(np.mean([g["p"][ax] for g in grp]))
    lo = min(g["p"][sx] - g["od"] / 2 for g in grp) - ss.C_SIDE_MM
    hi = max(g["p"][sx] + g["od"] / 2 for g in grp) + ss.C_SIDE_MM
    H = ctx.scenario.block.extent[2]
    if face == "floor":
        zc = min(g["surf"] for g in grp)
        L2 = zc
        zf = 0.0
    else:
        zc = max(g["surf"] for g in grp)
        L2 = H - zc
        zf = H

    def pt(s_, z):
        q = [0.0, 0.0, z]
        q[ax], q[sx] = a0, s_
        return np.array(q)
    mem = [(pt(lo, zc), pt(hi, zc), "crossbar"), (pt(lo, zc), pt(lo, zf), "leg"), (pt(hi, zc), pt(hi, zf), "leg")]
    basis = max((g["nd"] for g in grp), key=ss.SIZES.index)
    return mem, hi - lo, L2, basis


def _blocked(ctx, mem, half, carried):
    ext = np.array(ctx.scenario.block.extent, dtype=float)
    A = np.array([[m[0] for m in mem]])
    B = np.array([[m[1] for m in mem]])
    v = np.ones((1, len(mem)), dtype=bool)
    oA, oB, oR, oS = [], [], [], []
    for pid, cls in ctx.centerlines.items():
        if pid in carried:
            continue
        for cl in cls:
            a, b, slack, _ = cl.arrays()
            oA.append(a); oB.append(b); oS.append(slack); oR.append(np.full(len(a), ctx.pipes[pid].radius))
    others = (np.concatenate(oA), np.concatenate(oB), np.concatenate(oR), np.concatenate(oS)) if oA else None
    from .. import blockpair as bp
    holes = bp.hole_boxes(ctx.scenario)   # D76: 관통 창 = 구멍
    box_lo = [o.box.min for o in ctx.scenario.obstacles] + [h[0] for h in holes]
    box_hi = [o.box.max for o in ctx.scenario.obstacles] + [h[1] for h in holes]
    sp = bp.splits(ctx.scenario)
    bounds = bp.local_bounds(ext, np.array([m[0] for m in mem[:1]]), sp) if sp else None
    return bool(ss.members_blocked(A, B, v, half, ext, box_lo, box_hi, others, exact=True, bounds=bounds)[0])


def _try(ctx, grp, face):
    mem, L1, L2, basis = _group_geom(ctx, grp, face)
    r = ss.d_weight(ctx.std, basis, L1, L2)
    if r is None:
        return None
    kg, ang = r
    if _blocked(ctx, mem, ss.ANGLE_HALF[ang], {g["pid"] for g in grp}):
        return None
    return kg, ang, mem, L1, L2, basis


def _spacing_table(ctx, pid):
    """배관 주 중심선의 호 길이·정규화 길이 누적 (modules.support 와 같은 식)."""
    key = ("spacing_tbl", pid)
    if key not in ctx.cache:
        P = ctx.centerlines[pid][0].points
        seg = P[1:] - P[:-1]
        lens = np.linalg.norm(seg, axis=1)
        hmax, vmax = SUPPORT_SPACING[ctx.pipes[pid].nominal_size]
        smax = np.where(np.abs(seg[:, 2]) >= lens * (1 - 1e-9), vmax, hmax)
        cum = np.concatenate([[0], np.cumsum(lens)])
        ucum = np.concatenate([[0], np.cumsum(lens / smax)])
        ctx.cache[key] = (cum, ucum, smax, lens)
    return ctx.cache[key]


def _u_at(tbl, s):
    cum, ucum, smax, lens = tbl
    i = int(min(max(np.searchsorted(cum, s, side="right") - 1, 0), len(lens) - 1))
    return ucum[i] + (s - cum[i]) / smax[i]


def spacing_ok(ctx, pid, s_list, k, s_new) -> bool:
    """D82 보완 (2026-10-19): 서포트 k 를 호 위치 s_new 로 옮겨도 앞뒤(이웃 서포트 또는 단자 고정점) 간격이
    §3.6 최대 간격 안인가. s_list = 그 배관 자기 서포트들의 현재 호 위치 (이미 묶여 옮겨진 것은 옮긴 값)."""
    tbl = _spacing_table(ctx, pid)
    total = float(tbl[0][-1])
    if not (0.0 < s_new < total):
        return False
    from .modules import support_fixed_points
    fx = support_fixed_points(ctx, pid)   # D94 관통부 고정점
    prev = max([x for j, x in enumerate(s_list) if j != k and x < s_list[k]] + [f for f in fx if f < s_list[k]],
               default=0.0)
    nxt = min([x for j, x in enumerate(s_list) if j != k and x > s_list[k]] + [f for f in fx if f > s_list[k]],
              default=total)
    if not (prev < s_new < nxt):
        return False
    return (_u_at(tbl, s_new) - _u_at(tbl, prev) <= 1 + 1e-9) and (_u_at(tbl, nxt) - _u_at(tbl, s_new) <= 1 + 1e-9)


def _own_s(reports, pid):
    """배관 자기 서포트(자기 배치에서 놓은 것, s_mm 있음 — 남의 D89 사본 제외)의 (인덱스, 호 위치) 목록."""
    return [(i, s["s_mm"]) for i, s in enumerate(reports[pid].supports) if "s_mm" in s]


# ---------------------------------------------------------------- D89 막히면 함께 받친다

def together_pieces(ctx, pid):
    """P 와 같은 축으로 나란할 수 있는 다른 배관의 수평 축 정렬 직관 조각 [(Q, ax, A, B)] — 배관별 캐시."""
    key = ("together_pieces", pid)
    if key not in ctx.cache:
        out = []
        for q, cls in ctx.centerlines.items():
            if q == pid:
                continue
            for cl in cls[:1]:   # 주관만 (분기 다리는 S0 범위 밖, D27)
                A, B, _, _ = cl.arrays(("straight",))
                for a, b in zip(A, B):
                    d = b - a
                    L = float(np.linalg.norm(d))
                    if L < 1e-6 or abs(d[2]) > 1e-6 * L:
                        continue
                    ax = int(np.argmax(np.abs(d)))
                    if abs(abs(d[ax]) - L) > 1e-6 * L:
                        continue
                    out.append((q, ax, a, b))
        ctx.cache[key] = out
    return ctx.cache[key]


def l1_max(std) -> float:
    return max(float(r.bounds[-1]) for (f, _), r in std.rows.items() if f == "D1")


def together_cands(ctx, pid, p, u):
    """D89 후보 거르기: P 의 수평 축 정렬 직관(점 p, 방향 u) 과 같은 축이고 가로 거리 ≤ L1 표 끝이며 바닥·천장 어느 쪽이든
    지지면 쪽 표면 높이 차 ≤ 50 mm 인 다른 배관 조각. 같은 직관 위 점들은 가로·높이가 같으므로 직관마다 한 번 거른다."""
    if abs(u[2]) > 1e-9:
        return []
    ax = int(np.argmax(np.abs(u)))
    if abs(abs(u[ax]) - 1) > 1e-9:
        return []
    sx = 1 - ax
    odP = ctx.pipes[pid].od
    lim = l1_max(ctx.std)
    out = []
    for q, qax, a, b in together_pieces(ctx, pid):
        if qax != ax or abs(a[sx] - p[sx]) > lim:
            continue
        odQ = ctx.pipes[q].od
        if min(abs((p[2] - odP / 2) - (a[2] - odQ / 2)), abs((p[2] + odP / 2) - (a[2] + odQ / 2))) > SURFACE_TOL_MM:
            continue
        out.append((q, qax, a, b))
    return out


def together_at(ctx, pid, p, u, s_mm, pieces=None):
    """D89: 위치 p (P 의 수평 축 정렬 직관 위, A·B·C 모두 불가) 에서 나란한 배관 Q 하나를 함께 얹는 D 중 가장 가벼운 것.
    조건 = D82 전부 (같은 면, 표면 차 ≤ 50 mm, L1·L2 표 범위, 가로대·다리 간섭 없음 — P·Q 제외).
    가로대 축 위치 = P 의 위치 (P 의 탐욕 위치 그대로라 P 간격은 그대로 성립, Q 는 자기 서포트와 무관하게 얹힌다).
    면: 천장 무게 ≤ 바닥 × 1.05 이면 천장 (D79 ④). 동률은 (무게, Q id). 없으면 None."""
    if abs(u[2]) > 1e-9:
        return None
    ax = int(np.argmax(np.abs(u)))
    if abs(abs(u[ax]) - 1) > 1e-9:
        return None
    sx = 1 - ax
    pP = ctx.pipes[pid]
    lim = l1_max(ctx.std)
    best = None
    for q, qax, a, b in (together_pieces(ctx, pid) if pieces is None else pieces):
        if qax != ax:
            continue
        lo, hi = sorted((a[ax], b[ax]))
        if not (lo - 1e-6 <= p[ax] <= hi + 1e-6) or abs(a[sx] - p[sx]) > lim:
            continue
        qq = a + (p[ax] - a[ax]) / (b[ax] - a[ax]) * (b - a)
        pQ = ctx.pipes[q]
        opts = []
        for face in ("ceiling", "floor"):
            sp = p[2] - pP.od / 2 if face == "floor" else p[2] + pP.od / 2
            sq = qq[2] - pQ.od / 2 if face == "floor" else qq[2] + pQ.od / 2
            if abs(sp - sq) > SURFACE_TOL_MM:
                continue
            grp = [{"pid": pid, "ax": ax, "p": np.asarray(p, float), "od": pP.od, "surf": sp, "nd": pP.nominal_size},
                   {"pid": q, "ax": ax, "p": qq, "od": pQ.od, "surf": sq, "nd": pQ.nominal_size}]
            r = _try(ctx, grp, face)
            if r is not None:
                opts.append((face, r))
        if not opts:
            continue
        kg_c = next((r[0] for f, r in opts if f == "ceiling"), None)
        kg_f = next((r[0] for f, r in opts if f == "floor"), None)
        face = "ceiling" if kg_c is not None and (kg_f is None or kg_c <= kg_f * ss.CEILING_PREFER) else "floor"
        r = dict(opts)[face]
        if best is None or (r[0], q) < (best[1][0], best[0]):
            best = (q, r, face)
    if best is None:
        return None
    q, (kg, ang, mem, L1, L2, basis), face = best
    w = {pid: PIPE_SPECS[pP.nominal_size].kg_per_m, q: PIPE_SPECS[ctx.pipes[q].nominal_size].kg_per_m}
    shares = {k: round(kg * v / sum(w.values()), 4) for k, v in w.items()}
    # 같은 D 를 P·Q 양쪽이 찾으면 (서로 막는 쌍) 같은 id — attach_together 가 실물 하나로 합친다
    gid = f"T-{'+'.join(sorted([pid, q]))}-{face}-{ax}-{round(float(p[ax]), 1)}"
    return {"form": "D", "variant": "standard", "face": face, "section": ang, "origin": "D89", "group": gid,
            "carried": sorted([pid, q]), "basis_nd": basis, "dims": {"L1": round(L1, 1), "L2": round(L2, 1)},
            "members": [{"a": [round(float(v), 3) for v in a_], "b": [round(float(v), 3) for v in b_], "role": role}
                        for a_, b_, role in mem],
            "member_length_mm": round(2 * L2 + L1, 1), "angle_length_mm": round(2 * L2 + L1, 1),
            "kg_total": round(kg, 4), "kg": shares[pid], "kg_share": shares[pid], "shares": shares}


def attach_together(reports) -> int:
    """D89 D 를 함께 얹힌 배관 Q 의 서포트 목록에도 붙인다 (Q 몫 = D81 단위중량 비례, 부재는 얹은 배관 쪽에만 기록).
    Q 의 서포트 배치·support 판정은 바꾸지 않는다 (Q 는 붙은 사본에 기대지 않는다). 같은 D 를 양쪽이 각자 찾았으면
    (서로 막는 쌍, 같은 id) 시나리오 순서상 앞 배관 것이 실물, 뒤 배관 것은 사본(attached, 부재 없음)으로 바꾼다 —
    뒤 배관의 그 위치 서포트(s_mm)는 그대로 자기 서포트로 친다. 반환 = 붙인 사본 수."""
    found = {}
    for pid in reports:
        for s in reports[pid].supports:
            if s.get("origin") == "D89" and not s.get("attached"):
                found.setdefault(s["group"], []).append((pid, s))
    n = 0
    for gid, lst in found.items():
        pid, s = lst[0]
        holders = {x for x, _ in lst}
        for _, twin in lst[1:]:
            twin.update({"members": [], "attached": True})
        for q in s["carried"]:
            if q in holders or q not in reports or not reports[q].routed:
                continue
            c = {k: v for k, v in s.items() if k not in ("members", "s_mm")}
            c.update({"members": [], "attached": True, "kg": s["shares"][q], "kg_share": s["shares"][q]})
            reports[q].supports.append(c)
            n += 1
    return n


def _shifted_s(reports, g, a0):
    sup = reports[g["pid"]].supports[g["i"]]
    u = sup["axis"]
    return sup["s_mm"] + (a0 - g["p"][g["ax"]]) * (1.0 if u[g["ax"]] > 0 else -1.0)


def _group_spacing_ok(ctx, reports, grp, shifted) -> bool:
    """D82 보완: 가로대 축 위치(평균)로 옮긴 각 배관의 서포트가 앞뒤 간격(§3.6, 단자 고정점 포함)을 지키는가."""
    a0 = float(np.mean([g["p"][g["ax"]] for g in grp]))
    for g in grp:
        own = _own_s(reports, g["pid"])
        idx = [i for i, _ in own]
        if g["i"] not in idx:
            return False
        s_list = [shifted.get((g["pid"], i), s) for i, s in own]
        if not spacing_ok(ctx, g["pid"], s_list, idx.index(g["i"]), _shifted_s(reports, g, a0)):
            return False
    return True


def group_d(ctx, reports) -> int:
    """C 서포트를 D 로 병합. 반환 = 만든 묶음 수. reports[pid].supports 를 제자리에서 고친다."""
    cands = _cands(ctx, reports)
    if len(cands) < 2:
        return 0
    cands.sort(key=lambda c: (c["face"], c["ax"], round(c["p"][c["ax"]], 3), round(c["p"][1 - c["ax"] if c["ax"] < 2
                                                                                                else 0], 3), c["pid"]))
    used = set()
    shifted = {}   # (pid, i) → 묶여 옮겨진 호 위치 (D82 보완: 간격 재확인에 쓴다)
    n_groups = 0
    for c0 in cands:
        if (c0["pid"], c0["i"]) in used:
            continue
        sx = 1 - c0["ax"] if c0["ax"] < 2 else 0
        near = [c for c in cands if (c["pid"], c["i"]) not in used and c["pid"] != c0["pid"]
                and c["face"] == c0["face"] and c["ax"] == c0["ax"]
                and abs(c["p"][c0["ax"]] - c0["p"][c0["ax"]]) <= WINDOW_MM
                and abs(c["surf"] - c0["surf"]) <= SURFACE_TOL_MM]
        near.sort(key=lambda c: (abs(c["p"][sx] - c0["p"][sx]), c["pid"]))
        grp = [c0]
        best = None
        for c in near:
            if any(g["pid"] == c["pid"] for g in grp):
                continue
            trial = grp + [c]
            if max(g["surf"] for g in trial) - min(g["surf"] for g in trial) > SURFACE_TOL_MM:
                continue
            r = _try(ctx, trial, c0["face"])
            if r is None:
                continue
            if r[0] < sum(g["kg"] for g in trial) - 1e-9 and _group_spacing_ok(ctx, reports, trial, shifted):
                grp, best = trial, r
        if best is None:
            continue
        kg, ang, mem, L1, L2, basis = best
        n_groups += 1
        gid = f"D{n_groups:03d}"
        w = {g["pid"]: PIPE_SPECS[g["nd"]].kg_per_m for g in grp}
        wsum = sum(w.values())
        carried = sorted(g["pid"] for g in grp)
        a0 = float(np.mean([g["p"][g["ax"]] for g in grp]))
        for j, g in enumerate(grp):
            share = kg * w[g["pid"]] / wsum
            used.add((g["pid"], g["i"]))
            shifted[(g["pid"], g["i"])] = _shifted_s(reports, g, a0)
            s = reports[g["pid"]].supports[g["i"]]
            s.update({"form": "D", "variant": "standard", "section": ang, "group": gid, "carried": carried,
                      "basis_nd": basis, "dims": {"L1": round(L1, 1), "L2": round(L2, 1)},
                      "members": [{"a": [round(float(v), 3) for v in a], "b": [round(float(v), 3) for v in b],
                                   "role": role} for a, b, role in mem] if j == 0 else [],
                      "member_length_mm": round(2 * L2 + L1, 1), "angle_length_mm": round(2 * L2 + L1, 1),
                      "kg_total": round(kg, 4), "kg": round(share, 4), "kg_share": round(share, 4)})
    return n_groups
