"""D 형식 묶음 (D80 ⑥) — 배치가 끝난 C 서포트들을 병합만 한다 (위치 불변, 결정적).

`추정:` 묶음 조건 (D80 ⑥):
  같은 바닥·천장 면, 수평으로 평행한 배관(축 방향이 같은 수평 축), 서포트 위치를 배관 축에 투영해 ±250 mm 안,
  지지면 쪽 표면 높이(바닥 = 아랫면 z − OD/2, 천장 = 윗면 z + OD/2) 차이 ≤ 50 mm, 가로대 폭 L1·다리 높이 L2 가
  D 표(L1 = D1 표, L2 = D2 표, 기준 구경 = 묶음 최대 구경) 범위 안, 가로대·다리가 장애물·묶이지 않은 배관과 간섭 없음,
  묶음 무게 < 개별 합일 때만. 좌표 순서 탐욕 (같은 축 위치의 서포트를 가로 방향 순서로 하나씩 붙여 본다).
배관별 kg_share = 묶음 무게를 얹힌 배관들의 직관 단위중량(§3.1 kg/m) 비례로 나눈 값 (D81 J 귀속 규칙 — 물리적 유일해가
없는 회계 규약), J 서포트 항은 이 값을 쓴다.
"""
import numpy as np

from .. import support_std as ss
from ..constants import PIPE_SPECS

ATTRIBUTION_RULE = "unit_weight_proportional"   # D81
WINDOW_MM = 250.0
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
    box_lo = [o.box.min for o in ctx.scenario.obstacles]
    box_hi = [o.box.max for o in ctx.scenario.obstacles]
    return bool(ss.members_blocked(A, B, v, half, ext, box_lo, box_hi, others, exact=True)[0])


def _try(ctx, grp, face):
    mem, L1, L2, basis = _group_geom(ctx, grp, face)
    r = ss.d_weight(ctx.std, basis, L1, L2)
    if r is None:
        return None
    kg, ang = r
    if _blocked(ctx, mem, ss.ANGLE_HALF[ang], {g["pid"] for g in grp}):
        return None
    return kg, ang, mem, L1, L2, basis


def group_d(ctx, reports) -> int:
    """C 서포트를 D 로 병합. 반환 = 만든 묶음 수. reports[pid].supports 를 제자리에서 고친다."""
    cands = _cands(ctx, reports)
    if len(cands) < 2:
        return 0
    cands.sort(key=lambda c: (c["face"], c["ax"], round(c["p"][c["ax"]], 3), round(c["p"][1 - c["ax"] if c["ax"] < 2
                                                                                                else 0], 3), c["pid"]))
    used = set()
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
            if r[0] < sum(g["kg"] for g in trial) - 1e-9:
                grp, best = trial, r
        if best is None:
            continue
        kg, ang, mem, L1, L2, basis = best
        n_groups += 1
        gid = f"D{n_groups:03d}"
        w = {g["pid"]: PIPE_SPECS[g["nd"]].kg_per_m for g in grp}
        wsum = sum(w.values())
        carried = sorted(g["pid"] for g in grp)
        for j, g in enumerate(grp):
            share = kg * w[g["pid"]] / wsum
            used.add((g["pid"], g["i"]))
            s = reports[g["pid"]].supports[g["i"]]
            s.update({"form": "D", "variant": "standard", "section": ang, "group": gid, "carried": carried,
                      "basis_nd": basis, "dims": {"L1": round(L1, 1), "L2": round(L2, 1)},
                      "members": [{"a": [round(float(v), 3) for v in a], "b": [round(float(v), 3) for v in b],
                                   "role": role} for a, b, role in mem] if j == 0 else [],
                      "member_length_mm": round(2 * L2 + L1, 1), "angle_length_mm": round(2 * L2 + L1, 1),
                      "kg_total": round(kg, 4), "kg": round(share, 4), "kg_share": round(share, 4)})
    return n_groups
