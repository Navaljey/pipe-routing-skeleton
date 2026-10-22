"""D109 진단 도구: 경계 단자 관통부 이후 서포트 후보 위치별 형식 판정 (결과 불변 — 읽기만 한다).

배관의 경계 단자마다 ① 강제 직관(D94·D96) 위 몇 점 ② 90° 꺾은 뒤 네 방향으로 남은 간격(u) 안 후보 위치에서
`support_std.evaluate` 와 같은 함수로 (면, 형식) 조합마다 L·표 범위 안 여부·부재 간섭·선택 결과를 출력한다.

주의 (D78): 출력에 선정표에서 나온 값(무게·범위 판정)이 들어간다. 비공개 표로 돌린 출력은 공개 레포의
out/ 이나 보고서에 남기지 않는다 — 터미널로만 본다.

PIPE_SUPPORT_STANDARD=<비공개 레포> python -m pipe_routing.support_scan scenarios/procedural/proc_000.json P002
"""
import argparse

import numpy as np

from . import support_std as ss
from .blockpair import boundary_straight
from .constants import SUPPORT_SPACING, elbow_tangent
from .scenario import load


def options_at(std, sc, p, X, U):
    """위치 X·축 U 에서 (면, 형식) 조합별 (면, 형식, L, 사유) 와 evaluate 선택 결과."""
    ext = np.array(sc.block.extent, float)
    blo = np.array([o.box.min for o in sc.obstacles], float).reshape(-1, 3)
    bhi = np.array([o.box.max for o in sc.obstacles], float).reshape(-1, 3)
    X = np.atleast_2d(np.asarray(X, float))
    U = np.atleast_2d(np.asarray(U, float))
    dist = ss.face_distances(ext, X, U)
    rows = []
    for o, (face, frm) in enumerate(ss.OPTIONS):
        L = float(dist[0, ss.OPT_FACE[o]])
        if not np.isfinite(L):
            continue
        kg, k, rf = ss.form_weight(std, frm, p.nominal_size, p.od, np.array([L]))
        if not np.isfinite(kg[0]):
            rows.append((face, frm, round(L), "표 범위 밖"))
            continue
        A, B, v = ss.member_arrays(frm, X, U, face, np.array([L]), p.od, rf)
        half = ss.ANGLE_HALF[std.angle_of(frm, p.nominal_size, int(k[0]))]
        bl = ss.members_blocked(A, B, v, float(half), ext, blo, bhi, None, True)[0]
        rows.append((face, frm, round(L), "부재 간섭" if bl else f"가능 {kg[0]:.1f} kg"))
    ev = ss.evaluate(std, ext, blo, bhi, p.nominal_size, p.od, X, U)
    sel = int(ev["opt"][0])
    return rows, (ss.OPTIONS[sel] if sel >= 0 else None)


def _fmt(rows):
    return "; ".join(f"{f}/{m} L={L} {r}" for f, m, L, r in rows)


def scan(sc, p, std, step=250.0):
    hmax, vmax = SUPPORT_SPACING[p.nominal_size]
    t90 = elbow_tangent(p.nominal_size, 90)
    print(f"=== {p.id} {p.nominal_size} start {p.start.kind} {p.start.pos} {p.start.dir}"
          f" / end {p.end.kind} {p.end.pos} {p.end.dir}")
    for which, sgn in (("start", 1), ("end", -1)):
        t = getattr(p, which)
        if t.kind != "boundary":
            continue
        s0 = boundary_straight(p)
        T = np.array(t.pos, float)
        dvec = sgn * np.array(t.dir, float)
        ax = int(np.argmax(np.abs(dvec)))
        s_d = vmax if ax == 2 else hmax
        print(f"  [{which}] 관통 축 {'xyz'[ax]}, 강제 직관 {s0:.0f} + t90 {t90:.0f} mm, 간격 {s_d:.0f}")
        for s in sorted({s0 / 2, s0}):
            rows, ch = options_at(std, sc, p, T + s * dvec, dvec)
            print(f"    직관 s={s:.0f}: 선택 {ch} | {_fmt(rows)}")
        Bp = T + (s0 + t90) * dvec
        u_used = s0 / s_d + 2 * t90 / hmax
        for e_ax in range(3):
            if e_ax == ax:
                continue
            for e_s in (1, -1):
                e = np.zeros(3)
                e[e_ax] = e_s
                reach = max(0.0, (1 - u_used) * (vmax if e_ax == 2 else hmax))
                found, first = None, None
                for s in np.arange(t90, t90 + reach + 1, step):
                    X = Bp + s * e
                    if np.any(X < 0) or np.any(X > np.array(sc.block.extent)):
                        break
                    rows, ch = options_at(std, sc, p, X, e)
                    first = first or f"s={s:.0f}: {_fmt(rows)}"
                    if ch and found is None:
                        found = (round(s), ch)
                print(f"    꺾은 뒤 {'+-'[e_s < 0]}{'xyz'[e_ax]} (u 남음 {1 - u_used:.2f}, {reach:.0f} mm):"
                      f" 첫 가능 {found}")
                if first:
                    print("       ", first)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scenario")
    ap.add_argument("pipes", nargs="+")
    ap.add_argument("--step", type=float, default=250.0, help="꺾은 뒤 후보 간격 mm")
    a = ap.parse_args(argv)
    std = ss.load_standard()
    print("support_standard", std.source, std.sha256[:12])
    sc = load(a.scenario)
    by = {p.id: p for p in sc.pipes}
    for pid in a.pipes:
        scan(sc, by[pid], std, a.step)


if __name__ == "__main__":
    main()
