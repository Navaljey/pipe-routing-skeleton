"""라우터용 서포트 모델 (D80 ⑨) — 엣지마다 추정 서포트 비용과 "서포트 가능 구간" 요약.

축 엣지: 같은 격자선 위에서는 배관 축이 같고 면 거리(배관 축과 평행하지 않은 면)가 일정하므로 형식별 무게도 일정하다.
부재는 선을 따라 평행이동만 하므로 장애물 박스마다 막히는 구간이 하나 — 축 정렬 부재는 닫힌 식, B 경사재는
(선이 경사재 평면에 수직이면) 닫힌 식, 수직 배관의 B 경사재만 볼록 함수 이분 탐색. 그래서 선 위 서포트 가능 집합을
정확히 구하고(고정 장애물·블록 범위·표 범위 — 검증기 evaluate 와 같은 규칙), 엣지는 그 선의 부분 구간으로 읽는다.
45° 엣지: 중점 한 곳만 검증기와 같은 함수(support_std.evaluate)로 본다 — `추정:` 보수적 (중점이 가능하면 그 점만 가능).

엣지 요약 (단위 mm, 정수 올림 — 보수적): a = 시작에서 첫 가능 점까지, b = 마지막 가능 점에서 끝까지,
g = 엣지 안 불가 구간 최대 길이. 가능 점이 없으면 a = NONE. 21비트씩 하나의 int64 로 묶는다 (pack).
비용 = 엣지 중점에서 가장 가벼운 가능한 형식 무게 × 길이 / 최대 간격. `추정:` 중점이 불가면 막힘을 무시한 표 범위 안
최소 무게, 그것도 없으면 0 (그 구간은 서포트를 달 수 없고 길이는 상태 u 로 제한된다).
라우터는 놓인 배관에 의한 막힘·D 묶음은 보지 않는다 (D80 ⑨ 근사, D57 과 같음).
"""
import math

import numpy as np

from . import support_std as ss

NONE = (1 << 21) - 1
MASK = NONE
_EPS = 1e-6


def pack(a, b, g) -> np.ndarray:
    a = np.minimum(np.ceil(np.asarray(a, float) - 1e-6), NONE).astype(np.int64)
    b = np.minimum(np.ceil(np.asarray(b, float) - 1e-6), NONE).astype(np.int64)
    g = np.minimum(np.ceil(np.asarray(g, float) - 1e-6), NONE).astype(np.int64)
    return a | (b << 21) | (g << 42)


def unpack(p):
    p = np.asarray(p, dtype=np.int64)
    return p & MASK, (p >> 21) & MASK, (p >> 42) & MASK


def reverse(p) -> np.ndarray:
    """반대 방향 엣지의 요약 (a ↔ b). 가능 점이 없으면 그대로."""
    a, b, g = unpack(p)
    none = a == NONE
    out = np.asarray(b, np.int64) | (np.asarray(a, np.int64) << 21) | (np.asarray(g, np.int64) << 42)
    return np.where(none, p, out)


def _merge(iv):
    if not iv:
        return []
    iv = sorted(iv)
    out = [list(iv[0])]
    for s, e in iv[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def _intersect(A, B):
    out, i, j = [], 0, 0
    while i < len(A) and j < len(B):
        s, e = max(A[i][0], B[j][0]), min(A[i][1], B[j][1])
        if s <= e:
            out.append([s, e])
        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1
    return out


def _seg_rect_2d(p0, p1, lo, hi):
    """2D 선분 (N,2) ↔ 직사각형 거리 (닫힌 식: 교차면 0, 아니면 끝점→사각형 · 꼭짓점→선분 최소)."""
    def pt_rect(p):
        g = np.maximum(0.0, np.maximum(lo - p, p - hi))
        return np.sqrt((g * g).sum(-1))

    def pt_seg(q, a, b):
        ab = b - a
        dd = np.maximum((ab * ab).sum(-1), 1e-12)
        t = np.clip(((q - a) * ab).sum(-1) / dd, 0, 1)
        return np.linalg.norm(q - (a + t[..., None] * ab), axis=-1)

    d = np.minimum(pt_rect(p0), pt_rect(p1))
    for cx in (lo[..., 0], hi[..., 0]):
        for cy in (lo[..., 1], hi[..., 1]):
            c = np.stack(np.broadcast_arrays(cx, cy), -1)
            d = np.minimum(d, pt_seg(c, p0, p1))
    # 교차 (선분이 사각형을 지남): 사각형 변과 교차 또는 끝점이 안 → 위 값이 0 이 아닐 수 있으므로 별도 판정
    inter = _seg_hits_rect(p0, p1, lo, hi)
    return np.where(inter, 0.0, d)


def _seg_hits_rect(p0, p1, lo, hi):
    """Liang–Barsky."""
    d = p1 - p0
    t0 = np.zeros(p0.shape[:-1])
    t1 = np.ones(p0.shape[:-1])
    ok = np.ones(p0.shape[:-1], dtype=bool)
    for k in range(2):
        for pk, qk in ((-d[..., k], p0[..., k] - lo[..., k]), (d[..., k], hi[..., k] - p0[..., k])):
            par = np.abs(pk) < 1e-12
            ok &= ~(par & (qk < 0))
            r = np.where(par, 0.0, qk / np.where(par, 1.0, pk))
            neg = (pk < 0) & ~par
            pos = (pk > 0) & ~par
            t0 = np.where(neg, np.maximum(t0, r), t0)
            t1 = np.where(pos, np.minimum(t1, r), t1)
    return ok & (t0 <= t1)


class LineSupport:
    """고정 레이어 하나(구경별)의 격자선 서포트 모델. 선 데이터는 처음 쓸 때 계산해 캐시한다."""

    def __init__(self, std, ext, box_lo, box_hi, nd: str, od: float, spacing):
        self.std, self.nd, self.od = std, nd, od
        self.ext = np.asarray(ext, dtype=float)
        self.box_lo = np.asarray(box_lo, dtype=float).reshape(-1, 3)
        self.box_hi = np.asarray(box_hi, dtype=float).reshape(-1, 3)
        self.hmax, self.vmax = spacing
        self.cache = {}

    # ------------------------------------------------------------ 선 데이터

    def _lines(self, ax: int, keys: np.ndarray):
        """keys (n, 2) = 선의 다른 두 좌표 (o1, o2 순). 반환 [(opts: [(kg, 막힘 구간 목록)], ...)] — 선마다.
        막힘 구간은 선 좌표 t (= ax 좌표) 의 닫힌 구간."""
        o1, o2 = [k for k in range(3) if k != ax]
        n = len(keys)
        X0 = np.zeros((n, 3))
        X0[:, o1], X0[:, o2] = keys[:, 0], keys[:, 1]
        U = np.zeros((n, 3))
        U[:, ax] = 1.0
        dist = ss.face_distances(self.ext, X0, U)
        res = [[] for _ in range(n)]
        for o, (face, frm) in enumerate(ss.OPTIONS):
            L = dist[:, ss.OPT_FACE[o]]
            fin = np.isfinite(L)
            if not fin.any():
                continue
            kg, k, rf = ss.form_weight(self.std, frm, self.nd, self.od, np.where(fin, L, -1.0))
            live = np.nonzero(np.isfinite(kg))[0]
            if not len(live):
                continue
            A, B, V = ss.member_arrays(frm, X0[live], U[live], face, L[live], self.od, rf[live])
            half = np.array([ss.ANGLE_HALF[self.std.angle_of(frm, self.nd, int(kk))] for kk in k[live]])
            blocked = [[] for _ in range(len(live))]
            # 블록 밖 (선 방향 이동과 무관한 축은 상수, 선 방향(ax) 은 구간)
            other = [q for q in range(3) if q != ax]
            outc = np.zeros(len(live), dtype=bool)
            for q in other:
                outc |= ((np.minimum(A[..., q], B[..., q]) < -1) | (np.maximum(A[..., q], B[..., q]) > self.ext[q] + 1)
                         ).any(1)
            # ax 방향: 부재 끝점 범위 [m0 + t, m1 + t] ⊂ [−1, ext + 1]
            m0 = np.where(V, np.minimum(A[..., ax], B[..., ax]), np.inf).min(1)
            m1 = np.where(V, np.maximum(A[..., ax], B[..., ax]), -np.inf).max(1)
            for i in range(len(live)):
                if outc[i]:
                    blocked[i].append([-np.inf, np.inf])
                    continue
                lo_t, hi_t = -1 - m0[i], self.ext[ax] + 1 - m1[i]
                blocked[i].append([-np.inf, lo_t - _EPS])
                blocked[i].append([hi_t + _EPS, np.inf])
            # 장애물
            for bl, bh in zip(self.box_lo, self.box_hi):
                for m in range(A.shape[1]):
                    a, b, v = A[:, m], B[:, m], V[:, m]
                    if not v.any():
                        continue
                    diag = np.count_nonzero(np.abs(b - a) > 1e-9, axis=1) > 1
                    # 축 정렬 부재 (평행이동 방향 ax): 다른 두 축 간격 g⊥, ax 축은 구간
                    g2 = np.zeros(len(live))
                    for q in other:
                        lo_q, hi_q = np.minimum(a[:, q], b[:, q]), np.maximum(a[:, q], b[:, q])
                        g = np.maximum(0.0, np.maximum(bl[q] - hi_q, lo_q - bh[q]))
                        g2 += g * g
                    mm0 = np.minimum(a[:, ax], b[:, ax])
                    mm1 = np.maximum(a[:, ax], b[:, ax])
                    straight = v & ~diag
                    w2 = half * half - g2
                    sel = np.nonzero(straight & (w2 >= 0))[0]
                    for i in sel:
                        w = math.sqrt(w2[i])
                        blocked[i].append([bl[ax] - mm1[i] - w, bh[ax] - mm0[i] + w])
                    # 경사 부재 (B 경사재 · 45° 의 C 가로대 — 축 엣지에서는 B 경사재만)
                    dsel = np.nonzero(v & diag)[0]
                    if len(dsel):
                        span_ax = np.abs(b[dsel, ax] - a[dsel, ax]) > 1e-9
                        # (i) 부재가 ax 방향 성분이 없음: 선에 수직인 평면 안 → 2D 거리 닫힌 식, ax 는 구간
                        flat = dsel[~span_ax]
                        if len(flat):
                            q1, q2 = other
                            p0 = np.stack([a[flat, q1], a[flat, q2]], -1)
                            p1 = np.stack([b[flat, q1], b[flat, q2]], -1)
                            d2 = _seg_rect_2d(p0, p1, np.array([bl[q1], bl[q2]]), np.array([bh[q1], bh[q2]]))
                            for j, i in enumerate(flat):
                                if d2[j] <= half[i]:
                                    w = math.sqrt(half[i] ** 2 - d2[j] ** 2)
                                    blocked[i].append([bl[ax] - a[i, ax] - w, bh[ax] - a[i, ax] + w])
                        # (ii) ax 방향 성분이 있음 (수직 배관의 B 경사재, 평면 = 면 법선 축 × ax): 반폭 w 만큼 사각형
                        #      (L∞) 으로 키운 박스와 교차하는 t 구간 — 닫힌 식. `추정:` 모서리에서 유클리드보다 약간
                        #      보수적 (라우터가 더 막힌 것으로 본다)
                        for i in dsel[span_ax]:
                            iv = _slanted_interval(a[i], b[i], ax, bl, bh, half[i])
                            if iv is not None:
                                blocked[i].append(iv)
            for j, i in enumerate(live):
                res[i].append((float(kg[i]), _merge(blocked[j])))
        return res

    def line(self, ax: int, c1: float, c2: float):
        key = (ax, round(c1, 3), round(c2, 3))
        hit = self.cache.get(key)
        if hit is None:
            hit = self._summarize(self._lines(ax, np.array([[c1, c2]]))[0])
            self.cache[key] = hit
        return hit

    def prefetch(self, ax: int, keys: np.ndarray):
        keys = np.unique(np.round(keys, 3), axis=0)
        new = [k for k in keys if (ax, float(k[0]), float(k[1])) not in self.cache]
        if not new:
            return
        new = np.array(new)
        for k0 in range(0, len(new), 4096):
            chunk = new[k0:k0 + 4096]
            for k, data in zip(chunk, self._lines(ax, chunk)):
                self.cache[(ax, float(k[0]), float(k[1]))] = self._summarize(data)

    @staticmethod
    def _summarize(opts):
        """선 요약: 불가 구간 (모든 가능 형식이 막힌 곳 = ∩ 막힘) 과 형식별 (kg, 막힘)."""
        if not opts:
            return ([[-np.inf, np.inf]], [])
        un = opts[0][1]
        for _, bl in opts[1:]:
            un = _intersect(un, bl)
        return (un, sorted(opts, key=lambda x: x[0]))

    # ------------------------------------------------------------ 엣지 질의

    def axis_edges(self, ax: int, P0: np.ndarray, P1: np.ndarray):
        """축 엣지들 (같은 ax 방향, P0[ax] < P1[ax]) → (비용 kg, pack)."""
        n = len(P0)
        cost = np.zeros(n)
        pk = np.zeros(n, dtype=np.int64)
        if not n:
            return cost, pk
        o1, o2 = [k for k in range(3) if k != ax]
        keys = np.round(np.stack([P0[:, o1], P0[:, o2]], -1), 3)
        self.prefetch(ax, keys)
        smax = self.vmax if ax == 2 else self.hmax
        order = np.lexsort((keys[:, 1], keys[:, 0]))
        ks = keys[order]
        brk = np.nonzero(np.any(np.diff(ks, axis=0) != 0, axis=1))[0] + 1
        for grp in np.split(order, brk):
            un, opts = self.cache[(ax, float(keys[grp[0], 0]), float(keys[grp[0], 1]))]
            x0 = np.minimum(P0[grp, ax], P1[grp, ax])
            x1 = np.maximum(P0[grp, ax], P1[grp, ax])
            L = x1 - x0
            a = np.zeros(len(grp))
            b = np.zeros(len(grp))
            g = np.zeros(len(grp))
            none = np.zeros(len(grp), dtype=bool)
            for s, e in un:
                cov0 = (s <= x0 + 1e-9) & (e >= x0 - 1e-9)
                cov1 = (s <= x1 + 1e-9) & (e >= x1 - 1e-9)
                none |= cov0 & cov1
                a = np.where(cov0, np.maximum(a, np.minimum(e, x1) - x0), a)
                b = np.where(cov1, np.maximum(b, x1 - np.maximum(s, x0)), b)
                inside = (s > x0 + 1e-9) & (e < x1 - 1e-9)
                g = np.where(inside, np.maximum(g, e - s), g)
            mid = (x0 + x1) / 2
            w = np.full(len(grp), np.nan)
            wmin = opts[0][0] if opts else 0.0
            for kg, bl in opts:      # 가벼운 순서
                free = np.ones(len(grp), dtype=bool)
                for s, e in bl:
                    free &= ~((mid >= s - 1e-9) & (mid <= e + 1e-9))
                w = np.where(np.isnan(w) & free, kg, w)
            w = np.where(np.isnan(w), wmin, w)
            cost[grp] = L / smax * w
            pa = np.where(none, NONE, a)
            pk[grp] = pack(pa, np.where(none, 0, b), np.where(none, 0, g))
            # P0 이 큰 쪽이면 (음의 방향) 뒤집는다
            rev = P0[grp, ax] > P1[grp, ax]
            if rev.any():
                pk[grp[rev]] = reverse(pk[grp[rev]])
        return cost, pk

    def diag_edges(self, P0: np.ndarray, P1: np.ndarray):
        """45° 엣지 → (비용, pack). 중점만 검증기와 같은 함수로 본다 (`추정:` 보수적)."""
        n = len(P0)
        if not n:
            return np.zeros(0), np.zeros(0, dtype=np.int64)
        D = P1 - P0
        L = np.linalg.norm(D, axis=1)
        U = D / L[:, None]
        M = (P0 + P1) / 2
        ev = ss.evaluate(self.std, self.ext, self.box_lo, self.box_hi, self.nd, self.od, M, U, others=None,
                         exact=True, prefer_ceiling=False)
        ok = ev["opt"] >= 0
        smax = self.hmax   # 45° = 수평 간격 (검증기와 같음)
        w = np.where(ok, ev["kg"], 0.0)
        if (~ok).any():   # 막힘 무시 표 범위 안 최소 무게
            free = ss.evaluate(self.std, self.ext, np.zeros((0, 3)), np.zeros((0, 3)), self.nd, self.od, M[~ok],
                               U[~ok], others=None, exact=False, prefer_ceiling=False)
            w[~ok] = np.where(free["opt"] >= 0, free["kg"], 0.0)
        half = L / 2
        return L / smax * w, np.where(ok, pack(half, half, np.zeros(n)), pack(np.full(n, NONE), 0, 0))


def _slanted_interval(a, b, ax, bl, bh, h):
    """선분 a→b (ax 방향 성분 있음) 를 ax 로 t 만큼 옮길 때 박스(+ 반폭, 수직 축 간격 반영)와 닿는 t 구간."""
    d = b - a
    q_in = [q for q in range(3) if q != ax and abs(d[q]) > 1e-9]
    q_out = [q for q in range(3) if q != ax and abs(d[q]) <= 1e-9]
    g2 = 0.0
    for q in q_out:
        g = max(0.0, bl[q] - a[q], a[q] - bh[q])
        g2 += g * g
    if g2 > h * h:
        return None
    w = math.sqrt(h * h - g2)
    s0, s1 = 0.0, 1.0
    for q in q_in:
        lo, hi = (bl[q] - w - a[q]) / d[q], (bh[q] + w - a[q]) / d[q]
        if lo > hi:
            lo, hi = hi, lo
        s0, s1 = max(s0, lo), min(s1, hi)
    if s0 > s1:
        return None
    z0, z1 = a[ax] + s0 * d[ax], a[ax] + s1 * d[ax]
    zmin, zmax = min(z0, z1), max(z0, z1)
    return [bl[ax] - w - zmax, bh[ax] + w - zmin]


def _convex_interval(a, b, ax, bl, bh, h):
    """선분 a→b 를 ax 방향으로 t 만큼 옮겼을 때 박스와 거리 ≤ h 인 t 구간 (거리는 t 에 대해 볼록)."""
    from .verifier.geom import segment_box_distance

    def f(t):
        t = np.atleast_1d(t)
        sh = np.zeros((len(t), 3))
        sh[:, ax] = t
        d, _ = segment_box_distance(a[None] + sh, b[None] + sh, bl, bh)
        return d - h
    lo_r = bl[ax] - max(a[ax], b[ax]) - h - 1
    hi_r = bh[ax] - min(a[ax], b[ax]) + h + 1
    x, y = lo_r, hi_r
    gr = (math.sqrt(5) - 1) / 2
    for _ in range(80):
        c = y - gr * (y - x)
        d = x + gr * (y - x)
        if f(c)[0] < f(d)[0]:
            y = d
        else:
            x = c
    tm = (x + y) / 2
    if f(tm)[0] > 0:
        return None
    lo, hi = lo_r, tm
    for _ in range(50):
        m = (lo + hi) / 2
        if f(m)[0] <= 0:
            hi = m
        else:
            lo = m
    s = hi
    lo, hi = tm, hi_r
    for _ in range(50):
        m = (lo + hi) / 2
        if f(m)[0] <= 0:
            lo = m
        else:
            hi = m
    return [s, lo]


def support_step(u: float, pk: int, length: float, smax: float, bend_u: float = 0.0):
    """D80 ⑨ 상태 갱신 (라우터 파이썬판, 컴파일판과 같은 식). 반환 (새 u, 가능 여부).
    가능 점이 없으면 u + L/smax ≤ 1, 있으면 u + a/smax ≤ 1 이고 g/smax ≤ 1, 새 u = b/smax.
    bend_u = 이 엣지 시작에서 꺾으면 2·t_θ / 수평 간격 — 검증기 중심선은 꺾임점을 엘보 호로 바꾸므로 꺾임점 앞뒤 t 안은
    서포트 가능 점으로 치지 않는다 (보수적, 호 조각은 수평 간격으로 정규화 — 검증기와 같음)."""
    u = u + bend_u
    a = pk & MASK
    if a == NONE:
        nu = u + length / smax
        if not (nu <= 1 + 1e-9):
            return u, False
        return round(nu, 6), True
    b = (pk >> 21) & MASK
    g = (pk >> 42) & MASK
    if not (u + a / smax <= 1 + 1e-9) or not (g / smax <= 1 + 1e-9):
        return u, False
    return round(b / smax, 6), True
