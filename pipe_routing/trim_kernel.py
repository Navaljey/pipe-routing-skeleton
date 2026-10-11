"""D65 엣지 ↔ 박스 막힌 구간 계산 (고정 레이어 꺾임 등급 마스크용).

엣지 위 점 p(s) = Q0 + s·U (0 ≤ s ≤ L) 와 박스 사이 거리 − r 은 s 에 대해 볼록 → 막힌 구간은 하나 [b0, b1].
최소점은 황금분할(80회), 구간 끝은 이분 탐색(50회). 점–박스 거리는 정확 식. numba 가 있으면 컴파일, 없으면 같은 식을
파이썬으로 돈다 (스칼라 연산 순서 같음).
"""
import math

import numpy as np

try:
    from numba import njit
except ImportError:   # pragma: no cover
    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not (a and callable(a[0])) else a[0]


@njit(cache=True)
def _f(q0, q1, q2, u0, u1, u2, s, lo0, lo1, lo2, hi0, hi1, hi2, r):
    x0, x1, x2 = q0 + s * u0, q1 + s * u1, q2 + s * u2
    g0 = max(0.0, max(lo0 - x0, x0 - hi0))
    g1 = max(0.0, max(lo1 - x1, x1 - hi1))
    g2 = max(0.0, max(lo2 - x2, x2 - hi2))
    return math.sqrt(g0 * g0 + g1 * g1 + g2 * g2) - r


@njit(cache=True)
def blocked_intervals(Q0, U, L, blo, bhi, r, eps):
    """쌍마다 (막힘 여부, b0, b1). Q0·U·blo·bhi: (n, 3), L: (n,)."""
    n = Q0.shape[0]
    hit = np.zeros(n, dtype=np.bool_)
    b0 = np.zeros(n)
    b1 = np.zeros(n)
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    for k in range(n):
        q0, q1, q2 = Q0[k, 0], Q0[k, 1], Q0[k, 2]
        u0, u1, u2 = U[k, 0], U[k, 1], U[k, 2]
        lo0, lo1, lo2 = blo[k, 0], blo[k, 1], blo[k, 2]
        hi0, hi1, hi2 = bhi[k, 0], bhi[k, 1], bhi[k, 2]
        Lk = L[k]
        a, b = 0.0, Lk
        for _ in range(80):
            c1 = b - phi * (b - a)
            c2 = a + phi * (b - a)
            if _f(q0, q1, q2, u0, u1, u2, c1, lo0, lo1, lo2, hi0, hi1, hi2, r) <= \
                    _f(q0, q1, q2, u0, u1, u2, c2, lo0, lo1, lo2, hi0, hi1, hi2, r):
                b = c2
            else:
                a = c1
        s_star = (a + b) / 2
        if not (_f(q0, q1, q2, u0, u1, u2, s_star, lo0, lo1, lo2, hi0, hi1, hi2, r) < -eps):
            continue
        hit[k] = True
        if _f(q0, q1, q2, u0, u1, u2, 0.0, lo0, lo1, lo2, hi0, hi1, hi2, r) < -eps:
            b0[k] = 0.0
        else:
            lo_s, hi_s = 0.0, s_star
            for _ in range(50):
                mid = (lo_s + hi_s) / 2
                if _f(q0, q1, q2, u0, u1, u2, mid, lo0, lo1, lo2, hi0, hi1, hi2, r) < -eps:
                    hi_s = mid
                else:
                    lo_s = mid
            b0[k] = lo_s
        if _f(q0, q1, q2, u0, u1, u2, Lk, lo0, lo1, lo2, hi0, hi1, hi2, r) < -eps:
            b1[k] = Lk
        else:
            lo_s, hi_s = s_star, Lk
            for _ in range(50):
                mid = (lo_s + hi_s) / 2
                if _f(q0, q1, q2, u0, u1, u2, mid, lo0, lo1, lo2, hi0, hi1, hi2, r) < -eps:
                    lo_s = mid
                else:
                    hi_s = mid
            b1[k] = hi_s
    return hit, b0, b1
