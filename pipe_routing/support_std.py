"""서포트 표준 (D78·D79·D80) — 선정표 로더, 형식별 무게 규칙, 부재 기하.

선정표 원본(사내 표준 기반, 비공개)은 공개 레포에 두지 않는다 (D78). 환경 변수 PIPE_SUPPORT_STANDARD 가
비공개 레포 경로를 가리키면 그 `tables/` 를 읽고, 없으면 공개 임시 표(PUBLIC_APPROX, `추정:` 근사)로 동작한다.

형식 (D79·D80 ②), L = 배관 중심 → 지지면 거리, w = 선정표 앵글 kg/m:
  A 캔틸레버 (벽)       w × (L + 100)
  B 브래킷+경사재 (벽)  w × [(L + 100) + √(L² + (L/2)²) + 보강형이면 L/4]
  C 문형 1배관 (바닥·천장) w × [2·L + (OD + 200) + 보강형이면 (OD + 200)]
  D 문형 다배관 (바닥·천장) w × [2·L2 + L1]   L1 = 가로대 폭, L2 = 다리 높이, 앵글 = 두 표 결과 중 큰 규격
"""
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

SIZES = ("15A", "25A", "50A", "65A", "100A", "150A", "200A", "300A", "400A", "500A")
FORMS = ("A", "B", "C", "D1", "D2")
WALL_FORMS = ("A", "B")
DECK_FORMS = ("C",)
MARGIN_MM = 100.0               # A·B 수평재 여유 (D23 그대로)
C_SIDE_MM = 100.0               # C 가로대 = OD + 2 × 100 (D79 ⑥)
CEILING_PREFER = 1.05           # D79 ④ `추정:` 5%
# 앵글 크기 순서 (D 형식: 두 표 결과 중 큰 규격)
ANGLE_ORDER = ("38x6FB", "40x40x5", "50x50x6", "65x65x6", "75x75x6", "100x100x10", "100x100x13", "130x130x12")
# 부재 단면 반폭 (mm) — 간섭 판정용 막대 반경 (D80 ⑧). 앵글 다리 폭 / 2, FB 는 폭 / 2
ANGLE_HALF = {"38x6FB": 19.0, "40x40x5": 20.0, "50x50x6": 25.0, "65x65x6": 32.5, "75x75x6": 37.5,
              "100x100x10": 50.0, "100x100x13": 50.0, "130x130x12": 65.0}

# 공개 임시 표 (D78 ⑪): `추정:` 근사 — 사내 표와 무관한 단순값. 모든 형식·구경 L50 한 종류, 범위 3,000 mm, 보강 없음
PUBLIC_APPROX_LMAX = 3000.0
PUBLIC_APPROX_ANGLE = ("50x50x6", 4.43)


@dataclass
class Row:
    bounds: np.ndarray          # 구간 경계 [0, b1, ..., L_max]
    angles: tuple               # 구간별 앵글
    reinforced_from: float      # 이 L 이상이면 보강형 (없으면 inf)

    @property
    def lmax(self) -> float:
        return float(self.bounds[-1])


@dataclass
class Standard:
    source: str                 # "private" | "public_approx"
    sha256: str
    rows: dict                  # (form, nd) → Row
    kg_m: dict                  # 앵글 → kg/m
    path: str = ""
    _cache: dict = field(default_factory=dict, repr=False)

    def info(self) -> dict:
        return {"source": self.source, "sha256": self.sha256}

    def pick(self, form: str, nd: str, L: np.ndarray):
        """L 배열 → (앵글 번호 (불가 -1), 보강형 bool). 앵글 번호는 rows[...].angles 인덱스."""
        row = self.rows[(form, nd)]
        L = np.asarray(L, dtype=float)
        k = np.searchsorted(row.bounds, L, side="right") - 1
        ok = (L >= 0) & (L < row.lmax)
        k = np.where(ok, np.clip(k, 0, len(row.angles) - 1), -1)
        return k, ok & (L >= row.reinforced_from)

    def w_of(self, form: str, nd: str):
        """구간 번호 → kg/m 배열 (마지막 칸 = 불가 → nan)."""
        key = ("w", form, nd)
        if key not in self._cache:
            row = self.rows[(form, nd)]
            self._cache[key] = np.array([self.kg_m[a] for a in row.angles] + [np.nan])
        return self._cache[key]

    def angle_of(self, form: str, nd: str, k: int) -> str:
        return self.rows[(form, nd)].angles[k]


def _from_dir(root: Path) -> Standard:
    files = sorted((root / "tables").glob("*.json"))
    h = hashlib.sha256()
    for f in files:
        h.update(f.read_bytes())
    aw = json.loads((root / "tables" / "angle_weights.json").read_text())
    kg_m = {k: float(v["kg_m"]) for k, v in aw["angles"].items()}
    rows = {}
    for form in FORMS:
        d = json.loads((root / "tables" / f"type_{form}.json").read_text())
        for nd, r in d["rows"].items():
            segs = r["segments"]
            b = [float(segs[0]["L_from"])] + [float(s["L_to"]) for s in segs]
            rf = r.get("reinforced_from") or []
            rows[(form, nd)] = Row(np.array(b), tuple(s["angle"] for s in segs),
                                   float(rf[0]) if rf else math.inf)
    return Standard("private", h.hexdigest(), rows, kg_m, str(root))


def public_approx() -> Standard:
    ang, kg = PUBLIC_APPROX_ANGLE
    rows = {(f, nd): Row(np.array([0.0, PUBLIC_APPROX_LMAX]), (ang,), math.inf) for f in FORMS for nd in SIZES}
    h = hashlib.sha256(json.dumps({"lmax": PUBLIC_APPROX_LMAX, "angle": ang, "kg_m": kg}).encode()).hexdigest()
    return Standard("public_approx", h, rows, {ang: kg})


@lru_cache(maxsize=None)
def _load(path: str) -> Standard:
    if path:
        root = Path(path)
        if (root / "tables" / "type_A.json").exists():
            return _from_dir(root)
    return public_approx()


def load_standard(path: str | None = None) -> Standard:
    """비공개 표 경로(인자 또는 PIPE_SUPPORT_STANDARD) → Standard. 없으면 공개 임시 표 (D78)."""
    return _load(path if path is not None else os.environ.get("PIPE_SUPPORT_STANDARD", ""))


# --------------------------------------------------------------------------- 형식별 무게 (D80 ②)

def member_length(form: str, L: np.ndarray, od: float, reinf: np.ndarray) -> np.ndarray:
    """A·B·C 부재 길이 합 (mm). L 은 배관 중심 → 지지면 거리."""
    L = np.asarray(L, dtype=float)
    if form == "A":
        return L + MARGIN_MM
    if form == "B":
        return (L + MARGIN_MM) + np.sqrt(L * L + (L / 2) ** 2) + np.where(reinf, L / 4, 0.0)
    if form == "C":
        bar = od + 2 * C_SIDE_MM
        return 2 * L + bar + np.where(reinf, bar, 0.0)
    raise ValueError(form)


def form_weight(std: Standard, form: str, nd: str, od: float, L: np.ndarray):
    """(kg (불가 = inf), 앵글 구간 번호, 보강형). 벡터."""
    k, reinf = std.pick(form, nd, L)
    w = std.w_of(form, nd)[k]           # k = -1 → nan
    kg = w * member_length(form, L, od, reinf) / 1000
    return np.where(k >= 0, kg, np.inf), k, reinf


def d_weight(std: Standard, nd: str, L1: float, L2: float):
    """D 형식 (kg, 앵글, 보강 없음) 또는 None (표 밖). 앵글 = L1 표·L2 표 결과 중 큰 규격 (D80 ②)."""
    k1, _ = std.pick("D1", nd, np.array([L1]))
    k2, _ = std.pick("D2", nd, np.array([L2]))
    if k1[0] < 0 or k2[0] < 0:
        return None
    a1, a2 = std.angle_of("D1", nd, int(k1[0])), std.angle_of("D2", nd, int(k2[0]))
    ang = max(a1, a2, key=ANGLE_ORDER.index)
    return std.kg_m[ang] * (2 * L2 + L1) / 1000, ang


# --------------------------------------------------------------------------- 부재 기하 (D80 ⑦⑧)

FACE_AXIS = {"floor": (2, 0), "ceiling": (2, 1), "wall_x0": (0, 0), "wall_x1": (0, 1), "wall_y0": (1, 0),
             "wall_y1": (1, 1)}
FACE_ORDER = ("floor", "ceiling", "wall_x0", "wall_x1", "wall_y0", "wall_y1")


def face_normal(face: str) -> np.ndarray:
    """배관 → 지지면 방향 단위벡터."""
    ax, side = FACE_AXIS[face]
    n = np.zeros(3)
    n[ax] = 1.0 if side else -1.0
    return n


def side_dir(U: np.ndarray, n: np.ndarray) -> np.ndarray:
    """문형 가로대 방향 = 배관 축과 면 법선에 수직."""
    s = np.cross(n, U)
    nn = np.linalg.norm(s)
    if nn < 1e-9:
        return np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.5 else np.array([0.0, 1.0, 0.0])
    return s / nn


def members(form: str, P, U, face: str, L: float, od: float, reinf: bool) -> list:
    """A·B·C 부재 선분 [(시작, 끝, 역할)]. P = 배관 중심, U = 배관 축, L = 중심 → 면 거리.

    `추정:` 기하 단순화 — A·B 수평재 = 배관 중심에서 면까지, B 경사재 = 벽면에서 L/2 아래 점 → 배관 중심,
    B 보강 수직재 = 벽에서 L/2 지점에서 아래로 L/4, C 가로대 = 배관 면 쪽 표면(중심에서 OD/2), 폭 OD + 200,
    다리 = 가로대 양 끝 → 면, C 보강 = 다리 중간 높이 가로 연결재.
    """
    P = np.asarray(P, dtype=float)
    n = face_normal(face)
    Q = P + L * n                                   # 면 위 점
    if form == "A":
        return [(P, Q, "arm")]
    if form == "B":
        down = np.array([0.0, 0.0, -1.0])
        W = Q + (L / 2) * down
        out = [(P, Q, "arm"), (W, P, "strut")]
        if reinf:
            M = P + (L / 2) * n
            out.append((M, M + (L / 4) * down, "stiffener"))
        return out
    if form == "C":
        s = side_dir(np.asarray(U, dtype=float), n)
        b = (od + 2 * C_SIDE_MM) / 2
        C0 = P + (od / 2) * n
        e0, e1 = C0 - b * s, C0 + b * s
        h = max(L - od / 2, 0.0)
        out = [(e0, e1, "crossbar"), (e0, e0 + h * n, "leg"), (e1, e1 + h * n, "leg")]
        if reinf:
            out.append((e0 + (h / 2) * n, e1 + (h / 2) * n, "tie"))
        return out
    raise ValueError(form)


# --------------------------------------------------------------------------- 선택 (D80 ③④⑧) — 검증기·라우터 공용

OPTIONS = tuple((f, frm) for f in FACE_ORDER for frm in (DECK_FORMS if f in ("floor", "ceiling") else WALL_FORMS))
OPT_FACE = np.array([FACE_ORDER.index(f) for f, _ in OPTIONS])
_EPS = 1e-6


def face_distances(ext, X: np.ndarray, U: np.ndarray) -> np.ndarray:
    """(N, 6) 배관 중심 → 면 거리. 배관 축과 평행한 법선의 면은 inf (D46 과 같은 규칙)."""
    out = np.full((len(X), 6), np.inf)
    for f, face in enumerate(FACE_ORDER):
        ax, side = FACE_AXIS[face]
        ok = np.abs(U[:, ax]) < 1 - 1e-9
        out[ok, f] = np.abs(X[ok, ax] - side * ext[ax])
    return out


def member_arrays(form: str, X, U, face: str, L, od: float, reinf):
    """members() 의 벡터판. 반환 A, B (N, M, 3), valid (N, M)."""
    N = len(X)
    n = face_normal(face)
    Q = X + L[:, None] * n
    if form == "A":
        return X[:, None], Q[:, None], np.ones((N, 1), dtype=bool)
    if form == "B":
        down = np.array([0.0, 0.0, -1.0])
        W = Q + (L / 2)[:, None] * down
        M = X + (L / 2)[:, None] * n
        A = np.stack([X, W, M], 1)
        B = np.stack([Q, X, M + (L / 4)[:, None] * down], 1)
        return A, B, np.stack([np.ones(N, bool), np.ones(N, bool), np.asarray(reinf, bool)], 1)
    if form == "C":
        s = np.cross(n[None], U)
        sn = np.linalg.norm(s, axis=1)
        fb = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.5 else np.array([0.0, 1.0, 0.0])
        s = np.where(sn[:, None] > 1e-9, s / np.where(sn > 1e-9, sn, 1)[:, None], fb[None])
        b = (od + 2 * C_SIDE_MM) / 2
        C0 = X + (od / 2) * n
        e0, e1 = C0 - b * s, C0 + b * s
        h = np.maximum(L - od / 2, 0.0)[:, None] * n
        A = np.stack([e0, e0, e1, e0 + h / 2], 1)
        B = np.stack([e1, e0 + h, e1 + h, e1 + h / 2], 1)
        return A, B, np.stack([np.ones(N, bool)] * 3 + [np.asarray(reinf, bool)], 1)
    raise ValueError(form)


def _aabb_gap(A, B, lo, hi):
    """선분 AABB ↔ 박스 거리 (축 정렬 선분이면 정확, 아니면 하한)."""
    a, b = np.minimum(A, B), np.maximum(A, B)
    g = np.maximum(0.0, np.maximum(lo - b, a - hi))
    return np.sqrt((g * g).sum(-1))


def members_blocked(A, B, valid, half: float, ext, box_lo, box_hi, others=None, exact=True) -> np.ndarray:
    """부재 (N, M) 중 하나라도 간섭하면 True (D80 ⑧). 장애물: 거리 ≤ 반폭, 블록 밖: 끝점이 블록 밖,
    others = (oA, oB, oR, oS) 얹히지 않은 배관 조각: 거리 − slack < r_o + 반폭.
    exact=False (라우터): 축 정렬이 아닌 부재(B 경사재, 45° 배관의 C 가로대)는 장애물 판정을 AABB 하한으로만 한다 — 
    `추정:` 라우터 근사 (검증기는 exact=True, 황금분할 정밀 판정)."""
    from .verifier.geom import segment_box_distance, segment_segment_distance
    N, M = valid.shape
    bad = np.zeros(N, dtype=bool)
    ext = np.asarray(ext, dtype=float)
    out = ((A < -1).any(-1) | (A > ext + 1).any(-1) | (B < -1).any(-1) | (B > ext + 1).any(-1)) & valid
    bad |= out.any(1)
    Af, Bf = A.reshape(-1, 3), B.reshape(-1, 3)
    vf = valid.reshape(-1)
    diag = (np.count_nonzero(np.abs(Bf - Af) > 1e-9, axis=1) > 1) & vf
    hit = np.zeros(len(Af), dtype=bool)
    for lo, hi in zip(box_lo, box_hi):
        lo = np.asarray(lo, dtype=float); hi = np.asarray(hi, dtype=float)
        d = _aabb_gap(Af, Bf, lo, hi)
        near = vf & ~hit & (d <= half + _EPS)
        if exact:
            ref = np.nonzero(near & diag)[0]
            if len(ref):
                dd, _ = segment_box_distance(Af[ref], Bf[ref], lo, hi)
                near[ref] = dd <= half + _EPS
        hit |= near
    if others is not None and len(others[0]):
        oA, oB, oR, oS = others
        idx = np.nonzero(vf & ~hit)[0]
        for k0 in range(0, len(idx), 256):
            ii = idx[k0:k0 + 256]
            d, _, _ = segment_segment_distance(Af[ii][:, None], Bf[ii][:, None], oA[None], oB[None])
            hit[ii] |= np.any(d - oS[None] < oR[None] + half - _EPS, 1)
    bad |= hit.reshape(N, M).any(1)
    return bad


def evaluate(std: Standard, ext, box_lo, box_hi, nd: str, od: float, X: np.ndarray, U: np.ndarray,
             others=None, exact=True, prefer_ceiling=True) -> dict:
    """서포트 후보 점들 → 가장 가벼운 가능한 (면, 형식) (D80 ③). 반환 배열 dict:
    kg (불가 = inf), opt (OPTIONS 번호, 불가 = -1), L, k (앵글 구간), reinf."""
    N = len(X)
    dist = face_distances(ext, X, U)
    no = len(OPTIONS)
    KG = np.full((N, no), np.inf)
    K = np.full((N, no), -1)
    RF = np.zeros((N, no), dtype=bool)
    for o, (face, frm) in enumerate(OPTIONS):
        Lf = dist[:, OPT_FACE[o]]
        fin = np.isfinite(Lf)
        if not fin.any():
            continue
        kg, k, rf = form_weight(std, frm, nd, od, np.where(fin, Lf, -1.0))
        KG[:, o] = np.where(fin, kg, np.inf)
        K[:, o], RF[:, o] = k, rf
    order = np.lexsort((np.broadcast_to(np.arange(no), (N, no)), KG), axis=1)
    best = np.full(N, -1)
    half_cache = {}

    def blocked(rows, o):
        face, frm = OPTIONS[o]
        Lr = dist[rows, OPT_FACE[o]]
        A, B, v = member_arrays(frm, X[rows], U[rows], face, Lr, od, RF[rows, o])
        halves = np.array([ANGLE_HALF[std.angle_of(frm, nd, int(k))] for k in K[rows, o]])
        res = np.zeros(len(rows), dtype=bool)
        for hv in np.unique(halves):
            m = halves == hv
            o_sub = others
            res[m] = members_blocked(A[m], B[m], v[m], float(hv), ext, box_lo, box_hi, o_sub, exact)
        return res

    for r in range(no):
        todo = np.nonzero((best < 0) & np.isfinite(KG[np.arange(N), order[:, r]]))[0]
        if not len(todo):
            continue
        opts = order[todo, r]
        for o in np.unique(opts):
            rows = todo[opts == o]
            ok = ~blocked(rows, o)
            best[rows[ok]] = o
    if prefer_ceiling:   # D79 ④ `추정:` 천장 무게 ≤ 바닥 × 1.05 이면 천장
        o_floor = OPTIONS.index(("floor", "C"))
        o_ceil = OPTIONS.index(("ceiling", "C"))
        rows = np.nonzero(best == o_floor)[0]
        if len(rows):
            rows = rows[KG[rows, o_ceil] <= CEILING_PREFER * KG[rows, o_floor] + 1e-12]
            if len(rows):
                ok = ~blocked(rows, o_ceil)
                best[rows[ok]] = o_ceil
    sel = best >= 0
    ar = np.arange(N)
    bo = np.where(sel, best, 0)
    return {"kg": np.where(sel, KG[ar, bo], np.inf), "opt": best,
            "L": np.where(sel, dist[ar, OPT_FACE[bo]], np.nan),
            "k": np.where(sel, K[ar, bo], -1), "reinf": np.where(sel, RF[ar, bo], False)}


# --------------------------------------------------------------------------- 실현 가능성 분류 (D80 ⑩)

def band_lmax(std: Standard, nd: str) -> dict:
    """면별 서포트 가능 거리 상한 (그 면에 쓰는 형식 중 표 끝 최대)."""
    wall = max(std.rows[(f, nd)].lmax for f in WALL_FORMS)
    deck = max(std.rows[(f, nd)].lmax for f in DECK_FORMS)
    return {f: (deck if f in ("floor", "ceiling") else wall) for f in FACE_ORDER}


def support_feasibility(pipe, ext, std: Standard) -> dict:
    """규칙상 서포트 불가 (D80 ⑩, D64 방식의 장애물 무시 하한): 어느 단자에서든 최대 간격(수평·수직 중 큰 값) 안에
    서포트 가능 띠(어느 면에서든 표 범위 안)에 닿을 수 없고 다른 단자도 그 거리 밖이면 불가."""
    from .constants import SUPPORT_SPACING
    smax = float(max(SUPPORT_SPACING[pipe.nominal_size]))
    lm = band_lmax(std, pipe.nominal_size)
    T = [np.array(pipe.start.pos, float), np.array(pipe.end.pos, float)]
    gap = []
    for t in T:
        d = []
        for f in FACE_ORDER:
            ax, side = FACE_AXIS[f]
            d.append(max(0.0, abs(t[ax] - side * ext[ax]) - lm[f] + 1.0))
        gap.append(min(d))
    span = float(np.linalg.norm(T[0] - T[1]))
    bad = [g > smax and span > smax for g in gap]
    return {"feasible": not any(bad), "band_gap_mm": [round(g, 1) for g in gap], "smax_mm": smax,
            "blocked_terminal": [n for n, b in zip(("start", "end"), bad) if b]}
