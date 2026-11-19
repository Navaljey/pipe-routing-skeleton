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


def face_distances(ext, X: np.ndarray, U: np.ndarray, sp=()) -> np.ndarray:
    """(N, 6) 배관 중심 → 면 거리. 배관 축과 평행한 법선의 면은 inf (D46 과 같은 규칙).
    sp = 블록 쌍 공유 격벽 x 목록 (D76) — 면은 점이 속한 블록의 면 (공유 격벽은 양쪽 블록의 벽)."""
    out = np.full((len(X), 6), np.inf)
    if sp:
        from .blockpair import local_bounds
        lo, hi = local_bounds(ext, X, sp)
    for f, face in enumerate(FACE_ORDER):
        ax, side = FACE_AXIS[face]
        ok = np.abs(U[:, ax]) < 1 - 1e-9
        if sp:
            out[ok, f] = (hi[ok, ax] - X[ok, ax]) if side else (X[ok, ax] - lo[ok, ax])
        else:
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


def members_blocked(A, B, valid, half: float, ext, box_lo, box_hi, others=None, exact=True, bounds=None) -> np.ndarray:
    """부재 (N, M) 중 하나라도 간섭하면 True (D80 ⑧). 장애물: 거리 ≤ 반폭, 블록 밖: 끝점이 블록 밖,
    others = (oA, oB, oR, oS) 얹히지 않은 배관 조각: 거리 − slack < r_o + 반폭.
    exact=False (라우터): 축 정렬이 아닌 부재(B 경사재, 45° 배관의 C 가로대)는 장애물 판정을 AABB 하한으로만 한다 — 
    `추정:` 라우터 근사 (검증기는 exact=True, 황금분할 정밀 판정).
    bounds = (lo, hi) (N, 3) — 서포트 점마다 그 블록 범위 (블록 쌍 D76, 부재는 블록 밖으로 못 나간다). 없으면 [0, ext]."""
    from .verifier.geom import segment_box_distance, segment_segment_distance
    N, M = valid.shape
    bad = np.zeros(N, dtype=bool)
    ext = np.asarray(ext, dtype=float)
    if bounds is None:
        out = ((A < -1).any(-1) | (A > ext + 1).any(-1) | (B < -1).any(-1) | (B > ext + 1).any(-1)) & valid
    else:
        blo, bhi = bounds[0][:, None] - 1, bounds[1][:, None] + 1
        out = ((A < blo).any(-1) | (A > bhi).any(-1) | (B < blo).any(-1) | (B > bhi).any(-1)) & valid
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
        olo, ohi = np.minimum(oA, oB), np.maximum(oA, oB)
        lim = oR + oS + half + 1.0      # 결과 불변 가속 (D84): 이보다 먼 조각은 판정을 바꿀 수 없다
        idx = np.nonzero(vf & ~hit)[0]
        for k0 in range(0, len(idx), 256):
            ii = idx[k0:k0 + 256]
            clo = np.minimum(Af[ii], Bf[ii]).min(0)
            chi = np.maximum(Af[ii], Bf[ii]).max(0)
            g = np.maximum(0.0, np.maximum(olo - chi, clo - ohi))
            jj = np.nonzero(np.sqrt((g * g).sum(-1)) <= lim)[0]   # 선분 거리 ≥ AABB 간격
            if not len(jj):
                continue
            d, _, _ = segment_segment_distance(Af[ii][:, None], Bf[ii][:, None], oA[jj][None], oB[jj][None])
            hit[ii] |= np.any(d - oS[jj][None] < oR[jj][None] + half - _EPS, 1)
    bad |= hit.reshape(N, M).any(1)
    return bad


# --------------------------------------------------------------------------- D113 supportable 장애물 면 (앵커)

@dataclass(frozen=True)
class Anchor:
    """supportable 장애물의 면 하나 (D113). face = 블록 면과 같은 의미의 이름 (배관 → 면 방향):
    윗면 = floor (배관이 위), 아랫면 = ceiling, x 최대면 = wall_x0 (배관이 +x 쪽), x 최소면 = wall_x1 …
    box = evaluate 에 넘기는 장애물 목록에서 그 장애물 번호 (자기 자신과의 부재 간섭은 보지 않는다)."""
    obs_id: str
    box: int
    face: str
    lo: tuple
    hi: tuple

    @property
    def axis(self) -> int:
        return FACE_AXIS[self.face][0]

    @property
    def plane(self) -> float:
        ax, side = FACE_AXIS[self.face]
        return float(self.hi[ax] if side == 0 else self.lo[ax])

    @property
    def label(self) -> str:
        return f"{self.face}@{self.obs_id}"

    def forms(self) -> tuple:
        return DECK_FORMS if self.face in ("floor", "ceiling") else WALL_FORMS


def anchors_of(sc) -> tuple:
    """시나리오의 supportable 장애물 면 (D113). 장애물 순서 · FACE_ORDER 순서 (동률 깨기 `추정:`)."""
    out = []
    ext = sc.block.extent
    wmin = 2 * min(ANGLE_HALF.values())
    for i, o in enumerate(sc.obstacles):
        if not o.supportable:
            continue
        for face in FACE_ORDER:
            a = Anchor(o.id, i, face, tuple(map(float, o.box.min)), tuple(map(float, o.box.max)))
            # 쓸 수 없는 면은 뺀다 (판정 결과 같음): 면 폭이 앵글 두 반폭보다 좁으면 부재 끝점이 면 안에 들어갈 수 없고
            # (anchor_contacts_ok 가 늘 거부), 블록 경계 위의 면은 그 바깥에 배관이 있을 수 없다
            if any(a.hi[q] - a.lo[q] < wmin for q in range(3) if q != a.axis):
                continue
            if a.plane <= 0 or a.plane >= ext[a.axis]:
                continue
            out.append(a)
    return tuple(out)


def anchor_distance(a: Anchor, X: np.ndarray, U: np.ndarray) -> np.ndarray:
    """배관 중심 → 앵커 면 거리 (면 바깥쪽에 있을 때만, 배관 축이 법선과 평행하면 inf)."""
    ax, side = FACE_AXIS[a.face]
    L = (X[:, ax] - a.plane) if side == 0 else (a.plane - X[:, ax])
    ok = (np.abs(U[:, ax]) < 1 - 1e-9) & (L > _EPS)
    return np.where(ok, L, np.inf)


def anchor_contacts_ok(a: Anchor, A, B, valid, half) -> np.ndarray:
    """부재가 면에 닿는 끝점 (면 평면 위 끝점) 이 면 사각형 안 (가장자리에서 반폭 안쪽) — D113 `추정:`.
    A, B (N, M, 3), valid (N, M), half (N,). 닿는 끝점이 없으면 False."""
    ax = a.axis
    pl = a.plane
    q = [k for k in range(3) if k != ax]
    ok = np.ones(A.shape[0], dtype=bool)
    any_c = np.zeros(A.shape[0], dtype=bool)
    hv = np.asarray(half, float)[:, None]
    for E in (A, B):
        c = valid & (np.abs(E[..., ax] - pl) <= 1e-6)
        any_c |= c.any(1)
        for k in q:
            inside = (E[..., k] >= a.lo[k] + hv - 1e-6) & (E[..., k] <= a.hi[k] - hv + 1e-6)
            ok &= ~(c & ~inside).any(1)
    return ok & any_c


def option_list(anchors=()) -> list:
    """[(면 이름 (기하용), 형식, 앵커 또는 None)] — 블록 면 OPTIONS 뒤에 앵커 면 (동률은 앞쪽)."""
    out = [(f, frm, None) for f, frm in OPTIONS]
    for a in anchors:
        out += [(a.face, frm, a) for frm in a.forms()]
    return out


def option_label(o: int, anchors=()) -> tuple:
    """evaluate 의 opt 번호 → (출력 면 이름, 형식, 기하용 면 이름)."""
    face, frm, a = option_list(anchors)[o]
    return (a.label if a is not None else face), frm, face


def evaluate(std: Standard, ext, box_lo, box_hi, nd: str, od: float, X: np.ndarray, U: np.ndarray,
             others=None, exact=True, prefer_ceiling=True, trace=None, sp=(), anchors=()) -> dict:
    """서포트 후보 점들 → 가장 가벼운 가능한 (면, 형식) (D80 ③). 반환 배열 dict:
    kg (불가 = inf), opt (option_list 번호 — 앵커가 없으면 OPTIONS 번호, 불가 = -1), L, k (앵글 구간), reinf.
    anchors = supportable 장애물 면 (D113) — 없으면 이전과 같은 연산."""
    if anchors:
        return _evaluate_anchors(std, ext, box_lo, box_hi, nd, od, X, U, others, exact, prefer_ceiling, trace, sp,
                                 anchors)
    N = len(X)
    dist = face_distances(ext, X, U, sp)
    if sp:   # 블록 쌍 (D76): 면 거리·부재 범위는 점이 속한 블록 기준
        from .blockpair import local_bounds
        bnd = local_bounds(ext, X, sp)
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
        if trace is not None and v.any():   # D85 증분 검증: 다른 배관과 판정한 부재 영역 (AABB, 반폭 최대)
            trace.append((np.minimum(A, B)[v].min(0), np.maximum(A, B)[v].max(0), float(halves.max())))
        for hv in np.unique(halves):
            m = halves == hv
            o_sub = others
            res[m] = members_blocked(A[m], B[m], v[m], float(hv), ext, box_lo, box_hi, o_sub, exact,
                                     bounds=(bnd[0][rows][m], bnd[1][rows][m]) if sp else None)
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


def support_feasibility(pipe, ext, std: Standard, sp=(), anchors=()) -> dict:
    """규칙상 서포트 불가 (D80 ⑩, D64 방식의 장애물 무시 하한): 어느 단자에서든 최대 간격 안에 서포트 가능 띠(어느 면에서든
    표 범위 안)에 닿을 수 없고 다른 단자도 그 거리 밖이면 불가.

    단자에서 띠까지 필요한 정규화 길이 하한 u (> 1 이면 불가):
      노즐 (D80 ⑩ 그대로) — u = 띠까지 거리 / max(수평·수직 간격).
      경계 단자 (D106 (a), D96 강제 직관) — 단자에서 단자 방향 직선 위가 이미 띠 안(그 방향과 평행하지 않은 법선의
        면 기준 — 직선을 따라 거리가 일정)이면 0, 아니면 첫 꺾임 각 θ (45·90·135) 중 최소의
        강제 직관 max(3D, r) / 그 방향 간격 + 첫 꺾임 구역 2·t_θ / 수평 간격 (D87 ②)
        + 엘보 호 끝에서 띠까지 거리 / max(간격) (호 끝 = 꺾임점에서 t_θ 만큼 새 방향, 옆 방향은 가장 유리하게).
        단자 면 자체는 배관 축과 평행한 법선이라 강제 직관 위에서는 지지면이 아니다. 장애물은 무시 (하한)."""
    from .constants import SUPPORT_SPACING, elbow_tangent
    from . import blockpair as bp
    hmax, vmax = (float(v) for v in SUPPORT_SPACING[pipe.nominal_size])
    smax = max(hmax, vmax)
    lm = band_lmax(std, pipe.nominal_size)

    def face_dist(t):
        if sp:
            from .blockpair import local_bounds
            lo, hi = (v[0] for v in local_bounds(ext, t[None], sp))
        out = {}
        for f in FACE_ORDER:
            ax, side = FACE_AXIS[f]
            out[f] = (hi[ax] - t[ax] if side else t[ax] - lo[ax]) if sp else abs(t[ax] - side * ext[ax])
        return out

    def gap_of(t):
        return min(max(0.0, d - lm[f] + 1.0) for f, d in face_dist(t).items())

    T = [np.array(pipe.start.pos, float), np.array(pipe.end.pos, float)]
    gap, ulb = [], []
    for t, term, sgn in ((T[0], pipe.start, 1.0), (T[1], pipe.end, -1.0)):
        g0 = gap_of(t)
        gap.append(g0)
        u = g0 / smax
        if term.kind == "boundary" and bp.D96:
            d = sgn * np.asarray(term.dir, dtype=float)
            ax = int(np.argmax(np.abs(d)))
            fd = face_dist(t)
            on_line = min((max(0.0, fd[f] - lm[f] + 1.0) for f in FACE_ORDER if FACE_AXIS[f][0] != ax), default=np.inf)
            if on_line > 0:
                s0 = bp.straight_min(pipe)
                s_d = vmax if ax == 2 else hmax
                u = np.inf
                for th in (45, 90, 135):   # 첫 꺾임: 접점 (s0) → 꺾임점 (s0 + t) → 호 끝 (꺾임점 + t·e, e 와 d 사이 각 θ)
                    tt = elbow_tangent(pipe.nominal_size, th)
                    along = s0 + tt + tt * math.cos(math.radians(th))   # d 방향 이동
                    lateral = tt * math.sin(math.radians(th))           # d 에 수직 이동 (방향은 가장 유리하게)
                    g = np.inf
                    for f in FACE_ORDER:
                        fa, side = FACE_AXIS[f]
                        if fa == ax:   # d 와 같은 축의 면: d 방향 이동만큼 거리 변화
                            toward = (d[ax] > 0) == bool(side)
                            dist = fd[f] - along if toward else fd[f] + along
                        else:
                            dist = fd[f] - lateral
                        g = min(g, max(0.0, dist - lm[f] + 1.0))
                    u = min(u, s0 / s_d + 2 * tt / hmax + g / smax)
            else:
                u = 0.0
        ulb.append(u)
    for k, t in enumerate(T):   # D113 앵커 면: 단자에서 면 사각형까지 유클리드 거리 − 표 끝 (하한, 장애물 무시)
        for a in anchors:
            g = np.maximum(0.0, np.maximum(np.asarray(a.lo) - t, t - np.asarray(a.hi)))
            ulb[k] = min(ulb[k], max(0.0, float(np.sqrt((g * g).sum())) - lm[a.face] + 1.0) / smax)
    span = float(np.linalg.norm(T[0] - T[1]))
    bad = [u > 1 + 1e-9 and span > smax for u in ulb]
    return {"feasible": not any(bad), "band_gap_mm": [round(g, 1) for g in gap], "smax_mm": smax,
            "u_lower_bound": [round(u, 3) for u in ulb],
            "blocked_terminal": [n for n, b in zip(("start", "end"), bad) if b]}

def _evaluate_anchors(std, ext, box_lo, box_hi, nd, od, X, U, others, exact, prefer_ceiling, trace, sp, anchors):
    """evaluate 의 D113 판 — 블록 면 옵션 + 앵커 면 옵션. 블록 면 옵션은 evaluate 와 같은 규칙.
    앵커 옵션: 거리 = anchor_distance, 부재가 면 사각형 안에 닿아야 하고 (anchor_contacts_ok), 부재 간섭에서 그 장애물
    자체는 뺀다. 천장 선호 (D79 ④) = 가장 가벼운 것이 바닥형 C (블록 바닥 또는 장애물 윗면) 이면 천장형 C 중
    무게 ≤ 1.05 × 그 무게인 것을 (무게, 번호) 순서로 시도."""
    N = len(X)
    opts = option_list(anchors)
    no = len(opts)
    bd = face_distances(ext, X, U, sp)
    if sp:
        from .blockpair import local_bounds
        bnd = local_bounds(ext, X, sp)
    D = np.full((N, no), np.inf)
    for o, (face, frm, a) in enumerate(opts):
        D[:, o] = bd[:, FACE_ORDER.index(face)] if a is None else anchor_distance(a, X, U)
    KG = np.full((N, no), np.inf)
    K = np.full((N, no), -1)
    RF = np.zeros((N, no), dtype=bool)
    for o, (face, frm, a) in enumerate(opts):
        Lf = D[:, o]
        fin = np.isfinite(Lf)
        if not fin.any():
            continue
        kg, k, rf = form_weight(std, frm, nd, od, np.where(fin, Lf, -1.0))
        KG[:, o] = np.where(fin, kg, np.inf)
        K[:, o], RF[:, o] = k, rf
    blo_all = np.asarray(box_lo, dtype=float).reshape(-1, 3)
    bhi_all = np.asarray(box_hi, dtype=float).reshape(-1, 3)
    order = np.lexsort((np.broadcast_to(np.arange(no), (N, no)), KG), axis=1)
    best = np.full(N, -1)

    def blocked(rows, o):
        face, frm, a = opts[o]
        Lr = D[rows, o]
        A, B, v = member_arrays(frm, X[rows], U[rows], face, Lr, od, RF[rows, o])
        halves = np.array([ANGLE_HALF[std.angle_of(frm, nd, int(k))] for k in K[rows, o]])
        res = np.zeros(len(rows), dtype=bool)
        if trace is not None and v.any():
            trace.append((np.minimum(A, B)[v].min(0), np.maximum(A, B)[v].max(0), float(halves.max())))
        if a is not None:
            res |= ~anchor_contacts_ok(a, A, B, v, halves)
            keep = np.ones(len(blo_all), dtype=bool)
            if 0 <= a.box < len(keep):
                keep[a.box] = False
            bl, bh = blo_all[keep], bhi_all[keep]
        else:
            bl, bh = blo_all, bhi_all
        for hv in np.unique(halves):
            m = (halves == hv) & ~res
            if not m.any():
                continue
            res[m] = members_blocked(A[m], B[m], v[m], float(hv), ext, bl, bh, others, exact,
                                     bounds=(bnd[0][rows][m], bnd[1][rows][m]) if sp else None)
        return res

    for r in range(no):
        todo = np.nonzero((best < 0) & np.isfinite(KG[np.arange(N), order[:, r]]))[0]
        if not len(todo):
            continue
        oo = order[todo, r]
        for o in np.unique(oo):
            rows = todo[oo == o]
            ok = ~blocked(rows, o)
            best[rows[ok]] = o
    if prefer_ceiling:
        floor_c = np.array([f == "floor" and frm == "C" for f, frm, _ in opts])
        ceil_c = [o for o, (f, frm, _) in enumerate(opts) if f == "ceiling" and frm == "C"]
        rows = np.nonzero((best >= 0) & floor_c[np.maximum(best, 0)])[0]
        for i in rows:
            cands = sorted((KG[i, o], o) for o in ceil_c if KG[i, o] <= CEILING_PREFER * KG[i, best[i]] + 1e-12)
            for _, o in cands:
                if not blocked(np.array([i]), o)[0]:
                    best[i] = o
                    break
    sel = best >= 0
    ar = np.arange(N)
    bo = np.where(sel, best, 0)
    return {"kg": np.where(sel, KG[ar, bo], np.inf), "opt": best,
            "L": np.where(sel, D[ar, bo], np.nan),
            "k": np.where(sel, K[ar, bo], -1), "reinf": np.where(sel, RF[ar, bo], False)}
