"""배관 사양 체계 (D116·D117) — 유체 서비스 → pipe class → 구경별 OD·kg/m·관이음 무게.

라우터·검증기·서포트·예약은 배관 사양 값을 **이 모듈로만** 읽는다 (CLAUDE.md §1.16 정합 원칙).
`DEFAULT` = §3.1·§3.2 상수 (constants.PIPE_SPECS·FITTINGS) 그대로 — 값과 계산 순서까지 같다 (E1 불변).
공개 임시 표 = `data/pipe_classes.json` (`추정:`), 사내 표 = 환경 변수 `PIPE_CLASS_STANDARD` (D78 방식).
"""
import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from .constants import FITTINGS, PIPE_SPECS

DATA = Path(__file__).resolve().parent.parent / "data"
CLASS_ENV = "PIPE_CLASS_STANDARD"
DEFAULT_CLASS = "DEFAULT"
DEFAULT_INSULATION_MM = 50.0   # D17·D117: 입력도 서비스 기본도 없을 때


@dataclass(frozen=True)
class Spec:
    od: float           # 외경 mm
    kg_per_m: float     # 직관 kg/m
    elbow90: float      # kg
    elbow45: float
    gate_valve: float
    reducer: Optional[float] = None   # D118 — 표에 없으면 None (큰 쪽 45° 엘보로 대체)


@dataclass(frozen=True)
class Service:
    code: str
    name: str
    gravity: bool
    pipe_class: str
    insulation_mm: float
    color: str


@lru_cache(maxsize=None)
def services() -> dict:
    d = json.loads((DATA / "services.json").read_text(encoding="utf-8"))["services"]
    return {c: Service(c, v["name"], bool(v["gravity"]), v["class"], float(v["insulation_mm"]), v["color"])
            for c, v in d.items()}


def _class_specs(c: dict) -> dict:
    out = {}
    for nd, s in c["sizes"].items():
        f = c.get("fittings", {}).get(nd, {})
        out[nd] = Spec(float(s["od"]), float(s["kg_per_m"]), float(f["elbow90"]), float(f["elbow45"]),
                       float(f["gate_valve"]), float(f["reducer"]) if "reducer" in f else None)
    return out


@lru_cache(maxsize=None)
def _tables() -> tuple:
    """(class → {nd → Spec}, source, sha256). DEFAULT 은 코드 상수에서."""
    classes = {DEFAULT_CLASS: {nd: Spec(PIPE_SPECS[nd].od, PIPE_SPECS[nd].kg_per_m, FITTINGS[nd].elbow90,
                                        FITTINGS[nd].elbow45, FITTINGS[nd].gate_valve) for nd in PIPE_SPECS}}
    pub = (DATA / "pipe_classes.json").read_bytes()
    for name, c in json.loads(pub)["classes"].items():
        classes[name] = _class_specs(c)
    src, raw = "public_approx", pub
    path = os.environ.get(CLASS_ENV)
    if path:
        p = Path(path)
        p = p / "pipe_classes.json" if p.is_dir() else p
        raw = p.read_bytes()
        for name, c in json.loads(raw)["classes"].items():
            classes[name] = _class_specs(c)
        src = "private"
    return classes, src, hashlib.sha256(raw).hexdigest()


def class_names() -> list:
    return list(_tables()[0])


def has(pipe_class: str, nd: str) -> bool:
    return nd in _tables()[0].get(pipe_class, {})


def spec(pipe_class: str, nd: str) -> Spec:
    return _tables()[0][pipe_class][nd]


def standard_info() -> dict:
    """출력 global `pipe_class_standard` (D117)."""
    _, src, sha = _tables()
    return {"source": src, "sha256": sha}


def max_od() -> float:
    return max(s.od for c in _tables()[0].values() for s in c.values())


def elbow_kg(sp: Spec, deflection_deg: int) -> float:
    """§3.4 엘보 청구 (constants.elbow_kg 와 같은 식·순서). 135° = 90° + 45° (D25)."""
    return {0: 0.0, 45: sp.elbow45, 90: sp.elbow90, 135: sp.elbow90 + sp.elbow45}[deflection_deg]


def reducer_kg(pipe_class: str, nd_a: str, nd_b: str) -> float:
    """D118: class 표 값, 없으면 큰 쪽 구경 45° 엘보 (`추정:`)."""
    big = max(nd_a, nd_b, key=lambda n: PIPE_SPECS[n].od)
    sp = spec(pipe_class, big)
    return sp.reducer if sp.reducer is not None else sp.elbow45


def resolve(service: Optional[str], pipe_class: Optional[str], insulation: Optional[float]) -> tuple:
    """(class, 보온 mm) — 입력 → 서비스 기본 → DEFAULT / 50 (D117)."""
    svc = services().get(service) if service else None
    cls = pipe_class or (svc.pipe_class if svc else DEFAULT_CLASS)
    ins = insulation if insulation is not None else (svc.insulation_mm if svc else DEFAULT_INSULATION_MM)
    return cls, ins


def line_numbers(sc) -> dict:
    """D116 배관 이름: 입력 line_no, 없으면 {서비스}-{start 호칭경}-{번호 3자리}-{class}. `추정:` 번호 = 시나리오 배관
    순서로 서비스마다 001 부터, 직접 이름을 적은 배관도 번호 하나를 쓴다. 서비스 없는 배관은 없음."""
    count, out = {}, {}
    for p in sc.pipes:
        if p.service is None:
            if p.line_no is not None:
                out[p.id] = p.line_no
            continue
        count[p.service] = count.get(p.service, 0) + 1
        out[p.id] = p.line_no or f"{p.service}-{p.nominal_size}-{count[p.service]:03d}-{p.class_name}"
    return out


def color(p, n: int, palette) -> str:
    """3D·리포트 색 (D116): 서비스가 있으면 서비스 색, 없으면 배관 순서 팔레트 (지금과 같음)."""
    s = services().get(p.service) if p.service else None
    return s.color if s else palette[n % len(palette)]


def label(p, names: dict) -> str:
    """hover 용 사양 문자열 (D116): 이름·서비스·class·보온 — 서비스·class 를 적은 배관만."""
    if p.service is None and p.pipe_class is None:
        return ""
    nm = names.get(p.id, "")
    return f"{nm + ' · ' if nm else ''}{p.service or '-'} · {p.class_name} · 보온 {p.insulation_thickness:g} mm"
