"""S2-T 템플릿 로더·펼치기 (D124 ③, D130~D133, **형식 2 = D136**, 평가/학습 구분 = D137) — 구역 배치 설명서 (YAML)
+ 시드 → 시나리오.

python -m pipe_routing.template <템플릿.yaml> --seed 30000001 -o out/tpl/x.json        # 변형 있음
python -m pipe_routing.template <템플릿.yaml> --base -o out/tpl/base.json              # 기준 배치 (변형 없음)

원칙:
  - 형식 2 만 읽는다 (`format: 2`, D136 ⑥). 최상위 `use: eval | train` 필수 (D137 ①), eval 템플릿 + 학습·검증 시드
    (3xxxxxxx · 4xxxxxxx) 는 거부 (D137 ②).
  - 같은 템플릿 바이트 · 같은 시드 · 같은 배관 수 → 같은 시나리오 (비트 동일). 난수는 `random.Random(seed)` 하나를
    정해진 순서로만 쓴다 (장비 변형 → 배관 수 → 복제 계통 → 단자 흔들기).
  - 성격 (supportable·penetrable·penetration_axis·access) 은 `attribute_defaults` (그룹) · `attribute_overrides`
    (장애물 id) 에서만 읽는다 (D136 ①). 펼친 장애물에는 최종 값을 적는다.
  - 단자 = `장비.노즐` · `DECK_DN(x, y)` · `DECK_UP(x, y)` · `BHD(격벽 id, u, z)` (D136 ③), 배관 = 계통 (D136 ④).
  - 문·곁가지 통로·트레이 내림은 템플릿에 적힌 규칙대로 (D136 ⑤) — 없으면 오류. 템플릿이 정하지 않아 남는 해석은
    `INTERPRETATIONS` (`추정:`).
  - 점검 (D136 ⑦) = 장애물 겹침 (의도된 겹침 규칙 `INTENDED_OVERLAPS`) · 단자 직진 구간 · 접근 구역 침범 → 경고.
    편집기 (`tools/template_editor/`) 는 이 로더의 펼친 결과와 점검을 그린다 (`editor_data`).
  - 출력 meta.template = {id, format, use, kind, sha256 (템플릿 파일 바이트), seed, base, pipes, runs, groups, …}.
"""
import argparse
import hashlib
import json
import math
import random
import re
from dataclasses import replace
from pathlib import Path

import yaml

from . import pipe_spec
from .constants import CLEARANCE_MM, PIPE_SPECS, SNAP_MM, default_drain_slope, effective_radius, elbow_tangent
from .geometry import Box, box_distance
from .scenario import (Block, Branch, Obstacle, Pipe, Scenario, Terminal, inside_block_with_margin, terminal_stub,
                       to_dict, validate)

FORMAT = 2
INTERP_VERSION = 2
# `추정:` 형식 2 가 정하지 않아 로더가 고른 해석 — 번호는 형식 1 초안과 같게 유지 (형식 1 의 T6 노즐 이동·T8 문 위치·
# T11 트레이 내림 계산·T12 노즐 자동 배치·T14 격벽 단자 위치는 형식으로 옮겨 지웠다, D136 ②③⑤)
INTERPRETATIONS = {
    "T1": "방위: east = +x, west = −x, north = +y, south = −y, top = +z, bottom = −z (D136 ② 와 같음), port = y 0 쪽, "
          "stbd = y 최대 쪽",
    "T2": "구조 부재 배치: 간격 s 의 k 번째 부재는 좌표 k·s 에서 시작해 + 방향으로 단면 두께만큼 (k = 1 … 구역 안)",
    "T3": "10 mm 격자 (D5) 밖 두께는 바깥쪽 10 mm 로 올린다: 격벽 12 → 20, 외판 늑골 15 → 20",
    "T4": "웹 면재 (face plate) = 웹 아래 끝에 붙은 수평 판, 웹 중심에 맞추되 10 mm 격자로",
    "T5": "외판 늑골 = 외판 (y = 0 · y = 폭) 에서 안쪽으로 단면 깊이, 높이 = 구역 전체",
    "T6": "케이싱 벽 = 케이싱 평면 사각형 안쪽 가장자리 두께 wall_thickness (10 mm 올림), 높이 = 구역 전체",
    "T7": "고리 통로 = 케이싱 사각형에서 offset 만큼 떨어진 안쪽 가장자리, 폭 width 의 사각 고리 (상자 4개), 높이 height",
    "T8": "곁가지 통로 상자 = 목표 중심 좌표를 가운데로 폭 width, 목표 중심에서 고리 바깥 가장자리까지 (높이 = 고리 높이). "
          "목표 중심이 고리의 x 범위 안이면 y 방향, y 범위 안이면 x 방향, 둘 다 밖이면 오류",
    "T9": "계단 = 평면 상자 × 구역 전체 높이 (위·아래 데크로 이어짐)",
    "T10": "난간 = 개구부 둘레 바깥쪽 폭 width 띠 × 높이 height (상자 4개); 개구부 평면 자체는 장애물 아님 (D130)",
    "T11": "트레이 follow 통로: 통로 중심선을 따라 같은 평면 모양, 높이 = z_top − height … z_top. 수직 내림 "
           "{drop_from, at, to_z, width, depth} = at 을 평면 중심으로, width = 출발 트레이 (at 에 가장 가까운 상자) 진행 "
           "방향, depth = 그 직각, z = to_z … 출발 트레이 아랫면",
    "T15": "복제 계통의 데크·격벽 단자 흔들기 = 각 축 ±terminal_jitter 균등 (10 mm 격자) 뒤, 그 점에서 가장 가까운 빈 점 "
           "(직진 구간이 장애물·다른 단자 직진 구간과 r 이상). 템플릿에 적힌 계통의 단자는 옮기지 않는다",
    "T16": "배관 수 = 주관 + 다리 수 (variation.pipe_count, 없으면 템플릿 그대로). 적으면 앞에서부터 계통 단위로 (넘기기 전까지), "
           "많으면 단자가 모두 데크·격벽인 계통 (노즐은 하나뿐이라 복제 불가) 을 균등 선택해 복제 + 흔들기 (T15). "
           "복제할 계통이 없으면 템플릿 계통만 (meta.template.pipe_count_short)",
    "T17": "장비 변형: 위치 = x·y 각각 ±vary.pos 균등, 크기 = 각 축 ×(1 ± equipment_size) 균등 (바닥 고정, 평면 중심 기준), "
           "10 mm 격자, 노즐은 장비와 함께 (at mm 그대로, D136 ②). 블록 밖·구조·통로·계단·케이싱·다른 장비와 겹치거나, 노즐이 "
           "면 밖이거나, 노즐 직진 구간 (max(§3.3, t₉₀), r = 보온 50 기준) 이 블록 밖·다른 상자와 r 미만이면 다시 뽑기 "
           "(최대 20 번), 끝내 안 되면 기준 위치·크기 그대로 (meta.template.fallback)",
    "T18": "장비 정비 공간 (D119 생성기 규칙) = 장비 수평 면 중 노즐 없는 첫 면 (west, east, south, north 순) 앞 깊이 800 × "
           "장비 높이 × 면 폭; 블록 밖은 자르고, 다른 장비·통로·구조·단자 직진 구간과 겹치면 다음 면, 네 면 모두 안 되면 없음. "
           "성격 그룹 = maintenance",
    "T19": "케이싱 정비 공간 (above, height_to: ceiling) = above 장비 평면 × 그 윗면 … 구역 높이, 성격 그룹 = maintenance",
    "T20": "격벽 단자 BHD(id, u, z) = 격벽 면 (다른 쪽 단자가 있는 쪽, 모르면 − 쪽) 에서 r 만큼 띄운 노즐형 단자, host = 격벽. "
           "u·z 는 관 중심, 면 안 r 여유 밖이면 오류",
}

GROUPS = ("ceiling_longitudinals", "ceiling_web_frames", "side_shell_frames", "pillars", "bulkheads", "casing_wall",
          "casing_contents", "stairs", "equipment", "cable_trays", "passages", "maintenance", "handrails")
ATTR_KEYS = ("supportable", "penetrable", "penetration_axis", "access")
STRUCT_GROUPS = {"ceiling_longitudinals", "ceiling_web_frames", "side_shell_frames", "pillars", "bulkheads",
                 "casing_wall"}
CEILING_GROUPS = {"ceiling_longitudinals", "ceiling_web_frames"}
ACCESS_GROUPS = {"passages", "stairs", "maintenance", "handrails"}
# D136 ⑦ 의도된 겹침 (점검에서 뺀다) — 사람이 읽는 목록, `_intended` 가 같은 규칙
INTENDED_OVERLAPS = (
    "구조끼리 (종늑골·웹·외판 늑골·필라·격벽·케이싱 벽)",
    "접근 구역끼리 (통로·계단·정비 공간·난간)",
    "트레이끼리 (고리·곁가지·수직 내림이 맞닿음)",
    "천장 구조 ↔ 천장까지 닿는 것 (계단·업테이크·케이싱 정비 공간 등 z 최대 = 구역 높이)",
    "통로 ↔ 격벽, 겹친 평면 안에 그 격벽의 문 (passages.doors) 이 있을 때",
)
FACES = {"west": (0, -1), "east": (0, 1), "south": (1, -1), "north": (1, 1), "bottom": (2, -1), "top": (2, 1)}
SEEDS = {"eval": (20260000, 20269999), "train": (30000000, 39999999), "val": (40000000, 49999999)}
MAINT_D = 800
MAX_TRIES = 20
MAX_LEGS = 5
CATKO = {"ceiling_longitudinals": "종늑골", "ceiling_web_frames": "트랜스 웹", "side_shell_frames": "외판 수직 웹",
         "pillars": "필라", "bulkheads": "내부 격벽", "casing_wall": "케이싱 벽", "casing_contents": "케이싱 안 장비",
         "stairs": "계단", "equipment": "장비", "cable_trays": "케이블 트레이", "passages": "통로",
         "maintenance": "정비 공간", "handrails": "난간", "floor_openings": "바닥 개구부"}


class TemplateError(ValueError):
    """형식 2 위반 — 로더 오류 (메시지 목록)."""

    def __init__(self, errors: list):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors[:10]) + (f" … 외 {len(self.errors) - 10}" if len(self.errors) > 10 else ""))


def _s(v: float) -> int:
    return int(round(v / SNAP_MM)) * SNAP_MM


def _up(v: float) -> int:
    return int(math.ceil(v / SNAP_MM - 1e-9)) * SNAP_MM


def _dn(v: float) -> int:
    return int(math.floor(v / SNAP_MM + 1e-9)) * SNAP_MM


def _box(lo, hi) -> Box:
    return Box(tuple(float(c) for c in lo), tuple(float(c) for c in hi))


def _overlap(a: Box, b: Box) -> bool:
    return all(a.min[k] < b.max[k] - 1e-6 and b.min[k] < a.max[k] - 1e-6 for k in range(3))


def load_template(path) -> tuple:
    raw = Path(path).read_bytes()
    return yaml.safe_load(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def seed_range(seed) -> str:
    """평가 · 학습 · 검증 · 기타 (D110 ⑤)."""
    if seed is None:
        return "none"
    for k, (lo, hi) in SEEDS.items():
        if lo <= seed <= hi:
            return k
    return "other"


def check_header(t: dict, seed, base: bool) -> list:
    """형식 2 · use (D136 ⑥, D137 ①②) — 오류 목록."""
    err = []
    if not isinstance(t, dict):
        return ["템플릿 최상위가 사전이 아니다"]
    if t.get("format") != FORMAT:
        err.append(f"format: {t.get('format')!r} — 로더는 형식 2 만 읽는다 (D136 ⑥, v1 파일은 v2 로 바꾼다)")
    use = t.get("use")
    if use not in ("eval", "train"):
        err.append(f"use: {use!r} — eval 또는 train 을 적어야 한다 (D137 ①)")
    if use == "eval" and not base and seed_range(seed) in ("train", "val"):
        err.append(f"eval 템플릿을 학습·검증 시드 {seed} 로 펼칠 수 없다 (D137 ②)")
    if not base and seed is None:
        err.append("시드가 없다 (--seed, 또는 기준 배치 --base)")
    return err


# --------------------------------------------------------------------------- 성격 (D136 ①)

class _Attrs:
    def __init__(self, t: dict, err: list):
        self.defaults = t.get("attribute_defaults") or {}
        self.overrides = t.get("attribute_overrides") or {}
        for g, a in self.defaults.items():
            if g not in GROUPS:
                err.append(f"attribute_defaults: 모르는 그룹 {g} (그룹 = {', '.join(GROUPS)})")
            self._keys(f"attribute_defaults.{g}", a, err)
        for i, a in self.overrides.items():
            self._keys(f"attribute_overrides.{i}", a, err)

    @staticmethod
    def _keys(where, a, err):
        if not isinstance(a, dict):
            err.append(f"{where}: 사전이 아니다")
            return
        for k, v in a.items():
            if k not in ATTR_KEYS:
                err.append(f"{where}: 모르는 성격 {k} ({', '.join(ATTR_KEYS)})")
            elif k == "penetration_axis":
                if v is not None and v not in ("x", "y", "z", 0, 1, 2):
                    err.append(f"{where}.penetration_axis: {v!r} (x·y·z)")
            elif not isinstance(v, bool):
                err.append(f"{where}.{k}: true/false 가 아니다")

    def final(self, oid: str, group: str) -> dict:
        a = {"supportable": False, "penetrable": False, "penetration_axis": None, "access": False}   # D93 기본
        a.update(self.defaults.get(group) or {})
        a.update(self.overrides.get(oid) or {})
        ax = a["penetration_axis"]
        a["penetration_axis"] = "xyz".index(ax) if isinstance(ax, str) else ax
        return a


def _apply_attrs(obs: list, groups: dict, attrs: _Attrs, err: list) -> list:
    out = []
    ids = {o.id for o in obs}
    for i in attrs.overrides:
        if i not in ids:
            err.append(f"attribute_overrides: 펼친 장애물에 없는 id {i}")
    for o in obs:
        a = attrs.final(o.id, groups[o.id])
        if a["access"] and (a["supportable"] or a["penetrable"]):
            err.append(f"{o.id}: access 와 supportable·penetrable 을 함께 true 로 적을 수 없다 (D119)")
        out.append(replace(o, **a))
    return out


# --------------------------------------------------------------------------- 구조 (변형 없음)

def _structure(t: dict, W: float, Lg: float, H: float, G: dict) -> list:
    st = t.get("structure", {}) or {}
    out = []

    def add(o, g):
        out.append(o)
        G[o.id] = g
    cl = st.get("ceiling_longitudinals")
    if cl:   # x 방향으로 뻗음, y 간격 (T2)
        h, w = cl["section"]
        w = _up(w)
        k = 1
        while k * cl["spacing_y"] + w <= Lg:
            y = k * cl["spacing_y"]
            add(Obstacle(f"CL_{k:03d}", "stiffener", _box((0, y, H - h), (W, y + w, H))), "ceiling_longitudinals")
            k += 1
    wf = st.get("ceiling_web_frames")
    if wf:   # y 방향으로 뻗음, x 간격 (T2·T4, D132)
        h, w = wf["section"]
        w = _up(w)
        fw, ft = wf.get("face_plate", (0, 0))
        k = 1
        while k * wf["spacing_x"] + w <= W:
            x = k * wf["spacing_x"]
            add(Obstacle(f"WF_{k:03d}", "web_frame", _box((x, 0, H - h), (x + w, Lg, H))), "ceiling_web_frames")
            if fw:
                x0 = _dn(x + w / 2 - fw / 2)
                add(Obstacle(f"WFP_{k:03d}", "web_frame", _box((x0, 0, H - h - _up(ft)), (x0 + _up(fw), Lg, H - h))),
                    "ceiling_web_frames")
            k += 1
    sf = st.get("side_shell_frames")
    if sf:   # 외판 쪽 수직 늑골 (T5)
        d, th = sf["section"]
        th = _up(th)
        for side in sf.get("sides", ["port", "stbd"]):
            y0, y1 = (0, d) if side == "port" else (Lg - d, Lg)
            k = 1
            while k * sf["spacing_x"] + th <= W:
                x = k * sf["spacing_x"]
                add(Obstacle(f"SF_{side[0].upper()}{k:03d}", "stiffener", _box((x, y0, 0), (x + th, y1, H))),
                    "side_shell_frames")
                k += 1
    pl = st.get("pillars")
    if pl:
        sx, sy = pl["size"]
        k = 0
        for x in pl["grid_x"]:
            for y in pl["grid_y"]:
                k += 1
                add(Obstacle(f"PIL_{k:02d}", "pillar", _box((_s(x - sx / 2), _s(y - sy / 2), 0),
                                                            (_s(x - sx / 2) + sx, _s(y - sy / 2) + sy, H))), "pillars")
    for b in st.get("bulkheads", []) or []:   # 내부 격벽 = 판 (D113, T3)
        th = _up(b["thickness"])
        a = "xyz".index(b["axis"])
        lo, hi = [0.0, 0.0, 0.0], [W, Lg, H]
        c0 = _dn(b["at"] - th / 2)
        lo[a], hi[a] = c0, c0 + th
        q = 1 - a if a < 2 else 0
        span = b.get("span_y") if a == 0 else b.get("span_x")
        if span:
            lo[q], hi[q] = span
        add(Obstacle(b["id"], "bulkhead", _box(lo, hi)), "bulkheads")
    return out


# --------------------------------------------------------------------------- 케이싱·통로·개구부·트레이

def _rect_ring(x0, y0, x1, y1, w, z0, z1, pre: str, typ: str) -> list:
    """사각 고리 (안쪽 가장자리 x0..x1 × y0..y1, 폭 w 바깥쪽) — 상자 4개 (남·북 = 전체 길이, 서·동 = 사이)."""
    boxes = [((x0 - w, y0 - w), (x1 + w, y0)), ((x0 - w, y1), (x1 + w, y1 + w)),
             ((x0 - w, y0), (x0, y1)), ((x1, y0), (x1 + w, y1))]
    return [Obstacle(f"{pre}_{n}", typ, _box((a[0], a[1], z0), (b[0], b[1], z1))) for n, (a, b) in zip("SNWE", boxes)]


def _casing(t, H, G, err):
    c = t.get("casing")
    if not c:
        return [], [], [], None
    (x0, y0), (x1, y1) = c["box"]
    if "wall_thickness" not in c:
        err.append("casing.wall_thickness 가 없다 (형식 2)")
        return [], [], [], None
    th = _up(c["wall_thickness"])
    walls = _rect_ring(x0 + th, y0 + th, x1 - th, y1 - th, th, 0, H, c["id"], "casing")
    for o in walls:
        G[o.id] = "casing_wall"
    contents = []
    for e in c.get("contents", []) or []:
        contents.append(Obstacle(e["id"], e.get("type", "equipment"), _box(*e["box"])))
        G[e["id"]] = "casing_contents"
    return walls, contents, list(c.get("contents", []) or []), (x0, y0, x1, y1)


def _casing_maint(t, H, objs, G, err):
    c = t.get("casing") or {}
    m = c.get("maintenance")
    if not m:
        return []
    host = objs.get(m.get("above"))
    if host is None:
        err.append(f"casing.maintenance.above: 모르는 장비 {m.get('above')}")
        return []
    if m.get("height_to") != "ceiling":
        err.append(f"casing.maintenance.height_to: {m.get('height_to')!r} (형식 2 = ceiling)")
        return []
    o = Obstacle(f"ACC_{c['id']}", "maintenance", _box((host.box.min[0], host.box.min[1], host.box.max[2]),
                                                       (host.box.max[0], host.box.max[1], H)))
    G[o.id] = "maintenance"
    return [o]


def _door_on(d: dict):
    """문의 `on` — YAML 1.1 은 따옴표 없는 키 on 을 참 (True) 으로 읽는다 (yaml.safe_load). 둘 다 받는다."""
    return d.get("on", d.get(True))


def _door_point(d: dict, bulk: dict, err) -> tuple:
    b = bulk.get(_door_on(d))
    if b is None:
        err.append(f"passages.doors {d.get('id')}: on = 모르는 격벽 {_door_on(d)}")
        return None
    x, y = (float(v) for v in d["at"])
    a = b.pen_axis()
    c = (x, y)[a]
    q = 1 - a
    if not (b.box.min[a] - 1e-6 <= c <= b.box.max[a] + 1e-6 and b.box.min[q] <= (x, y)[q] <= b.box.max[q]):
        err.append(f"passages.doors {d['id']}: at {d['at']} 가 격벽 {b.id} 위가 아니다")
    return x, y


def _spoke_box(c, ring_outer, w, err, tag):
    """목표 중심 c → 고리 바깥 가장자리 한 축 직선 (D136 ⑤, T8). 반환 ((x0, y0), (x1, y1), 축)."""
    rx0, ry0, rx1, ry1 = ring_outer
    cx, cy = c
    if rx0 <= cx <= rx1 and not (ry0 <= cy <= ry1):   # y 방향
        x0 = _s(cx - w / 2)
        return ((x0, cy), (x0 + w, ry0), 1) if cy < ry0 else ((x0, ry1), (x0 + w, cy), 1)
    if ry0 <= cy <= ry1 and not (rx0 <= cx <= rx1):   # x 방향
        y0 = _s(cy - w / 2)
        return ((cx, y0), (rx0, y0 + w), 0) if cx < rx0 else ((rx1, y0), (cx, y0 + w), 0)
    err.append(f"passages.spokes {tag}: 목표 중심 ({cx:g}, {cy:g}) 에서 고리까지 한 축 직선이 없다")
    return None


def _passages(t, casing_rect, H, bulk: dict, G, err):
    p = t.get("passages", {}) or {}
    acc, paths, openings, doors = [], {}, [], {}
    ring_outer = None
    rg = p.get("ring")
    if rg:
        if not casing_rect:
            err.append("passages.ring: 케이싱이 없다 (around)")
        else:
            x0, y0, x1, y1 = casing_rect
            o, w, h = rg["offset"], rg["width"], rg["height"]
            ring_outer = (x0 - o - w, y0 - o - w, x1 + o + w, y1 + o + w)
            for ob in _rect_ring(x0 - o, y0 - o, x1 + o, y1 + o, w, 0, h, "PASS_RING", "passage"):
                acc.append(ob)
                G[ob.id] = "passages"
            paths["ring"] = {"kind": "ring", "inner": (x0 - o, y0 - o, x1 + o, y1 + o), "w": w, "h": h}
    stairs = {}
    for s in p.get("stairs", []) or []:
        ob = Obstacle(s["id"], "stair", _box((*s["box"][0], 0), (*s["box"][1], H)))
        acc.append(ob)
        G[ob.id] = "stairs"
        stairs[s["id"]] = s
    for d in p.get("doors", []) or []:
        pt = _door_point(d, bulk, err)
        if pt:
            doors[d["id"]] = {"at": pt, "on": _door_on(d)}
    for i, sp in enumerate(p.get("spokes", []) or []):
        tgt = sp["to"]
        if tgt in stairs:
            (a0, b0), (a1, b1) = stairs[tgt]["box"]
            c = ((a0 + a1) / 2, (b0 + b1) / 2)
        elif tgt in doors:
            c = doors[tgt]["at"]
        else:
            err.append(f"passages.spokes[{i}]: to = {tgt} 가 계단·문 목록에 없다 (D136 ⑤ 문은 doors 에 적는다)")
            continue
        if ring_outer is None:
            err.append(f"passages.spokes[{i}]: 고리 통로가 없다")
            continue
        sb = _spoke_box(c, ring_outer, sp["width"], err, f"[{i}] {tgt}")
        if sb is None:
            continue
        lo, hi, ax = sb
        h = paths.get("ring", {}).get("h", rg["height"] if rg else 2000)
        ob = Obstacle(f"PASS_SPOKE_{i}", "passage", _box((*lo, 0), (*hi, h)))
        acc.append(ob)
        G[ob.id] = "passages"
        paths[f"spokes[{i}]"] = {"kind": "spoke", "lo": lo, "hi": hi, "axis": ax, "w": sp["width"]}
    for hch in p.get("hatches", []) or []:
        (hx0, hy0), (hx1, hy1) = hch["box"]
        if hch.get("floor_opening"):
            openings.append({"id": hch["id"], "min": [hx0, hy0], "max": [hx1, hy1]})
        hr = hch.get("handrail")
        if hr is not None:
            if not isinstance(hr, dict) or "height" not in hr or "width" not in hr:
                err.append(f"passages.hatches {hch['id']}.handrail: {{height, width}} (형식 2)")
                continue
            for ob in _rect_ring(hx0, hy0, hx1, hy1, _up(hr["width"]), 0, hr["height"], f"RAIL_{hch['id']}",
                                 "handrail"):
                acc.append(ob)
                G[ob.id] = "handrails"
    return acc, paths, openings, doors


def _trays(t, paths, G, err):
    out, by_id = [], {}

    def add(o):
        out.append(o)
        G[o.id] = "cable_trays"
    for tr in t.get("cable_trays", []) or []:
        if "drop_from" in tr:
            continue
        key = str(tr.get("follow", ""))
        if not key.startswith("passages.") or key.split(".", 1)[1] not in paths:
            err.append(f"cable_trays {tr.get('id')}: follow = {key!r} 가 펼친 통로에 없다")
            continue
        p = paths[key.split(".", 1)[1]]
        z1 = tr["z_top"]
        z0 = z1 - tr["height"]
        w = tr["width"]
        if p["kind"] == "ring":
            x0, y0, x1, y1 = p["inner"]
            pw = p["w"]
            if tr.get("side") == "outer":
                ix0, iy0, ix1, iy1 = x0 - pw + w, y0 - pw + w, x1 + pw - w, y1 + pw - w
            else:
                ix0, iy0, ix1, iy1 = x0, y0, x1, y1
            boxes = _rect_ring(ix0, iy0, ix1, iy1, w, z0, z1, tr["id"], "tray")
        else:
            (lx, ly), (hx, hy), ax = p["lo"], p["hi"], p["axis"]
            if ax == 1:
                c = (lx + hx) / 2
                lo, hi = (_s(c - w / 2), ly), (_s(c - w / 2) + w, hy)
            else:
                c = (ly + hy) / 2
                lo, hi = (lx, _s(c - w / 2)), (hx, _s(c - w / 2) + w)
            boxes = [Obstacle(tr["id"], "tray", _box((*lo, z0), (*hi, z1)))]
        by_id[tr["id"]] = boxes
        for b in boxes:
            add(b)
    for tr in t.get("cable_trays", []) or []:
        if "drop_from" not in tr:
            continue
        miss = [k for k in ("at", "to_z", "width", "depth") if k not in tr]
        if miss or tr["drop_from"] not in by_id:
            err.append(f"cable_trays {tr.get('id')}: 수직 내림은 drop_from (펼친 트레이) · at · to_z · width · depth "
                       f"(D136 ⑤, 빠짐 {miss or tr['drop_from']})")
            continue
        px, py = (float(v) for v in tr["at"])
        src = by_id[tr["drop_from"]]
        b = min(src, key=lambda s: (math.hypot(px - min(max(px, s.box.min[0]), s.box.max[0]),
                                               py - min(max(py, s.box.min[1]), s.box.max[1])), s.id))
        run_x = (b.box.max[0] - b.box.min[0]) >= (b.box.max[1] - b.box.min[1])   # 출발 트레이 진행 방향 (T11)
        w, dpt = (tr["width"], tr["depth"]) if run_x else (tr["depth"], tr["width"])
        x0, y0 = _s(px - w / 2), _s(py - dpt / 2)
        if not tr["to_z"] < b.box.min[2]:
            err.append(f"cable_trays {tr['id']}: to_z {tr['to_z']} 가 출발 트레이 아랫면 {b.box.min[2]:g} 보다 낮지 않다")
            continue
        add(Obstacle(tr["id"], "tray", _box((x0, y0, tr["to_z"]), (x0 + w, y0 + dpt, b.box.min[2]))))
    return out


# --------------------------------------------------------------------------- 장비·노즐 (D136 ②)

def _nozzle_point(box: Box, nz: dict):
    """노즐 점·면 (축, 부호) — 면 밖이면 None."""
    ax, side = FACES[nz["face"]]
    p = [0.0, 0.0, 0.0]
    p[ax] = box.max[ax] if side > 0 else box.min[ax]
    a0, a1 = (float(v) for v in nz["at"])
    if ax == 2:
        p[0], p[1] = box.min[0] + a0, box.min[1] + a1
        ok = 0 <= a0 <= box.max[0] - box.min[0] and 0 <= a1 <= box.max[1] - box.min[1]
    else:
        q = 1 - ax
        p[q], p[2] = box.min[q] + a0, box.min[2] + a1
        ok = 0 <= a0 <= box.max[q] - box.min[q] and 0 <= a1 <= box.max[2] - box.min[2]
    return (tuple(p), ax, side) if ok else None


def _check_nozzles(entries: list, err: list):
    for e in entries:
        for n, nz in (e.get("nozzles") or {}).items():
            miss = [k for k in ("face", "at", "size") if k not in nz]
            if miss:
                err.append(f"{e['id']}.{n}: 노즐에 {miss} 가 없다")
            elif nz["face"] not in FACES:
                err.append(f"{e['id']}.{n}: face {nz['face']!r} ({', '.join(FACES)})")
            elif _nozzle_point(_box(*e["box"]), nz) is None:
                err.append(f"{e['id']}.{n}: 노즐 점 at {nz['at']} 이 {nz['face']} 면 밖 (D136 ②)")


def _nozzle_stub_clear(box: Box, nz: dict, others: list, ext, r: float) -> bool:
    """변형 후보 검사 (T17): 노즐 직진 구간 (max(§3.3, t₉₀), r = 그 노즐을 쓰는 계통 서비스의 r) 이 블록 안 r 여유 ·
    다른 상자와 r 이상."""
    p, ax, side = _nozzle_point(box, nz)
    sz = nz["size"]
    L = max(PIPE_SPECS[sz].min_straight, elbow_tangent(sz, 90))
    a, b = list(p), list(p)
    a[ax] += side * _up(r)
    b[ax] = a[ax] + side * L
    st = Box.of_points(tuple(a), tuple(b))
    return inside_block_with_margin(st, ext, r) and all(box_distance(st, o) >= r - 1e-6 for o in others)


def _equipment(t, rng, H, avoid: list, base: bool, G, nozzle_r: dict):
    var = t.get("variation", {}) or {}
    scale = float(var.get("equipment_size", 0.0))
    out, fallback = [], []
    W, Lg = t["zone"]["size"][0], t["zone"]["size"][1]
    for e in t.get("equipment", []) or []:
        (x0, y0, z0), (x1, y1, z1) = e["box"]
        basebox = _box((x0, y0, z0), (x1, y1, z1))
        vp = 0 if base else (e.get("vary") or {}).get("pos", 0)
        noz = list((e.get("nozzles") or {}).values())
        box = None
        for _ in range(MAX_TRIES):
            if not vp:
                break
            dx, dy = _s(rng.uniform(-vp, vp)), _s(rng.uniform(-vp, vp))
            sx = rng.uniform(1 - scale, 1 + scale) if scale else 1.0
            sy = rng.uniform(1 - scale, 1 + scale) if scale else 1.0
            sz = rng.uniform(1 - scale, 1 + scale) if scale else 1.0
            cx, cy = (x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy
            wx, wy, wz = _s((x1 - x0) * sx), _s((y1 - y0) * sy), _s((z1 - z0) * sz)
            lx, ly = _s(cx - wx / 2), _s(cy - wy / 2)
            cand = _box((lx, ly, z0), (lx + wx, ly + wy, min(H, z0 + wz)))
            others = avoid + [o.box for o in out]
            if all(c >= 0 for c in cand.min) and cand.max[0] <= W and cand.max[1] <= Lg and \
                    all(_nozzle_point(cand, nz) is not None for nz in noz) and \
                    not any(_overlap(cand, a) for a in others) and \
                    all(_nozzle_stub_clear(cand, nz, others, (W, Lg, H),
                                           nozzle_r.get((e["id"], n), effective_radius(nz["size"])))
                        for n, nz in (e.get("nozzles") or {}).items()):
                box = cand
                break
        if box is None:
            box = basebox
            if vp:
                fallback.append(e["id"])
        out.append(Obstacle(e["id"], e.get("kind", "equipment"), box))
        G[e["id"]] = "equipment"
    return out, fallback


# --------------------------------------------------------------------------- 단자 (D136 ③)

_DECK = re.compile(r"^(DECK_DN|DECK_UP)\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)$")
_BHD = re.compile(r"^BHD\(\s*([A-Za-z0-9_]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)$")
_NOZ = re.compile(r"^([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)$")


def parse_term(s: str, err: list, where: str):
    s = str(s).strip()
    m = _DECK.match(s)
    if m:
        return ("deck", 0 if m.group(1) == "DECK_DN" else 1, float(m.group(2)), float(m.group(3)))
    m = _BHD.match(s)
    if m:
        return ("bhd", m.group(1), float(m.group(2)), float(m.group(3)))
    m = _NOZ.match(s)
    if m:
        return ("nozzle", m.group(1), m.group(2))
    err.append(f"{where}: 단자 {s!r} — 장비.노즐 · DECK_DN(x, y) · DECK_UP(x, y) · BHD(id, u, z) (D136 ③)")
    return None


def _tmp_pipe(svc, size, term, which):
    grav = pipe_spec.services()[svc].gravity
    cls, ins = pipe_spec.resolve(svc, None, None)
    dummy = Terminal((0.0, 0.0, 0.0), "nozzle", (1, 0, 0))
    return Pipe("tmp", "T5" if grav else "T1", size, grav, default_drain_slope(size) if grav else None, ins,
                term if which == "start" else dummy, term if which == "end" else dummy, service=svc)


def _radius(svc: str, size: str) -> float:
    cls, ins = pipe_spec.resolve(svc, None, None)
    return pipe_spec.spec(cls, size).od / 2 + ins + CLEARANCE_MM


def stub_length(pipe: Pipe, term: Terminal, penetration: bool) -> float:
    """D136 ⑦ 점검 직진 길이: 노즐 = max(§3.3, t₉₀), 데크·격벽 관통 = max(3D, r) + t₉₀."""
    t90 = elbow_tangent(pipe.nominal_size, 90)
    if penetration:
        from .blockpair import straight_min
        return straight_min(pipe) + t90
    return max(pipe.min_straight, t90)


def _clear(stub, r, solid, done, block, skip_axis=None) -> bool:
    if not inside_block_with_margin(stub, block.extent, r, skip_axis=skip_axis):
        return False
    if any(box_distance(stub, o.box) < r - 1e-6 for o in solid):
        return False
    return not any(box_distance(stub, st) < r + rr - 1e-6 for st, rr in done)


def _offsets(rng_mm: float):
    n = int(rng_mm // SNAP_MM)
    pts = [(dx * SNAP_MM, dy * SNAP_MM) for dx in range(-n, n + 1) for dy in range(-n, n + 1)]
    return sorted(pts, key=lambda p: (p[0] * p[0] + p[1] * p[1], p[0], p[1]))


class _Terms:
    """단자 만들기 — 노즐 (D136 ②) · 데크 (D35) · 격벽 (T20)."""

    def __init__(self, objs, bulk, H, block, solid, err):
        self.objs, self.bulk, self.H, self.block, self.solid, self.err = objs, bulk, H, block, solid, err
        self.noz = {}      # (장비, 노즐) → spec
        self.used = {}     # (장비, 노즐) → 쓰는 계통

    def nozzle(self, host, name, which, svc, size, where):
        o = self.objs.get(host)
        nz = self.noz.get((host, name))
        if o is None or nz is None:
            self.err.append(f"{where}: 노즐 {host}.{name} 이 없다")
            return None
        if nz["size"] != size:
            self.err.append(f"{where}: 노즐 {host}.{name} 구경 {nz['size']} ≠ 배관 {size} (D136 ③)")
        if (host, name) in self.used:
            self.err.append(f"{where}: 노즐 {host}.{name} 를 {self.used[(host, name)]} 도 쓴다")
        self.used[(host, name)] = where
        pt = _nozzle_point(o.box, nz)
        if pt is None:
            self.err.append(f"{where}: 노즐 {host}.{name} 이 면 밖")
            return None
        p, ax, side = pt
        r = _radius(svc, size)
        pos = list(p)
        pos[ax] += side * _up(r)
        nrm = [0, 0, 0]
        nrm[ax] = side
        d = tuple(nrm) if which == "start" else tuple(-c for c in nrm)
        return Terminal(tuple(float(v) for v in pos), "nozzle", d, host)

    def deck(self, x, y, up, which):
        z = self.H if up else 0.0
        sign = -1 if up else 1   # 하부 데크 start = +z, 상부 데크 start = −z (D35)
        d = (0, 0, sign) if which == "start" else (0, 0, -sign)
        return Terminal((float(x), float(y), z), "boundary", d)

    def bhd(self, bid, u, z, which, svc, size, other, where, err=None):
        err = self.err if err is None else err
        o = self.bulk.get(bid)
        if o is None:
            err.append(f"{where}: BHD({bid}) — 모르는 격벽")
            return None
        a = o.pen_axis()
        q = 1 - a if a < 2 else 0
        r = _radius(svc, size)
        if not (o.box.min[q] + r - 1e-6 <= u <= o.box.max[q] - r + 1e-6 and
                o.box.min[2] + r - 1e-6 <= z <= o.box.max[2] - r + 1e-6):
            err.append(f"{where}: BHD({bid}, {u:g}, {z:g}) 가 격벽 면 안 r {r:.0f} 여유 밖 (T20)")
        side = 1 if (other is not None and other.pos[a] > o.box.max[a]) else -1
        pos = [0.0, 0.0, float(z)]
        pos[q] = float(u)
        pos[a] = (o.box.max[a] if side > 0 else o.box.min[a]) + side * _up(r)
        nrm = [0, 0, 0]
        nrm[a] = side
        d = tuple(nrm) if which == "start" else tuple(-v for v in nrm)
        return Terminal(tuple(pos), "nozzle", d, bid)


def _systems(t, err):
    """계통 목록 검사 (D136 ④) — [(계통 dict, 서비스)]."""
    svcs = pipe_spec.services()
    out = []
    for i, ps in enumerate(t.get("pipes", []) or []):
        where = f"pipes[{i}]"
        miss = [k for k in ("svc", "size", "from", "to") if k not in ps]
        if miss:
            err.append(f"{where}: {miss} 가 없다")
            continue
        if ps["svc"] not in svcs:
            err.append(f"{where}: 모르는 서비스 {ps['svc']} (D116)")
            continue
        legs = ps.get("legs") or []
        if len(legs) > MAX_LEGS:
            err.append(f"{where}: 다리 {len(legs)} 개 > {MAX_LEGS} (D136 ④)")
        main_mm = int(ps["size"][:-1])
        for j, lg in enumerate(legs):
            if ("to" in lg) == ("from" in lg) or "size" not in lg:
                err.append(f"{where}.legs[{j}]: {{size, to}} 또는 {{size, from}} 하나 (D136 ④)")
                continue
            if int(lg["size"][:-1]) > main_mm:
                err.append(f"{where}.legs[{j}]: 다리 구경 {lg['size']} > 주관 {ps['size']} (D112)")
            if svcs[ps["svc"]].gravity and "to" in lg:
                err.append(f"{where}.legs[{j}]: 중력 서비스 {ps['svc']} 의 다리는 from (합류) 만 (D136 ④, D112 Q1)")
        out.append(ps)
    return out


def _ends_sized(ps):
    """(단자 규약 start|end, 단자 표기, 구경) — 주관 양 끝 + 다리 (합류 = start, 갈래 = end)."""
    yield "start", ps["from"], ps["size"]
    yield "end", ps["to"], ps["size"]
    for lg in ps.get("legs") or []:
        yield ("start", lg["from"], lg["size"]) if "from" in lg else ("end", lg["to"], lg["size"])


def _all_ends(ps):
    yield "start", ps["from"]
    yield "end", ps["to"]
    for lg in ps.get("legs") or []:
        yield ("legin", lg["from"]) if "from" in lg else ("legout", lg["to"])


def _duplicable(ps, err) -> bool:
    return all((parse_term(s, [], "") or ("?",))[0] in ("deck", "bhd") for _, s in _all_ends(ps))


def expand(t: dict, sha: str, seed=None, n_pipes=None, name: str = "", base: bool = False) -> Scenario:
    """템플릿 → 시나리오. 형식 위반은 TemplateError (모든 오류 목록)."""
    err = check_header(t, seed, base)
    if err:
        raise TemplateError(err)
    rng = random.Random(0 if base else seed)
    W, Lg, H = (float(v) for v in t["zone"]["size"])
    block = Block(W, Lg, H)
    var = {} if base else (t.get("variation", {}) or {})
    G = {}
    attrs = _Attrs(t, err)
    _check_nozzles(list(t.get("equipment", []) or []) + list((t.get("casing") or {}).get("contents", []) or []), err)
    systems = _systems(t, err)
    if err:
        raise TemplateError(err)
    struct = _structure(t, W, Lg, H, G)
    walls, contents, content_specs, casing_rect = _casing(t, H, G, err)
    bulk = {o.id: o for o in struct if o.type == "bulkhead"}
    acc, paths, openings, doors = _passages(t, casing_rect, H, bulk, G, err)
    avoid = [o.box for o in acc] + [o.box for o in walls] + [o.box for o in struct]
    if casing_rect:
        x0, y0, x1, y1 = casing_rect
        avoid.append(_box((x0, y0, 0), (x1, y1, H)))
    # 변형이 피할 것: 템플릿에 적힌 데크·격벽 단자의 직진 구간 (r 만큼 키움), 노즐마다 쓰는 계통의 r (T17)
    nozzle_r = {}
    for ps in systems:
        for which, spec, sz in _ends_sized(ps):
            e = parse_term(spec, [], "")
            if e is None:
                continue
            rr = _radius(ps["svc"], sz)
            if e[0] == "nozzle":
                nozzle_r[(e[1], e[2])] = rr
            elif e[0] == "deck":
                tt = Terminal((float(e[2]), float(e[3]), H if e[1] == 1 else 0.0), "boundary",
                              (0, 0, (-1 if e[1] == 1 else 1) * (1 if which == "start" else -1)))
                p_ = _tmp_pipe(ps["svc"], sz, tt, which)
                st = terminal_stub(p_, which, stub_length(p_, tt, True))
                avoid.append(_box([c - rr for c in st.min], [c + rr for c in st.max]))
    equip, fallback = _equipment(t, rng, H, avoid, base, G, nozzle_r)
    objs = {o.id: o for o in equip + contents + struct}
    acc += _casing_maint(t, H, objs, G, err)
    trays = _trays(t, paths, G, err)
    if err:
        raise TemplateError(err)

    # 배관 수 (T16) — 주관 + 다리
    runs = lambda ps: 1 + len(ps.get("legs") or [])
    total = sum(runs(ps) for ps in systems)
    target = total
    if not base and "pipe_count" in var and n_pipes is None:
        lo, hi = var["pipe_count"]
        target = rng.randint(lo, hi)
    elif n_pipes is not None:
        target = n_pipes
    chosen, short = [], 0
    if target <= total:
        k = 0
        for ps in systems:
            if k + runs(ps) > target:
                break
            chosen.append((ps, False))
            k += runs(ps)
    else:
        chosen = [(ps, False) for ps in systems]
        dup = [ps for ps in systems if _duplicable(ps, err)]
        k = total
        while k < target and dup:
            ps = dup[rng.randrange(len(dup))]
            if k + runs(ps) > target:
                small = [d for d in dup if k + runs(d) <= target]
                if not small:
                    break
                ps = small[rng.randrange(len(small))]
            chosen.append((ps, True))
            k += runs(ps)
        short = target - k
    jit = float(var.get("terminal_jitter", 0))

    # 단자
    tm = _Terms(objs, bulk, H, block, [], err)
    for e in list(t.get("equipment", []) or []) + content_specs:
        for n, nz in (e.get("nozzles") or {}).items():
            tm.noz[(e["id"], n)] = nz
    solid = [o for o in struct + walls + contents + equip + acc + trays
             if not _final_pen(attrs, o, G)]
    done = []      # 이미 정한 단자 직진 구간 [(Box, r)]
    moved = {}
    pipes = []
    sources = []   # 펼친 계통마다 [템플릿 pipes 순번, 복제 여부] — 편집기 표 (D136 ⑦)
    svcs = pipe_spec.services()
    for i, (ps, is_dup) in enumerate(chosen):
        pid = f"P{i + 1:03d}"
        svc, size = ps["svc"], ps["size"]
        grav = svcs[svc].gravity
        r = _radius(svc, size)

        def make(spec, which, sz, where, other=None):
            e = parse_term(spec, err, where)
            if e is None:
                return None, False
            rr = _radius(svc, sz)
            if e[0] == "nozzle":
                return tm.nozzle(e[1], e[2], which, svc, sz, where), False
            if e[0] == "deck":
                x, y = e[2], e[3]
                if is_dup and jit:
                    x, y = x + _s(rng.uniform(-jit, jit)), y + _s(rng.uniform(-jit, jit))
                    for dx, dy in _offsets(jit):
                        cand = tm.deck(_s(x + dx), _s(y + dy), e[1] == 1, which)
                        p_ = _tmp_pipe(svc, sz, cand, which)
                        st = terminal_stub(p_, which, stub_length(p_, cand, True))
                        if _clear(st, rr, solid, done, block, skip_axis=2):
                            moved[f"{where}"] = [float(_s(x + dx)), float(_s(y + dy))]
                            return cand, True
                    moved[f"{where}"] = None
                return tm.deck(_s(x), _s(y), e[1] == 1, which), True
            u, z = e[2], e[3]
            if is_dup and jit:
                u, z = u + _s(rng.uniform(-jit, jit)), z + _s(rng.uniform(-jit, jit))
                own = [o for o in solid if o.id != e[1]]
                for du, dz in _offsets(jit):
                    q = []
                    cand = tm.bhd(e[1], float(_s(u + du)), float(_s(z + dz)), which, svc, sz, other, where, q)
                    if q or cand is None:
                        continue
                    p_ = _tmp_pipe(svc, sz, cand, which)
                    st = terminal_stub(p_, which, stub_length(p_, cand, True))
                    if _clear(st, rr, own, done, block):
                        moved[where] = [float(_s(u + du)), float(_s(z + dz))]
                        return cand, True
                moved[where] = None
            return tm.bhd(e[1], float(_s(u)), float(_s(z)), which, svc, sz, other, where), True
        s_t, s_pen = make(ps["from"], "start", size, f"{pid}.start")
        e_t, e_pen = make(ps["to"], "end", size, f"{pid}.end", s_t)
        branches = []
        for j, lg in enumerate(ps.get("legs") or []):
            flow = "in" if "from" in lg else "out"
            which = "start" if flow == "in" else "end"
            bt, _ = make(lg["from"] if flow == "in" else lg["to"], which, lg["size"], f"{pid}.B{j + 1}")
            if bt is not None:
                branches.append(Branch(f"B{j + 1}", lg["size"], bt, flow))
        if s_t is None or e_t is None:
            continue
        cls, ins = pipe_spec.resolve(svc, None, None)
        typ = ("T7" if branches else "T5") if grav else ("T3" if branches else "T1")
        p = Pipe(pid, typ, size, grav, default_drain_slope(size) if grav else None, ins, s_t, e_t,
                 branches=branches, service=svc)
        pipes.append(p)
        sources.append([systems.index(ps), is_dup])
        for which, term, pen in (("start", s_t, s_pen), ("end", e_t, e_pen)):
            done.append((terminal_stub(p, which, stub_length(p, term, pen)), r))
        for b in branches:
            lp = _tmp_pipe(svc, b.nominal_size, b.terminal, b.which)
            done.append((terminal_stub(lp, b.which, stub_length(lp, b.terminal, b.terminal.kind == "boundary")),
                         lp.radius))
    if err:
        raise TemplateError(err)

    # 장비 정비 공간 (T18, D119)
    noz_faces = {(h, nz["face"]) for (h, _n), nz in tm.noz.items()}
    maint = []
    solid_all = [o.box for o in equip + contents + struct + walls + acc + trays]
    for o in equip:
        for face in ("west", "east", "south", "north"):
            if (o.id, face) in noz_faces:
                continue
            ax, side = FACES[face]
            lo, hi = list(o.box.min), list(o.box.max)
            if side > 0:
                lo[ax], hi[ax] = o.box.max[ax], min((W, Lg)[ax], o.box.max[ax] + MAINT_D)
            else:
                lo[ax], hi[ax] = max(0.0, o.box.min[ax] - MAINT_D), o.box.min[ax]
            if hi[ax] - lo[ax] < 100:
                continue
            b = _box(lo, hi)
            if any(_overlap(b, s) for s in solid_all + [m.box for m in maint] if s is not o.box):
                continue
            if any(box_distance(b, st) < rr - 1e-6 for st, rr in done):
                continue
            mo = Obstacle(f"MAINT_{o.id}", "maintenance", b)
            G[mo.id] = "maintenance"
            maint.append(mo)
            break

    obstacles = _apply_attrs(struct + walls + contents + equip + acc + maint + trays, G, attrs, err)
    if err:
        raise TemplateError(err)
    meta = {"name": name or (f"{t['template']}_base" if base else f"{t['template']}_s{seed}"),
            "template": {"id": t["template"], "format": FORMAT, "use": t["use"], "kind": t.get("kind"),
                         "version": t.get("version"), "sha256": sha, "seed": None if base else seed,
                         "seed_range": "none" if base else seed_range(seed), "base": base,
                         "pipes": sum(1 + len(p.branches) for p in pipes), "systems": len(pipes),
                         "pipe_count_short": short, "interpretations": INTERP_VERSION, "fallback": fallback,
                         "moved_terminals": moved, "sources": sources,
                         "groups": {o.id: G[o.id] for o in obstacles}}}
    sc = Scenario(block, obstacles, pipes, meta=meta)
    sc.floor_openings = openings
    names = pipe_spec.line_numbers(sc)
    for p in sc.pipes:
        p.line_no = names.get(p.id)
    sc.meta["template"]["doors"] = {k: {"on": v["on"], "at": list(v["at"])} for k, v in doors.items()}
    return sc


def _final_pen(attrs: _Attrs, o: Obstacle, G: dict) -> bool:
    return bool(attrs.final(o.id, G[o.id])["penetrable"])


# --------------------------------------------------------------------------- 점검 (D136 ⑦, 경고)

def _intended(a, b, ga, gb, H, doors) -> bool:
    if ga in STRUCT_GROUPS and gb in STRUCT_GROUPS:
        return True
    if ga in ACCESS_GROUPS and gb in ACCESS_GROUPS:
        return True
    if ga == gb == "cable_trays":
        return True
    for x, gx, y in ((a, ga, b), (b, gb, a)):
        if gx in CEILING_GROUPS and y.box.max[2] >= H - 1e-6:
            return True
    for x, gx, y, gy in ((a, ga, b, gb), (b, gb, a, ga)):
        if gx == "passages" and gy == "bulkheads":
            lo = [max(x.box.min[k], y.box.min[k]) for k in range(2)]
            hi = [min(x.box.max[k], y.box.max[k]) for k in range(2)]
            if any(d["on"] == y.id and all(lo[k] - 1e-6 <= d["at"][k] <= hi[k] + 1e-6 for k in range(2))
                   for d in doors.values()):
                return True
    return False


def check(sc: Scenario, doors: dict = None) -> dict:
    """D136 ⑦ 점검 → {"overlaps": [(a, b)], "terminals": [문장], "validate": (오류, 경고)}. 시나리오는 바꾸지 않는다."""
    H = sc.block.height
    G = sc.meta["template"]["groups"]
    doors = doors if doors is not None else sc.meta["template"].get("doors", {})
    obs = sc.obstacles
    ov = []
    for i, a in enumerate(obs):
        for b in obs[i + 1:]:
            if _overlap(a.box, b.box) and not _intended(a, b, G[a.id], G[b.id], H, doors):
                ov.append((a.id, b.id))
    msgs = []
    for p in sc.pipes:
        ends = [("start", p.start, p), ("end", p.end, p)]
        for b in p.branches:
            ends.append((b.which, b.terminal, _tmp_pipe(p.service, b.nominal_size, b.terminal, b.which), b.id))
        for e in ends:
            which, term, pp = e[:3]
            tag = e[3] if len(e) > 3 else which
            pen = term.kind == "boundary" or (term.host in G and G[term.host] == "bulkheads")
            L = stub_length(pp, term, pen)
            st = terminal_stub(pp, which, L)
            r = pp.radius
            for o in obs:
                if o.id == term.host:
                    continue
                d = box_distance(st, o.box)
                if d < r - 1e-6:
                    kind = "접근 구역 침범" if o.access else "단자 직진 구간"
                    msgs.append(f"{p.id}.{tag} {term.host or 'DECK'} ({pp.nominal_size}): "
                                f"{kind} {L:.0f} mm 가 {o.id} ({CATKO.get(G.get(o.id), o.type)}) 와 {d:.0f} mm (< r {r:.0f})")
    return {"overlaps": ov, "terminals": msgs, "validate": validate(sc)}


# --------------------------------------------------------------------------- 편집기 자료 (D136 ⑦)

def editor_data(t: dict, sc: Scenario, chk: dict) -> dict:
    """편집기 page.html 이 그리는 자료 — 이 로더의 펼친 결과 그대로 (펼치기 코드 하나)."""
    G = sc.meta["template"]["groups"]
    boxes = []
    for o in sc.obstacles:
        boxes.append({"key": G[o.id], "cat": CATKO[G[o.id]], "id": o.id, "min": [round(v, 1) for v in o.box.min],
                      "max": [round(v, 1) for v in o.box.max],
                      "attrs": {"supportable": o.supportable, "penetrable": o.penetrable,
                                "penetration_axis": o.penetration_axis, "access": o.access}})
    for f in sc.floor_openings:
        boxes.append({"key": "floor_openings", "cat": CATKO["floor_openings"], "id": f["id"],
                      "min": [*f["min"], 0.0], "max": [*f["max"], 20.0],
                      "attrs": {"supportable": False, "penetrable": False, "penetration_axis": None, "access": False}})
    systems = []
    src = sc.meta["template"].get("sources") or [[None, False]] * len(sc.pipes)
    for k, p in enumerate(sc.pipes):
        ps = (t.get("pipes") or [])[src[k][0]] if src[k][0] is not None else {}
        dup = " (복제)" if src[k][1] else ""
        runs = [{"role": "주관" + dup, "size": p.nominal_size, "line_no": p.line_no,
                 "from": ps.get("from", ""), "to": ps.get("to", "")}]
        terms = [{"name": f"{p.id}.start", "pos": list(p.start.pos), "dir": list(p.start.dir), "size": p.nominal_size,
                  "host": p.start.host or ("DECK_DN" if p.start.pos[2] == 0 else "DECK_UP"), "role": "주관",
                  "side": "from"},
                 {"name": f"{p.id}.end", "pos": list(p.end.pos), "dir": list(p.end.dir), "size": p.nominal_size,
                  "host": p.end.host or ("DECK_DN" if p.end.pos[2] == 0 else "DECK_UP"), "role": "주관", "side": "to"}]
        for j, b in enumerate(p.branches):
            role = f"다리 {j + 1} ({'합류' if b.flow == 'in' else '갈래'})"
            lg = (ps.get("legs") or [{}] * (j + 1))[j]
            runs.append({"role": role, "size": b.nominal_size, "from": lg.get("from", "(티)"), "to": lg.get("to", "(티)")})
            terms.append({"name": f"{p.id}.{b.id}", "pos": list(b.terminal.pos), "dir": list(b.terminal.dir),
                          "size": b.nominal_size, "host": b.terminal.host or "DECK", "role": role,
                          "side": "from" if b.flow == "in" else "to"})
        systems.append({"no": k + 1, "id": p.id, "svc": p.service, "runs": runs, "terms": terms})
    errs, warns = chk["validate"]
    return {"zone": [sc.block.width, sc.block.length, sc.block.height], "boxes": boxes, "systems": systems,
            "clash": [[CATKO[G[a]], a, CATKO[G[b]], b] for a, b in chk["overlaps"]],
            "issues": chk["terminals"] + [f"로더 오류: {e}" for e in errs] +
                      [f"로더 경고: {w}" for w in warns if "(D27)" not in w],   # D27 (S0 라우터 대상) 은 편집기에 필요 없음
            "defaults": t.get("attribute_defaults") or {}, "overrides": t.get("attribute_overrides") or {},
            "catko": CATKO, "meta": {k: v for k, v in sc.meta["template"].items() if k != "groups"}}


def scenario_bytes(sc: Scenario) -> bytes:
    return (json.dumps(to_dict(sc), ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("template")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--base", action="store_true", help="기준 배치 — 장비 변형·배관 늘리기·단자 흔들기 없음")
    ap.add_argument("--pipes", type=int, default=None, help="배관 수 = 주관 + 다리 (없으면 variation.pipe_count)")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--name", default="")
    a = ap.parse_args(argv)
    t, sha = load_template(a.template)
    try:
        sc = expand(t, sha, a.seed, a.pipes, a.name, base=a.base)
    except TemplateError as e:
        for m in e.errors:
            print("ERR", m)
        return 2
    chk = check(sc)
    errs, warns = chk["validate"]
    data = scenario_bytes(sc)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_bytes(data)
    print(json.dumps({"out": a.out, "template_sha256": sha, "scenario_sha256": hashlib.sha256(data).hexdigest(),
                      "obstacles": len(sc.obstacles), "systems": len(sc.pipes), "pipes": sc.meta["template"]["pipes"],
                      "errors": len(errs), "warnings": len(warns), "overlaps": len(chk["overlaps"]),
                      "terminal_checks": len(chk["terminals"]), "fallback": sc.meta["template"]["fallback"]},
                     ensure_ascii=False))
    for e in errs[:40]:
        print("ERR", e)
    for w in warns[:40]:
        print("WARN", w)
    for o in chk["overlaps"][:40]:
        print("OVERLAP", *o)
    for m in chk["terminals"][:40]:
        print("CHECK", m)
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
