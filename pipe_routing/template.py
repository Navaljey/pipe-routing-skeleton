"""S2-T 템플릿 로더·펼치기 (D124 ③, D130~D133) — 구역 배치 설명서 (YAML) + 시드 → 시나리오.

python -m pipe_routing.template templates/er_2nd_generic_a.yaml --seed 40000001 --pipes 40 -o out/tpl/er_40.json

원칙:
  - 같은 템플릿 바이트 · 같은 시드 · 같은 배관 수 → 같은 시나리오 (비트 동일). 난수는 `random.Random(seed)` 하나를
    정해진 순서로만 쓴다 (장비 변형 → 배관 늘리기 → 단자 흔들기).
  - 구조 규칙 (종늑골·웹·외판 늑골·필라) 은 같은 좌표를 공유하도록 펼친다 — 같은 간격의 부재는 같은 좌표 열에 놓인다
    (D124 ③). 구조는 흔들지 않는다 (`variation.structure: fixed`).
  - 통로·계단·난간·정비 공간 → 접근 구역 (D119), 개구부 → floor_openings (D130), 트레이 → tray 장애물 (D131),
    천장 웹·종늑골·외판 늑골 → 스티프너 계열 supportable (D113·D132), z 범위 = 데크 윗면 → 윗데크 아랫면 (D133).
  - 템플릿 형식이 모호하거나 빠진 곳은 가장 단순한 해석을 골랐다 — 아래 `INTERPRETATIONS` 와 `docs/reports/s2_t.md` §2
    (`추정:` 목록). 값을 바꾼 곳은 없다 — 10 mm 격자 (D5) 밖 두께만 바깥쪽으로 올린다 (목록 T3).
  - 출력 meta.template = {id, version, sha256 (템플릿 파일 바이트), seed, pipes, interpretations 판}, 시나리오 sha256 은
    저장한 JSON 바이트 (`scenario_sha256`).
"""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import yaml

from . import pipe_spec
from .constants import CLEARANCE_MM, SNAP_MM, default_drain_slope
from .geometry import Box, box_distance
from .scenario import (Block, Obstacle, Pipe, Scenario, Terminal, ScenarioError, terminal_stub, to_dict,
                       validate)

INTERP_VERSION = 1
# `추정:` 템플릿 해석 — 보고서 s2_t.md §2 목록과 같은 번호
INTERPRETATIONS = {
    "T1": "방위: east = +x, west = −x, north = +y, south = −y, port = y 0 쪽, stbd = y 최대 쪽",
    "T2": "구조 부재 배치: 간격 s 의 k 번째 부재는 좌표 k·s 에서 시작해 + 방향으로 단면 두께만큼 (k = 1 … 구역 안)",
    "T3": "10 mm 격자 (D5) 밖 두께는 바깥쪽 10 mm 로 올린다: 격벽 12 → 20, 외판 늑골 15 → 20",
    "T4": "웹 면재 (face plate) = 웹 아래 끝에 붙은 수평 판, 웹 중심에 맞추되 10 mm 격자로 (웹 x 에서 −60 … +90)",
    "T5": "외판 늑골 = 외판 (y = 0 · y = 폭) 에서 안쪽으로 단면 깊이 300, 높이 = 구역 전체; band_y 는 깊이 상한 확인에만",
    "T6": "케이싱 벽 = 케이싱 평면 사각형 안쪽 가장자리 두께 10, 높이 = 구역 전체, 관통 불가 (템플릿 값). 케이싱 안 장비에 "
          "붙는 노즐은 그 면 쪽 케이싱 벽 바깥면에 둔다 (케이싱 안은 배관 금지 구역 — 템플릿 주석)",
    "T7": "고리 통로 = 케이싱 사각형에서 offset 만큼 떨어진 안쪽 가장자리, 폭 width 의 사각 고리 (상자 4개), 높이 height",
    "T8": "문 위치 = 격벽 위에서 고리 통로에 가장 가까운 점 (겹치는 구간의 가운데, 10 mm 격자); 바퀴살 통로 = 대상 (계단 상자·"
          "문) 에서 고리 바깥 가장자리까지 축 방향 곧은 상자 (폭 width, 높이 = 고리 높이)",
    "T9": "계단 = 평면 상자 × 구역 전체 높이 접근 구역 (위·아래 데크로 이어짐)",
    "T10": "난간 = 개구부 둘레 바깥쪽 폭 50 mm 띠 × 난간 높이 접근 구역 (상자 4개); 개구부 평면 자체는 장애물 아님 (D130)",
    "T11": "트레이 follow 통로: 통로 중심선을 따라 같은 평면 모양, side outer = 고리 바깥쪽 가장자리에서 안쪽으로 폭만큼, "
           "높이 = z_top − height … z_top. 수직 트레이 (from → to) = from 트레이에서 to 장비 중심에 가장 가까운 점, "
           "평면 width (트레이 진행 방향) × depth, z = to 장비 윗면 … from 트레이 아랫면",
    "T12": "노즐 = 장비 면 가운데 높이 (면 높이 범위의 가운데, 10 mm 격자), 같은 면의 노즐 n 개는 면 폭을 n + 1 등분한 점에, "
           "면에서 유효 반경 (10 mm 올림) 만큼 띄움 (D28). start = 바깥 법선, end = 안쪽 (D29)",
    "T13": "DECK_DN(x, y) = 하부 데크 (z = 0) 경계 단자, DECK_UP(x, y) = 상부 데크 (z = H) 경계 단자 (D35 dir 규약)",
    "T14": "BHD(id) = 그 격벽 면 (먼저 정해진 쪽 단자가 있는 쪽 면) 의 노즐형 단자 — host = 격벽, 위치 = 격벽 면 가운데 "
           "(폭 가운데 · 높이 가운데)",
    "T15": "같은 점을 쓰는 단자 (같은 DECK 점) 는 배관 순서대로 +x 쪽으로 r₁ + r₂ (10 mm 올림) 씩 옮긴다; 데크 단자 흔들기 "
           "= 각 축 ±terminal_jitter 균등 (10 mm 격자) 후 같은 규칙",
    "T16": "배관 수 n < 목록 수면 앞 n 개, n > 목록 수면 목록에서 균등 선택한 배관을 복제 (같은 서비스·구경·연결, 단자는 같은 "
           "규칙으로 다시 배치 + 데크 단자 흔들기). 서비스 SD·DD = 중력관 (T5, 최소 구배 = 배수관 기본값), 나머지 T1",
    "T17": "장비 변형: 위치 = x·y 각각 ±vary.pos 균등, 크기 = 각 축 ×(1 ± equipment_size) 균등 (바닥 고정, 평면 중심 기준), "
           "10 mm 격자. 통로·계단·케이싱·다른 장비와 겹치면 다시 뽑기 (최대 20 번), 끝내 겹치면 기준 위치·크기 그대로 "
           "(meta.template.fallback 에 기록)",
    "T18": "정비 공간 = 장비 수평 면 중 노즐 없는 첫 면 (west, east, south, north 순) 앞 깊이 800 × 장비 높이 × 면 폭; "
           "블록 밖은 자르고, 다른 장비·통로·구조와 겹치면 다음 면, 네 면 모두 안 되면 없음",
    "T19": "케이싱 정비 공간 (above ME_TOP, height_to ceiling) = ME_TOP 평면 × ME_TOP 윗면 … 구역 높이 접근 구역",
}

FACE_AXIS = {"west": (0, -1), "east": (0, 1), "south": (1, -1), "north": (1, 1)}
MAINT_D = 800
HANDRAIL_W = 50
MAX_TRIES = 20


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


# --------------------------------------------------------------------------- 구조 (변형 없음)

def _structure(t: dict, W: float, Lg: float, H: float) -> list:
    st = t.get("structure", {})
    out = []
    cl = st.get("ceiling_longitudinals")
    if cl:   # x 방향으로 뻗음, y 간격 (T2)
        h, w = cl["section"]
        w = _up(w)
        k = 1
        while k * cl["spacing_y"] + w <= Lg:
            y = k * cl["spacing_y"]
            out.append(Obstacle(f"CL_{k:03d}", "stiffener", _box((0, y, H - h), (W, y + w, H)), supportable=True))
            k += 1
    wf = st.get("ceiling_web_frames")
    if wf:   # y 방향으로 뻗음, x 간격 (T2·T4, D132)
        h, w = wf["section"]
        w = _up(w)
        fw, ft = wf.get("face_plate", (0, 0))
        attrs = wf.get("attributes", {})
        sup = bool(attrs.get("supportable", True))
        k = 1
        while k * wf["spacing_x"] + w <= W:
            x = k * wf["spacing_x"]
            out.append(Obstacle(f"WF_{k:03d}", "web_frame", _box((x, 0, H - h), (x + w, Lg, H)), supportable=sup))
            if fw:
                x0 = _dn(x + w / 2 - fw / 2)
                out.append(Obstacle(f"WFP_{k:03d}", "web_frame", _box((x0, 0, H - h - _up(ft)), (x0 + _up(fw), Lg, H - h)),
                                    supportable=sup))
            k += 1
    sf = st.get("side_shell_frames")
    if sf:   # 외판 쪽 수직 늑골 (T5)
        d, th = sf["section"]
        th = _up(th)
        assert d <= sf.get("band_y", d), "외판 늑골 깊이가 band_y 보다 깊다"
        for side in sf.get("sides", ["port", "stbd"]):
            y0, y1 = (0, d) if side == "port" else (Lg - d, Lg)
            k = 1
            while k * sf["spacing_x"] + th <= W:
                x = k * sf["spacing_x"]
                out.append(Obstacle(f"SF_{side[0].upper()}{k:03d}", "stiffener", _box((x, y0, 0), (x + th, y1, H)),
                                    supportable=True))
                k += 1
    pl = st.get("pillars")
    if pl:
        sx, sy = pl["size"]
        k = 0
        for x in pl["grid_x"]:
            for y in pl["grid_y"]:
                k += 1
                out.append(Obstacle(f"PIL_{k:02d}", "pillar", _box((_s(x - sx / 2), _s(y - sy / 2), 0),
                                                                     (_s(x - sx / 2) + sx, _s(y - sy / 2) + sy, H)),
                                    supportable=True))
    for b in st.get("bulkheads", []):   # 내부 격벽 = 판 (D113, T3)
        th = _up(b["thickness"])
        a = "xyz".index(b["axis"])
        lo, hi = [0.0, 0.0, 0.0], [W, Lg, H]
        c0 = _dn(b["at"] - th / 2)
        lo[a], hi[a] = c0, c0 + th
        q = 1 - a if a < 2 else 0
        span = b.get("span_y") if a == 0 else b.get("span_x")
        if span:
            lo[q], hi[q] = span
        out.append(Obstacle(b["id"], "bulkhead", _box(lo, hi), supportable=True, penetrable=True, penetration_axis=a))
    return out


# --------------------------------------------------------------------------- 케이싱·통로·개구부·트레이

def _rect_ring(x0, y0, x1, y1, w, z0, z1, pre: str, typ: str, access: bool, sup=False) -> list:
    """사각 고리 (안쪽 가장자리 x0..x1 × y0..y1, 폭 w 바깥쪽) — 상자 4개 (남·북 = 전체 길이, 서·동 = 사이)."""
    boxes = [((x0 - w, y0 - w), (x1 + w, y0)), ((x0 - w, y1), (x1 + w, y1 + w)),
             ((x0 - w, y0), (x0, y1)), ((x1, y0), (x1 + w, y1))]
    return [Obstacle(f"{pre}_{n}", typ, _box((a[0], a[1], z0), (b[0], b[1], z1)), supportable=sup, access=access)
            for n, (a, b) in zip("SNWE", boxes)]


def _casing(t, H):
    c = t.get("casing")
    if not c:
        return [], [], None
    (x0, y0), (x1, y1) = c["box"]
    th = _up(c["wall"]["thickness"])
    sup = bool(c["wall"].get("supportable", True))
    walls = _rect_ring(x0 + th, y0 + th, x1 - th, y1 - th, th, 0, H, c["id"], "casing", False, sup)
    contents = [Obstacle(e["id"], e.get("type", "equipment"), _box(*e["box"])) for e in c.get("contents", [])]
    acc = []
    m = c.get("maintenance")
    if m and m.get("access"):
        host = next(o for o in contents if o.id == m["above"])
        acc.append(Obstacle(f"ACC_{c['id']}", "maintenance",
                            _box((host.box.min[0], host.box.min[1], host.box.max[2]),
                                 (host.box.max[0], host.box.max[1], H)), access=True))
    return walls, contents + acc, (x0, y0, x1, y1)


def _door(b: dict, ring_outer) -> tuple:
    """문 = 격벽 위 고리 통로에 가장 가까운 점 (T8)."""
    rx0, ry0, rx1, ry1 = ring_outer
    a = "xyz".index(b["axis"])
    if a == 0:
        s0, s1 = b["span_y"]
        lo, hi = max(s0, ry0), min(s1, ry1)
        y = _s((lo + hi) / 2) if lo <= hi else (s1 if s1 < ry0 else s0)
        return float(b["at"]), float(y)
    s0, s1 = b["span_x"]
    lo, hi = max(s0, rx0), min(s1, rx1)
    x = _s((lo + hi) / 2) if lo <= hi else (s1 if s1 < rx0 else s0)
    return float(x), float(b["at"])


def _spoke_box(target, ring_outer, w):
    """대상 (점 또는 평면 상자) → 고리 바깥 가장자리 축 방향 곧은 상자 (T8). 반환 ((x0, y0), (x1, y1), 축)."""
    rx0, ry0, rx1, ry1 = ring_outer
    (tx0, ty0), (tx1, ty1) = target
    cx, cy = (tx0 + tx1) / 2, (ty0 + ty1) / 2
    hw = w / 2
    if rx0 <= cx <= rx1:   # y 방향
        x0, x1 = _s(cx - hw), _s(cx - hw) + w
        if cy < ry0:
            return (x0, ty1), (x1, ry0), 1
        return (x0, ry1), (x1, ty0), 1
    y0, y1 = _s(cy - hw), _s(cy - hw) + w
    if cx < rx0:
        return (tx1, y0), (rx0, y1), 0
    return (rx1, y0), (tx0, y1), 0


def _passages(t, casing_rect, H, bulkheads):
    p = t.get("passages", {})
    acc, paths, openings = [], {}, []
    ring_outer = None
    rg = p.get("ring")
    if rg and casing_rect:
        x0, y0, x1, y1 = casing_rect
        o, w, h = rg["offset"], rg["width"], rg["height"]
        ix0, iy0, ix1, iy1 = x0 - o - w, y0 - o - w, x1 + o + w, y1 + o + w   # 바깥 가장자리
        acc += _rect_ring(x0 - o, y0 - o, x1 + o, y1 + o, w, 0, h, "PASS_RING", "passage", True)
        ring_outer = (ix0, iy0, ix1, iy1)
        paths["ring"] = {"kind": "ring", "inner": (x0 - o, y0 - o, x1 + o, y1 + o), "w": w, "h": h}
    stairs = {s["id"]: s for s in p.get("stairs", [])}
    for s in p.get("stairs", []):
        acc.append(Obstacle(s["id"], "stair", _box((*s["box"][0], 0), (*s["box"][1], H)), access=True))
    doors = {}
    for b in bulkheads:
        tag = b["id"].split("_", 1)[1] if "_" in b["id"] else b["id"]
        doors[f"DOOR_{tag}"] = b
    for i, sp in enumerate(p.get("spokes", [])):
        tgt = sp["to"]
        if tgt in stairs:
            target = tuple(map(tuple, stairs[tgt]["box"]))
        else:
            dx, dy = _door(doors[tgt], ring_outer)
            target = ((dx, dy), (dx, dy))
        lo, hi, ax = _spoke_box(target, ring_outer, sp["width"])
        h = paths.get("ring", {}).get("h", 2000)
        acc.append(Obstacle(f"PASS_SPOKE_{i}", "passage", _box((*lo, 0), (*hi, h)), access=True))
        paths[f"spokes[{i}]"] = {"kind": "spoke", "lo": lo, "hi": hi, "axis": ax, "w": sp["width"]}
    for hch in p.get("hatches", []):
        (hx0, hy0), (hx1, hy1) = hch["box"]
        if hch.get("floor_opening"):
            openings.append({"id": hch["id"], "min": [hx0, hy0], "max": [hx1, hy1]})
        if hch.get("handrail"):
            acc += _rect_ring(hx0, hy0, hx1, hy1, HANDRAIL_W, 0, hch["handrail"], f"RAIL_{hch['id']}", "handrail", True)
    return acc, paths, openings, ring_outer


def _trays(t, paths, objs_by_id):
    out = []
    by_id = {}
    for tr in t.get("cable_trays", []):
        if tr.get("vertical"):
            continue
        p = paths[tr["follow"].split(".", 1)[1]]
        z1 = tr["z_top"]
        z0 = z1 - tr["height"]
        w = tr["width"]
        if p["kind"] == "ring":
            x0, y0, x1, y1 = p["inner"]
            pw = p["w"]
            if tr.get("side") == "outer":   # 고리 바깥쪽 가장자리에서 안쪽으로 폭 w (T11)
                ix0, iy0, ix1, iy1 = x0 - pw + w, y0 - pw + w, x1 + pw - w, y1 + pw - w
            else:
                ix0, iy0, ix1, iy1 = x0, y0, x1, y1
            boxes = _rect_ring(ix0, iy0, ix1, iy1, w, z0, z1, tr["id"], "tray", False)
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
        out += boxes
    for tr in t.get("cable_trays", []):
        if not tr.get("vertical"):
            continue
        src = by_id[tr["from"]]
        dst = objs_by_id[tr["to"]]
        cx, cy = (dst.box.min[0] + dst.box.max[0]) / 2, (dst.box.min[1] + dst.box.max[1]) / 2
        best = None
        for b in src:   # 트레이 상자 위 대상 중심에 가장 가까운 점 (T11)
            px = min(max(cx, b.box.min[0]), b.box.max[0])
            py = min(max(cy, b.box.min[1]), b.box.max[1])
            d = math.hypot(px - cx, py - cy)
            if best is None or d < best[0]:
                best = (d, px, py, b)
        _, px, py, b = best
        run_x = (b.box.max[0] - b.box.min[0]) >= (b.box.max[1] - b.box.min[1])   # 트레이 진행 방향
        w, dpt = tr["width"], tr["depth"]
        if run_x:
            x0 = _s(px - w / 2)
            y0 = _s(min(max(py - dpt / 2, b.box.min[1]), b.box.max[1] - dpt))
            lo, hi = (x0, y0), (x0 + w, y0 + dpt)
        else:
            y0 = _s(py - w / 2)
            x0 = _s(min(max(px - dpt / 2, b.box.min[0]), b.box.max[0] - dpt))
            lo, hi = (x0, y0), (x0 + dpt, y0 + w)
        out.append(Obstacle(tr["id"], "tray", _box((*lo, dst.box.max[2]), (*hi, b.box.min[2]))))
    return out


# --------------------------------------------------------------------------- 장비 변형

def _equipment(t, rng, H, avoid: list):
    var = t.get("variation", {})
    scale = float(var.get("equipment_size", 0.0))
    out, fallback = [], []
    for e in t.get("equipment", []):
        (x0, y0, z0), (x1, y1, z1) = e["box"]
        base = _box((x0, y0, z0), (x1, y1, z1))
        vp = e.get("vary", {}).get("pos", 0)
        box = None
        for _ in range(MAX_TRIES):
            dx = _s(rng.uniform(-vp, vp)) if vp else 0
            dy = _s(rng.uniform(-vp, vp)) if vp else 0
            sx = rng.uniform(1 - scale, 1 + scale) if scale and vp else 1.0
            sy = rng.uniform(1 - scale, 1 + scale) if scale and vp else 1.0
            sz = rng.uniform(1 - scale, 1 + scale) if scale and vp else 1.0
            cx, cy = (x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy
            wx, wy, wz = _s((x1 - x0) * sx), _s((y1 - y0) * sy), _s((z1 - z0) * sz)
            lx, ly = _s(cx - wx / 2), _s(cy - wy / 2)
            cand = _box((lx, ly, z0), (lx + wx, ly + wy, min(H, z0 + wz)))
            if all(c >= 0 for c in cand.min) and cand.max[0] <= t["zone"]["size"][0] and \
                    cand.max[1] <= t["zone"]["size"][1] and \
                    not any(_overlap(cand, a) for a in avoid + [o.box for o in out]):
                box = cand
                break
            if not vp:
                break
        if box is None:
            box = base
            fallback.append(e["id"])
        out.append(Obstacle(e["id"], e.get("kind", "equipment"), box))
    return out, fallback


# --------------------------------------------------------------------------- 단자

def _face_nozzles(host: Obstacle, face: str, entries: list, casing_rect, casing_ids, H):
    """같은 면의 노즐 n 개 위치 (T12, T6) — entries = [(배관 순번, which, r)]."""
    ax, side = FACE_AXIS[face]
    q = 1 - ax
    b = host.box
    if host.id in casing_ids and casing_rect:   # 케이싱 안 장비 → 그 면 쪽 케이싱 벽 바깥면 (T6)
        x0, y0, x1, y1 = casing_rect
        plane = (x0, y0, x1, y1)[(0 if side < 0 else 2) + ax]
    else:
        plane = b.max[ax] if side > 0 else b.min[ax]
    lo_q, hi_q = b.min[q], b.max[q]
    zc = _s((b.min[2] + min(b.max[2], H)) / 2)
    out = {}
    n = len(entries)
    for k, (idx, which, r) in enumerate(entries):
        pos = [0.0, 0.0, float(zc)]
        pos[ax] = plane + side * _up(r)
        pos[q] = float(_s(lo_q + (hi_q - lo_q) * (k + 1) / (n + 1)))
        nrm = [0, 0, 0]
        nrm[ax] = side
        d = tuple(nrm) if which == "start" else tuple(-c for c in nrm)
        out[(idx, which)] = Terminal(tuple(pos), "nozzle", d, host.id)
    return out


def _stub_of(pos, d, which, svc, size, kind, host=None):
    grav = pipe_spec.services()[svc].gravity
    cls, ins = pipe_spec.resolve(svc, None, None)
    t = Terminal(tuple(pos), kind, d, host)
    p = Pipe("tmp", "T5" if grav else "T1", size, grav, default_drain_slope(size) if grav else None, ins,
             t if which == "start" else Terminal((0.0, 0.0, 0.0), "nozzle", (1, 0, 0)),
             t if which == "end" else Terminal((0.0, 0.0, 0.0), "nozzle", (1, 0, 0)), service=svc)
    return t, terminal_stub(p, which)


def _clear(stub, r, solid, done, block, skip_axis=None) -> bool:
    from .scenario import inside_block_with_margin
    if not inside_block_with_margin(stub, block.extent, r, skip_axis=skip_axis):
        return False
    if any(box_distance(stub, o.box) < r - 1e-6 for o in solid):
        return False
    return not any(box_distance(stub, st) < r + rr - 1e-6 for st, rr in done)


def _offsets(rng_mm: float):
    n = int(rng_mm // SNAP_MM)
    pts = [(dx * SNAP_MM, dy * SNAP_MM) for dx in range(-n, n + 1) for dy in range(-n, n + 1)]
    return sorted(pts, key=lambda p: (p[0] * p[0] + p[1] * p[1], p[0], p[1]))


def _free_deck(x, y, z, d, which, svc, size, r, jit, solid, done, block, moved, tag):
    """데크 관통 단자: 템플릿 점에서 흔들기 범위 (±terminal_jitter) 안의 가장 가까운 빈 점 (T15) — 직진 구간이
    장애물·통로·다른 단자 직진 구간과 유효 반경 이상. 없으면 템플릿 점 그대로 (로더가 오류를 낸다)."""
    for dx, dy in _offsets(jit):
        t, st = _stub_of((x + dx, y + dy, z), d, which, svc, size, "boundary")
        if _clear(st, r, solid, done, block, skip_axis=2):
            if dx or dy:
                moved[tag] = round(math.hypot(dx, dy), 1)
            return t
    moved[tag] = None
    return _stub_of((x, y, z), d, which, svc, size, "boundary")[0]


def _free_bhd(o, a, side, which, svc, size, r, solid, done, block, moved, tag):
    """격벽 노즐형 단자 (T14): 격벽 면 가운데에서 가장 가까운 빈 점 (면 안, 10 mm 격자) — 가운데 높이 고정."""
    q = [k for k in range(2) if k != a][0]
    c = [float(_s((o.box.min[k] + o.box.max[k]) / 2)) for k in range(3)]
    span = o.box.max[q] - o.box.min[q]
    nrm = [0, 0, 0]
    nrm[a] = side
    d = tuple(nrm) if which == "start" else tuple(-v for v in nrm)
    steps = sorted(range(-int(span // 2 // SNAP_MM), int(span // 2 // SNAP_MM) + 1), key=lambda k: (abs(k), k))
    others = [s for s in solid if s.id != o.id]
    for k in steps:
        pos = list(c)
        pos[q] = c[q] + k * SNAP_MM
        if not (o.box.min[q] + r <= pos[q] <= o.box.max[q] - r):
            continue
        pos[a] = (o.box.max[a] if side > 0 else o.box.min[a]) + side * _up(r)
        t, st = _stub_of(pos, d, which, svc, size, "nozzle", o.id)
        if _clear(st, r, others, done, block):
            if k:
                moved[tag] = abs(k) * SNAP_MM
            return t
    moved[tag] = None
    pos = list(c)
    pos[a] = (o.box.max[a] if side > 0 else o.box.min[a]) + side * _up(r)
    return Terminal(tuple(pos), "nozzle", d, o.id)


def _parse_term(s: str):
    s = s.strip()
    if s.startswith("DECK_DN(") or s.startswith("DECK_UP("):
        x, y = (float(v) for v in s[8:-1].split(","))
        return ("deck", 0 if s.startswith("DECK_DN") else 1, x, y)
    if s.startswith("BHD("):
        return ("bhd", s[4:-1].strip())
    host, face = s.split(".")
    return ("nozzle", host, face)


def _radius(svc: str, size: str) -> float:
    cls, ins = pipe_spec.resolve(svc, None, None)
    return pipe_spec.spec(cls, size).od / 2 + ins + CLEARANCE_MM


def expand(t: dict, sha: str, seed: int, n_pipes=None, name: str = "") -> Scenario:
    rng = random.Random(seed)
    W, Lg, H = (float(v) for v in t["zone"]["size"])
    block = Block(W, Lg, H)
    var = t.get("variation", {})
    struct = _structure(t, W, Lg, H)
    walls, contents, casing_rect = _casing(t, H)
    bulk_specs = t.get("structure", {}).get("bulkheads", [])
    acc, paths, openings, ring_outer = _passages(t, casing_rect, H, bulk_specs)
    casing_ids = {o.id for o in contents}
    avoid = [o.box for o in acc] + [o.box for o in walls]
    if casing_rect:
        x0, y0, x1, y1 = casing_rect
        avoid.append(_box((x0, y0, 0), (x1, y1, H)))
    equip, fallback = _equipment(t, rng, H, avoid)
    objs = {o.id: o for o in equip + contents + struct}
    trays = _trays(t, paths, objs)

    # 배관 목록 (T16)
    base = list(t.get("pipes", []))
    if n_pipes is None:
        lo, hi = var.get("pipe_count", [len(base), len(base)])
        n_pipes = rng.randint(lo, hi)
    specs = base[:n_pipes]
    extra = []
    while len(specs) + len(extra) < n_pipes:
        extra.append(dict(base[rng.randrange(len(base))]))
    specs = specs + extra
    jit = float(var.get("terminal_jitter", 0))

    # 노즐 묶기: (장비, 면) → [(배관 순번, which, r)]
    rows = []
    face_groups = {}
    for i, ps in enumerate(specs):
        svc, size = ps["svc"], ps["size"]
        r = _radius(svc, size)
        ends = {}
        for which, key in (("start", "from"), ("end", "to")):
            ends[which] = _parse_term(ps[key])
            if ends[which][0] == "nozzle":
                face_groups.setdefault((ends[which][1], ends[which][2]), []).append((i, which, r))
        rows.append((ps, svc, size, r, ends))
    terms = {}
    for (host, face), ent in face_groups.items():
        terms.update(_face_nozzles(objs[host], face, ent, casing_rect, casing_ids, H))
    bhd = {o.id: o for o in struct if o.penetrable}
    solid = [o for o in struct + walls + contents + equip + acc + trays if not o.penetrable]
    done = []      # 이미 정한 단자 직진 구간 [(Box, r)]
    moved = {}     # 기준 좌표에서 옮긴 단자 {이름: mm}
    pipes = []
    for i, (ps, svc, size, r, ends) in enumerate(rows):
        tm = {}
        for which in ("start", "end"):
            e = ends[which]
            if e[0] == "nozzle":
                tm[which] = terms[(i, which)]
            elif e[0] == "deck":
                z = 0.0 if e[1] == 0 else H
                x, y = e[2], e[3]
                if jit and i >= len(base):   # 복제 배관만 흔들기 — 목록 배관은 템플릿 좌표 그대로 (T15)
                    x += _s(rng.uniform(-jit, jit))
                    y += _s(rng.uniform(-jit, jit))
                sign = 1 if z == 0 else -1   # 하부 데크 start = +z, 상부 데크 start = −z (D35)
                d = (0, 0, sign) if which == "start" else (0, 0, -sign)
                tm[which] = _free_deck(float(_s(x)), float(_s(y)), z, d, which, svc, size, r, jit, solid, done,
                                       block, moved, f"P{i + 1:03d}.{which}")
            else:   # 격벽 노즐형 단자 (T14)
                o = bhd[e[1]]
                a = o.pen_axis()
                other = tm.get("start") if which == "end" else None
                side = -1
                if other is not None and other.pos[a] > o.box.max[a]:
                    side = 1
                tm[which] = _free_bhd(o, a, side, which, svc, size, r, solid, done, block, moved,
                                      f"P{i + 1:03d}.{which}")
        cls, ins = pipe_spec.resolve(svc, None, None)
        grav = pipe_spec.services()[svc].gravity
        pipes.append(Pipe(f"P{i + 1:03d}", "T5" if grav else "T1", size, grav,
                          default_drain_slope(size) if grav else None, ins, tm["start"], tm["end"], service=svc))
        for which in ("start", "end"):
            done.append((terminal_stub(pipes[-1], which), r))

    # 정비 공간 (T18)
    noz_faces = {(h, f) for (h, f) in face_groups}
    maint = []
    solid_all = [o.box for o in equip + contents + struct + walls + acc + trays]
    for o in equip:
        for face in ("west", "east", "south", "north"):
            if (o.id, face) in noz_faces:
                continue
            ax, side = FACE_AXIS[face]
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
            if any(box_distance(b, st) < rr - 1e-6 for st, rr in done):   # 단자 직진 구간과 겹치지 않게 (D119)
                continue
            maint.append(Obstacle(f"MAINT_{o.id}", "maintenance", b, access=True))
            break

    obstacles = struct + walls + contents + equip + acc + maint + trays
    meta = {"name": name or f"{t['template']}_s{seed}_n{n_pipes}",
            "template": {"id": t["template"], "version": t.get("version"), "sha256": sha, "seed": seed,
                         "pipes": n_pipes, "interpretations": INTERP_VERSION, "fallback": fallback, "moved_terminals": moved}}
    sc = Scenario(block, obstacles, pipes, meta=meta)
    sc.floor_openings = openings
    return sc


def scenario_bytes(sc: Scenario) -> bytes:
    return (json.dumps(to_dict(sc), ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("template")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--pipes", type=int, default=None, help="배관 수 (없으면 variation.pipe_count 범위에서 시드로)")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--name", default="")
    a = ap.parse_args(argv)
    t, sha = load_template(a.template)
    sc = expand(t, sha, a.seed, a.pipes, a.name)
    errs, warns = validate(sc)
    data = scenario_bytes(sc)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_bytes(data)
    print(json.dumps({"out": a.out, "template_sha256": sha, "scenario_sha256": hashlib.sha256(data).hexdigest(),
                      "obstacles": len(sc.obstacles), "pipes": len(sc.pipes), "errors": len(errs),
                      "warnings": len(warns), "fallback": sc.meta["template"]["fallback"]}, ensure_ascii=False))
    for e in errs[:40]:
        print("ERR", e)
    for w in warns[:40]:
        print("WARN", w)
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
