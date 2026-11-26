"""템플릿 v2 시각화용 펼치기 + 점검 — 공식 로더(S2-T)가 아니다."""
import json, math, re, sys, yaml
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from pipe_routing.constants import PIPE_SPECS, effective_radius, elbow_tangent

T = yaml.safe_load(open(sys.argv[1]))
X, Y, H = T["zone"]["size"]
DEF = T["attribute_defaults"]; OVR = T.get("attribute_overrides") or {}
boxes = []
CATKO = {"ceiling_longitudinals": "종늑골", "ceiling_web_frames": "트랜스 웹", "side_shell_frames": "외판 수직 웹",
         "pillars": "필라", "bulkheads": "내부 격벽", "casing_wall": "케이싱 벽", "casing_contents": "케이싱 안 장비",
         "stairs": "계단", "equipment": "장비", "cable_trays": "케이블 트레이", "passages": "통로",
         "maintenance": "정비 공간", "handrails": "난간", "floor_openings": "바닥 개구부"}
def box(key, id_, mn, mx, **kw):
    boxes.append({"key": key, "cat": CATKO[key], "id": id_, "min": [round(float(v), 1) for v in mn],
                  "max": [round(float(v), 1) for v in mx], **kw})

S = T["structure"]
L = S["ceiling_longitudinals"]; h, w = L["section"]; k = 1
while L["spacing_y"] * k < Y:
    y = L["spacing_y"] * k; box("ceiling_longitudinals", f"LONG_{k:02d}", [0, y - w / 2, H - h], [X, y + w / 2, H]); k += 1
W = S["ceiling_web_frames"]; d, t = W["section"]; fw, ft = W["face_plate"]; k = 1
while W["spacing_x"] * k < X:
    x = W["spacing_x"] * k
    box("ceiling_web_frames", f"WEB_{k:02d}", [x - t / 2, 0, H - d], [x + t / 2, Y, H])
    box("ceiling_web_frames", f"WEB_{k:02d}_FP", [x - fw / 2, 0, H - d - ft], [x + fw / 2, Y, H - d]); k += 1
F = S["side_shell_frames"]; dep, th = F["section"]; k = 1
while F["spacing_x"] * k < X:
    x = F["spacing_x"] * k
    box("side_shell_frames", f"SF_P_{k:02d}", [x - th / 2, 0, 0], [x + th / 2, dep, H])
    box("side_shell_frames", f"SF_S_{k:02d}", [x - th / 2, Y - dep, 0], [x + th / 2, Y, H]); k += 1
P = S["pillars"]; ps = P["size"]
for gx in P["grid_x"]:
    for gy in P["grid_y"]:
        box("pillars", f"PIL_{gx}_{gy}", [gx - ps[0] / 2, gy - ps[1] / 2, 0], [gx + ps[0] / 2, gy + ps[1] / 2, H])
for b in S["bulkheads"]:
    tt = b["thickness"]
    if b["axis"] == "x": box("bulkheads", b["id"], [b["at"] - tt / 2, b["span_y"][0], 0], [b["at"] + tt / 2, b["span_y"][1], H])
    else: box("bulkheads", b["id"], [b["span_x"][0], b["at"] - tt / 2, 0], [b["span_x"][1], b["at"] + tt / 2, H])
C = T["casing"]; (cx0, cy0), (cx1, cy1) = C["box"]; ct = C["wall_thickness"]
box("casing_wall", "CAS_S", [cx0, cy0 - ct, 0], [cx1, cy0, H]); box("casing_wall", "CAS_N", [cx0, cy1, 0], [cx1, cy1 + ct, H])
box("casing_wall", "CAS_W", [cx0 - ct, cy0, 0], [cx0, cy1, H]); box("casing_wall", "CAS_E", [cx1, cy0, 0], [cx1 + ct, cy1, H])
eq = {}
for c in C["contents"]:
    box("casing_contents", c["id"], c["box"][0], c["box"][1]); eq[c["id"]] = c
me = eq["ME_TOP"]["box"]; box("maintenance", "MAINT_ME_TOP", [me[0][0], me[0][1], me[1][2]], [me[1][0], me[1][1], H])
R = T["passages"]["ring"]; o, wd, hh = R["offset"], R["width"], R["height"]
ix0, iy0, ix1, iy1 = cx0 - o - wd, cy0 - o - wd, cx1 + o + wd, cy1 + o + wd
box("passages", "RING_S", [ix0, iy0, 0], [ix1, iy0 + wd, hh]); box("passages", "RING_N", [ix0, iy1 - wd, 0], [ix1, iy1, hh])
box("passages", "RING_W", [ix0, iy0, 0], [ix0 + wd, iy1, hh]); box("passages", "RING_E", [ix1 - wd, iy0, 0], [ix1, iy1, hh])
tgt = {}
for s in T["passages"]["stairs"]:
    (a0, b0), (a1, b1) = s["box"]; box("stairs", s["id"], [a0, b0, 0], [a1, b1, H]); tgt[s["id"]] = ((a0 + a1) / 2, (b0 + b1) / 2)
for dd in T["passages"]["doors"]: tgt[dd["id"]] = tuple(dd["at"])
spk = []
for i, s in enumerate(T["passages"]["spokes"]):
    tx, ty = tgt[s["to"]]; wv = s["width"]
    if ty < iy0: box("passages", f"SPOKE_{i}", [tx - wv / 2, ty, 0], [tx + wv / 2, iy0, hh]); spk.append(("y", tx, ty, iy0))
    elif ty > iy1: box("passages", f"SPOKE_{i}", [tx - wv / 2, iy1, 0], [tx + wv / 2, ty, hh]); spk.append(("y", tx, iy1, ty))
    elif tx < ix0: box("passages", f"SPOKE_{i}", [tx, ty - wv / 2, 0], [ix0, ty + wv / 2, hh]); spk.append(("x", ty, tx, ix0))
    else: box("passages", f"SPOKE_{i}", [ix1, ty - wv / 2, 0], [tx, ty + wv / 2, hh]); spk.append(("x", ty, ix1, tx))
for hc in T["passages"]["hatches"]:
    (a0, b0), (a1, b1) = hc["box"]; box("floor_openings", hc["id"], [a0, b0, 0], [a1, b1, 20])
    r_, e = hc["handrail"]["height"], hc["handrail"]["width"]
    for nm, mn, mx in [("S", [a0, b0 - e, 0], [a1, b0, r_]), ("N", [a0, b1, 0], [a1, b1 + e, r_]),
                       ("W", [a0 - e, b0, 0], [a0, b1, r_]), ("E", [a1, b0, 0], [a1 + e, b1, r_])]:
        box("handrails", f"{hc['id']}_RAIL_{nm}", mn, mx)
for tr in T["cable_trays"]:
    if "drop_from" in tr:
        ax_, ay_ = tr["at"]; box("cable_trays", tr["id"], [ax_ - tr["width"] / 2, ay_ - tr["depth"] / 2, tr["to_z"]], [ax_ + tr["width"] / 2, ay_ + tr["depth"] / 2, 4300]); continue
    zt, th_, wid = tr["z_top"], tr["height"], tr["width"]
    if tr["follow"] == "passages.ring":
        box("cable_trays", tr["id"] + "_S", [ix0, iy0, zt - th_], [ix1, iy0 + wid, zt]); box("cable_trays", tr["id"] + "_N", [ix0, iy1 - wid, zt - th_], [ix1, iy1, zt])
        box("cable_trays", tr["id"] + "_W", [ix0, iy0, zt - th_], [ix0 + wid, iy1, zt]); box("cable_trays", tr["id"] + "_E", [ix1 - wid, iy0, zt - th_], [ix1, iy1, zt])
    else:
        i = int(re.search(r"\[(\d+)\]", tr["follow"]).group(1)); ax, c, a, b = spk[i]
        if ax == "y": box("cable_trays", tr["id"], [c - wid / 2, min(a, b), zt - th_], [c + wid / 2, max(a, b), zt])
        else: box("cable_trays", tr["id"], [min(a, b), c - wid / 2, zt - th_], [max(a, b), c + wid / 2, zt])
for e_ in T["equipment"]:
    box("equipment", e_["id"], e_["box"][0], e_["box"][1]); eq[e_["id"]] = e_

# 성격
for b in boxes:
    a = {"supportable": False, "penetrable": False, "access": False, "penetration_axis": None}
    a.update(DEF.get(b["key"], {})); a.update(OVR.get(b["id"], {})); b["attrs"] = a

# 노즐
NORM = {"east": (0, 1), "west": (0, -1), "north": (1, 1), "south": (1, -1), "top": (2, 1), "bottom": (2, -1)}
def nozzle(eid, nn):
    e = eq[eid]; (a, b) = e["box"]; nz = e["nozzles"][nn]; ax, sg = NORM[nz["face"]]
    p = [0.0, 0.0, 0.0]; p[ax] = b[ax] if sg > 0 else a[ax]
    if ax == 2: p[0], p[1] = a[0] + nz["at"][0], a[1] + nz["at"][1]
    else:
        hax = 1 if ax == 0 else 0; p[hax] = a[hax] + nz["at"][0]; p[2] = a[2] + nz["at"][1]
    dvec = [0, 0, 0]; dvec[ax] = sg
    return p, dvec, nz["size"], eid
def term(s):
    m = re.match(r"(DECK_DN|DECK_UP)\((\d+),\s*(\d+)\)", s)
    if m:
        up = m.group(1) == "DECK_UP"; return [float(m.group(2)), float(m.group(3)), float(H) if up else 0.0], [0, 0, -1 if up else 1], None, m.group(1)
    m = re.match(r"BHD\((\w+),\s*(\d+),\s*(\d+)\)$", s)     # D136 ③: BHD(격벽 id, u, z)
    if m:
        bh = next(b for b in S["bulkheads"] if b["id"] == m.group(1)); u, z = float(m.group(2)), float(m.group(3))
        c = [bh["at"], u, z] if bh["axis"] == "x" else [u, bh["at"], z]
        return c, "BHD", None, bh["id"]           # dir 는 배관의 다른 끝 쪽으로 (아래에서 정함)
    if s.startswith("BHD(") or "." not in s: raise SystemExit(f"단자 표기 오류 (D136 ③): {s}")
    eid, nn = s.split("."); return nozzle(eid, nn)
systems, issues = [], []
for i, p in enumerate(T["pipes"]):
    runs = [{"role": "주관", "size": p["size"], "from": p["from"], "to": p["to"]}]
    for j, lg in enumerate(p.get("legs", [])):
        runs.append({"role": f"다리 {j+1} ({'합류' if 'from' in lg else '갈래'})", "size": lg["size"], "from": lg.get("from", "(티)"), "to": lg.get("to", "(티)")})
    sys_ = {"no": i + 1, "svc": p["svc"], "runs": runs, "terms": []}
    for rr in runs:
        for side in ("from", "to"):
            s = rr[side]
            if s == "(티)": continue
            pos, dvec, nsize, host = term(s)
            if dvec == "BHD":   # 격벽 관통: 직진 구간은 배관이 오는 쪽 (이 줄의 다른 끝 쪽) — 편집기 해석, 로더 규칙 아님
                bh = next(b for b in S["bulkheads"] if b["id"] == host); ax = 0 if bh["axis"] == "x" else 1
                other = rr["to" if side == "from" else "from"]
                if other == "(티)": other = p["from"] if side == "to" else p["to"]
                op = term(other)[0]; dvec = [0, 0, 0]; dvec[ax] = 1 if op[ax] > bh["at"] else -1
            if nsize and nsize != rr["size"]: issues.append(f"계통 {i+1} {s}: 노즐 구경 {nsize} ≠ 배관 {rr['size']}")
            sys_["terms"].append({"name": s, "pos": pos, "dir": dvec, "size": rr["size"], "host": host, "role": rr["role"], "side": side})
    systems.append(sys_)

# 점검 ①: 겹침
solid = {"equipment", "casing_contents", "pillars", "stairs", "bulkheads", "casing_wall"}
skip = lambda a, b: ({a["key"], b["key"]} <= {"side_shell_frames", "ceiling_longitudinals", "ceiling_web_frames", "pillars", "bulkheads", "casing_wall"}
    or {a["key"], b["key"]} <= {"casing_contents", "casing_wall"} or "SPOKE" in a["id"] + b["id"]
    or (("stairs" in (a["key"], b["key"]) or "UPTAKE" in a["id"] + b["id"]) and {a["key"], b["key"]} & {"ceiling_longitudinals", "ceiling_web_frames"})
    or {a["key"], b["key"]} <= {"passages", "cable_trays", "handrails", "floor_openings", "maintenance"})
def ov(a, b): return all(a["min"][i] < b["max"][i] - 1e-6 and b["min"][i] < a["max"][i] - 1e-6 for i in range(3))
clash = []
for i, a in enumerate(boxes):
    for b in boxes[i + 1:]:
        if not (a["key"] in solid or b["key"] in solid): continue
        if skip(a, b): continue
        if ov(a, b): clash.append([a["cat"], a["id"], b["cat"], b["id"]])
# 점검 ②: 단자 직진 구간 (노즐 = max(§3.3, t90), 데크 = max(3D, r) + t90) 이 다른 박스와 r 이상
def segbox(p, q, b):
    best = 1e18
    for k in range(41):
        u = k / 40; x = [p[i] + (q[i] - p[i]) * u for i in range(3)]
        dd = math.sqrt(sum(max(b["min"][i] - x[i], 0, x[i] - b["max"][i]) ** 2 for i in range(3))); best = min(best, dd)
    return best
for s_ in systems:
    for t_ in s_["terms"]:
        if t_["dir"] is None: continue
        sz = t_["size"]; r = effective_radius(sz); D = int(sz[:-1])
        thru = t_["host"].startswith("DECK") or t_["name"].startswith("BHD(")   # 데크·격벽 관통 = max(3D, r) + t90 (D136 ⑦)
        L = (max(3 * D, r) + elbow_tangent(sz, 90)) if thru else max(PIPE_SPECS[sz].min_straight, elbow_tangent(sz, 90))
        q = [t_["pos"][i] + t_["dir"][i] * L for i in range(3)]
        for b in boxes:
            if b["id"] == t_["host"] or b["key"] in ("floor_openings",): continue
            dd = segbox(t_["pos"], q, b)
            lim = 0 if b["attrs"]["access"] else r
            if dd < lim - 1e-6 or (b["attrs"]["access"] and dd <= 0):
                issues.append(f"계통 {s_['no']} {s_['svc']} {t_['name']} ({sz}): 단자 직진 {L:.0f} mm 가 {b['id']} ({b['cat']}) 와 {dd:.0f} mm (< {lim:.0f})")
json.dump({"zone": [X, Y, H], "boxes": boxes, "systems": systems, "clash": clash, "issues": issues,
           "defaults": DEF, "overrides": OVR, "catko": CATKO}, open(sys.argv[2], "w"), ensure_ascii=False)
print(len(boxes), "boxes", len(systems), "systems", sum(len(s["runs"]) for s in systems), "runs", len(clash), "clash", len(issues), "issues")
for c in clash: print("CLASH", c)
for x in issues: print("ISSUE", x)
