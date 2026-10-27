"""서포트 힌트 — 조회 전용 (S1-7b, D100). 검증기 채점과 같은 함수를 부른다.

form_at     — 한 위치에서 검증기가 놓을 형식 (D80 ③: `modules.support_candidates` → `support_std.evaluate`)
groupable   — 두 C 서포트가 D 로 묶일 수 있는가 (D82 조건, `support_group._try`)
together_at — A·B·C 가 모두 불가할 때 함께 받칠 D (D89, `support_group.together_at`)

결과 불변: 힌트는 Context(배관 순서 = 시나리오 순서, 놓인 배관 = 주어진 경로) 를 따로 만들어 부를 뿐 검증기·Session 상태를
바꾸지 않는다. 같은 경로 집합에서 검증기가 실제로 놓은 서포트는 힌트 답과 같다 (`tests/test_s1_hint.py`).
힌트가 보지 않는 것: 탐욕 배치 순서(D80 ⑤ — 어느 위치에 놓을지는 간격·도달성으로 정해진다), D82 보완 간격 재확인
(묶인 서포트를 가로대 위치로 옮긴 뒤 앞뒤 간격 — 그 배관의 서포트 목록 전체가 있어야 한다), 묶음 좌표 순서 탐욕.
"""
import numpy as np

from .. import support_std as ss
from .core import Context, PipeRoute, centerlines_of



class SupportHint:
    """SupportHint(scenario, routes) — routes = {pipe_id: waypoints} 또는 [PipeRoute]. 그 경로들이 놓인 상태에서 묻는다.
    묻는 배관 자신의 경로는 없어도 된다 (계획 중인 경로 위 점을 물을 때) — 검증기도 자기 배관은 '다른 배관' 에서 뺀다."""

    def __init__(self, scenario, routes=(), std=None):
        from . import modules as _  # noqa: F401  (모듈 등록)
        self.scenario = scenario
        self.std = std or ss.load_standard()
        self.pipes = {p.id: p for p in scenario.pipes}
        if isinstance(routes, dict):
            routes = [PipeRoute(k, v) for k, v in routes.items()]
        given = {r.pipe_id: r for r in routes if r.waypoints}
        rmap = {pid: given[pid] for pid in self.pipes if pid in given}   # 시나리오 순서 (검증기와 같음)
        cls = {pid: centerlines_of(self.pipes[pid], r) for pid, r in rmap.items()}
        self.ctx = Context(scenario, self.pipes, rmap, cls, std=self.std)

    @classmethod
    def from_session(cls, session):
        """증분 검증 Session 의 지금 배치로 만든다 (Session 은 바꾸지 않는다)."""
        return cls(session.scenario, [session.routes[p] for p in session.order if p in session.routes], session.std)

    # ---------------------------------------------------------------- form_at
    def form_at(self, pipe_id: str, pos, axis) -> dict:
        """위치 pos (배관 중심선 위 점), 배관 축 axis 에서 검증기가 고르는 서포트 (D80 ③④⑧) 와 (면, 형식) 조합별 사유.
        반환 {"chosen": 서포트 dict (검증기 supports[] 항목과 같은 필드) 또는 None,
              "options": [{"face", "form", "L", "status": "ok"|"out_of_range"|"blocked"|"parallel", "kg"}]}"""
        from .modules import support_candidates
        P = np.asarray(pos, dtype=float)[None]
        U = np.asarray(axis, dtype=float)[None]
        chosen = support_candidates(self.ctx, pipe_id, P, U)[0]
        return {"chosen": chosen, "options": self._options(pipe_id, P, U)}

    def _options(self, pid, P, U):
        from .. import blockpair as bp
        pipe = self.pipes[pid]
        ext = np.array(self.scenario.block.extent, dtype=float)
        oA, oB, oR, oS, _ = self.ctx.others(pid)
        holes = bp.hole_boxes(self.scenario)
        box_lo = [o.box.min for o in self.scenario.obstacles] + [h[0] for h in holes]
        box_hi = [o.box.max for o in self.scenario.obstacles] + [h[1] for h in holes]
        sp = bp.splits(self.scenario)
        dist = ss.face_distances(ext, P, U, sp)
        bounds = bp.local_bounds(ext, P, sp) if sp else None
        out = []
        for o, (face, frm) in enumerate(ss.OPTIONS):
            L = float(dist[0, ss.OPT_FACE[o]])
            row = {"face": face, "form": frm, "L": round(L, 1) if np.isfinite(L) else None, "kg": None}
            if not np.isfinite(L):
                row["status"] = "parallel"          # 배관 축과 평행한 법선의 면 (D80 ③)
                out.append(row)
                continue
            kg, k, rf = ss.form_weight(self.std, frm, pipe.nominal_size, pipe.od, np.array([L]))
            if not np.isfinite(kg[0]):
                row["status"] = "out_of_range"      # 표 범위 밖 (D79 ③)
                out.append(row)
                continue
            A, B, v = ss.member_arrays(frm, P, U, face, np.array([L]), pipe.od, rf)
            half = ss.ANGLE_HALF[self.std.angle_of(frm, pipe.nominal_size, int(k[0]))]
            bl = ss.members_blocked(A, B, v, float(half), ext, box_lo, box_hi, (oA, oB, oR, oS), True,
                                    bounds=bounds)[0]
            row["kg"] = round(float(kg[0]), 4)
            row["status"] = "blocked" if bl else "ok"
            out.append(row)
        return out

    # ---------------------------------------------------------------- groupable
    def groupable(self, a: tuple, b: tuple, face: str = None) -> dict:
        """두 서포트 위치 a = (pipe_id, pos, axis), b = 같은 형식 — 둘 다 C 로 놓인다고 볼 때 D 로 묶일 수 있는가 (D82).
        face = "floor"|"ceiling" (None 이면 둘 다 보고 묶을 수 있는 가벼운 쪽, 천장 선호 D79 ④).
        반환 {"ok", "face", "reason", "kg" (묶음), "kg_individual" (각자 C 무게 합), "L1", "L2", "basis_nd", "section"}.
        간격 재확인 (D82 보완) 은 보지 않는다 — 서포트 목록 전체가 있어야 해서, 검증기만 한다."""
        from . import support_group as sg
        (pa, xa, ua), (pb, xb, ub) = a, b
        xa, xb = np.asarray(xa, float), np.asarray(xb, float)
        ua, ub = np.asarray(ua, float), np.asarray(ub, float)
        res = {"ok": False, "face": None, "reason": None, "spacing_recheck": "not_checked"}   # D82 보완은 검증기만
        if pa == pb:
            return {**res, "reason": "같은 배관"}
        ax = int(np.argmax(np.abs(ua)))
        if abs(ua[2]) > 1e-9 or abs(ub[2]) > 1e-9 or abs(abs(ua[ax]) - 1) > 1e-9 or abs(abs(ub[ax]) - 1) > 1e-9 \
                or int(np.argmax(np.abs(ub))) != ax:
            return {**res, "reason": "축 정렬 수평 평행 아님"}
        if abs(xa[ax] - xb[ax]) > sg.WINDOW_MM:
            return {**res, "reason": f"축 방향 위치 차 > ±{sg.WINDOW_MM:.0f} mm"}
        faces = [face] if face else ["ceiling", "floor"]
        tried = {}
        for f in faces:
            grp, kg_ind = [], 0.0
            ok_c = True
            for pid, x, u in ((pa, xa, ua), (pb, xb, ub)):
                od = self.pipes[pid].od
                surf = x[2] - od / 2 if f == "floor" else x[2] + od / 2
                c = self.form_at(pid, x, u)
                opt = next(o for o in c["options"] if o["face"] == f and o["form"] == "C")
                if opt["status"] != "ok":
                    ok_c = False
                kg_ind += opt["kg"] or 0.0
                grp.append({"pid": pid, "ax": ax, "p": x, "od": od, "surf": surf,
                            "nd": self.pipes[pid].nominal_size})
            if abs(grp[0]["surf"] - grp[1]["surf"]) > sg.SURFACE_TOL_MM:
                tried[f] = {"reason": f"지지면 쪽 표면 높이 차 > {sg.SURFACE_TOL_MM:.0f} mm"}
                continue
            if not ok_c:
                tried[f] = {"reason": "C 형식으로 놓이지 않는 위치 (표 범위 밖·부재 간섭)"}
                continue
            grp.sort(key=lambda g: (round(g["p"][1 - ax if ax < 2 else 0], 3), g["pid"]))
            r = sg._try(self.ctx, grp, f)
            if r is None:
                tried[f] = {"reason": "L1·L2 표 범위 밖 또는 가로대·다리 간섭"}
                continue
            kg, ang, mem, L1, L2, basis = r
            if not kg < kg_ind - 1e-9:
                tried[f] = {"reason": "묶은 무게 ≥ 개별 무게 합", "kg": round(kg, 4), "kg_individual": round(kg_ind, 4)}
                continue
            tried[f] = {"ok": True, "kg": round(kg, 4), "kg_individual": round(kg_ind, 4), "L1": round(L1, 1),
                        "L2": round(L2, 1), "basis_nd": basis, "section": ang}
        good = {f: v for f, v in tried.items() if v.get("ok")}
        if not good:
            f0 = faces[0]
            return {**res, **tried.get(f0, {}), "face": f0, "ok": False, "by_face": tried}
        kc, kf = good.get("ceiling", {}).get("kg"), good.get("floor", {}).get("kg")
        f = "ceiling" if kc is not None and (kf is None or kc <= kf * ss.CEILING_PREFER) else "floor"
        return {**res, **good[f], "face": f, "ok": True, "by_face": tried}

    # ---------------------------------------------------------------- together_at
    def together_at(self, pipe_id: str, pos, axis) -> dict:
        """D89: 위치 pos 에서 나란한 배관 하나를 함께 얹는 D 중 가장 가벼운 것 (없으면 None). 검증기는 A·B·C 가 모두
        불가한 위치에서만 이 답을 쓴다 — `form_at(...)["chosen"]` 이 None 일 때."""
        from . import support_group as sg
        return sg.together_at(self.ctx, pipe_id, np.asarray(pos, float), np.asarray(axis, float), 0.0)
