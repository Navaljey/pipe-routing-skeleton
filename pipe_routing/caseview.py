"""8단계: 간섭-차단 실패 사례 3D 보기 (D51·D58).

최종 배치 경로 + 실패 배관의 단독 경로(빨간 점선) + 그 단독 경로와 충돌하는 놓인 배관(굵게)을 그리고,
가장 가까운 지점 사이를 선으로 이어 실제 거리와 필요 이격(r₁ + r₂)을 표시한다.
"실제로 공간이 없는지" vs "격자선이 없어 나란히 못 가는지"를 눈으로 구분하는 용도.

python -m pipe_routing.caseview <scenario.json> <output.json> <pipe_id> -o out/case.html [--png]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from .layered import LayeredGraph, _route, clear_cache
from .multi import _pieces, routes_conflict
from .router_astar import RouteResult
from .scenario import load
from .verifier.geom import segment_segment_distance
from .viz import add_pipe_tubes, add_routes, scenario_figure, write_html


def closest(p1, w1, p2, w2):
    A1, B1, S1 = _pieces(p1, w1)
    A2, B2, S2 = _pieces(p2, w2)
    d, c1, c2 = segment_segment_distance(A1[:, None], B1[:, None], A2[None], B2[None])   # 거리, 양쪽 최근접점
    d = d - S1[:, None] - S2[None]
    i, j = np.unravel_index(np.argmin(d), d.shape)
    return float(d[i, j]), c1[i, j], c2[i, j]


def case_figure(sc, out: dict, pid: str, margin_mm: float = 4000.0, close: bool = False):
    """close=True 면 최근접 지점 주변(± margin)만, 아니면 단자·최근접 지점 전체를 화면에 담는다."""
    pipes = {p.id: p for p in sc.pipes}
    p = pipes[pid]
    routes = {x["pipe_id"]: x["waypoints"] for x in out["routes"] if x["waypoints"]}
    clear_cache()
    solo = _route(LayeredGraph(sc, p), p, 60)
    blockers = [q for q in routes if q != pid and solo.status == "ok"
                and routes_conflict(p, solo.waypoints, pipes[q], routes[q])]
    fig = scenario_figure(sc, connect=False)
    add_routes(fig, sc, [RouteResult(q, "ok", waypoints=w) for q, w in routes.items() if q not in blockers], line_width=2)
    # 실제 굵기 관 (§0 보고 규칙): 놓인 배관 전부 + 실패 배관 단독 경로(빨강). 유효 반경 관은 범례에서 켠다
    tubes = dict(routes)
    if solo.status == "ok":
        tubes[pid] = solo.waypoints
    add_pipe_tubes(fig, sc, tubes, colors={**{q: ("black" if q in blockers else "#888") for q in routes}, pid: "red"})
    pts = [np.array(p.start.pos), np.array(p.end.pos)]
    notes = []
    for q in blockers:
        x, y, z = zip(*routes[q])
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", name=f"{q} {pipes[q].nominal_size} (막는 배관)",
                                   line=dict(color="black", width=12)))
        d, q1, q2 = closest(p, solo.waypoints, pipes[q], routes[q])
        need = p.radius + pipes[q].radius
        fig.add_trace(go.Scatter3d(x=[q1[0], q2[0]], y=[q1[1], q2[1]], z=[q1[2], q2[2]], mode="lines+markers",
                                   name=f"최근접 {d:.0f} mm < 필요 {need:.0f} mm", line=dict(color="red", width=5)))
        notes.append(f"{q}: {d:.0f}/{need:.0f} mm")
        pts += [q1, q2]
    if solo.status == "ok":
        x, y, z = zip(*solo.waypoints)
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines+markers", name=f"{pid} {p.nominal_size} 단독 경로",
                                   line=dict(color="red", width=8, dash="dash"), marker=dict(size=3, color="red")))
    if close and len(pts) > 2:
        pts = pts[2:]
    P = np.array(pts)
    lo, hi = P.min(0) - margin_mm, P.max(0) + margin_mm
    ext = sc.block.extent
    rng = [[max(0, lo[k]), min(ext[k], hi[k])] for k in range(3)]
    fig.update_layout(
        title=dict(text=f"{out['scenario']} {pid} ({p.nominal_size}) 간섭 — " + ", ".join(notes), font=dict(size=14)),
        scene=dict(xaxis=dict(range=rng[0], autorange=False), yaxis=dict(range=rng[1], autorange=False),
                   zaxis=dict(range=rng[2], autorange=False), aspectmode="manual",
                   aspectratio=dict(zip("xyz", [(r[1] - r[0]) / max(rr[1] - rr[0] for rr in rng) * 1.6 for r in rng])),
                   camera=dict(eye=dict(x=1.3, y=-1.5, z=1.0))))
    for tr in fig.data:   # 장애물 면은 흐리게 — 배관이 보이도록
        if isinstance(tr, go.Mesh3d):
            tr.opacity = min(tr.opacity or 1.0, 0.15)
    return fig, blockers


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="간섭 실패 사례 3D")
    ap.add_argument("scenario")
    ap.add_argument("output")
    ap.add_argument("pipe")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--png", action="store_true", help="같은 이름 PNG 도 저장 (playwright 필요)")
    args = ap.parse_args(argv)
    sc = load(args.scenario)
    out = json.loads(Path(args.output).read_text(encoding="utf-8"))
    outs = []
    for close in (False, True):
        fig, blockers = case_figure(sc, out, args.pipe, margin_mm=2500.0 if close else 4000.0, close=close)
        path = args.out if not close else str(Path(args.out).with_name(Path(args.out).stem + "_close.html"))
        write_html(fig, path, offline=args.png)
        outs.append(path)
    print(f"{args.pipe}: 막는 배관 {blockers} → {outs}")
    for path in (outs if args.png else []):
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            import os
            exe = os.environ.get("CHROMIUM_PATH", "/opt/pw-browsers/chromium")
            b = pw.chromium.launch(executable_path=exe if os.path.exists(exe) else None,
                                   args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader"])
            pg = b.new_page(viewport={"width": 1400, "height": 900})
            pg.goto("file://" + str(Path(path).resolve()))
            pg.wait_for_timeout(2500)
            pg.screenshot(path=str(Path(path).with_suffix(".png")))
            b.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
