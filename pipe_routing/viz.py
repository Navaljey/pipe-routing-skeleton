"""3D 시각화 (2단계, D36). plotly → 독립 HTML.

레이어 함수(add_*)가 Figure 에 trace 를 더하는 구조다. 3단계 이후 그래프·경로·위반 위치도
같은 방식으로 레이어를 추가한다.
"""
import argparse
from pathlib import Path

import plotly.graph_objects as go

from .constants import PIPE_SPECS
from .geometry import Box
from .scenario import Scenario, load, terminal_stub

# 블록 큐브의 12개 모서리 (꼭짓점 인덱스, 아래 _corners 순서)
_EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
          (0, 4), (1, 5), (2, 6), (3, 7)]
# 박스 6면 → 삼각형 12개
_TRIS = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
         (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
ARROW_MM = 1500   # 단자 방향 화살표 길이 (표시 전용)
_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
            "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]


def _corners(b: Box) -> list[tuple[float, float, float]]:
    return [(x, y, z) for x in (b.min[0], b.max[0]) for y in (b.min[1], b.max[1])
            for z in (b.min[2], b.max[2])]


def _edge_lines(boxes: list[Box]) -> tuple[list, list, list]:
    """박스 모서리를 None 으로 끊은 선 좌표 하나로 합친다 (trace 수 절약)."""
    xs, ys, zs = [], [], []
    for b in boxes:
        c = _corners(b)
        for i, j in _EDGES:
            for p in (c[i], c[j], (None, None, None)):
                xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
    return xs, ys, zs


def add_block(fig: go.Figure, sc: Scenario) -> None:
    xs, ys, zs = _edge_lines([Box((0.0, 0.0, 0.0), sc.block.extent)])
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name="블록",
                               line=dict(color="black", width=3), hoverinfo="skip"))


def add_obstacles(fig: go.Figure, sc: Scenario) -> None:
    for o in sc.obstacles:
        c = _corners(o.box)
        x, y, z = zip(*c)
        i, j, k = zip(*_TRIS)
        hover = (f"<b>{o.id}</b> ({o.type})<br>min {list(map(int, o.box.min))}"
                 f"<br>max {list(map(int, o.box.max))}")
        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color="#8a8f98", opacity=0.35,
                                flatshading=True, name=o.id, legendgroup="obstacles",
                                showlegend=False, hovertext=hover, hoverinfo="text"))
    xs, ys, zs = _edge_lines([o.box for o in sc.obstacles])
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=f"장애물 ({len(sc.obstacles)})",
                               legendgroup="obstacles", line=dict(color="#4a4f58", width=2),
                               hoverinfo="skip"))


def add_terminals(fig: go.Figure, sc: Scenario, connect: bool = True) -> None:
    """배관별 단자: start ●, end ◆, 노즐은 채움·경계는 빈 마커. 굵은 선 = 최소 직진 구간(§3.3).

    connect=True 면 start–end 를 점선으로 잇는다 (경로가 아니라 짝 표시).
    """
    for n, p in enumerate(sc.pipes):
        color = _PALETTE[n % len(_PALETTE)]
        group = p.id
        label = f"{p.id} {p.type_id} {p.nominal_size}" + (f" 구배≥{p.min_slope:g}" if p.gravity_pipe else "")
        width = 4 + 8 * PIPE_SPECS[p.nominal_size].od / PIPE_SPECS["500A"].od
        for which, symbol in (("start", "circle"), ("end", "diamond")):
            t = getattr(p, which)
            filled = t.kind == "nozzle"
            host = f" host={t.host}" if t.host else ""
            fig.add_trace(go.Scatter3d(
                x=[t.pos[0]], y=[t.pos[1]], z=[t.pos[2]], mode="markers+text" if which == "start" else "markers",
                text=[p.id] if which == "start" else None, textposition="top center",
                marker=dict(size=6, symbol=symbol if filled else f"{symbol}-open", color=color,
                            line=dict(color="black", width=1)),
                name=label, legendgroup=group, showlegend=which == "start",
                hovertext=(f"<b>{p.id}.{which}</b> {t.kind}{host}<br>pos {list(map(int, t.pos))}"
                           f"<br>dir {list(t.dir)}<br>{p.type_id} {p.nominal_size} r={p.radius:.2f}mm"),
                hoverinfo="text"))
            s = terminal_stub(p, which)
            fig.add_trace(go.Scatter3d(
                x=[s.min[0], s.max[0]], y=[s.min[1], s.max[1]], z=[s.min[2], s.max[2]], mode="lines",
                line=dict(color=color, width=width, dash="dash" if p.gravity_pipe else "solid"),
                legendgroup=group, showlegend=False, hoverinfo="skip"))
            # dir 화살표 (D29 흐름 방향). 직진 구간은 블록 대비 짧아 방향이 안 보이므로 별도 표시
            fig.add_trace(go.Cone(
                x=[t.pos[0]], y=[t.pos[1]], z=[t.pos[2]], u=[t.dir[0]], v=[t.dir[1]], w=[t.dir[2]],
                anchor="tail" if which == "start" else "tip", sizemode="absolute", sizeref=ARROW_MM,
                colorscale=[[0, color], [1, color]], showscale=False,
                legendgroup=group, showlegend=False, hoverinfo="skip"))
        if connect:
            fig.add_trace(go.Scatter3d(
                x=[p.start.pos[0], p.end.pos[0]], y=[p.start.pos[1], p.end.pos[1]],
                z=[p.start.pos[2], p.end.pos[2]], mode="lines",
                line=dict(color=color, width=1, dash="dot"), opacity=0.5,
                legendgroup=group, showlegend=False, hoverinfo="skip"))


def scenario_figure(sc: Scenario, connect: bool = True) -> go.Figure:
    fig = go.Figure()
    add_block(fig, sc)
    add_obstacles(fig, sc)
    add_terminals(fig, sc, connect)
    s = sc.summary()
    title = (f"{s['name'] or '시나리오'} — 장애물 {s['n_obstacles']}개 (채움 {s['obstacle_fill']:.1%}), "
             f"배관 {s['n_pipes']}개 {s['pipe_types']}")
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        scene=dict(aspectmode="data", xaxis_title="X 폭 (mm)", yaxis_title="Y 길이 (mm)",
                   zaxis_title="Z 높이 (mm)", camera=dict(eye=dict(x=1.4, y=-1.6, z=1.1))),
        legend=dict(title="● start ◆ end / 채움=노즐 빈=경계 / 화살표=흐름 방향 / 점선=중력관",
                    itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def add_graph(fig: go.Figure, g, axis: int = 2, value: float = None) -> dict:
    """escape graph 한 단면(axis = value 평면)의 노드·축 엣지·45° 엣지와 팽창 장애물 (3단계).

    전체 그래프(노드 10만 단위)는 HTML 로 그리기에 너무 커서 단면만 그린다.
    value 기본값 = 해당 배관 start 단자 좌표. 격자 좌표가 아니면 가장 가까운 격자 평면.
    """
    import numpy as np
    from .space import DIRS
    coords = g.axes[axis]
    if value is None:
        value = g.pipe.start.pos[axis]
    k = int(np.abs(coords - value).argmin())
    plane = coords[k]
    sl = [slice(None)] * 3
    sl[axis] = k
    sl = tuple(sl)

    def pts(idx_arrays):
        out = [None, None, None]
        it = iter(idx_arrays)
        for ax in range(3):
            out[ax] = np.full(len(idx_arrays[0]), plane) if ax == axis else g.axes[ax][next(it)]
        return out

    ia, ib = np.nonzero(g.node_ok[sl])
    x, y, z = pts((ia, ib))
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="markers", name=f"노드 ({len(ia)})",
                               legendgroup="graph", marker=dict(size=1.6, color="#333"), hoverinfo="skip"))

    def lines(name, color, width, segs):
        xs, ys, zs = [], [], []
        for p0, p1 in segs:
            for q in (p0, p1, (None, None, None)):
                xs.append(q[0]); ys.append(q[1]); zs.append(q[2])
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=name, legendgroup="graph",
                                   line=dict(color=color, width=width), hoverinfo="skip"))
        return len(segs)

    other = [ax for ax in range(3) if ax != axis]
    axis_segs = []
    for ax in other:
        for idx in zip(*np.nonzero(g.axis_ok[ax][sl])):
            node = list(idx); node.insert(axis, k)
            nxt = list(node); nxt[ax] += 1
            axis_segs.append((g.position(tuple(node)), g.position(tuple(nxt))))
    diag_segs = []
    for d, vec in enumerate(DIRS):
        # 이 평면 안의 45° 방향 중 반쪽만 (반대 방향은 같은 엣지)
        if d < 6 or vec[axis] != 0 or vec[other[0]] < 0:
            continue
        for idx in zip(*np.nonzero(g.diag_ok[d][sl])):
            node = list(idx); node.insert(axis, k)
            diag_segs.append((g.position(tuple(node)), g.position(g._step(tuple(node), d))))
    n_axis = lines(f"축 엣지 ({len(axis_segs)})", "#4c78a8", 1.5, axis_segs)
    n_diag = lines(f"45° 엣지 ({len(diag_segs)})", "#e45756", 1, diag_segs)

    from .escape_graph import _snap_down, _snap_up
    inflated = [Box(tuple(_snap_down(v - g.r) for v in lo), tuple(_snap_up(v + g.r) for v in hi))
                for lo, hi in zip(g.box_lo, g.box_hi)]
    xs, ys, zs = _edge_lines(inflated)
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=f"팽창 장애물 (r={g.r:.2f})",
                               legendgroup="graph", visible="legendonly",
                               line=dict(color="#e45756", width=1, dash="dash"), hoverinfo="skip"))
    return {"axis": "xyz"[axis], "value": float(plane), "nodes": len(ia), "edges_axis": n_axis, "edges_45": n_diag}


def add_routes(fig: go.Figure, sc: Scenario, results) -> None:
    """라우터 결과 경로 (4단계). 성공 배관은 배관 색 굵은 선, 꺾임점 표시. 실패는 범례에만 상태 표기."""
    color_of = {p.id: _PALETTE[n % len(_PALETTE)] for n, p in enumerate(sc.pipes)}
    for r in results:
        color = color_of.get(r.pipe_id, "black")
        if r.status != "ok":
            fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="lines", name=f"{r.pipe_id} {r.status}",
                                       line=dict(color=color, width=2, dash="dot"), legendgroup=r.pipe_id))
            continue
        x, y, z = zip(*r.waypoints)
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="lines+markers", legendgroup=r.pipe_id,
            name=f"{r.pipe_id} 경로" + (f" J={r.J:.1f}kg" if r.J else ""),
            line=dict(color=color, width=7), marker=dict(size=3, color=color),
            hovertext=[f"{r.pipe_id} [{i}] {list(map(int, p))}" for i, p in enumerate(r.waypoints)],
            hoverinfo="text"))


MODULE_COLORS = {"collision": "#d62728", "boundary": "#ff7f0e", "bend": "#9467bd", "gravity_slope": "#17becf",
                 "valve": "#e377c2", "branch": "#8c564b", "support": "#bcbd22"}


def add_verification(fig: go.Figure, report: dict, show_supports: bool = True) -> None:
    """검증 결과 (5단계): 위반 위치 ✕ (모듈별 색, hover = 메시지), 설치된 서포트 ■."""
    by_module = {}
    for rep in report["pipes"].values():
        for v in rep.violations:
            by_module.setdefault(v.module, []).append(v)
    for module, vs in by_module.items():
        x, y, z = zip(*[v.pos for v in vs])
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="markers", name=f"위반 {module} ({len(vs)})", legendgroup="violations",
            marker=dict(size=7, symbol="x", color=MODULE_COLORS.get(module, "red"), line=dict(width=2)),
            hovertext=[f"<b>{v.pipe_id} {v.module}</b><br>{v.message}<br>{v.pos}" for v in vs], hoverinfo="text"))
    if show_supports:
        sup = [(rep.pipe_id, s) for rep in report["pipes"].values() for s in rep.supports]
        if sup:
            x, y, z = zip(*[s["pos"] for _, s in sup])
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z, mode="markers", name=f"서포트 ({len(sup)})", visible="legendonly",
                marker=dict(size=3, symbol="square", color="#555"),
                hovertext=[f"{pid} 서포트 → {s['face']} {s['angle_length_mm']:.0f}mm {s['kg']:.2f}kg" for pid, s in sup],
                hoverinfo="text"))


def write_html(fig: go.Figure, path, offline: bool = False) -> None:
    """offline=True 면 plotly.js(약 3.5MB)를 파일에 넣는다. 기본은 CDN 참조."""
    fig.write_html(str(path), include_plotlyjs=True if offline else "cdn", full_html=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="시나리오 3D 표시 → HTML")
    ap.add_argument("paths", nargs="+", help="시나리오 JSON")
    ap.add_argument("-o", "--out", default="out/viz", help="HTML 출력 폴더")
    ap.add_argument("--offline", action="store_true", help="plotly.js 를 HTML 에 포함")
    ap.add_argument("--no-connect", action="store_true", help="start–end 점선 숨김")
    ap.add_argument("--graph", metavar="PIPE_ID", help="이 배관의 escape graph 단면을 겹쳐 그린다 (3단계)")
    ap.add_argument("--slice", default="z", metavar="AXIS[=MM]",
                    help="그래프 단면. 예: z (start 단자 높이), z=1500, x=20000")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in args.paths:
        sc = load(path)
        fig = scenario_figure(sc, not args.no_connect)
        suffix = ""
        if args.graph:
            from .escape_graph import EscapeGraph
            pipe = next((p for p in sc.pipes if p.id == args.graph), None)
            if pipe is None:
                print(f"{path}: 배관 {args.graph} 없음 — 건너뜀")
                continue
            g = EscapeGraph(sc, pipe)
            ax_name, _, val = args.slice.partition("=")
            info = add_graph(fig, g, "xyz".index(ax_name), float(val) if val else None)
            st = g.stats()
            fig.update_layout(title=dict(text=fig.layout.title.text + (
                f"<br>{pipe.id} escape graph: 노드 {st['nodes']:,} · 엣지 {st['edges']:,} "
                f"(축 {st['edges_axis']:,}, 45° {st['edges_45']:,}) · {st['build_sec']:.2f}s — "
                f"단면 {info['axis']}={info['value']:.0f}: 노드 {info['nodes']:,}, "
                f"축 {info['edges_axis']:,}, 45° {info['edges_45']:,}")))
            suffix = f"_{pipe.id}_{info['axis']}{info['value']:.0f}"
        dst = out / (Path(path).stem + suffix + ".html")
        write_html(fig, dst, args.offline)
        print(f"{path} → {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
