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


def write_html(fig: go.Figure, path, offline: bool = False) -> None:
    """offline=True 면 plotly.js(약 3.5MB)를 파일에 넣는다. 기본은 CDN 참조."""
    fig.write_html(str(path), include_plotlyjs=True if offline else "cdn", full_html=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="시나리오 3D 표시 → HTML")
    ap.add_argument("paths", nargs="+", help="시나리오 JSON")
    ap.add_argument("-o", "--out", default="out/viz", help="HTML 출력 폴더")
    ap.add_argument("--offline", action="store_true", help="plotly.js 를 HTML 에 포함")
    ap.add_argument("--no-connect", action="store_true", help="start–end 점선 숨김")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in args.paths:
        sc = load(path)
        dst = out / (Path(path).stem + ".html")
        write_html(scenario_figure(sc, not args.no_connect), dst, args.offline)
        print(f"{path} → {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
