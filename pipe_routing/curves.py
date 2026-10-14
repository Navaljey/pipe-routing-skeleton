"""S1-5 (D68): 시간-품질 곡선 집계·그림.

입력 = `anytime` 이 시나리오마다 쓴 `<stem>_curve.json` (궤적: 최선 해가 바뀔 때마다 (누적 확장, 벽시계, 품질)).
한 시나리오의 품질은 예산 b 에서 "b 이하에서 마지막으로 바뀐 점" 의 값이다 (계단 함수).

두 축 (D68 보완):
  (i) 결정적 예산 = 누적 확장 상태 수 — 재현용. 같은 시드·같은 코드면 머신·부하와 무관하게 같은 곡선 (D72).
  (ii) 벽시계 초 — V3 비교용. 측정 조건(동시 실행 수·머신)이 같은 곡선끼리만 비교한다.

세트 집계 `추정:` — 세트 곡선은 세트의 모든 시나리오가 첫 해를 낸 시점(max t₀)부터 그린다. 그 전에는 일부 시나리오에
해가 없어 합계가 의미 없다. 예산 격자 = 절대 1·5·15·60분 + 시나리오별 t₀ 의 1·2·4·8배 (D68). 시나리오가 그 예산 전에
멈췄으면(벽시계 상한) 마지막 값을 쓰고 `censored` 로 표시한다.

사용: python -m pipe_routing.curves <label>=<dir> ... -o out/curves/name [--png]
  label 예: base:G0=out/s15/base_g0  (세트:곡선 이름)
"""
import json
from pathlib import Path

WALL_GRID = (60, 300, 900, 3600)
T0_MULT = (1, 2, 4, 8)
METRICS = ("success", "success_rate", "J_total", "routed", "interference_block")
SUM_METRICS = ("success", "J_total", "routed", "interference_block")


def load_curves(d) -> list:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(d).glob("*_curve.json"))]


def value_at(curve: dict, budget: float, axis: str = "sec"):
    """예산 budget 에서의 품질 (계단). 첫 해 전이면 None. 반환 (점, censored)."""
    key = "sec" if axis == "sec" else "expanded"
    total = curve["total_sec"] if axis == "sec" else curve["total_expanded"]
    pt = None
    for p in curve["points"]:
        if p[key] <= budget + 1e-9:
            pt = p
        else:
            break
    return pt, budget > total + 1e-9


def n_pipes(curve: dict) -> int:
    p = curve["points"][0]
    return round(p["success"] / p["success_rate"]) if p["success_rate"] else curve.get("n_pipes", 0)


def set_value(curves: list, budget: float, axis: str = "sec") -> dict:
    """세트 합계 (모든 시나리오가 첫 해를 낸 뒤에만)."""
    agg = {m: 0 for m in SUM_METRICS}
    n, cens = 0, 0
    for c in curves:
        pt, ce = value_at(c, budget, axis)
        if pt is None:
            return None
        for m in SUM_METRICS:
            agg[m] += pt[m]
        n += c.get("n_pipes") or n_pipes(c)
        cens += int(ce)
    agg["J_total"] = round(agg["J_total"], 1)
    agg["success_rate"] = round(agg["success"] / n, 4) if n else 0.0
    agg["pipes"], agg["censored"] = n, cens
    return agg


def relative_value(curves: list, mult: float, axis: str = "sec") -> dict:
    """시나리오별 t₀(또는 e₀) × mult 에서의 합계."""
    agg = {m: 0 for m in SUM_METRICS}
    n, cens = 0, 0
    for c in curves:
        b = (c["t0_sec"] if axis == "sec" else c["e0"]) * mult
        pt, ce = value_at(c, b, axis)
        for m in SUM_METRICS:
            agg[m] += pt[m]
        n += c.get("n_pipes") or n_pipes(c)
        cens += int(ce)
    agg["J_total"] = round(agg["J_total"], 1)
    agg["success_rate"] = round(agg["success"] / n, 4) if n else 0.0
    agg["pipes"], agg["censored"] = n, cens
    return agg


def set_series(curves: list, axis: str = "sec"):
    """세트 계단 곡선: max t₀ 부터 모든 변화점."""
    key = "sec" if axis == "sec" else "expanded"
    start = max(c["points"][0][key] for c in curves)
    end = max(c["total_sec"] if axis == "sec" else c["total_expanded"] for c in curves)
    xs = sorted({start, end} | {p[key] for c in curves for p in c["points"] if p[key] >= start})
    return [(x, set_value(curves, x, axis)) for x in xs]


def summary(groups: dict) -> dict:
    """groups = {label: [curve, ...]} → 표 데이터."""
    out = {}
    for label, curves in groups.items():
        out[label] = {
            "scenarios": len(curves),
            "seeds": sorted({c["seed"] for c in curves}),
            "concurrency": sorted({c.get("concurrency") for c in curves}, key=str),
            "machine": curves[0].get("machine"),
            "search_budget": curves[0].get("search_budget"),
            "t0_max_sec": max(c["t0_sec"] for c in curves),
            "e0_max": max(c["e0"] for c in curves),
            "total_sec_max": max(c["total_sec"] for c in curves),
            "moves": sum(c["moves"] for c in curves),
            "improvements": sum(len(c["points"]) - 1 for c in curves),
            "wall": {str(b): set_value(curves, b, "sec") for b in WALL_GRID},
            "wall_t0x": {str(k): relative_value(curves, k, "sec") for k in T0_MULT},
            "exp_e0x": {str(k): relative_value(curves, k, "expanded") for k in T0_MULT},
            "final": set_value(curves, float("inf"), "sec"),
        }
    return out


def figure(groups: dict, axis: str = "sec"):
    """세트별 열 × 지표별 행 small multiples, 곡선(G0·G1) = 선. 한 축만 (이중 축 없음)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    sets = sorted({lab.split(":")[0] for lab in groups}, key=lambda s: (s != "base", s))
    rows = [("first", "첫 해를 낸 시나리오 수 (t₀ 분포)"), ("success", "Layer 0 통과 배관 수"),
            ("J_total", "J_total (통과 배관 합, kg)"), ("routed", "경로 배관 수"), ("interference_block", "간섭-차단 수")]
    colors = {"G0": "#2a6fdb", "G1": "#d9822b"}
    fig = make_subplots(rows=len(rows), cols=len(sets), shared_xaxes=False, vertical_spacing=0.06,
                        horizontal_spacing=0.08,
                        subplot_titles=[f"{s} · {r[1]}" for r in rows for s in sets])
    for label, curves in sorted(groups.items()):
        s, name = label.split(":")
        col = sets.index(s) + 1
        ser = set_series(curves, axis)
        xs = [x / 60 if axis == "sec" else x / 1e6 for x, _ in ser]
        key = "sec" if axis == "sec" else "expanded"
        t0s = sorted(c["points"][0][key] for c in curves)     # t₀ (또는 e₀) 누적 분포 — 세트 곡선 시작 전 구간
        end = max(c["total_sec"] if axis == "sec" else c["total_expanded"] for c in curves)
        cx = [x / 60 if axis == "sec" else x / 1e6 for x in t0s + [end]]
        fig.add_trace(go.Scatter(x=cx, y=list(range(1, len(t0s) + 1)) + [len(t0s)], mode="lines+markers",
                                 line=dict(shape="hv", width=2, color=colors.get(name)), marker=dict(size=8),
                                 name=name, legendgroup=name, showlegend=(col == 1),
                                 hovertemplate=f"{name} %{{x:.2f}} · 첫 해 %{{y}}<extra></extra>"), row=1, col=col)
        for ri, (m, _) in enumerate(rows[1:], start=2):
            fig.add_trace(go.Scatter(
                x=xs, y=[v[m] for _, v in ser], mode="lines+markers", line=dict(shape="hv", width=2,
                                                                                    color=colors.get(name)),
                marker=dict(size=8), name=name, legendgroup=name, showlegend=False,
                hovertemplate=f"{name} %{{x:.2f}} · {m} %{{y}}<extra></extra>"), row=ri, col=col)
    xt = "벽시계 (분, 로그)" if axis == "sec" else "누적 확장 상태 수 (백만, 로그)"
    for ri in range(1, len(rows) + 1):
        for ci in range(1, len(sets) + 1):
            fig.update_xaxes(type="log", title_text=xt if ri == len(rows) else None, row=ri, col=ci,
                             gridcolor="#e6e6e6")
            fig.update_yaxes(gridcolor="#e6e6e6", row=ri, col=ci)
    fig.update_layout(height=260 * len(rows), width=520 * len(sets), template="plotly_white",
                      title=dict(text=f"시간-품질 곡선 (D68) — x = {xt}", y=0.995, yanchor="top"),
                      legend=dict(orientation="h", y=1.035, yanchor="bottom", x=0, xanchor="left"),
                      margin=dict(t=140),
                      font=dict(size=12))
    return fig


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="S1-5 시간-품질 곡선 집계·그림 (D68)")
    ap.add_argument("groups", nargs="+", help="<세트:곡선>=<curve.json 폴더>")
    ap.add_argument("-o", "--out", required=True, help="출력 접두사 (…_summary.json, …_wall.html, …_exp.html)")
    ap.add_argument("--png", action="store_true", help="PNG 도 저장 (playwright 크로미움)")
    args = ap.parse_args(argv)
    groups = {}
    for g in args.groups:
        label, d = g.split("=", 1)
        groups[label] = load_curves(d)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summ = summary(groups)
    Path(f"{out}_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    for axis, suf in (("sec", "wall"), ("expanded", "exp")):
        fig = figure(groups, axis)
        html = Path(f"{out}_{suf}.html")
        fig.write_html(str(html), include_plotlyjs="cdn")
        if args.png:
            _png(fig, Path(f"{out}_{suf}.png"))
    for label, s in summ.items():
        print(label, "최종", s["final"], "| 1분", s["wall"]["60"], "| 60분", s["wall"]["3600"])
    return 0


def _png(fig, path: Path):
    import tempfile
    from playwright.sync_api import sync_playwright
    with tempfile.TemporaryDirectory() as td:
        h = Path(td) / "f.html"
        fig.write_html(str(h), include_plotlyjs=True, full_html=True)
        with sync_playwright() as p:
            exe = Path("/opt/pw-browsers/chromium")
            b = p.chromium.launch(executable_path=str(exe)) if exe.is_file() else p.chromium.launch()
            pg = b.new_page(viewport={"width": int(fig.layout.width) + 40, "height": int(fig.layout.height) + 40})
            pg.goto(h.as_uri())
            pg.wait_for_timeout(1500)
            pg.screenshot(path=str(path), full_page=True)
            b.close()


if __name__ == "__main__":
    raise SystemExit(main())
