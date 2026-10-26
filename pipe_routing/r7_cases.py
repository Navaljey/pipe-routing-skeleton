"""R7 부재 막힘 사례 목록 (docs/reports/r7_cases.md·.json) — 출력 JSON 재분석, 새 라우팅 없음.

경우마다 (예: 첫 해 = 순차 배치, 예산 끝 = anytime 최선 해) support 위반 배관을 `support_diag` 와 같은 판정으로
나눈다: 단독 검증 통과 (= 다른 배관의 부재 막힘) 여부 · 막는 배관 · 배치 순서 (가) 막는 배관이 나중 / (나) 먼저 ·
막힘 유형 (relation) · 막힌 부재 형식·면. 출력 global.member_blockage 와 같은 집계를 경우·세트별로 묶는다.

D78: 목록에는 형식(A~D)·면·배관 id·구경·거리만 쓴다 — 선정표 값(무게·보강 여부·표 범위)은 쓰지 않는다.

python -m pipe_routing.r7_cases first:base=out/s1_7_base_d105 first:dense30=out/s1_7_d30_d105 \\
    budget_end:base=out/r7_any_base budget_end:dense30=out/r7_any_d30 -o docs/reports/r7_cases
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from .support_diag import diagnose
from .support_std import load_standard

CASE_KO = {"first": "첫 해 (순차 배치, D50)", "budget_end": "예산 끝 (anytime 최선 해, D68·D75)"}
ORDER_KO = {"after": "(가) 막는 배관이 나중", "before": "(나) 막는 배관이 먼저"}


def collect(specs: list, std) -> dict:
    out = {"support_standard": std.info(), "cases": []}
    for spec in specs:
        key, d = spec.split("=", 1)
        case, set_name = key.split(":", 1)
        files = sorted(Path(d).glob("*_output.json"))
        rows, scen = [], 0
        for f in files:
            scen += 1
            g = json.loads(f.read_text(encoding="utf-8"))["global"]
            if g.get("support_standard", {}).get("source") != "private":
                raise SystemExit(f"{f}: 공개 임시 표로 돈 결과 — R7 목록에 쓰지 않는다 (D78)")
            for r in diagnose(str(f), std):
                rows.append({k: r[k] for k in ("scenario", "pipe", "size", "type", "gaps", "solo_pass", "blockers",
                                               "blocker_kind", "order", "blocker_sizes", "surface_gap_mm", "relation")}
                            | {"blocked": sorted({f"{b['form']}/{b['face']}" for b in r["blocked_supports"]})})
        mb = [r for r in rows if r["solo_pass"]]
        out["cases"].append({
            "case": case, "set": set_name, "dir": d, "scenarios": scen,
            "support_violation": len(rows), "member_blockage": len(mb),
            "after": sum(1 for r in mb if r["order"] == ["after"]),
            "before": sum(1 for r in mb if r["order"] == ["before"]),
            "both": sum(1 for r in mb if len(r["order"]) == 2),
            "relation": dict(sorted(Counter(r["relation"] for r in mb).items())),
            "blocked_forms": dict(sorted(Counter(x for r in mb for x in r["blocked"]).items())),
            "rows": rows})
    return out


def to_md(data: dict) -> str:
    L = ["# R7 부재 막힘 사례 목록", "",
         "다른 배관의 서포트 부재 자리 때문에 support 위반이 된 배관 (단독 검증 통과) 의 목록. "
         "생성: `python -m pipe_routing.r7_cases` (출력 재분석, 결과 불변). "
         f"서포트 표: 사내 표준 기반(비공개), sha256 `{data['support_standard']['sha256'][:12]}…`. "
         "선정표 값은 쓰지 않는다 (D78).", "",
         "막힘 유형 (relation, `support_diag.relation`): p_vertical_45 = 막힌 배관 구간이 수직·45° · q_vertical_45 = "
         "막는 배관이 수직·45° · crossing = 둘 다 축 정렬 수평이고 축이 다름 · parallel_le50 / _50_300 / _gt300 = 같은 축 "
         "수평 평행, 지지면 쪽 표면 높이 차 (mm).", "",
         "## 요약", "",
         "| 경우 | 세트 | 시나리오 | support 위반 | 부재 막힘 | (가) 나중 | (나) 먼저 | 양쪽 | 막힘 유형 | 막힌 부재 (형식/면) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for c in data["cases"]:
        L.append(f"| {CASE_KO.get(c['case'], c['case'])} | {c['set']} | {c['scenarios']} | {c['support_violation']} | "
                 f"{c['member_blockage']} | {c['after']} | {c['before']} | {c['both']} | "
                 + (", ".join(f"{k} {v}" for k, v in c["relation"].items()) or "-") + " | "
                 + (", ".join(f"{k} {v}" for k, v in c["blocked_forms"].items()) or "-") + " |")
    for c in data["cases"]:
        L += ["", f"## {CASE_KO.get(c['case'], c['case'])} — {c['set']} (`{c['dir']}`)", ""]
        if not c["rows"]:
            L.append("support 위반 없음.")
            continue
        L += ["| 시나리오 | 배관 | 구경 | 타입 | 단독 통과 | 막는 배관 (구경, 표면 간격 mm) | 순서 | 유형 | 막힌 부재 |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in c["rows"]:
            bl = ", ".join(f"{q} ({r['blocker_sizes'][q]}, {r['surface_gap_mm'][q]:.0f})" for q in r["blockers"]) or "-"
            L.append(f"| {r['scenario']} | {r['pipe']} | {r['size']} | {r['type']} | {'예' if r['solo_pass'] else '아니오'} | "
                     f"{bl} | {' · '.join(ORDER_KO[o] for o in r['order']) or '-'} | {r['relation']} | "
                     f"{', '.join(r['blocked']) or '-'} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("specs", nargs="+", help="<경우>:<세트>=<출력 폴더>")
    ap.add_argument("-o", "--out", required=True, help="확장자 없는 경로 (.md·.json 을 쓴다)")
    a = ap.parse_args(argv)
    std = load_standard()
    if std.source != "private":
        raise SystemExit("비공개 서포트 표가 필요하다 (PIPE_SUPPORT_STANDARD, D78)")
    data = collect(a.specs, std)
    Path(a.out + ".json").write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    Path(a.out + ".md").write_text(to_md(data), encoding="utf-8")
    for c in data["cases"]:
        print(f"{c['case']}:{c['set']} support 위반 {c['support_violation']} · 부재 막힘 {c['member_blockage']} "
              f"(가 {c['after']} · 나 {c['before']} · 양쪽 {c['both']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
