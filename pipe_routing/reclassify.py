"""D135 ② — 평가판 E1.1 (분류 정정판): 경로·J·곡선은 다시 돌리지 않고, E1 출력에서 분류만 정정 기준으로 다시 계산한다.

분류 필드 (CLASS_FIELDS) = 분모 제외·필수 배관·완전 성공 — routes[].terminal_feasibility, global.terminal_blocked ·
full_success · required_pipes · required_passed · excluded_infeasible, 기준선 JSON 의 excluded·곡선 점 full_success·
required_passed. fail_class (D51·D74) 는 경로·진단 재라우팅 결과라 바뀌지 않는다 (제외됐던 배관도 이미 분류됨).
그 밖의 필드는 E1 과 바이트 동일해야 한다 (strip_class 로 비교).

  python -m pipe_routing.reclassify baseline docs/eval_set_E1_baseline.json -o docs/eval_set_E1.1_baseline.json
  python -m pipe_routing.reclassify output <E1 출력 JSON>... -o <dir>
"""
import argparse
import copy
import json
from pathlib import Path

from .pipeline import full_success_summary, required_pipe_ids
from .scenario import load, terminal_straight_blocked

ROOT = Path(__file__).resolve().parents[1]
CLASS_FIELDS = {"terminal_feasibility", "terminal_blocked", "full_success", "required_pipes", "required_passed",
                "excluded_infeasible", "excluded"}


def strip_class(obj):
    if isinstance(obj, dict):
        return {k: strip_class(v) for k, v in obj.items() if k not in CLASS_FIELDS}
    if isinstance(obj, list):
        return [strip_class(v) for v in obj]
    return obj


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def reclassify_output(out: dict, sc) -> dict:
    """pipeline 출력 하나 (sc = 그 시나리오) → 분류만 다시 계산한 사본."""
    new = copy.deepcopy(out)
    pipes = {p.id: p for p in sc.pipes}
    for x in new["routes"]:
        if "terminal_feasibility" in x:
            x["terminal_feasibility"] = terminal_straight_blocked(pipes[x["pipe_id"]], sc.obstacles)
    g = new["global"]
    if "terminal_blocked" in g:
        g["terminal_blocked"] = [x["pipe_id"] for x in new["routes"]
                                 if x.get("terminal_feasibility") and not x["terminal_feasibility"]["feasible"]]
    if "full_success" in g:
        g.update(full_success_summary(new["routes"]))
    return new


def _scenario_paths() -> dict:
    H = json.loads((ROOT / "docs/reports/s1_8_scenario_sha256.json").read_text())
    return {Path(f).stem: ROOT / f for fs in H.values() for f in fs}


def reclassify_baseline(base: dict) -> dict:
    """docs/eval_set_E1_baseline.json → E1.1 사본 (첫 해·예산 끝·곡선 점의 분류만)."""
    new = copy.deepcopy(base)
    paths = _scenario_paths()
    for sets in new["sets"].values():
        for stem, rec in sets.items():
            sc = load(paths[stem])
            req, exc = required_pipe_ids(sc)
            for key in ("first", "budget_end"):
                s = rec[key]
                passed = {pid for pid, v in s["pipes"].items() if v["pass"]}
                s["excluded"] = exc
                s["required_pipes"] = len(req)
                s["required_passed"] = sum(1 for pid in req if pid in passed)
                s["full_success"] = s["required_passed"] == len(req)
            # 곡선 점: 제외가 풀린 배관은 모든 점에서 통과하지 못했다 (첫 해·예산 끝에서 미통과 확인) →
            # required_passed 는 그대로, full_success 는 필수 배관이 늘어 참이 될 수 없다 — 점마다 다시 계산
            for pt in rec["curve"]["points"]:
                if "full_success" in pt:
                    pt["full_success"] = int(pt["required_passed"] == len(req))
    return new


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("kind", choices=["baseline", "output"])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)
    if a.kind == "baseline":
        base = json.loads(Path(a.paths[0]).read_text())
        new = reclassify_baseline(base)
        assert dumps(strip_class(new)) == dumps(strip_class(base)), "분류 밖 필드가 바뀜 (D135 ② 멈춤 조건)"
        Path(a.out).write_text(json.dumps(new, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        return 0
    paths = _scenario_paths()
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    for f in a.paths:
        out = json.loads(Path(f).read_text())
        stem = Path(f).name.removesuffix("_output.json").removesuffix("_first")
        new = reclassify_output(out, load(paths[stem]))
        assert dumps(strip_class(new)) == dumps(strip_class(out)), f"{f}: 분류 밖 필드가 바뀜"
        (od / Path(f).name).write_text(json.dumps(new, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
