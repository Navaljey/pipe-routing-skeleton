"""D129 (나) 보정 계수 — 보정 묶음 (세트마다 1 시나리오, 기준선 첫 해, 동시 2) 의 첫 해 벽시계를 이 머신에서 재고
동결 머신 값 (`docs/eval_set_E1_baseline.json` curve.t0_sec, E1 코드 `cd9c13e`) 과의 비를 구한다.

python -m pipe_routing.calibrate --code <E1 코드 작업 트리 (cd9c13e)> -o out/calib [--repeat 2]

- 묶음 = BUNDLE (세트마다 동결 t₀ 가 세트 중앙에 가장 가까운 시나리오, `docs/v3_contract.md` §e3).
- 같은 코드로 재야 머신 비가 된다 — `--code` 는 E1 코드 커밋의 작업 트리 (지금 코드가 결과는 같아도 속도는 다를 수 있다).
- 실행 = `pipe_routing.anytime --max-sec 0.001` (첫 해만), 동시 2, 서포트 표 source = private 확인 (D78·D87).
- 비 r = Σ 이 머신 t₀ / Σ 동결 t₀. 두 번 재서 비가 ±3 % 안이면 사용, 0.8~1.25 밖이면 (가) 같은 컨테이너 짝 실행 (D129).
- 출력: 시나리오별 t₀ · 회차별 r · 머신 사양 (CPU 모델·클럭·코어·Python·numba·numpy, D68).
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUNDLE = {"manual": "scenarios/manual/manual_01.json", "base": "scenarios/procedural/proc_012.json",
          "stair": "scenarios/stair/stair10_003.json", "dense30": "scenarios/dense/dense30_003.json",
          "dense50": "scenarios/dense/dense50_001.json", "pair": "scenarios/pair/pair_004.json"}


def machine() -> dict:
    cpu = ""
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except (OSError, StopIteration):
        pass
    out = {"cpu": cpu or platform.processor(), "cores": os.cpu_count(), "python": platform.python_version()}
    for m in ("numba", "numpy"):
        try:
            out[m] = __import__(m).__version__
        except ImportError:
            out[m] = None
    return out


def frozen_t0() -> dict:
    d = json.loads((REPO / "docs/eval_set_E1_baseline.json").read_text(encoding="utf-8"))
    return {s: d["sets"][s][Path(p).stem]["curve"]["t0_sec"] for s, p in BUNDLE.items()}


def run_round(code: str, out: Path, conc: int = 2) -> dict:
    """묶음 한 회차 — 무거운 것부터 동시 conc 개 (D110 ①: 밀집 50 이 혼자 남지 않게 큰 것부터 채운다)."""
    fro = frozen_t0()
    jobs = sorted(BUNDLE.items(), key=lambda kv: -fro[kv[0]])
    res, lock = {}, threading.Lock()
    env = dict(os.environ, PYTHONPATH=code)

    def worker():
        while True:
            with lock:
                if not jobs:
                    return
                s, p = jobs.pop(0)
            od = out / s
            r = subprocess.run([sys.executable, "-m", "pipe_routing.anytime", str(REPO / p), "-o", str(od),
                                "--max-sec", "0.001", "--concurrency", str(conc)], cwd=code, env=env,
                               capture_output=True, text=True)
            curve = next(od.glob("*_curve.json"), None)
            t0 = json.loads(curve.read_text())["t0_sec"] if curve and r.returncode == 0 else None
            src = None
            fo = next(od.glob("*_first_output.json"), None)
            if fo:
                src = json.loads(fo.read_text())["global"]["support_standard"]["source"]
            with lock:
                res[s] = {"t0_sec": t0, "frozen_t0_sec": fro[s], "rc": r.returncode, "support_source": src}
    ts = [threading.Thread(target=worker) for _ in range(conc)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    ok = all(v["t0_sec"] is not None for v in res.values())
    ratio = (sum(v["t0_sec"] for v in res.values()) / sum(v["frozen_t0_sec"] for v in res.values())) if ok else None
    return {"scenarios": res, "ratio": ratio}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--code", required=True, help="E1 코드 커밋 (cd9c13e) 작업 트리")
    ap.add_argument("-o", "--out", default="out/calib")
    ap.add_argument("--repeat", type=int, default=2)
    a = ap.parse_args(argv)
    out = Path(a.out)
    rounds = []
    for k in range(a.repeat):
        t = time.time()
        r = run_round(a.code, out / f"round{k + 1}")
        r["sec"] = round(time.time() - t, 1)
        rounds.append(r)
        bad = [s for s, v in r["scenarios"].items() if v["support_source"] != "private"]
        print(f"회차 {k + 1}: r = {r['ratio']} · {r['sec']} s" + (f" · 서포트 표 private 아님 {bad}" if bad else ""),
              flush=True)
        if bad:
            break
    rs = [r["ratio"] for r in rounds if r["ratio"]]
    stable = len(rs) >= 2 and max(rs) / min(rs) - 1 <= 0.03
    use = stable and all(0.8 <= x <= 1.25 for x in rs)
    summary = {"machine": machine(), "bundle": BUNDLE, "rounds": rounds,
               "ratio": (sum(rs) / len(rs)) if rs else None, "stable_3pct": stable,
               "usable": use, "verdict": "보정 계수 사용 (나)" if use else "같은 컨테이너 짝 실행 (가)"}
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("machine", "ratio", "stable_3pct", "usable", "verdict")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
