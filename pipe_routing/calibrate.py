"""D129 (나) 보정 계수 — 보정 묶음 (세트마다 1 시나리오, 기준선 첫 해, 동시 2) 의 첫 해 벽시계를 이 머신에서 재고
동결 머신 값 (`docs/eval_set_E1_baseline.json` curve.t0_sec, E1 코드 `cd9c13e`) 과의 비를 구한다.

python -m pipe_routing.calibrate --code <E1 코드 작업 트리 (cd9c13e)> -o out/calib [--repeat 2]

- 묶음 = BUNDLE (세트마다 동결 t₀ 가 세트 중앙에 가장 가까운 시나리오, `docs/v3_contract.md` §e3).
- 같은 코드로 재야 머신 비가 된다 — `--code` 는 E1 코드 커밋의 작업 트리 (지금 코드가 결과는 같아도 속도는 다를 수 있다).
- 실행 = `pipe_routing.anytime --max-sec 0.001` (첫 해만), 동시 실행 수 = ⌊코어 ÷ 2⌋ (최소 1, D140 ①, `--concurrency` 로 더 줄임),
  서포트 표 source = private 확인 (D78·D87).
- 비 r = Σ 이 머신 t₀ / Σ 동결 t₀. 두 번 재서 비가 ±3 % 안이면 사용, 0.8~1.25 밖이면 (가) 같은 컨테이너 짝 실행 (D129).
- 출력: 시나리오별 t₀ · 회차별 r · 머신 사양 (CPU 모델·클럭·코어·메모리 한도·동시 실행 수·Python·numba·numpy, D68·D140 ③).
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
    out = {"cpu": cpu or platform.processor(), "cores": os.cpu_count(), "mem_limit_gb": mem_limit_gb(),
           "python": platform.python_version()}
    for m in ("numba", "numpy"):
        try:
            out[m] = __import__(m).__version__
        except ImportError:
            out[m] = None
    return out


def mem_limit_gb():
    """컨테이너 메모리 한도 (cgroup v2 → v1 → MemTotal), GB."""
    for f in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(f).read().strip()
            if v != "max" and int(v) < 1 << 60:
                return round(int(v) / 2 ** 30, 1)
        except (OSError, ValueError):
            pass
    try:
        kb = next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemTotal"))
        return round(kb / 2 ** 20, 1)
    except (OSError, StopIteration):
        return None


def default_concurrency() -> int:
    """D140 ①: 동시 실행 수 = ⌊코어 ÷ 2⌋, 최소 1."""
    return max(1, (os.cpu_count() or 1) // 2)


def frozen_t0() -> dict:
    d = json.loads((REPO / "docs/eval_set_E1_baseline.json").read_text(encoding="utf-8"))
    return {s: d["sets"][s][Path(p).stem]["curve"]["t0_sec"] for s, p in BUNDLE.items()}


# D129 보완 --no-classify: 첫 해 시각만 재는 실행기 — E1 코드 (cd9c13e) 에 플래그가 없어도 쓸 수 있게, 첫 해 뒤 실패 분류
# (D51·D74 진단 재라우팅, 예산 밖) 를 건너뛰도록 anytime 모듈의 classify_sequential 을 빈 함수로 바꿔 부른다. t₀ 는 분류 전에
# 재므로 값은 같다. 서포트 표 source 도 함께 적는다.
RUNNER = r"""
import json, sys
from pathlib import Path
from pipe_routing import anytime
from pipe_routing.layered import make_layered_router, parse_grid
from pipe_routing.scenario import load
from pipe_routing.support_std import load_standard
anytime.classify_sequential = lambda *a, **k: None
sc = load(sys.argv[1])
tr = anytime.anytime_run(sc, make_layered_router(grid=parse_grid("g0")), seed=0, max_sec=0.001)
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
(out / (Path(sys.argv[1]).stem + "_t0.json")).write_text(json.dumps(
    {"t0_sec": round(tr.t0_sec, 3), "e0": tr.e0, "support_source": load_standard().source, "classify": False}))
"""


def warmup(code: str, out: Path) -> float:
    """numba 캐시 데우기 — 수작업 시나리오 첫 해를 한 번 돌려 버린다 (재지 않음). 새 컨테이너·numba 판이 바뀐 경우 첫 프로세스는
    컴파일 시간을 t₀ 에 넣게 된다 (수작업 첫 해 1.9 s → 컴파일 포함 4.4 s, 2026-11-28 측정)."""
    t = time.time()
    subprocess.run([sys.executable, "-c", RUNNER, str(REPO / BUNDLE["manual"]), str(out / "warmup")], cwd=code,
                   env=dict(os.environ, PYTHONPATH=code), capture_output=True, text=True)
    return round(time.time() - t, 1)


def run_round(code: str, out: Path, conc: int, classify: bool = True) -> dict:
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
            if not classify:
                r = subprocess.run([sys.executable, "-c", RUNNER, str(REPO / p), str(od)], cwd=code, env=env,
                                   capture_output=True, text=True)
                f = od / f"{Path(p).stem}_t0.json"
                d = json.loads(f.read_text()) if f.exists() and r.returncode == 0 else {}
                with lock:
                    res[s] = {"t0_sec": d.get("t0_sec"), "e0": d.get("e0"), "frozen_t0_sec": fro[s], "rc": r.returncode,
                              "support_source": d.get("support_source")}
                continue
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
    ap.add_argument("--no-classify", action="store_true",
                    help="D129 보완: 첫 해 뒤 실패 분류 (D51·D74 진단 재라우팅) 를 건너뛴다 — t₀ 같음, 메모리 작은 컨테이너용")
    ap.add_argument("--concurrency", type=int, default=None,
                    help="동시 실행 수 (기본 ⌊코어 ÷ 2⌋, D140 ① — 메모리가 모자라면 더 줄인다)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    rounds = []
    conc = a.concurrency or default_concurrency()
    warm = warmup(a.code, out)   # 재기 전 numba 캐시 데우기 (회차 시간에 넣지 않는다)
    for k in range(a.repeat):
        t = time.time()
        r = run_round(a.code, out / f"round{k + 1}", conc, classify=not a.no_classify)
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
    summary = {"machine": {**machine(), "concurrency": conc}, "bundle": BUNDLE, "classify": not a.no_classify, "warmup_sec": warm,
               "rounds": rounds,
               "ratio": (sum(rs) / len(rs)) if rs else None, "stable_3pct": stable,
               "usable": use, "verdict": "보정 계수 사용 (나)" if use else "같은 컨테이너 짝 실행 (가)"}
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("machine", "ratio", "stable_3pct", "usable", "verdict")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
