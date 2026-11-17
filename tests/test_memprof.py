"""측정 도구 결과 불변 (2026-11-17 소유자: "측정 도구 추가는 결과 불변이어야 한다 (출력 바이트 동일)").

memprof 로 감싼 실행과 감싸지 않은 실행의 출력 JSON 이 (벽시계·메모리 필드를 뺀 deterministic_view 로) 같고,
기록에 고정층·라우팅·단계가 들어 있는지 본다. 감싸기는 프로세스 전역이라 하위 프로세스로 돌린다."""
import json
import subprocess
import sys
from pathlib import Path

from pipe_routing.pipeline import deterministic_view

ROOT = Path(__file__).resolve().parent.parent
SC = ROOT / "scenarios" / "manual" / "manual_01.json"


def _run(args, cwd):
    subprocess.run([sys.executable, "-m", *args], cwd=cwd, check=True, capture_output=True)


def test_memprof_output_identical(tmp_path):
    _run(["pipe_routing.pipeline", str(SC), "-o", str(tmp_path / "a"), "--no-viz"], ROOT)
    log = tmp_path / "prof.jsonl"
    _run(["pipe_routing.memprof", "--log", str(log), "--", str(SC), "-o", str(tmp_path / "b"), "--no-viz"], ROOT)
    a = json.loads((tmp_path / "a" / "manual_01_output.json").read_text(encoding="utf-8"))
    b = json.loads((tmp_path / "b" / "manual_01_output.json").read_text(encoding="utf-8"))
    assert json.dumps(deterministic_view(a), sort_keys=True) == json.dumps(deterministic_view(b), sort_keys=True)
    recs = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    kinds = {r["kind"] for r in recs}
    assert {"layer", "route", "caches", "phase", "done"} <= kinds
    layers = [r for r in recs if r["kind"] == "layer"]
    assert all(r["coord_sources"]["union_matches"] for r in layers)   # 좌표 출처 합집합 = 고정층 축
    routes = [r for r in recs if r["kind"] == "route"]
    assert all(0 < r["edges_read"] <= r["edge_entries"] for r in routes if r["expanded"])
