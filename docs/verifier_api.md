# 검증기 공개 API (D84·D85)

> V3 가 보상·판정에 쓰는 심판 API. 결과는 `pipeline` 출력과 같은 검증기다. V3 쪽에서 고치지 않는다 (`docs/v3_contract.md` §f).

서포트 선정표는 사내 표준 기반(비공개, D78)이다. 환경 변수 `PIPE_SUPPORT_STANDARD` 가 비공개 표 경로를 가리키면 그 표를 쓰고, 없으면 공개 임시 표(근사)로 돈다. 어느 표인지와 데이터 해시는 결과의 `support_standard` 에 남는다. 비교·제출은 E1 동결 해시와 같은 표로 한다.

---

## 1. 전체 검증 — `verify`

```python
from pipe_routing.scenario import load
from pipe_routing.verifier import PipeRoute, verify

sc = load("scenarios/procedural/proc_009.json")
rep = verify(sc, [PipeRoute("P001", waypoints), ...])     # 경로 없는 배관은 빼거나 waypoints = []
rep["pipes"]["P001"].success          # Layer 0 전부 통과 (D44)
rep["pipes"]["P001"].layer0           # {"collision": True, ..., "gravity_slope": None, "support": False}
rep["pipes"]["P001"].violations       # [Violation(module, pipe_id, pos, message, value, other)]
rep["pipes"]["P001"].supports         # D80 ⑦ 부재 묶음, kg_share (D81). D89 함께 받친 D: origin "D89", 얹힌 쪽은 attached 사본
rep["pipes"]["P001"].fittings         # 엘보·티·밸브 (J 분해용)
rep["timing"]                         # 모듈별 시간 (초) — centerline · 7 모듈 · group_d
rep["support_standard"]               # {"source": "private" | "public_approx", "sha256": ...}
```

- `waypoints` = start 단자 → end 단자 꺾임점 (mm). 라우터·표현과 무관한 순수 기하다.
- J 분해: `pipe_routing.pipeline.j_breakdown(pipe, waypoints, rep.fittings, rep.supports)`.
- `J_total` = Layer 0 통과 배관 J 합 (D49). 공유 서포트(D 묶음)는 배관 단위중량 비례로 귀속한다 (D81).

**모듈 단독 호출** — `verify(sc, routes, modules=["support"])`
- `MODULES` 레지스트리(`pipe_routing.verifier.MODULES`)의 이름 목록을 넘긴다. 순서는 `MODULE_ORDER` 를 따른다.
- 모듈끼리 의존하지 않는다. `support` 단독 결과(서포트 목록·support 판정)는 전체 검증과 같다.
- `support` 가 들어 있으면 D 묶음(D80 ⑥)도 수행한다.

---

## 2. 증분 검증 — `Session` (S1-6b, D85)

```python
from pipe_routing.verifier import Session

s = Session(sc)                         # std=None → 환경 변수 / 공개 임시 표
r = s.add("P001", waypoints)            # 배관 놓기
r = s.replace("P001", new_waypoints)    # 경로 바꾸기
r = s.remove("P001")                    # 들어내기
r["pipes"]["P001"]                      # {"routed", "success", "layer0", "J", "supports", "violations"}
r["J_total"]                            # Layer 0 통과 배관 J 합 (D49)
r["recomputed"]                         # 이번 호출에서 다시 판정한 배관 id
s.reports()                             # verify()["pipes"] 와 같은 형식
```

**결과 불변 (D53·D67 원칙).** 어떤 add / remove / replace 순서 뒤에도 `s.reports()` 는 같은 배관 집합을 `verify(sc, routes)` (routes = 시나리오 배관 순서) 로 전체 검증한 것과 비트 동일하다 (`tests/test_s1_session.py`, 무작위 순서 매 단계 대조).

**다시 판정하는 범위 (모듈별, D87 ⑤)**
- 바뀐 배관은 전 모듈을 다시 판정한다.
- 이웃 배관은 영향받을 수 있는 모듈만 다시 판정한다 (`r["recomputed_modules"]`).
  - collision·bend: 바뀐 배관(이전·새 중심선)과의 거리 ≤ r₁ + r₂ + 1 mm 일 때.
  - valve: 밸브가 있고 거리 ≤ r₁ + r₂ + 1,250 mm 일 때.
  - support: 바뀐 배관이 그 이웃의 서포트 배치에서 판정한 부재 영역에 닿을 때.
- D 묶음(좌표 순서 탐욕 — 연쇄 영향 가능)은 매 호출 전체를 다시 한다.

**비용** (S1-6b 측정, `docs/reports/s1_6.md` §7.4·7.5, Intel Xeon 2.8 GHz 단일 프로세스)

| 세트 | 전체 `verify` 중앙 / p90 | 증분 1회 중앙 / p90 | 다시 판정한 배관 중앙 |
|---|---|---|---|
| 기본 (10배관) | 1.26 / 1.84 s | 0.46 / 0.83 s | 3 |
| 밀집 30 | 4.00 / 5.14 s | 1.67 / 2.78 s | 9 |
| 밀집 50 | 6.69 / 8.25 s | 2.86 / 4.40 s | 14 |

- 전체 검증 시간은 배관 수에 거의 선형이다 (로그-로그 기울기 1.11).
- 증분 호출 시간은 영향 범위(서포트 부재 도달 약 4 m) 안 이웃 수에 비례한다.

---

## 3. 측정 도구

```
python -m pipe_routing.verifier.bench <scenario.json>=<output.json> ... [--incremental N] [--check K] -o out.json
```

- 전체 검증 시간과 모듈별 분해를 잰다.
- `--incremental N`: 무작위 배관 N 개를 remove → add 하며 호출별 시간을 잰다.
- `--check K`: K 번째마다 전체 검증과 비트 동일인지 확인한다.
