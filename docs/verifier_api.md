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

---

## 4. 서포트 힌트 — `SupportHint` (S1-7b, D100)

조회 전용·결과 불변. 검증기 채점과 **같은 함수**를 부른다 — 힌트가 말하는 형식·묶임은 검증기가 실제로 놓는 것과 같다.

```python
from pipe_routing.verifier.hint import SupportHint

h = SupportHint(sc, {"P001": wps1, "P002": wps2})   # 이 경로들이 놓인 상태에서 묻는다 (std=None → 환경 변수 / 공개 임시 표)
h = SupportHint.from_session(session)                # 증분 검증 Session 의 지금 배치 (Session 은 바꾸지 않는다)

r = h.form_at("P003", pos, axis)        # 묻는 배관 자신의 경로는 없어도 된다 (계획 중 경로 위 점)
r["chosen"]                             # 검증기 supports[] 항목과 같은 필드 (form·face·variant·section·L·kg·members) 또는 None
r["options"]                            # (면, 형식) 조합마다 status: ok | out_of_range (표 밖, D79 ③) | blocked (부재 간섭, D80 ⑧) | parallel

g = h.groupable(("P001", pos1, axis1), ("P002", pos2, axis2), face=None)
g["ok"], g["face"], g["kg"], g["kg_individual"], g["reason"]   # D82 조건 — 아니면 처음 걸린 이유

t = h.together_at("P001", pos, axis)    # D89 함께 받치는 D (id·면·규격·kg_total·shares·members) 또는 None
```

| 함수 | 공유하는 채점 함수 | 검증기와의 관계 |
|---|---|---|
| `form_at` | `modules.support_candidates` → `support_std.evaluate` (D80 ③④⑧) | 검증기가 그 위치에 놓는 서포트와 같다 |
| `groupable` | `support_group._try` (D82: 같은 면·수평 평행·±500 mm·표면 차 ≤ 50 mm·L1/L2 표 범위·간섭 없음·묶은 무게 < 개별 합) | 검증기 묶음은 모두 `ok`. **간격 재확인 (D82 보완) 은 보지 않는다** (`spacing_recheck: "not_checked"`) — 그 배관 서포트 목록 전체가 있어야 한다 |
| `together_at` | `support_group.together_at` (D89) | 검증기는 `form_at(...)["chosen"]` 이 None 인 위치에서만 이 답을 쓴다 |

힌트가 보지 않는 것: 어느 위치에 서포트를 둘지 (D80 ⑤ 탐욕 — 간격·도달성), D82 보완 간격 재확인, 묶음 좌표 순서 탐욕.

**대조 (2026-10-27, `python -m pipe_routing.verifier.hint_bench`, 비공개 표):** 기본·밀집 30 (D96 + D105 첫 해), 밀집 50 (S1-6c 경로), 블록 쌍 (S1-7) 경로를 지금 검증기로 다시 판정하고, 검증기가 놓은 서포트마다 힌트를 물었다. 검증기가 판정에 쓴 반올림 전 위치·축으로 묻는다 (출력 좌표는 반올림돼 있어 45° 배관에서 무게 끝자리가 달라진다).

| | A·B·C 서포트 | D89 | D 묶음 후보 쌍 (같은 면 C, ±500 mm) |
|---|---|---|---|
| 호출 | 6,829 (수작업 42 · 기본 1,770 · 밀집 30 1,403 · 밀집 50 2,086 · 블록 쌍 1,528) | 1 | 107 |
| 검증기와 다름 | **0** (형식·면·보강·규격·무게·L·부재 정확히) | **0** | groupable ok 2 — 둘 다 검증기 간격 재확인(D82 보완)에서 묶지 않음. 나머지 105 는 아니라고 답함 (표면 차 79 · 묶은 무게 ≥ 합 20 · L1/L2·간섭 6) |

이 세트들의 검증기 결과에는 D 묶음이 없어 `groupable` 의 "묶음 = ok" 대조는 합성 시험 (`tests/test_s1_hint.py`) 으로 한다.

**호출 비용** (단일 프로세스, Intel Xeon 2.8 GHz, 중앙 / p90 / 최대 ms)

| 세트 | 생성 | `form_at` | `groupable` | `together_at` |
|---|---|---|---|---|
| 수작업 (5배관) | 1.3 | 0.84 / 0.94 / 1.2 | — | — |
| 기본 (10배관) | 4.7 / 7.1 / 8.2 | 1.35 / 1.96 / 3.3 | 3.0 / 3.6 / 3.7 | — |
| 밀집 30 | 17 / 20 / 20 | 1.71 / 2.61 / 5.4 | 3.3 / 5.1 / 6.3 | — |
| 밀집 50 | 31 / 34 / 34 | 2.45 / 4.91 / 9.6 | 4.3 / 8.4 / 9.4 | 13 (1회) |
| 블록 쌍 (24배관) | 15 / 18 / 18 | 2.29 / 3.49 / 8.0 | 4.6 / 6.9 / 7.8 | — |

- `form_at` 은 다른 배관 조각 배열을 처음 한 번 만들고(생성 비용과 별도, 첫 호출에 포함) 이후 재사용한다.
- 전체 검증 1회(밀집 50 중앙 6.7 s) 대비 `form_at` 은 약 1/2,700. 경로를 바꾸면 힌트 객체를 새로 만든다 (생성 수 ms~수십 ms).
