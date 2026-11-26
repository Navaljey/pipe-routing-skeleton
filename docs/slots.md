# 교체 슬롯 인터페이스 (CLAUDE.md §9-5, D3·D49·D50)

S0 는 두 자리에 다른 방법(V3 / GNARL 등)을 꽂아 **같은 시나리오 · 같은 검증기 · 같은 지표**로 비교한다.
검증기(`pipe_routing/verifier`)와 지표(`pipe_routing/pipeline.py`)는 고정이다.

```
pipeline.run(scenario, router=..., planner=..., time_limit=60)
          │
          ├─ planner(scenario, router, time_limit) ─→ PlanResult      ← 다중 배관 슬롯 (순서·rip-up)
          │       └─ router(scenario, pipe, time_limit, placed) ─→ RouteResult   ← 라우터 슬롯 (배관 1개)
          ├─ verify(...)            검증기 Layer 0 (D44)
          └─ J 분해 · 출력 JSON · 리포트 (D49)
```

## 1. 라우터 슬롯

```python
def router(scenario: Scenario, pipe: Pipe, time_limit: float, placed=()) -> RouteResult: ...
```

| 입력 | 내용 |
|---|---|
| `scenario` | `pipe_routing.scenario.Scenario` — 블록, 장애물(축 정렬 박스), 배관 목록 (§7.1) |
| `pipe` | 라우팅할 배관 1개 (`Pipe`: id, nominal_size, radius(유효 반경 D17), start/end 단자 (pos, kind, dir), gravity_pipe …) |
| `time_limit` | 초. 넘기면 `status="timeout"` 으로 돌려준다 (D40②) |
| `placed` | 이미 놓인 배관 `[(Pipe, waypoints), ...]` — 장애물로 본다. 이격 r₁ + r₂ (D50②) |

출력 = `pipe_routing.router_astar.RouteResult` 또는 같은 속성을 가진 객체:

| 속성 | 필수 | 내용 |
|---|---|---|
| `pipe_id` | ✔ | 배관 id |
| `status` | ✔ | `"ok"` · `"timeout"` · `"unreachable"` (D34·D40) |
| `waypoints` | ✔ (ok) | 꺾임점 좌표 목록 mm, start·end 포함 — 검증기 입력 (D44) |
| `search_sec` | ✔ | 탐색 시간 |
| `expanded` | | 확장 상태 수 (없으면 0) |
| `J` | | 라우터가 최소화한 값 (참고용 — 출력 J 는 검증기 기하로 다시 계산한다) |
| `graph_sec` | | 표현 준비 시간 (D50③ 기록) |
| `J_support_est` | | 라우터 추정 서포트 (D57, §9-6 비교용) |

규약: 경로는 단자 좌표에서 시작·끝나고, 첫·끝 구간 방향은 단자 `dir` (D29). 편향각은 45·90·135° (D15).
검증기가 실제 기하(엘보 R = 1.5D 호 포함)로 판정하므로, 라우터가 규칙을 어기면 그 배관은 실패로 집계된다.

**기본 구현:** `pipe_routing.layered.layered_router_a` (D55 2층 구조 A + D57 추정 서포트, 컴파일 A*).
비교용: `layered_router_b` (국소 이격선), `pipe_routing.multi.astar_router` (배관마다 escape graph 재생성, 7단계 방식).

예 — 직선만 시도하는 장난감 라우터:

```python
from pipe_routing.router_astar import RouteResult

def straight_router(scenario, pipe, time_limit, placed=()):
    a, b = pipe.start.pos, pipe.end.pos
    if sum(x != y for x, y in zip(a, b)) == 1 and tuple(pipe.start.dir) == tuple(pipe.end.dir):
        return RouteResult(pipe.id, "ok", waypoints=[list(a), list(b)], search_sec=0.0)
    return RouteResult(pipe.id, "unreachable", search_sec=0.0)

from pipe_routing.pipeline import run
from pipe_routing.scenario import load
out = run(load("scenarios/manual/manual_01.json"), router=straight_router)
```

### 1.1 분기 계통 (S2-3, D112·D134)

분기 배관 (`pipe.branches` 1~5개) 의 라우터는 같은 `RouteResult` 에 **나무** `tree` 속성을 더해 돌려준다. 분기 없는 배관은 지금과 같다 (`tree` 없음).

| `tree` 키 | 내용 |
|---|---|
| `method` | `exact` (분기 1개 정확 풀이) · `exact_retry` (자기 간섭으로 티 후보를 막고 다시, 최대 8회) · `sequential` (순차 근사, D112 Q9) |
| `legs[]` | `{leg_id, nominal_size, flow, found, waypoints, tee {pos, angle}}` — `waypoints` 는 **기하 순서** (티 → 다리 단자), 흐름은 `flow` (D134 보완 2) |
| `leg_search` · `leg_search_final` | 다리 표 상태 (`done` 끝까지 · `bounded` 한계 c 에서 멈춤 · `budget` 예산 소진) — 처음 / 순차 근사 전 다시 만든 뒤 (F1) |
| `search` | 확장 수 분해 `{main_solo, leg_tables[], tee_search[], leg_rebuild[]}` (보고용) |

- 주관 경로 = `waypoints` (분기 없는 배관과 같은 뜻). 계통 성공 = 주관과 모든 다리가 Layer 0 통과 (D112 Q10), 찾은 다리는 실패해도 출력에 남는다.
- 예산 = 배관당 라우팅 상한 × (다리 수 + 1) (D112 Q6), 진단 = D127 식.
- 다른 선수는 나무를 어떻게 만들든 같은 `tree` 형식으로 돌려주면 된다 — 심판은 최종 나무만 채점한다 (D134 보완 1).
- `leg_router(scenario, pipe, main_waypoints, leg_id, placed, budget)` (주관 고정, 주관 직관 위 목표 집합 — 기준선 순차 근사와 같은 함수) 는 S2-4 에서 공개한다 (D134 보완 4).

## 2. 다중 배관 슬롯

```python
def planner(scenario: Scenario, router, time_limit: float) -> PlanResult: ...
```

`router` 는 위 라우터 슬롯 함수. planner 는 순서·재배치(rip-up)를 정하고 router 를 부른다.
출력 = `pipe_routing.multi.PlanResult`:

| 속성 | 필수 | 내용 |
|---|---|---|
| `routes` | ✔ | `{pipe_id: RouteResult}` — 최종 결과 (모든 배관) |
| `order` | ✔ | 배치 순서 (배관 id 목록) |
| `planner` | ✔ | 이름 (출력 JSON 기록용) |
| `ripups`, `reverted` | | rip-up 횟수·되돌린 횟수 (§6.3 reroute_count) |
| `fail_class`, `fail_detail` | | 최종 실패 배관 3분류 (D51): `interference_block` · `interference_search` · `individual` |
| `attempts` | | `{pipe_id: [{"status", "graph_sec", "search_sec", "n_obstacle_pipes"}]}` — 라우팅 시도 기록 |
| `events` | | 로그 문자열 |

`multi._route(res, ...)` 를 거쳐 router 를 부르면 `attempts` 가 자동으로 쌓인다.

**기본 구현:** `sequential_ripup_planner` (D50: 호칭경·맨해튼 거리 내림차순, 놓인 배관 = 장애물, rip-up 최대 3회,
성공 수가 늘 때만 유지 D52). 비교용: `independent_planner` (배관 단독, 6단계 방식).

예 — 입력 순서대로 깔기만 하는 planner:

```python
from pipe_routing.multi import PlanResult, _route

def in_order(scenario, router, time_limit):
    res = PlanResult(routes={}, order=[p.id for p in scenario.pipes], planner="in_order")
    placed = []
    for p in scenario.pipes:
        r = _route(res, scenario, p, router, time_limit, list(placed))
        res.routes[p.id] = r
        if r.status == "ok":
            placed.append((p, r.waypoints))
    return res

out = run(load("scenarios/manual/manual_01.json"), planner=in_order)
```

## 3. 표현 (선택)

라우터는 표현 독립 인터페이스(§4.1, `pipe_routing/space.py` `SpaceRepresentation`: `start_state` · `is_goal` ·
`neighbors` · `cost` · `position`) 위에서 동작한다 (D9). 새 휴리스틱만 바꾸려면 기존 표현에
`router_astar.astar_route(space, pipe, time_limit, heuristic=h)` 로 넘기면 된다 (파이썬 구현으로 실행).

## 4. CLI

```
python -m pipe_routing.pipeline <json>... -o out/run [--router layered-a|layered-b|astar] [--no-support-cost]
                                                     [--planner sequential|independent] [--time-limit 60]
```
