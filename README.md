# pipe-routing-skeleton

Pipe Routing Walking Skeleton (S0). 사양·결정 기록은 [CLAUDE.md](CLAUDE.md).

## 1단계 — 입력 스키마 + 시나리오

```
pip install -r requirements.txt

# 시나리오 로드 검사 (요약 출력, -v 는 경고 출력)
python -m pipe_routing.scenario scenarios/manual/*.json scenarios/procedural/*.json

# procedural 세트 재생성 (기본 20개, 시드 20261008)
python -m pipe_routing.generator --out scenarios/procedural --n 20

# 테스트
python -m unittest discover -s tests -t .
```

| 경로 | 내용 |
|---|---|
| `schemas/scenario_input.schema.json` | 입력 JSON 스키마 (CLAUDE.md §7.1) |
| `pipe_routing/constants.py` | 규격 상수 (§3, D17·D18) |
| `pipe_routing/geometry.py` | 축 정렬 박스 거리·합집합 부피 |
| `pipe_routing/scenario.py` | 로드·검증(D33)·저장 |
| `pipe_routing/generator.py` | procedural 생성기 (v3 `gen_obstacle_grid` → 박스 목록, D30·D31) |
| `scenarios/manual/manual_01.json` | 수작업 시나리오 (기관실형, 장애물 6 · 배관 5) |
| `scenarios/procedural/` | procedural 세트 20개 (장애물 21~29 · 배관 10) |

## 2단계 — 3D 시각화

```
python -m pipe_routing.viz scenarios/manual/manual_01.json            # → out/viz/manual_01.html
python -m pipe_routing.viz --offline -o out/viz scenarios/procedural/*.json
```

장애물(회색 박스), 단자(● start / ◆ end, 채움 = 노즐 · 빈 마커 = 경계), 흐름 방향 화살표(D29),
최소 직진 구간(굵은 선, 중력관은 점선), start–end 짝 표시(가는 점선 — 경로 아님). 범례 클릭으로 배관별 켜고 끄기.

## 3단계 — 인터페이스 + escape graph

```
python -m pipe_routing.escape_graph scenarios/manual/manual_01.json        # 배관별 노드·엣지 수
python -m pipe_routing.viz scenarios/manual/manual_01.json --graph P002 --slice z=1500   # 그래프 단면 표시
```

| 경로 | 내용 |
|---|---|
| `pipe_routing/space.py` | 라우터 슬롯 인터페이스 (`SpaceRepresentation`, `State`, `route`) — D37 |
| `pipe_routing/escape_graph.py` | S0 표현 escape graph (축 + 평면 내 45°) — D38·D39 |

## 4단계 — 단일 배관 A*

```
python -m pipe_routing.router_astar scenarios/manual/manual_01.json --viz out/viz   # 경로·J·탐색 통계 + 경로 HTML
python -m pipe_routing.router_astar scenarios/procedural/*.json --json out/astar.jsonl
python -m pipe_routing.router_astar --no-45 ...                                       # M9 비교 실험 (45° 엣지 없음)
```

`pipe_routing/router_astar.py` — 인터페이스(D37)만 쓰는 A*. 비용 = J(직관 + 엘보), 허용 휴리스틱으로 그래프 위 J 최적(D40·D41),
배관당 60초 초과 시 timeout, 탐색 완료 후 경로 없음은 unreachable.
