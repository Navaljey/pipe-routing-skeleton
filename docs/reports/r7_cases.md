# R7 부재 막힘 사례 목록

다른 배관의 서포트 부재 자리 때문에 support 위반이 된 배관 (단독 검증 통과) 의 목록. 생성: `python -m pipe_routing.r7_cases` (출력 재분석, 결과 불변). 서포트 표: 사내 표준 기반(비공개), sha256 `ac2ba6a2974b…`. 선정표 값은 쓰지 않는다 (D78).

막힘 유형 (relation, `support_diag.relation`): p_vertical_45 = 막힌 배관 구간이 수직·45° · q_vertical_45 = 막는 배관이 수직·45° · crossing = 둘 다 축 정렬 수평이고 축이 다름 · parallel_le50 / _50_300 / _gt300 = 같은 축 수평 평행, 지지면 쪽 표면 높이 차 (mm).

## 요약

| 경우 | 세트 | 시나리오 | support 위반 | 부재 막힘 | (가) 나중 | (나) 먼저 | 양쪽 | 막힘 유형 | 막힌 부재 (형식/면) |
|---|---|---|---|---|---|---|---|---|---|
| 첫 해 (순차 배치, D50) | manual | 1 | 0 | 0 | 0 | 0 | 0 | - | - |
| 예산 끝 (anytime 최선 해, D68·D75) | manual | 1 | 0 | 0 | 0 | 0 | 0 | - | - |
| 첫 해 (순차 배치, D50) | base | 20 | 1 | 1 | 0 | 1 | 0 | parallel_gt300 1 | C/floor 1 |
| 예산 끝 (anytime 최선 해, D68·D75) | base | 20 | 1 | 1 | 0 | 1 | 0 | parallel_gt300 1 | C/floor 1 |
| 첫 해 (순차 배치, D50) | stair | 5 | 0 | 0 | 0 | 0 | 0 | - | - |
| 예산 끝 (anytime 최선 해, D68·D75) | stair | 5 | 0 | 0 | 0 | 0 | 0 | - | - |
| 첫 해 (순차 배치, D50) | dense30 | 5 | 5 | 5 | 0 | 5 | 0 | p_vertical_45 5 | B/wall_y1 1, C/ceiling 3, C/floor 1 |
| 예산 끝 (anytime 최선 해, D68·D75) | dense30 | 5 | 2 | 2 | 0 | 2 | 0 | p_vertical_45 2 | B/wall_y1 1, C/ceiling 1 |
| 첫 해 (순차 배치, D50) | dense50 | 5 | 13 | 13 | 0 | 13 | 0 | crossing 1, p_vertical_45 6, parallel_50_300 1, parallel_gt300 4, q_vertical_45 1 | B/wall_x1 1, B/wall_y0 1, C/ceiling 8, C/floor 4 |
| 예산 끝 (anytime 최선 해, D68·D75) | dense50 | 5 | 12 | 12 | 0 | 12 | 0 | crossing 1, p_vertical_45 6, parallel_50_300 1, parallel_gt300 3, q_vertical_45 1 | B/wall_x1 1, B/wall_y0 1, C/ceiling 8, C/floor 3 |
| 첫 해 (순차 배치, D50) | pair | 5 | 1 | 1 | 0 | 1 | 0 | parallel_gt300 1 | C/floor 1 |
| 예산 끝 (anytime 최선 해, D68·D75) | pair | 5 | 1 | 1 | 0 | 1 | 0 | parallel_gt300 1 | C/floor 1 |

## 첫 해 (순차 배치, D50) — manual (`out/e1_ana/first/manual`)

support 위반 없음.

## 예산 끝 (anytime 최선 해, D68·D75) — manual (`out/e1_ana/final/manual`)

support 위반 없음.

## 첫 해 (순차 배치, D50) — base (`out/e1_ana/first/base`)

| 시나리오 | 배관 | 구경 | 타입 | 단독 통과 | 막는 배관 (구경, 표면 간격 mm) | 순서 | 유형 | 막힌 부재 |
|---|---|---|---|---|---|---|---|---|
| proc_000 | P010 | 25A | T5 | 예 | P008 (50A, 250) | (나) 막는 배관이 먼저 | parallel_gt300 | C/floor |

## 예산 끝 (anytime 최선 해, D68·D75) — base (`out/e1_ana/final/base`)

| 시나리오 | 배관 | 구경 | 타입 | 단독 통과 | 막는 배관 (구경, 표면 간격 mm) | 순서 | 유형 | 막힌 부재 |
|---|---|---|---|---|---|---|---|---|
| proc_000 | P010 | 25A | T5 | 예 | P008 (50A, 250) | (나) 막는 배관이 먼저 | parallel_gt300 | C/floor |

## 첫 해 (순차 배치, D50) — stair (`out/e1_ana/first/stair`)

support 위반 없음.

## 예산 끝 (anytime 최선 해, D68·D75) — stair (`out/e1_ana/final/stair`)

support 위반 없음.

## 첫 해 (순차 배치, D50) — dense30 (`out/e1_ana/first/dense30`)

| 시나리오 | 배관 | 구경 | 타입 | 단독 통과 | 막는 배관 (구경, 표면 간격 mm) | 순서 | 유형 | 막힌 부재 |
|---|---|---|---|---|---|---|---|---|
| dense30_000 | P006 | 50A | T5 | 예 | P008 (150A, 2280) | (나) 막는 배관이 먼저 | p_vertical_45 | C/ceiling |
| dense30_001 | P009 | 65A | T5 | 예 | P016 (500A, 281) | (나) 막는 배관이 먼저 | p_vertical_45 | C/ceiling |
| dense30_002 | P026 | 15A | T1 | 예 | P001 (65A, 100) | (나) 막는 배관이 먼저 | p_vertical_45 | B/wall_y1 |
| dense30_003 | P007 | 50A | T1 | 예 | P008 (200A, 32) | (나) 막는 배관이 먼저 | p_vertical_45 | C/floor |
| dense30_004 | P025 | 15A | T1 | 예 | P021 (50A, 408) | (나) 막는 배관이 먼저 | p_vertical_45 | C/ceiling |

## 예산 끝 (anytime 최선 해, D68·D75) — dense30 (`out/e1_ana/final/dense30`)

| 시나리오 | 배관 | 구경 | 타입 | 단독 통과 | 막는 배관 (구경, 표면 간격 mm) | 순서 | 유형 | 막힌 부재 |
|---|---|---|---|---|---|---|---|---|
| dense30_000 | P006 | 50A | T5 | 예 | P008 (150A, 2280) | (나) 막는 배관이 먼저 | p_vertical_45 | C/ceiling |
| dense30_002 | P026 | 15A | T1 | 예 | P001 (65A, 100) | (나) 막는 배관이 먼저 | p_vertical_45 | B/wall_y1 |

## 첫 해 (순차 배치, D50) — dense50 (`out/e1_ana/first/dense50`)

| 시나리오 | 배관 | 구경 | 타입 | 단독 통과 | 막는 배관 (구경, 표면 간격 mm) | 순서 | 유형 | 막힌 부재 |
|---|---|---|---|---|---|---|---|---|
| dense50_000 | P007 | 65A | T5 | 예 | P034 (100A, 81) | (나) 막는 배관이 먼저 | p_vertical_45 | B/wall_y0 |
| dense50_000 | P017 | 50A | T5 | 예 | P007 (65A, 52) | (나) 막는 배관이 먼저 | crossing | B/wall_x1, C/ceiling |
| dense50_000 | P033 | 400A | T5 | 예 | P024 (500A, 17), P026 (400A, 214) | (나) 막는 배관이 먼저 | parallel_gt300 | C/floor |
| dense50_000 | P040 | 400A | T1 | 예 | P002 (400A, 114) | (나) 막는 배관이 먼저 | p_vertical_45 | C/ceiling |
| dense50_001 | P012 | 15A | T1 | 예 | P028 (65A, 11) | (나) 막는 배관이 먼저 | p_vertical_45 | C/ceiling |
| dense50_002 | P005 | 65A | T1 | 예 | P037 (200A, 436) | (나) 막는 배관이 먼저 | p_vertical_45 | C/ceiling |
| dense50_002 | P024 | 400A | T1 | 예 | P008 (500A, 1535) | (나) 막는 배관이 먼저 | p_vertical_45 | C/ceiling |
| dense50_002 | P040 | 400A | T1 | 예 | P024 (400A, 835) | (나) 막는 배관이 먼저 | parallel_gt300 | C/floor |
| dense50_002 | P041 | 50A | T5 | 예 | P034 (65A, 192) | (나) 막는 배관이 먼저 | parallel_gt300 | C/floor |
| dense50_002 | P043 | 50A | T1 | 예 | P012 (65A, 100) | (나) 막는 배관이 먼저 | p_vertical_45 | C/ceiling |
| dense50_004 | P007 | 25A | T1 | 예 | P003 (50A, 43) | (나) 막는 배관이 먼저 | parallel_50_300 | C/floor |
| dense50_004 | P020 | 100A | T5 | 예 | P041 (150A, 151) | (나) 막는 배관이 먼저 | parallel_gt300 | C/ceiling |
| dense50_004 | P024 | 25A | T1 | 예 | P006 (400A, 7) | (나) 막는 배관이 먼저 | q_vertical_45 | C/ceiling |

## 예산 끝 (anytime 최선 해, D68·D75) — dense50 (`out/e1_ana/final/dense50`)

| 시나리오 | 배관 | 구경 | 타입 | 단독 통과 | 막는 배관 (구경, 표면 간격 mm) | 순서 | 유형 | 막힌 부재 |
|---|---|---|---|---|---|---|---|---|
| dense50_000 | P007 | 65A | T5 | 예 | P034 (100A, 81) | (나) 막는 배관이 먼저 | p_vertical_45 | B/wall_y0 |
| dense50_000 | P017 | 50A | T5 | 예 | P007 (65A, 52) | (나) 막는 배관이 먼저 | crossing | B/wall_x1, C/ceiling |
| dense50_000 | P040 | 400A | T1 | 예 | P002 (400A, 114) | (나) 막는 배관이 먼저 | p_vertical_45 | C/ceiling |
| dense50_001 | P012 | 15A | T1 | 예 | P028 (65A, 11) | (나) 막는 배관이 먼저 | p_vertical_45 | C/ceiling |
| dense50_002 | P005 | 65A | T1 | 예 | P037 (200A, 436) | (나) 막는 배관이 먼저 | p_vertical_45 | C/ceiling |
| dense50_002 | P024 | 400A | T1 | 예 | P008 (500A, 1535) | (나) 막는 배관이 먼저 | p_vertical_45 | C/ceiling |
| dense50_002 | P040 | 400A | T1 | 예 | P024 (400A, 835) | (나) 막는 배관이 먼저 | parallel_gt300 | C/floor |
| dense50_002 | P041 | 50A | T5 | 예 | P034 (65A, 192) | (나) 막는 배관이 먼저 | parallel_gt300 | C/floor |
| dense50_002 | P043 | 50A | T1 | 예 | P012 (65A, 100) | (나) 막는 배관이 먼저 | p_vertical_45 | C/ceiling |
| dense50_004 | P007 | 25A | T1 | 예 | P003 (50A, 43) | (나) 막는 배관이 먼저 | parallel_50_300 | C/floor |
| dense50_004 | P020 | 100A | T5 | 예 | P041 (150A, 151) | (나) 막는 배관이 먼저 | parallel_gt300 | C/ceiling |
| dense50_004 | P024 | 25A | T1 | 예 | P006 (400A, 7) | (나) 막는 배관이 먼저 | q_vertical_45 | C/ceiling |

## 첫 해 (순차 배치, D50) — pair (`out/e1_ana/first/pair`)

| 시나리오 | 배관 | 구경 | 타입 | 단독 통과 | 막는 배관 (구경, 표면 간격 mm) | 순서 | 유형 | 막힌 부재 |
|---|---|---|---|---|---|---|---|---|
| pair_004 | P009 | 400A | T1 | 예 | P003 (400A, 1564) | (나) 막는 배관이 먼저 | parallel_gt300 | C/floor |

## 예산 끝 (anytime 최선 해, D68·D75) — pair (`out/e1_ana/final/pair`)

| 시나리오 | 배관 | 구경 | 타입 | 단독 통과 | 막는 배관 (구경, 표면 간격 mm) | 순서 | 유형 | 막힌 부재 |
|---|---|---|---|---|---|---|---|---|
| pair_004 | P009 | 400A | T1 | 예 | P003 (400A, 1564) | (나) 막는 배관이 먼저 | parallel_gt300 | C/floor |
