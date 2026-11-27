# 템플릿 편집기 (임시 — 사양 세션 인계본, 2026-11-24)

- `expand.py` — 템플릿 YAML 을 박스·노즐·계통으로 펼치고 겹침·단자 직진 구간을 점검한다. **공식 로더 (S2-T) 가 아니다.**
  S2-T 로더가 나오면 이 파일 대신 로더 출력을 그리도록 바꾼다 (펼치기 코드 두 벌 금지).
- `page.html` — 편집기 화면 틀 (평면도 편집 · 3D · 성격 편집 · YAML 복사). `__PLOTLY__`·`__DATA__` 자리를 build.py 가 채운다.
- `build.py` — `python tools/template_editor/build.py templates/er_2nd_generic_a.yaml out/editor.html`
  (먼저 `npm i plotly.js-dist-min@2.35.2`). 결과 HTML 을 아티팩트로 올린다.
- 점검 기준: 장애물 겹침 (의도된 것 — 계단·업테이크의 데크 관통, 천장 구조끼리 — 제외) ·
  단자 직진 구간 (노즐 = max(§3.3, t₉₀), 데크 관통 = max(3D, r) + t₉₀) 이 다른 장애물과 유효 반경 r 이상, 접근 구역 침범 없음.

`templates/er_2nd_generic_a.yaml` — 형식 2 (D136) · 학습용 (D137 ⑤). `expand.py` 는 임시 펼치기 — S2-T 로더가 형식 2 를 읽으면 편집기는 로더 출력을 그리고 이 파일은 지운다 (D136 ⑦).
