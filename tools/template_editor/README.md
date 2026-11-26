# 템플릿 편집기 (D136 ⑦)

편집기는 **S2-T 로더 (`pipe_routing/template.py`) 가 펼친 결과와 점검**을 그린다 — 펼치기 코드는 로더 하나뿐이다
(사양 세션 인계본의 임시 `expand.py` 는 지웠다).

- `build.py` — `python tools/template_editor/build.py <template.yaml> out/editor.html [--seed N]`
  (먼저 `npm i plotly.js-dist-min@2.35.2`). 시드가 없으면 기준 배치 (로더 `--base`). 결과 HTML 을 아티팩트로 올린다.
  로더 오류 (형식 2 위반, D136·D137) 가 있으면 페이지를 만들지 않고 오류를 찍는다.
- `page.html` — 화면 틀 (평면도 · 3D · 성격 편집 · YAML 복사). `__PLOTLY__`·`__DATA__` 자리를 build.py 가 채운다.
  자료 = `template.editor_data(...)`: 장애물마다 성격 그룹 (`attribute_defaults` 키) · 최종 성격, 계통 (주관·다리·단자),
  겹침 (`clash`), 점검 (`issues`).
- 점검 (로더 `template.check`, 경고 층): 장애물 겹침 (의도된 겹침 규칙 = `template.INTENDED_OVERLAPS`) ·
  단자 직진 구간 (노즐 = max(§3.3, t₉₀), 데크·격벽 관통 = max(3D, r) + t₉₀) 이 다른 장애물과 r 이상 ·
  접근 구역 침범 · 로더 시나리오 검사 (D33).
