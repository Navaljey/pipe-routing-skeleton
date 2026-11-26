"""템플릿 편집기 아티팩트 만들기 — S2-T 로더 (pipe_routing.template) 의 펼친 결과와 점검을 그린다 (D136 ⑦, 펼치기 코드 하나).

python tools/template_editor/build.py <template.yaml> <out.html> [--seed N] [plotly.min.js]

- 시드가 없으면 기준 배치 (--base: 장비 변형·배관 늘리기·단자 흔들기 없음).
- 로더 오류 (형식 2 위반) 가 있으면 페이지를 만들지 않고 오류를 찍는다.
- plotly.min.js 는 npm i plotly.js-dist-min@2.35.2 로 받는다 (페이지 안에 넣어 외부 주소 없이 열리게).
"""
import argparse
import json
import pathlib
import sys

here = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(here.parents[1]))
from pipe_routing import template  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("template")
    ap.add_argument("out")
    ap.add_argument("plotly", nargs="?", default="node_modules/plotly.js-dist-min/plotly.min.js")
    ap.add_argument("--seed", type=int, default=None, help="변형 시드 (없으면 기준 배치)")
    a = ap.parse_args(argv)
    t, sha = template.load_template(a.template)
    try:
        sc = template.expand(t, sha, a.seed, base=a.seed is None)
    except template.TemplateError as e:
        for m in e.errors:
            print("ERR", m)
        return 2
    D = template.editor_data(t, sc, template.check(sc))
    js = pathlib.Path(a.plotly).read_text().replace("</script>", "<\\/script>").replace("�", "\\ufffd")
    html = (here / "page.html").read_text().replace("<script>__PLOTLY__</script>", "<script>" + js + "</script>") \
        .replace("__DATA__", json.dumps(D, ensure_ascii=False))
    pathlib.Path(a.out).write_text(html)
    print(a.out, len(html) // 1024, "KB ·", len(D["boxes"]), "boxes ·", len(D["systems"]), "systems ·",
          len(D["clash"]), "clash ·", len(D["issues"]), "issues")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
