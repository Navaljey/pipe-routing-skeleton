"""템플릿 편집기 아티팩트 만들기.
python tools/template_editor/build.py <template.yaml> <out.html> [plotly.min.js]
plotly.min.js 는 npm i plotly.js-dist-min@2.35.2 로 받는다 (페이지 안에 넣어 외부 주소 없이 열리게)."""
import json, subprocess, sys, pathlib
here = pathlib.Path(__file__).parent
tpl, out = sys.argv[1], sys.argv[2]
plotly = sys.argv[3] if len(sys.argv) > 3 else "node_modules/plotly.js-dist-min/plotly.min.js"
tmp = pathlib.Path(out).with_suffix(".json")
subprocess.run([sys.executable, str(here / "expand.py"), tpl, str(tmp)], check=True)
D = json.load(open(tmp))
js = open(plotly).read().replace("</script>", "<\\/script>").replace("�", "\\ufffd")
html = (here / "page.html").read_text().replace("<script>__PLOTLY__</script>", "<script>" + js + "</script>").replace("__DATA__", json.dumps(D, ensure_ascii=False))
pathlib.Path(out).write_text(html)
print(out, len(html) // 1024, "KB")
