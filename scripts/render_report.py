"""주차 보고서 마크다운을 읽기용 HTML 로 바꾼다.

    python scripts/render_report.py docs/reports/week-02.md

같은 폴더에 같은 이름의 .html 을 쓴다. 발표 자료(docs/slides/)와 같은
색·글꼴을 쓰되, 슬라이드가 아니라 한 줄로 읽는 문서로 배치한다.

목차는 h2 를 훑어 자동으로 만든다. 인쇄하면(Ctrl+P) 목차와 상단 막대가
빠지고 본문만 나온다.
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]

CSS = """
:root{
  color-scheme: light;
  --ground:#f4f6f7; --surface:#fcfcfb; --surface-2:#eceef0;
  --ink:#12161a; --ink-2:#454b52; --ink-3:#7d848c;
  --rule:#c9ced4; --rule-soft:#e2e6e9;
  --accent:#2a78d6; --accent-soft:#dbe8f8;
  --warn:#e34948; --warn-soft:#f8dedd;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    color-scheme: dark;
    --ground:#141719; --surface:#1a1d20; --surface-2:#232729;
    --ink:#f2f4f5; --ink-2:#b9c0c7; --ink-3:#8b939b;
    --rule:#343a3f; --rule-soft:#282d31;
    --accent:#3987e5; --accent-soft:#1d2f45;
    --warn:#e66767; --warn-soft:#3d2526;
  }
}
:root[data-theme="dark"]{
  color-scheme: dark;
  --ground:#141719; --surface:#1a1d20; --surface-2:#232729;
  --ink:#f2f4f5; --ink-2:#b9c0c7; --ink-3:#8b939b;
  --rule:#343a3f; --rule-soft:#282d31;
  --accent:#3987e5; --accent-soft:#1d2f45;
  --warn:#e66767; --warn-soft:#3d2526;
}

*{box-sizing:border-box}
body{margin:0; background:var(--ground); color:var(--ink);
  font-family:"IBM Plex Sans KR","Malgun Gothic",system-ui,sans-serif;
  font-size:16px; line-height:1.78; -webkit-font-smoothing:antialiased}

.topbar{position:sticky; top:0; z-index:40; background:var(--ground);
  border-bottom:3px double var(--rule)}
.topbar-in{max-width:1120px; margin:0 auto; padding:12px 28px;
  display:flex; gap:16px; align-items:baseline; justify-content:space-between}
.brand{font-family:"Nanum Myeongjo","Batang",serif; font-size:17px;
  font-weight:800; letter-spacing:-.02em}
.docno{font-family:"IBM Plex Mono",monospace; font-size:10.5px;
  letter-spacing:.1em; color:var(--ink-3); text-transform:uppercase;
  white-space:nowrap}

.shell{max-width:1120px; margin:0 auto; padding:0 28px;
  display:grid; grid-template-columns:1fr 224px; gap:52px; align-items:start}
main{min-width:0; padding:40px 0 96px}

nav.toc{position:sticky; top:74px; padding:44px 0 0;
  font-size:12.5px; line-height:1.5}
nav.toc .lbl{font-family:"IBM Plex Mono",monospace; font-size:10px;
  letter-spacing:.12em; color:var(--ink-3); text-transform:uppercase;
  padding-bottom:10px; border-bottom:1px solid var(--rule-soft);
  margin-bottom:12px}
nav.toc a{display:block; padding:5px 0; color:var(--ink-3);
  text-decoration:none; border-left:2px solid transparent; padding-left:10px;
  margin-left:-12px; transition:color .15s, border-color .15s}
nav.toc a:hover{color:var(--ink)}
nav.toc a.on{color:var(--ink); border-left-color:var(--accent); font-weight:500}

h1,h2,h3{font-family:"Nanum Myeongjo","Batang",serif; text-wrap:balance;
  letter-spacing:-.01em; line-height:1.35}
h1{font-size:30px; font-weight:800; margin:0 0 6px}
h2{font-size:22px; font-weight:700; margin:56px 0 16px;
  padding-top:22px; border-top:1px solid var(--rule)}
h3{font-size:17px; font-weight:700; margin:32px 0 10px; color:var(--ink)}
h1 + p{color:var(--ink-2); font-size:16px; margin:0 0 4px}
p{margin:0 0 15px}
a{color:var(--accent); text-underline-offset:3px;
  text-decoration-thickness:1px}
strong{font-weight:600}
hr{border:0; border-top:1px solid var(--rule-soft); margin:36px 0}
hr + h2{border-top:0; padding-top:0; margin-top:30px}
ul,ol{margin:0 0 15px; padding-left:22px}
li{margin:0 0 7px}

table{border-collapse:collapse; width:100%; margin:18px 0 22px;
  font-size:14px; background:var(--surface);
  font-variant-numeric:tabular-nums}
th,td{border:1px solid var(--rule-soft); padding:7px 11px;
  text-align:left; vertical-align:top}
th{background:var(--surface-2); font-weight:600; font-size:13px;
  color:var(--ink-2)}
.tw{overflow-x:auto; margin:18px 0 22px}
.tw table{margin:0}

blockquote{margin:18px 0; padding:11px 16px; background:var(--accent-soft);
  border-left:3px solid var(--accent); font-size:14px; color:var(--ink-2)}
blockquote p{margin:0}
blockquote strong{color:var(--ink)}

code{font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:13px;
  background:var(--surface-2); padding:1px 5px; border-radius:3px}
pre{background:var(--surface); border:1px solid var(--rule-soft);
  padding:14px 16px; overflow-x:auto; font-size:13px; line-height:1.6}
pre code{background:none; padding:0}

@media (max-width:940px){
  .shell{grid-template-columns:1fr; gap:0}
  nav.toc{display:none}
}
@media print{
  .topbar, nav.toc{display:none}
  .shell{display:block; max-width:none; padding:0}
  body{background:#fff; font-size:11pt}
  h2{break-after:avoid} table,pre,blockquote{break-inside:avoid}
  a{color:#000; text-decoration:none}
}
"""

JS = """
const heads=[...document.querySelectorAll('main h2[id]')];
const links=[...document.querySelectorAll('nav.toc a')];
const io=new IntersectionObserver(es=>{
  es.forEach(e=>{if(!e.isIntersecting)return;
    const i=heads.indexOf(e.target);
    links.forEach((a,j)=>a.classList.toggle('on',i===j));});
},{rootMargin:'-70px 0px -75% 0px'});
heads.forEach(h=>io.observe(h));
"""

PAGE = """<title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Nanum+Myeongjo:wght@400;700;800&family=IBM+Plex+Sans+KR:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{css}</style>
<div class="topbar"><div class="topbar-in">
  <span class="brand">{brand}</span>
  <span class="docno">{docno}</span>
</div></div>
<div class="shell">
  <main>{body}</main>
  <nav class="toc"><div class="lbl">목차</div>{toc}</nav>
</div>
<script>{js}</script>
"""

_TAG = re.compile(r"<[^>]+>")


def slugify(text: str) -> str:
    """머리글 문자열을 앵커로. 한글은 그대로 두고 공백만 정리한다."""
    s = _TAG.sub("", text).strip()
    s = re.sub(r"[^\w가-힣]+", "-", s, flags=re.UNICODE)
    return s.strip("-").lower()


def render(src: Path) -> Path:
    text = src.read_text(encoding="utf-8")

    md = markdown.Markdown(extensions=["tables", "fenced_code", "attr_list", "sane_lists"])
    body = md.convert(text)

    # 머리글에 id 를 달고 목차를 모은다
    items: list[tuple[str, str]] = []

    def stamp(m: re.Match) -> str:
        inner = m.group(2)
        anchor = slugify(inner)
        if m.group(1) == "2":
            items.append((anchor, _TAG.sub("", inner)))
        return f'<h{m.group(1)} id="{anchor}">{inner}</h{m.group(1)}>'

    body = re.sub(r"<h([23])>(.*?)</h\1>", stamp, body, flags=re.S)

    # 넓은 표는 제 안에서 가로로 스크롤되게 감싼다
    body = body.replace("<table>", '<div class="tw"><table>')
    body = body.replace("</table>", "</table></div>")

    toc = "".join(
        f'<a href="#{a}">{html.escape(t)}</a>' for a, t in items
    )

    # 첫 h1 을 제목으로, 그 아래 메타 줄에서 문서 번호를 만든다
    h1 = re.search(r"<h1[^>]*>(.*?)</h1>", body, re.S)
    title = _TAG.sub("", h1.group(1)).strip() if h1 else src.stem
    docno = src.stem.replace("week-", "WEEK ")

    out = src.with_suffix(".html")
    out.write_text(
        PAGE.format(
            title=f"{title} · 서울 토허구역 인과추정",
            brand="서울 토허구역 인과추정",
            docno=docno,
            css=CSS,
            js=JS,
            body=body,
            toc=toc,
        ),
        encoding="utf-8",
    )
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("사용법: python scripts/render_report.py docs/reports/week-02.md")
    path = Path(sys.argv[1])
    if not path.is_absolute():
        path = ROOT / path
    result = render(path)
    print(f"{result.relative_to(ROOT)}  ({result.stat().st_size // 1024}KB)")
