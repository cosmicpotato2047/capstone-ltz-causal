"""[D10] 학과 제출용 결과보고서(.docx) 생성.

    python scripts/build_report_docx.py

입력  paper/docx/*.md        본문 (장별로 한 파일)
      paper/docx/refs.py     참고문헌 — 학과가 요구하는 형식으로 적어 둔다
      output/figures/*.png
출력  output/report/캡스톤디자인_결과보고서.docx

**`paper/main_ko.tex` 와 별개 문서다.** 학술 보고서(LaTeX, 50쪽)는 그대로
두고, 학과 양식에 맞춘 제출본을 따로 만든다. 내용은 같은 결과에서 오지만
목차 구조와 서식 규정이 다르다.

학과 규정 (받은 안내서):

  A4 · 본문 11pt · 줄간격 180% · 글자체 자유
  참고문헌 표시는 [1] [2] 번호식
  그림·표는 장별 번호 — [그림-1.1], [표-1.1]
  **표·그림의 별도 목차는 만들지 않는다**
  목차에 쪽번호
  PDF 로 제출

줄간격 180% 는 Word 의 '배수 1.8' 로 넣는다. 한글(HWP) 이 Word 의 배수 N 을
N×100% 로 읽으므로, 변환해도 '줄간격 180%' 로 보인다.

**왜 python-docx 인가.** 저장소의 다른 빌드 스크립트가 전부 Python 이고
`check_deps.py`·`run_all.py` 가 Python import 를 훑는다. JS 로 짜면 그 장치
밖으로 샌다.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import io  # noqa: E402

io.setup_stdout()

SRC = io.ROOT / "paper" / "docx"
OUTDIR = io.ROOT / "output" / "report"
OUT = OUTDIR / "캡스톤디자인_결과보고서.docx"

FONT_BODY = "맑은 고딕"
FONT_MONO = "D2Coding"        # 없으면 Word 가 대체한다
PT_BODY = 11
LINE = 1.8                    # 줄간격 180%
GREY = (0x59, 0x59, 0x59)

RX_CITE = re.compile(r"\[@([A-Za-z0-9_]+)\]")
RX_BOLD = re.compile(r"\*\*(.+?)\*\*")
RX_FIG = re.compile(r"^\[그림\]\s*(\S+)\s*\|\s*(.+?)\s*(?:\|\s*([\d.]+)\s*)?$")
RX_TAB = re.compile(r"^\[표\]\s*(.+)$")


# ─── 본문 조각 ──────────────────────────────────────────────────────────────
class Doc:
    """장별 그림·표 번호와 인용 번호를 세면서 문서를 쌓는다."""

    def __init__(self, refs: dict[str, str]):
        from docx import Document
        from docx.enum.text import WD_LINE_SPACING
        from docx.shared import Pt

        self.d = Document()
        self.refs = refs
        self.cited: list[str] = []        # 처음 나온 순서가 곧 [1][2][3]
        self.chapter = 0
        self.nfig = 0
        self.ntab = 0

        sec = self.d.sections[0]
        sec.page_width, sec.page_height = Pt(595.3), Pt(841.9)   # A4
        for m in ("left_margin", "right_margin"):
            setattr(sec, m, Pt(72))
        sec.top_margin = sec.bottom_margin = Pt(72)

        st = self.d.styles["Normal"]
        st.font.name = FONT_BODY
        st.font.size = Pt(PT_BODY)
        st.element.rPr.rFonts.set(
            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}eastAsia",
            FONT_BODY)
        pf = st.paragraph_format
        pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
        pf.line_spacing = LINE
        pf.space_after = Pt(0)

    # ── 낱낱 ────────────────────────────────────────────────────────────
    def cite(self, key: str) -> str:
        if key not in self.cited:
            self.cited.append(key)
        return f"[{self.cited.index(key) + 1}]"

    def runs(self, p, text: str, size=PT_BODY, color=None, bold=False):
        from docx.shared import Pt, RGBColor
        # LaTeX 본문에서 문장을 옮겨 오다 보면 $t$ 같은 수식 기호가 딸려 온다.
        # Word 에는 그대로 글자로 찍히므로 그 자리에서 잡는다.
        if "$" in text:
            sys.exit(f"수식 기호 '$' 가 본문에 남아 있다 — {text[:60]}")
        text = RX_CITE.sub(lambda m: self.cite(m.group(1)), text)
        for i, part in enumerate(RX_BOLD.split(text)):
            if not part:
                continue
            r = p.add_run(part)
            r.font.name = FONT_BODY
            r.font.size = Pt(size)
            r.bold = bold or (i % 2 == 1)
            if color:
                r.font.color.rgb = RGBColor(*color)
            r.element.rPr.rFonts.set(
                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}eastAsia",
                FONT_BODY)
        return p

    def para(self, text="", size=PT_BODY, color=None, bold=False, style=None,
             after=0, indent=True):
        from docx.shared import Pt
        p = self.d.add_paragraph(style=style)
        p.paragraph_format.space_after = Pt(after)
        if indent:
            p.paragraph_format.first_line_indent = Pt(11)
        if text:
            self.runs(p, text, size, color, bold)
        return p

    def heading(self, text: str, level: int):
        from docx.shared import Pt, RGBColor
        if level == 1:
            self.chapter += 1
            self.nfig = self.ntab = 0
            if self.chapter > 1:
                self.d.add_page_break()
        h = self.d.add_heading("", level=level)
        h.paragraph_format.space_before = Pt(0 if level == 1 else 14)
        h.paragraph_format.space_after = Pt(8)
        r = h.add_run(text)
        r.font.name = FONT_BODY
        r.font.size = Pt({1: 16, 2: 13, 3: 11.5}[level])
        r.bold = True
        r.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)
        r.element.rPr.rFonts.set(
            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}eastAsia",
            FONT_BODY)

    def figure(self, name: str, caption: str, width_cm: float):
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Cm, Pt
        self.nfig += 1
        p = self.d.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(10)
        p.add_run().add_picture(str(io.FIGURES / name), width=Cm(width_cm))
        c = self.d.add_paragraph()
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        c.paragraph_format.space_after = Pt(12)
        c.paragraph_format.line_spacing = 1.2
        self.runs(c, f"**[그림-{self.chapter}.{self.nfig}]** {caption}", size=9.5,
                  color=GREY)

    def table(self, caption: str, rows: list[list[str]]):
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Pt
        self.ntab += 1
        c = self.d.add_paragraph()
        c.paragraph_format.space_before = Pt(10)
        c.paragraph_format.space_after = Pt(3)
        c.paragraph_format.line_spacing = 1.2
        self.runs(c, f"**[표-{self.chapter}.{self.ntab}]** {caption}", size=10)

        t = self.d.add_table(rows=len(rows), cols=len(rows[0]))
        t.style = "Table Grid"
        t.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for i, row in enumerate(rows):
            for j, cell in enumerate(row):
                cp = t.cell(i, j).paragraphs[0]
                cp.paragraph_format.line_spacing = 1.2
                cp.paragraph_format.space_after = Pt(0)
                if j > 0:
                    cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
                self.runs(cp, cell, size=10, bold=(i == 0))
        self.d.add_paragraph().paragraph_format.space_after = Pt(6)


# ─── Markdown 읽기 ──────────────────────────────────────────────────────────
def render(doc: Doc, text: str) -> None:
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i].rstrip()
        if not ln or ln.startswith("%"):
            i += 1
            continue
        if ln.startswith("### "):
            doc.heading(ln[4:], 3)
        elif ln.startswith("## "):
            doc.heading(ln[3:], 2)
        elif ln.startswith("# "):
            doc.heading(ln[2:], 1)
        elif m := RX_FIG.match(ln):
            doc.figure(m.group(1), m.group(2), float(m.group(3) or 15))
        elif m := RX_TAB.match(ln):
            rows, i = [], i + 1
            while i < len(lines) and not lines[i].strip():
                i += 1                      # 캡션과 표 사이의 빈 줄
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(set(c) <= set("-: ") for c in cells):
                    rows.append(cells)
                i += 1
            if not rows:
                sys.exit(f"표에 줄이 없다 — '[표] {m.group(1)}' 다음에 | 로 "
                         "시작하는 줄이 있어야 한다")
            doc.table(m.group(1), rows)
            continue
        elif ln.startswith("- "):
            p = doc.para(style="List Bullet", after=2, indent=False)
            doc.runs(p, ln[2:])
            p.paragraph_format.line_spacing = 1.4
        else:
            doc.para(ln, after=6)
        i += 1


def toc(doc: Doc) -> None:
    """목차 — Word 가 열 때 쪽번호를 채우는 필드."""
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    from docx.shared import Pt

    # 제목은 Heading 으로 달지 않는다 — 달면 목차가 자기 자신을 싣는다.
    h = doc.para("목   차", size=16, bold=True, after=14, indent=False)
    h.alignment = 1
    p = doc.d.add_paragraph()
    r = p.add_run()
    for tag, attrs, text in (("w:fldChar", {"w:fldCharType": "begin"}, None),
                             ("w:instrText", {"xml:space": "preserve"},
                              r' TOC \o "1-3" \h \z \u '),
                             ("w:fldChar", {"w:fldCharType": "separate"}, None),
                             ("w:t", {}, "목차를 갱신하려면 F9 를 누르세요."),
                             ("w:fldChar", {"w:fldCharType": "end"}, None)):
        el = OxmlElement(tag)
        for k, v in attrs.items():
            el.set(qn(k), v)
        if text:
            el.text = text
        r._r.append(el)
    p.paragraph_format.space_after = Pt(0)


def titlepage(doc: Doc, meta: dict) -> None:
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt
    for _ in range(4):
        doc.para(after=0, indent=False)
    for text, size, bold in [
        ("2026학년도 2학기 캡스톤디자인(2)", 13, False),
        ("", 6, False),
        ("결과보고서", 26, True),
        ("", 10, False),
        (meta["제목"], 17, True),
        (meta["부제"], 12, False),
    ]:
        p = doc.para(text, size=size, bold=bold, after=4, indent=False)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for _ in range(6):
        doc.para(after=0, indent=False)
    for k, v in [("학    과", meta["학과"]), ("학    번", meta["학번"]),
                 ("성    명", meta["성명"]), ("지도교수", meta["지도교수"]),
                 ("제 출 일", meta["제출일"])]:
        p = doc.para(f"{k}    {v}", size=12, after=6, indent=False)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.d.add_page_break()
    doc.chapter = 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="이 파일 하나만 (디버그용)")
    a = ap.parse_args()

    sys.path.insert(0, str(io.ROOT / "paper"))
    import refs_kr as R                                     # noqa: E402

    doc = Doc(R.REFS)
    titlepage(doc, R.META)
    toc(doc)
    doc.d.add_page_break()

    files = sorted(SRC.glob("[0-9]*.md"))
    if a.only:
        files = [f for f in files if a.only in f.name]
    if not files:
        sys.exit(f"본문이 없다 — {SRC.relative_to(io.ROOT)}/*.md")
    for f in files:
        render(doc, f.read_text(encoding="utf-8"))

    # 참고문헌은 인용된 순서로 — 7장 자리에 쌓는다
    doc.heading("7. 참고 자료, URL, 문헌", 1)
    for n, key in enumerate(doc.cited, 1):
        if key not in doc.refs:
            print(f"  [없음] 참고문헌 '{key}' 가 refs.py 에 없다")
        doc.para(f"[{n}] {doc.refs.get(key, '??')}", after=6, indent=False)
    for key in doc.refs:
        if key not in doc.cited:
            print(f"  [안 쓰임] {key}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    doc.d.save(str(OUT))
    print(f"  {OUT.relative_to(io.ROOT)}")
    print(f"  본문 {len(files)}개 · 참고문헌 {len(doc.cited)}편 인용")
    print("  Word 에서 열어 목차 위에서 F9 (쪽번호 채우기) 후 PDF 로 저장")


if __name__ == "__main__":
    main()
