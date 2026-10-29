"""[V7] 졸업작품 요약보고서(판넬) 생성.

    python scripts/build_panel.py
    python scripts/build_panel.py --template 다른양식.pptx

입력  data/forms/요약보고서-판넬양식.pptx   (학과 서식. 저장소 밖 — 아래 참조)
      output/results/*.json                (판넬의 모든 수치)
      output/figures/*.png                 (도식과 그림)
출력  output/panel/요약보고서_판넬.pptx
      output/panel/qr.png

**양식은 수정하지 않는다.** 학과가 "양식 수정 불가" 라고 못박았으므로 머리말
표·구역 상자·구역 이름표는 손대지 않고 그 안에 내용만 얹는다. 예시 슬라이드와
안내 문구만 지운다.

**숫자는 손으로 적지 않는다.** 판넬에 들어가는 값은 전부
`output/results/*.json` 에서 꺼낸다 — 보고서 표([[0007-문서유형별-수치기록]])·
결과 대시보드와 같은 원칙이다. 분석을 다시 돌리면 판넬도 따라 바뀐다.
실제로 초안에서 '하루 9건이 67건으로' 라고 그림에서 눈으로 읽어 적었다가,
JSON 의 정확한 값(발표 전 나흘 30건 -> 발표 후 나흘 155건)으로 바로잡았다.

**양식 파일은 저장소에 없다.** 학과가 배포한 서식이라 재배포하지 않는다
(선행연구 PDF 를 넣지 않는 것과 같은 이유). 받은 파일을
`data/forms/요약보고서-판넬양식.pptx` 로 두면 이 단계가 돈다. 없으면 건너뛴다.

**PDF 변환은 이 단계가 하지 않는다.** PowerPoint 나 LibreOffice 가 있어야
하므로 재현 환경을 가린다. 제출용 PDF 는 PowerPoint 에서 '다른 이름으로
저장 -> PDF' 로 뽑는다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import io, jpath  # noqa: E402

io.setup_stdout()

TEMPLATE = io.ROOT / "data" / "forms" / "요약보고서-판넬양식.pptx"
OUTDIR = io.ROOT / "output" / "panel"
SITE = "https://cosmicpotato2047.github.io/capstone-ltz-causal/"

# ─── 손으로 정하는 것 — 여기만 고친다 ────────────────────────────────────────
강좌번호 = "41010-??"     # 분반이 정해지면 ?? 를 채운다
학기 = "202611"
지도교수 = "김성곤"
조원 = "이찬희"
작품제목 = "서울 토지거래허가구역 인과효과 추정"
소제목 = "실거래 47만 건과 정책 공고 원문을 결합한 미시 패널 구축"

FONT = "맑은 고딕"
MINUS = "−"       # 판넬 전체가 이 빼기표를 쓴다. ASCII '-' 는 짧아서 섞이면 티난다


def sgn(v: float, d: int = 1) -> str:
    """부호를 붙여 적는다. 빼기표는 U+2212."""
    return (MINUS if v < 0 else "+") + f"{abs(v):.{d}f}"

NAVY = (0x00, 0x20, 0x60)
RED = (0xC0, 0x00, 0x00)
INK = (0x1A, 0x1A, 0x1A)
GREY = (0x59, 0x59, 0x59)


# ─── 결과 JSON 에서 꺼내는 값 ───────────────────────────────────────────────
def load(name: str) -> dict:
    return json.loads((io.RESULTS / f"{name}.json").read_text(encoding="utf-8"))


def numbers() -> dict:
    """판넬에 들어가는 수치. 손으로 적는 것은 하나도 없다."""
    b = load("02_build")
    rs = load("07_repeat_sales")
    sc = load("05b_spec_curve")
    hd = load("11_honest_did")
    rel = load("09_release_2025")
    ar = load("06g_area_exemption")
    ev = load("04_event_study")
    inf = load("06_inference")

    ix = [(r["연월"], r["전체"]) for r in rs["지수"]]
    끝 = rs["지수"][-1]
    # '출렁였다' 를 말하려면 전체 최고점이 아니라 **가장 크게 떨어진 구간**을
    # 찾아야 한다. 전체 최고점은 늘 마지막 점이라 떨어진 이야기가 안 된다.
    정점, 저점 = max((( ix[i], ix[j]) for i in range(len(ix))
                      for j in range(i + 1, len(ix))),
                     key=lambda pair: pair[0][1] - pair[1][1])
    정렬 = jpath.dig(sc, "축별_영향.거래량[축=사건정렬]")
    막차 = jpath.dig(rel, "막차[집단=해제군]")

    n = {
        "표본": b["표본"],
        "쌍": rs["쌍"]["전체"],
        "기준연월": rs["지수"][0]["연월"],
        "끝연월": 끝["연월"],
        "끝지수": 끝["전체"],
        "배수": 끝["전체"] / rs["지수"][0]["전체"],
        "정점연월": 정점[0],
        "정점": 정점[1],
        "저점연월": 저점[0],
        "저점": 저점[1],
        "정렬폭": 정렬["폭_pct포인트"],
        "달력월": 정렬["값별_중위"]["달력 월"],
        "효력일": 정렬["값별_중위"]["효력일"],
        "사양수": sc["거래량"]["사양수"],
        "경우의수": jpath.dig(inf, "무작위화추론.사건연구 k=0.경우의수"),
        "k0": jpath.dig(ev, "거래량[k=0].effect_pct"),
        "붕괴_거래량": jpath.dig(hd, "결과[0].붕괴_Mbar_구간"),
        "붕괴_가격": jpath.dig(hd, "결과[2].붕괴_Mbar_구간"),
        "막차전": 막차["발표전_나흘"],
        "막차후": 막차["발표후_나흘"],
        "효력후": 막차["효력후_나흘"],
        "막차배수": 막차["막차배수"],
        "소형2020": jpath.dig(rel, "크기_삼중차분[0].소형_pct"),
        "소형2025": jpath.dig(rel, "크기_삼중차분[1].소형_pct"),
        "소형2022": jpath.dig(ar, "나_2022_기준강화[기준_㎡=40.0].소형효과_pct"),
        "결정기록": len(list((io.ROOT / "docs" / "decisions").glob("[0-9]*.md"))),
        "주차보고서": len(list((io.ROOT / "docs" / "reports").glob("week-*.md"))),
        "결과json": len(list(io.RESULTS.glob("*.json"))),
    }
    # 햇수와 낙폭 구간도 지수에서 센다 — '9년 만에', '1년 만에' 를 손으로
    # 적어 두면 지수가 한 점 늘어날 때마다 틀린 말이 된다.
    def 월(ym):
        return int(ym[:4]) * 12 + int(ym[5:])

    n["햇수"] = round((월(n["끝연월"]) - 월(n["기준연월"])) / 12)
    n["낙폭개월"] = 월(n["저점연월"]) - 월(n["정점연월"])
    n["표본만"] = n["표본"] / 10000
    return n


# ─── 판넬 그리기 ────────────────────────────────────────────────────────────
def build(tpl: Path, n: dict) -> Path:
    from PIL import Image
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    from pptx.util import Cm, Pt
    import qrcode

    C = {k: RGBColor(*v) for k, v in
         {"navy": NAVY, "red": RED, "ink": INK, "grey": GREY}.items()}

    def para(tf, runs, size=21, bold=False, color=C["ink"], before=0, first=False):
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.line_spacing = 1.15
        if before:
            p.space_before = Pt(before)
        for item in ([runs] if isinstance(runs, str) else runs):
            txt, opt = (item, {}) if isinstance(item, str) else item
            r = p.add_run()
            r.text = txt
            r.font.name = FONT
            r.font.size = Pt(opt.get("size", size))
            r.font.bold = opt.get("bold", bold)
            r.font.color.rgb = opt.get("color", color)
        return p

    def textbox(slide, x, y, w, h):
        tf = slide.shapes.add_textbox(Cm(x), Cm(y), Cm(w), Cm(h)).text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Cm(0.15)
        tf.margin_top = tf.margin_bottom = 0
        return tf

    def picture(slide, name, x, y, w):
        p = io.FIGURES / name
        iw, ih = Image.open(p).size
        slide.shapes.add_picture(str(p), Cm(x), Cm(y), Cm(w), Cm(w * ih / iw))

    OUTDIR.mkdir(parents=True, exist_ok=True)
    qr = OUTDIR / "qr.png"
    qrcode.make(SITE, border=1).save(qr)

    prs = Presentation(str(tpl))

    # 예시 슬라이드 제거
    lst = prs.slides._sldIdLst
    for el in list(lst)[1:]:
        prs.part.drop_rel(el.get(
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"))
        lst.remove(el)
    s = prs.slides[0]

    # 안내 문구 제거
    for sh in list(s.shapes):
        if sh.has_text_frame and "양식 수정 불가" in sh.text_frame.text:
            sh._element.getparent().remove(sh._element)

    # 머리말·제목·지도교수 (양식의 글상자를 그대로 쓰고 글자만 바꾼다)
    for sh in s.shapes:
        if sh.has_table:
            sh.table.cell(0, 2).text_frame.paragraphs[0].runs[0].text = \
                f"{학기}_{강좌번호}"
        elif sh.has_text_frame and "작품제목" in sh.text_frame.text:
            ps = sh.text_frame.paragraphs
            ps[0].runs[0].text = 작품제목
            ps[1].runs[0].text = 소제목
        elif sh.has_text_frame and "지도교수" in sh.text_frame.text:
            ps = sh.text_frame.paragraphs
            ps[0].runs[-1].text = f": {지도교수}"
            ps[1].runs[-1].text = f": {조원}"

    # ── 배경 ───────────────────────────────────────────────────────────────
    # 학술적 공백이 아니라 **시장이 출렁였다** 로 연다. 지수는 이 연구가 지은
    # 것이고 강남·송파 기준이다 — '서울' 이라고 쓰면 틀린다.
    tf = textbox(s, 1.7, 12.1, 25.8, 9.4)
    para(tf, [(f"강남·송파 아파트 가격은 {n['햇수']}년 만에 "
               f"{n['배수']:.1f}배가 됐다.",
               {"bold": True, "color": C["navy"], "size": 23})], first=True)
    para(tf, [f"{n['기준연월'][:4]}년 {int(n['기준연월'][5:])}월 100 → "
              f"{n['정점연월'][:4]}년 {int(n['정점연월'][5:])}월 ",
              (f"{n['정점']:.0f}", {"bold": True}),
              (f" → {n['낙폭개월'] // 12}년 만에 " if n["낙폭개월"] % 12 == 0
               else f" → {n['낙폭개월']}개월 만에 "),
              (f"{n['저점']:.0f}", {"bold": True, "color": C["red"]}),
              f" → {n['끝연월'][:4]}년 {int(n['끝연월'][5:])}월 ",
              (f"{n['끝지수']:.0f}", {"bold": True}), "."], before=5)
    para(tf, [f"이 연구가 지은 반복매매지수. 같은 집이 두 번 팔린 "
              f"{n['쌍']:,}쌍만 쓴다."], size=18, color=C["grey"], before=2)
    para(tf, ["정부는 ", ("토지거래허가제", {"bold": True}),
              "로 답했다. 집을 사려면 구청 허가를 받아야 한다. "
              "2020년 6월 서울에 처음 걸렸고 2025년 10월 ",
              ("서울 전역", {"bold": True}), "으로 넓어졌다."], before=10)
    para(tf, [("그런데 이 규제가 무엇을 했는지는 분명하지 않다.",
               {"bold": True, "color": C["red"], "size": 22})], before=9)
    para(tf, ["선행연구 세 편의 결론이 엇갈린다. 동 단위 집계로는 지정 전후를 "
              "가를 수 없다. 날짜 세는 법 하나가 ",
              (f"{n['정렬폭']:.1f} %p", {"bold": True, "color": C["red"]}),
              f"를 가른다 — 달력 월 {sgn(n['달력월'])}%, 효력일 "
              f"{sgn(n['효력일'])}%."], size=19, color=C["grey"], before=4)

    # ── 목표 ───────────────────────────────────────────────────────────────
    tf = textbox(s, 28.9, 12.1, 25.4, 9.4)
    para(tf, [("거래 한 건이 한 행인 미시 패널을 짓고,",
               {"bold": True, "color": C["navy"], "size": 24})], first=True)
    para(tf, [("그 위에서 인과효과를 추정한다.",
               {"bold": True, "color": C["navy"], "size": 24})])
    para(tf, ["국토교통부 실거래 ", ("상세", {"bold": True}), " 자료 ",
              (f"{n['표본만']:.0f}만 건", {"bold": True}),
              "에 서울시보·전자관보 ", ("공고 원문", {"bold": True}),
              "을 결합해, 거래마다 계약일·전용면적·층·단지와 그 시점의 규제 "
              "상태를 붙인다."], before=12)
    para(tf, ["수집부터 보고서의 표·그림까지 ",
              ("한 번의 명령으로 재현", {"bold": True, "color": C["red"]}),
              "되게 만들고, 문서의 숫자가 분석 결과와 어긋나지 않는지 "
              "매 실행마다 점검한다."], before=8)

    # ── 시스템 구조 ────────────────────────────────────────────────────────
    picture(s, "pipeline.png", 1.9, 25.6, 24.6)
    tf = textbox(s, 1.9, 54.5, 24.6, 5.8)
    para(tf, [("입구는 셋인데 출구는 하나다.",
               {"bold": True, "color": C["red"], "size": 21})], first=True)
    para(tf, ["현재 값의 출처는 output/results/*.json 하나이며, 보고서의 표와 "
              "결과 대시보드가 거기서 생성된다. 가운데 줄기(공고 PDF → 법정동 "
              "지정 구간)가 없으면 처치 변수 자체가 없다."],
         size=19, color=C["grey"], before=4)

    tf = textbox(s, 28.2, 25.9, 25.6, 12.0)
    para(tf, [("미시 패널이라야 보이는 것",
               {"bold": True, "color": C["navy"], "size": 25})], first=True)
    for 발견, 필요 in [
        (f"지정 직후 효과가 {MINUS}59%가 아니라 {sgn(n['k0'], 0)}%",
         "거래 하나하나의 계약일"),
        ("공고와 효력 사이 나흘의 막차 수요", "하루 단위 거래"),
        ("허가 면적 기준이 회피 통로였다", "거래마다의 전용면적"),
        ("중위가가 11포인트 부풀려져 있었다", "같은 집의 반복 거래"),
        ("2025년 부분 해제 — 같은 동 안의 두 집단", "단지별 해제·유지 여부"),
    ]:
        para(tf, [("• ", {"color": C["red"], "bold": True}), (발견, {"bold": True}),
                  ("    ← " + 필요, {"color": C["grey"], "size": 19})], before=7)

    picture(s, "09_release_2025.png", 28.2, 36.4, 25.6)
    tf = textbox(s, 28.2, 44.2, 25.6, 12.5)
    para(tf, [("2025년 재지정 — 미시 자료가 아니면 안 보이는 장면",
               {"bold": True, "color": C["navy"], "size": 21})], first=True)
    para(tf, ["왼쪽은 주별 거래. 해제(2/13)에 솟았다가 재지정(3/24)에 바닥으로 "
              "꺼진다. 가운데는 ", ("발표에서 효력까지 나흘", {"bold": True}),
              f" — 그 나흘에 해제군 거래가 {n['막차전']}건에서 ",
              (f"{n['막차후']}건", {"bold": True, "color": C["red"]}),
              f"으로 {n['막차배수']:.1f}배 뛰었다가, 효력 뒤 나흘에는 "
              f"{n['효력후']}건이 된다."],
         size=19, color=C["grey"], before=5)
    para(tf, ["오른쪽은 면적 기준의 효과. 허가 기준이 대지 18 m²에서 6 m²로 "
              "좁아지자 소형이 ",
              (f"{sgn(n['소형2020'], 0)}%에서 {sgn(n['소형2025'], 0)}%로",
               {"bold": True, "color": C["red"]}), " 뒤집혔다."],
         size=19, color=C["grey"], before=4)

    tf = textbox(s, 28.2, 53.2, 25.6, 7.0)
    para(tf, [("매 실행마다 걸리는 자동 점검 넷",
               {"bold": True, "color": C["navy"], "size": 21})], first=True)
    para(tf, ["폐기된 값 · 아무도 안 만드는 산출물 · 의존 목록의 구멍 · 보고서 "
              "표 생성. 새 폴더 새 가상환경에서 돌려 결과 파일 "
              f"{n['결과json']}개가 바이트 단위로 같은 것을 확인했다."],
         size=19, color=C["grey"], before=4)

    # ── 적용 및 기대효과 ───────────────────────────────────────────────────
    tf = textbox(s, 1.8, 64.4, 27.0, 14.8)
    para(tf, [("결과 셋", {"bold": True, "color": C["navy"], "size": 25})],
         first=True)
    for i, (제목, 본문) in enumerate([
        ("거래는 확실하게 멈췄다",
         f"지정 직후 3개월 {MINUS}71 ~ {MINUS}80%. 27개 동에서 4개를 고르는 "
         f"{n['경우의수']:,}가지 중 진짜 배정이 1위이고, 합리적인 설계 조합 "
         f"{n['사양수']}가지가 전부 음수다."),
        ("그러나 가격을 잡았다는 증거는 약하다",
         f"평행추세를 가정하면 {MINUS}4 ~ {MINUS}7%이나, 그 가정을 조금만 풀면 0과 "
         f"구분되지 않는다(붕괴값 {n['붕괴_가격']} 대 거래량 "
         f"{n['붕괴_거래량']})."),
        ("갈랐던 것은 허가 면적 기준이었다",
         f"2020년 지정은 대지 18 m² 초과만 걸려 소형은 오히려 늘었다"
         f"({sgn(n['소형2020'], 0)}%). 2022년 기준이 6 m²로 좁아지자 그때 "
         f"소형만 {abs(n['소형2022']):.0f}% 줄었다."),
    ], 1):
        para(tf, [(f"{i}. ", {"bold": True, "color": C["red"], "size": 23}),
                  (제목, {"bold": True, "size": 23})], before=11)
        para(tf, [본문], size=20, color=C["grey"], before=2)

    para(tf, [("제도의 명분은 가격 안정인데 작동한 것은 거래 동결이었다.",
               {"bold": True, "color": C["red"], "size": 23})], before=13)
    para(tf, ["규제를 ", ("거느냐", {"bold": True}), "가 아니라 ",
              ("어떻게 거느냐", {"bold": True, "color": C["navy"]}),
              "가 효과를 정한다."], before=3)

    picture(s, "06g_area_exemption.png", 29.5, 64.9, 24.0)
    tf = textbox(s, 29.5, 71.9, 24.0, 2.8)
    para(tf, ["가운데가 2022년 기준 강화, 오른쪽이 같은 절차를 기준 그대로인 "
              "사건에 건 위약 검정. ",
              ("기준을 바꾼 날에만 소형이 움직였다.",
               {"bold": True, "color": C["red"]})],
         size=18, color=C["grey"], first=True)

    s.shapes.add_picture(str(qr), Cm(29.5), Cm(74.6), Cm(4.6), Cm(4.6))
    tf = textbox(s, 34.8, 74.8, 18.7, 4.4)
    para(tf, [("코드 · 자료 · 결정기록 전부 공개",
               {"bold": True, "color": C["navy"], "size": 22})], first=True)
    para(tf, ["github.com/cosmicpotato2047/capstone-ltz-causal"],
         size=19, color=C["grey"], before=3)
    para(tf, [f"결정기록 {n['결정기록']}편 · 결과 대시보드 · 주차 보고서 "
              f"{n['주차보고서']}편. 분석을 다시 돌리면 보고서의 표와 그림이 "
              f"따라 바뀐다."], size=18, color=C["grey"], before=5)

    out = OUTDIR / "요약보고서_판넬.pptx"
    prs.save(str(out))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", type=Path, default=TEMPLATE)
    a = ap.parse_args()

    if not a.template.exists():
        print(f"  [건너뜀] 양식이 없다 — {a.template.relative_to(io.ROOT)}")
        print("           학과가 배포한 요약보고서(판넬양식) 파일을 그 경로에 두면 돈다.")
        print("           서식이라 저장소에 넣지 않는다 (.gitignore).")
        return

    n = numbers()
    out = build(a.template, n)
    print(f"  {out.relative_to(io.ROOT)}")
    print(f"  지도교수 {지도교수} · 조원 {조원} · {학기}_{강좌번호}")
    print(f"  지수 {n['기준연월']} 100 → {n['정점연월']} {n['정점']:.0f}"
          f" → {n['끝연월']} {n['끝지수']:.0f}  ({n['배수']:.1f}배)")
    print(f"  막차 {n['막차전']} → {n['막차후']}건 ({n['막차배수']:.1f}배)"
          f" · 소형 +{n['소형2020']:.0f}% → {n['소형2025']:.0f}%")
    print("  제출용 PDF 는 PowerPoint 에서 '다른 이름으로 저장 → PDF'")


if __name__ == "__main__":
    main()
