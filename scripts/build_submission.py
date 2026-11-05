r"""[D11] 제출본 PDF 합치기 — 앞 3쪽 + 본문.

    python scripts/build_submission.py

입력  data/forms/결과보고서_앞3쪽.pdf   요약서 · 표지 · 윤리서약서 (저장소 밖)
      data/forms/제출정보.json          학번과 이름 (저장소 밖)
      paper/main_ko_v2.pdf              본문 (쪽번호 4 부터)
출력  output/submit/캡스톤디자인_결과보고서_<학번>_<이름>.pdf

**개인정보는 저장소에 넣지 않는다.** 앞 3쪽에는 학번·연락처·서명이 들어 있고
합친 결과물도 마찬가지다. 그래서 입력은 `data/forms/`(git 제외)에서 읽고
출력도 `output/submit/`(git 제외)에 쓴다. 이 스크립트 자체에도 학번을 적지
않는다 — `제출정보.json` 에서 읽는다.

`main_ko_v2.tex` 가 목차부터 시작하고 쪽번호를 4 로 여는 것이 이 합치기를
전제로 한 것이다. 앞 3쪽의 인쇄된 쪽번호(1·2·3)와 그대로 이어진다.

없는 파일이 있으면 무엇을 어디에 두면 되는지 알리고 건너뛴다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import io  # noqa: E402

io.setup_stdout()

FRONT = io.ROOT / "data" / "forms" / "결과보고서_앞3쪽.pdf"
INFO = io.ROOT / "data" / "forms" / "제출정보.json"
BODY = io.ROOT / "paper" / "main_ko_v2.pdf"
OUTDIR = io.ROOT / "output" / "submit"

INFO_SAMPLE = {"학번": "00000000", "이름": "홍길동"}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--front", type=Path, default=FRONT)
    ap.add_argument("--body", type=Path, default=BODY)
    a = ap.parse_args()

    missing = [p for p in (a.front, a.body) if not p.exists()]
    if missing or not INFO.exists():
        for p in missing:
            print(f"  [건너뜀] 없는 파일 — {p.relative_to(io.ROOT)}")
        if not INFO.exists():
            print(f"  [건너뜀] 없는 파일 — {INFO.relative_to(io.ROOT)}")
            print("           이렇게 만들면 된다 (저장소에 올라가지 않는다):")
            print("           " + json.dumps(INFO_SAMPLE, ensure_ascii=False))
        print("           앞 3쪽(요약서·표지·윤리서약서)은 학과 서식으로 따로")
        print("           만들어 data/forms/ 에 둔다. 개인정보라 git 밖이다.")
        return

    import fitz

    info = json.loads(INFO.read_text(encoding="utf-8"))
    for k in ("학번", "이름"):
        if not info.get(k):
            sys.exit(f"{INFO.relative_to(io.ROOT)} 에 '{k}' 가 없다.")

    front, body = fitz.open(a.front), fitz.open(a.body)

    # 두 쪽 크기가 다르면 합쳤을 때 티가 난다. 넘기기 전에 알린다.
    fa, fb = front[0].rect, body[0].rect
    if abs(fa.width - fb.width) > 2 or abs(fa.height - fb.height) > 2:
        print(f"  [경고] 쪽 크기가 다르다 — 앞 {fa.width:.0f}x{fa.height:.0f}pt,"
              f" 본문 {fb.width:.0f}x{fb.height:.0f}pt")

    out = fitz.open()
    out.insert_pdf(front)
    out.insert_pdf(body)
    out.set_metadata({
        "title": "캡스톤디자인 결과보고서 — 서울 토지거래허가구역 인과효과 추정",
        "author": info["이름"],
        "subject": "2026학년도 2학기 캡스톤디자인(2) 결과보고서",
        "keywords": "토지거래허가구역, 이중차분, 사건연구, 미시 패널",
    })

    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / f"캡스톤디자인_결과보고서_{info['학번']}_{info['이름']}.pdf"
    out.save(str(path), garbage=3, deflate=True)

    print(f"  {path.relative_to(io.ROOT)}")
    print(f"  앞 {front.page_count}쪽 + 본문 {body.page_count}쪽"
          f" = {out.page_count}쪽 · {path.stat().st_size // 1024:,}KB")
    print(f"  인쇄된 쪽번호 1 ~ {3 + body.page_count}")


if __name__ == "__main__":
    main()
