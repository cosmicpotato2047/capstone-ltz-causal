"""서울시보 발췌본에서 토지거래허가구역 공고를 모두 뽑는다.

한 발췌본에 여러 공고가 들어 있을 수 있으므로 블록 단위로 나눠 처리한다.

    python scripts/parse_seoulsibo.py

입력  data/policy/notices/seoul/시보*.pdf
출력  data/policy/notices/parsed_seoulsibo.csv
      data/policy/notices/parsed_seoulsibo_blocks.txt  (원문 블록, 검증용)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import fitz
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import io  # noqa: E402

io.setup_stdout()

SEOUL = io.POLICY / "notices" / "seoul"
NOTICE_RE = re.compile(r"◈?\s*서울특별시\s*공고\s*제?\s*(20\d\d)\s*-\s*(\d+)\s*호")
DATE_RE = re.compile(r"(20\d\d)\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일")
KEY = "토지거래허가"

CATS = {
    "국제교류복합지구": ["국제교류복합지구"],
    "주요재건축단지": ["압구정", "여의도", "목동", "성수전략", "주요 재건축"],
    "신속통합기획": ["신속통합기획", "신통기획"],
    "공공재개발": ["공공재개발"],
    "모아타운": ["모아타운"],
    "자연녹지": ["자연녹지", "서리풀"],
    "강남3구용산": ["강남구, 서초구", "강남·서초·송파·용산", "강남3구"],
}
ACTS = ["신규지정", "재지정", "지정", "해제", "조정"]


def blocks(text: str) -> list[tuple[str, str]]:
    """(공고번호, 본문) 목록. 다음 공고 시작 전까지를 본문으로 본다."""
    ms = list(NOTICE_RE.finditer(text))
    out = []
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(text)
        out.append((f"{m.group(1)}-{m.group(2)}", text[m.start():end]))
    return out


def main() -> None:
    files = sorted(SEOUL.glob("시보*.pdf"))
    if not files:
        sys.exit("data/policy/notices/seoul/ 에 발췌본이 없습니다.")

    rows, dump = [], []
    for f in files:
        doc = fitz.open(f)
        text = re.sub(r"[ \t]+", " ", "\n".join(p.get_text() for p in doc))
        issue = re.search(r"시보(\d{8})", f.name).group(1)

        found = blocks(text)
        if not found:                       # 공고 헤더를 못 찾으면 파일 전체를 한 블록으로
            found = [("미상", text)]

        for no, body in found:
            if KEY not in body:
                continue
            d = DATE_RE.search(body)
            pub = (f"{d.group(1)}-{int(d.group(2)):02d}-{int(d.group(3)):02d}"
                   if d else f"{issue[:4]}-{issue[4:6]}-{issue[6:]}")
            cats = [k for k, kws in CATS.items() if any(w in body for w in kws)]
            acts = [a for a in ACTS if a in body[:1200]]
            gu = sorted(set(re.findall(r"(강남구|서초구|송파구|용산구|성동구|광진구|강동구|동작구)", body)))
            rows.append({
                "공고번호": no, "공고일": pub, "시보발행": issue,
                "구분": "|".join(cats), "행위": "|".join(acts),
                "관련자치구": "|".join(gu), "본문길이": len(body), "파일": f.name,
            })
            dump.append(f"\n{'='*90}\n[{no}] {pub}  {f.name}\n{'='*90}\n{body[:3000]}")

    df = pd.DataFrame(rows).sort_values("공고일", ascending=False)
    out = io.POLICY / "notices" / "parsed_seoulsibo.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    (io.POLICY / "notices" / "parsed_seoulsibo_blocks.txt").write_text(
        "\n".join(dump), encoding="utf-8")

    print(f"{'공고번호':<11}{'공고일':<12}{'구분':<34}{'행위':<20}관련 자치구")
    print("-" * 108)
    for _, r in df.iterrows():
        print(f"{r.공고번호:<11}{r.공고일:<12}{r.구분[:32]:<34}{r.행위[:18]:<20}"
              f"{r.관련자치구.replace('|', ',')[:30]}")
    print(f"\n{len(df)}건 · 저장 {out.name}")


if __name__ == "__main__":
    main()
