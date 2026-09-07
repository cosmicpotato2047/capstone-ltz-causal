"""공고 원문 PDF 에서 지정 이력을 구조화해 뽑는다.

    python scripts/parse_notices.py

data/policy/notices/{molit,seoul}/*.pdf 를 읽어
data/policy/notices/parsed_notices.csv 를 만든다.

공고문은 관보/시보 지면을 그대로 스캔한 것이라 다른 부처 공고가 섞여 있다.
따라서 국토교통부(또는 서울특별시) 공고 블록만 잘라 쓴다.
서울 관련 대상지역만 추출한다.
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

NOTICES = io.POLICY / "notices"
SEOUL_GU = ("종로구 중구 용산구 성동구 광진구 동대문구 중랑구 성북구 강북구 도봉구 "
            "노원구 은평구 서대문구 마포구 양천구 강서구 구로구 금천구 영등포구 동작구 "
            "관악구 서초구 강남구 송파구 강동구").split()


def clean(t: str) -> str:
    return re.sub(r"[ \t]+", " ", t)


def notice_block(text: str, issuer: str) -> str:
    """지면에 섞인 다른 공고를 걷어내고 해당 기관 공고 블록만 남긴다."""
    pat = rf"◉?\s*{issuer}\s*공\s*고\s*제?\s*20\d\d\s*-\s*\d+\s*호"
    m = list(re.finditer(pat, text))
    if not m:
        return text
    start = m[0].start()
    nxt = re.search(r"◉", text[start + 10:])
    return text[start: start + 10 + nxt.start()] if nxt else text[start:]


def parse(path: Path, issuer: str) -> dict:
    doc = fitz.open(path)
    full = clean("\n".join(p.get_text() for p in doc))
    head = notice_block(full, issuer)

    no = re.search(rf"{issuer}\s*공\s*고\s*제?\s*(20\d\d)\s*-\s*(\d+)\s*호", head)
    notice_no = f"{no.group(1)}-{no.group(2)}" if no else path.stem

    pub = re.search(r"(20\d\d)\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일", head)
    pub_date = (f"{pub.group(1)}-{int(pub.group(2)):02d}-{int(pub.group(3)):02d}"
                if pub else "")

    span = re.search(r"지정기간\s*:?\s*(20\d\d)\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일"
                     r"\s*부터\s*(20\d\d)\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일", head)
    if span:
        g = span.groups()
        eff = f"{g[0]}-{int(g[1]):02d}-{int(g[2]):02d}"
        exp = f"{g[3]}-{int(g[4]):02d}-{int(g[5]):02d}"
    else:
        eff = exp = ""

    act = ("해제" if "해제" in path.name else
           "조정" if "조정" in path.name else
           "재지정" if "재지정" in path.name else "지정")

    # 서울 관련 자치구 — 지면 전체에서 찾되 공고 블록 우선
    scope = head if len(head) > 400 else full
    gus = [g for g in SEOUL_GU if re.search(rf"서울\s*{g}|{g}", scope)]

    # 허가 대상 면적 기준 (주거지역)
    area = re.search(r"주거지역\s*(\d+)\s*㎡\s*초과", full)

    return {
        "기관": issuer,
        "공고번호": notice_no,
        "공고일": pub_date,
        "효력일": eff,
        "종료일": exp,
        "행위": act,
        "서울자치구": "|".join(gus),
        "주거지역기준_㎡초과": area.group(1) if area else "",
        "쪽수": doc.page_count,
        "파일": path.name,
    }


def main() -> None:
    rows = []
    for sub, issuer in (("molit", "국토교통부"), ("seoul", "서울특별시")):
        d = NOTICES / sub
        if not d.exists():
            continue
        for f in sorted(d.glob("*.pdf")):
            try:
                rows.append(parse(f, issuer))
            except Exception as e:
                print(f"  [실패] {f.name}: {e}")

    if not rows:
        sys.exit("공고 PDF 가 없습니다.")
    df = pd.DataFrame(rows).sort_values(["기관", "공고번호"])
    out = NOTICES / "parsed_notices.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"{'공고번호':<12}{'공고일':<12}{'효력일':<12}{'종료일':<12}{'행위':<6}{'기준':>5}  서울 자치구")
    print("-" * 100)
    for _, r in df.iterrows():
        gus = r.서울자치구.replace("|", ",")
        print(f"{r.공고번호:<12}{r.공고일:<12}{r.효력일:<12}{r.종료일:<12}"
              f"{r.행위:<6}{r['주거지역기준_㎡초과']:>5}  {gus[:44]}")
    print(f"\n저장: {out}  ({len(df)}건)")


if __name__ == "__main__":
    main()
