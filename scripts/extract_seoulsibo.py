"""서울시보 원본에서 토지거래허가구역 관련 쪽만 잘라 보관한다.

서울시보는 한 호가 수백~수천 쪽이고 그중 토허제 관련은 일부다.
원본을 전부 저장소에 넣을 수 없으므로 해당 쪽만 발췌해 커밋한다.

    python scripts/extract_seoulsibo.py

입력  data/policy/notices/seoulsibo_raw/*.pdf  (git 제외)
      data/policy/notices/seoulsibo_pages.csv  (파일명, 시작쪽, 끝쪽)
출력  data/policy/notices/seoul/시보YYYYMMDD_p<시작>-<끝>.pdf

쪽 번호는 사람이 세는 1부터 시작하는 번호로 받는다.
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
RAW = NOTICES / "seoulsibo_raw"
OUT = NOTICES / "seoul"
INDEX = NOTICES / "seoulsibo_pages.csv"


def issue_date(name: str) -> str:
    """seoulsibo_20250213092648_24250.pdf -> 20250213
    2020-제3590호-pdf.pdf -> 2020호3590 (형식이 다른 과년도 파일)"""
    m = re.search(r"_(\d{8})\d{6}_", name)
    if m:
        return m.group(1)
    m = re.search(r"(20\d\d)-제(\d+)호", name)
    if m:
        return f"{m.group(1)}호{m.group(2)}"
    return "unknown"


def main() -> None:
    if not INDEX.exists():
        sys.exit(f"{INDEX} 가 없습니다.")
    OUT.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(INDEX)

    made, missing, failed = [], [], []
    for _, r in idx.iterrows():
        src = RAW / r.파일명
        if not src.exists():
            missing.append(r.파일명)
            continue
        try:
            doc = fitz.open(src)
            a, b = int(r.시작쪽), int(r.끝쪽)
            if a < 1 or b > doc.page_count or a > b:
                failed.append(f"{r.파일명}: 쪽 범위 {a}-{b} / 전체 {doc.page_count}쪽")
                continue
            sub = fitz.open()
            sub.insert_pdf(doc, from_page=a - 1, to_page=b - 1)   # 0-기반으로 변환
            dst = OUT / f"시보{issue_date(r.파일명)}_p{a}-{b}.pdf"
            sub.save(dst)
            made.append((dst.name, b - a + 1, dst.stat().st_size,
                         doc.page_count, src.stat().st_size))
        except Exception as e:
            failed.append(f"{r.파일명}: {e}")

    print(f"{'발췌본':<34}{'쪽수':>5}{'크기':>9}   원본")
    print("-" * 72)
    for name, n, sz, tot, tsz in made:
        print(f"{name:<34}{n:>5}{sz/1e6:>8.1f}M   {tot}쪽 {tsz/1e6:.0f}M")

    print(f"\n발췌 {len(made)}건 / 원본 없음 {len(missing)}건 / 실패 {len(failed)}건")
    if missing:
        print("\n[원본 없음] seoulsibo_raw/ 에 넣어주세요")
        for m in missing[:12]:
            print(f"  {m}")
        if len(missing) > 12:
            print(f"  ... 외 {len(missing)-12}건")
    if failed:
        print("\n[실패]")
        for f in failed:
            print(f"  {f}")
    if made:
        tot_out = sum(m[2] for m in made) / 1e6
        tot_in = sum(m[4] for m in made) / 1e6
        print(f"\n용량 {tot_in:.0f}MB -> {tot_out:.1f}MB "
              f"({100*tot_out/tot_in:.1f}%)" if tot_in else "")


if __name__ == "__main__":
    main()
