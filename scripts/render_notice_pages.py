"""공고 PDF의 특정 쪽을 그림으로 뽑는다. 지형도면 확인용.

    python scripts/render_notice_pages.py <PDF파일명> <쪽번호...> [--tag 이름]

예)
    python scripts/render_notice_pages.py 시보2020호3590_p105-107.pdf 2 3 --tag 2020-1832

입력  data/policy/notices/seoul/<PDF파일명>   (또는 molit/)
출력  output/figures/notice_maps/<tag>_p<쪽>.png

허가구역 경계는 본문 표가 아니라 별첨 지형도면에만 있다. 그 도면이
법정동 경계와 일치하는지는 사람이 눈으로 봐야 하므로 그림으로 뽑아 둔다.
쪽 번호는 사람이 세는 1부터다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import io  # noqa: E402

io.setup_stdout()

OUT = io.FIGURES / "notice_maps"
DPI = 200


def find(name: str) -> Path:
    for sub in ("seoul", "molit"):
        p = io.POLICY / "notices" / sub / name
        if p.exists():
            return p
    sys.exit(f"{name} 을 molit/ 에서도 seoul/ 에서도 못 찾았습니다.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("pages", nargs="+", type=int)
    ap.add_argument("--tag", default=None, help="출력 파일 이름 앞부분")
    a = ap.parse_args()

    src = find(a.pdf)
    tag = a.tag or src.stem
    OUT.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(src)

    print(f"원본 {src.relative_to(io.ROOT)}  ({doc.page_count}쪽)")
    for n in a.pages:
        if not 1 <= n <= doc.page_count:
            print(f"  쪽 {n}: 범위 밖 (1~{doc.page_count})")
            continue
        pix = doc[n - 1].get_pixmap(dpi=DPI)
        dst = OUT / f"{tag}_p{n}.png"
        pix.save(dst)
        print(f"  쪽 {n} -> {dst.relative_to(io.ROOT)}  "
              f"{pix.width}x{pix.height}  {dst.stat().st_size/1e6:.2f}MB")


if __name__ == "__main__":
    main()
