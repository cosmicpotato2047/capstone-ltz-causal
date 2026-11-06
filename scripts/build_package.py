r"""[D12] 제출 꾸러미 만들기 — 폴더 일곱과 zip.

    python scripts/build_package.py

입력  data/forms/결과보고서_앞3쪽.pdf   요약서·표지·윤리서약서 (저장소 밖)
      data/forms/제출정보.json          학번과 이름 (저장소 밖)
      paper/main_ko_v2.pdf              보고서 본문
      output/panel/요약보고서_판넬.pptx  판넬
      docs/slides/2026-10-30.html       최종 발표 자료
      docs/proposal.md                  주제제안서
출력  output/submit/<색인번호>_<이름>/   폴더 일곱
      output/submit/<색인번호>_<이름>.zip

**개인정보가 들어가므로 저장소에 올리지 않는다.** 앞 3쪽에 학번·연락처·서명이
있고 꾸러미는 그것을 통째로 담는다. 입력은 `data/forms/`(git 제외), 출력은
`output/submit/`(git 제외)다. **올리는 것은 이 스크립트뿐이다.**

폴더 이름과 구성은 학과 안내서를 따른다.

    0_계획서                      주제제안서 (수업 발표자료로 대체 가능)
    1_캡스톤디자인_보고서본문       앞 3쪽 + 본문
    2_캡스톤디자인_실행파일         실행에 필요한 것
    3_캡스톤디자인_운영환경_기타     운영환경 · 실행 결과
    4_캡스톤디자인_실행파일소스      프로그램 소스
    5_캡스톤디자인_요약보고서       판넬
    6_캡스톤디자인_발표자료         최종 발표 자료

**변환에는 이 PC 의 프로그램이 필요하다.** pptx -> PDF 는 PowerPoint,
HTML -> PDF 는 Chrome 을 쓴다. 없으면 그 조각만 건너뛰고 무엇이 빠졌는지
끝에 모아 알린다 — 꾸러미를 반쯤 만들어 놓고 조용히 끝내지 않는다.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import io  # noqa: E402

io.setup_stdout()

FORMS = io.ROOT / "data" / "forms"
FRONT = FORMS / "결과보고서_앞3쪽.pdf"
INFO = FORMS / "제출정보.json"
BODY = io.ROOT / "paper" / "main_ko_v2.pdf"
PANEL = io.ROOT / "output" / "panel" / "요약보고서_판넬.pptx"
DECK = io.ROOT / "docs" / "slides" / "2026-10-30.html"
PROPOSAL = io.ROOT / "docs" / "proposal.md"
OUTDIR = io.ROOT / "output" / "submit"

색인번호 = "202611_41010-02"

CHROME = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
]

FOLDERS = ["0_계획서", "1_캡스톤디자인_보고서본문", "2_캡스톤디자인_실행파일",
           "3_캡스톤디자인_운영환경_기타", "4_캡스톤디자인_실행파일소스",
           "5_캡스톤디자인_요약보고서", "6_캡스톤디자인_발표자료"]

skipped: list[str] = []


def note(msg: str) -> None:
    skipped.append(msg)
    print(f"  [건너뜀] {msg}")


# ─── 변환 ───────────────────────────────────────────────────────────────────
def html_to_pdf(src: Path, dst: Path) -> bool:
    """Chrome 이나 Edge 로 인쇄. 덱의 @page 가 16:9 쪽을 만든다."""
    exe = next((p for p in CHROME if p.exists()), None)
    if exe is None:
        note(f"{dst.name} — Chrome/Edge 를 찾지 못했다")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(exe), "--headless", "--disable-gpu", "--no-sandbox",
         "--no-pdf-header-footer", f"--print-to-pdf={dst}",
         "--virtual-time-budget=15000", src.resolve().as_uri()],
        capture_output=True, check=False)
    if not dst.exists():
        note(f"{dst.name} — 인쇄가 끝났는데 파일이 없다")
        return False
    return True


def pptx_to_pdf(src: Path, dst: Path) -> bool:
    """PowerPoint 로 내보낸다 (ppSaveAsPDF = 32)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    ps = (
        "$ErrorActionPreference='Stop';"
        "$pp = New-Object -ComObject PowerPoint.Application;"
        f"$d = $pp.Presentations.Open('{src}', $true, $false, $false);"
        f"$d.SaveAs('{dst}', 32); $d.Close(); $pp.Quit()"
    )
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       capture_output=True, text=True)
    if not dst.exists():
        note(f"{dst.name} — PowerPoint 변환 실패 "
             f"({(r.stderr or '').strip().splitlines()[:1]})")
        return False
    return True


def merge_pdf(parts: list[Path], dst: Path, meta: dict) -> bool:
    import fitz
    out = fitz.open()
    sizes = []
    for p in parts:
        d = fitz.open(p)
        sizes.append((d[0].rect.width, d[0].rect.height))
        out.insert_pdf(d)
    if len({(round(w), round(h)) for w, h in sizes}) > 1:
        print(f"  [경고] {dst.name} — 쪽 크기가 섞였다: {sizes}")
    out.set_metadata(meta)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(str(dst), garbage=3, deflate=True)
    return True


def copytree(src: Path, dst: Path, *, skip_suffix=(), skip_dir=()) -> int:
    n = 0
    for p in src.rglob("*"):
        if p.is_dir() or p.suffix.lower() in skip_suffix:
            continue
        if any(d in p.parts for d in ("__pycache__", ".git", *skip_dir)):
            continue
        t = dst / p.relative_to(src)
        t.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, t)
        n += 1
    return n


# ─── 안내문 ─────────────────────────────────────────────────────────────────
RUN_TXT = """캡스톤디자인 실행 방법

이 작품에는 **컴파일된 실행 파일이 없다.** 분석 파이프라인이므로 Python 으로
실행한다. 이 폴더에 실행에 필요한 것(진입점·의존 목록·환경 파일 예시)을 담았고,
전체 소스는 4_캡스톤디자인_실행파일소스 에 있다.

1. Python 3.13 설치

2. 의존 설치
       pip install -r requirements.txt

   윈도에서는 폴더 경로가 짧아야 한다. 경로 길이 한계(260자) 때문에 깊은
   폴더에서는 statsmodels 설치가 조용히 깨진다.

3. 인증키 넣기
       .env.example 을 .env 로 복사하고 공공데이터포털 Decoding 키를 적는다.
       (이미 수집된 자료로만 돌릴 것이라면 건너뛴다)

4. 원자료 두기
       data/raw/ 에 실거래 원자료 스냅샷(2026-09-03 수집, 902MB)을 둔다.
       **--collect 로 다시 받으면 같은 결과가 나오지 않는다** — 실거래 자료는
       취소와 지연신고로 소급 갱신되기 때문이다. 재현은 같은 스냅샷 위에서 한다.

5. 실행
       python run_all.py              수집 빼고 전부 (약 16분)
       python run_all.py --list       단계 목록
       python run_all.py --only 04    한 단계만

실행이 끝나면 output/results/*.json 에 모든 수치가, output/figures/ 에 그림이,
output/dashboard.html 에 결과 대시보드가 생긴다. 보고서의 표도 같은 JSON 에서
생성된다.

전체 과정은 공개 저장소에서도 볼 수 있다.
    https://github.com/cosmicpotato2047/capstone-ltz-causal
"""

ENV_TXT = """캡스톤디자인 개발 및 운영 환경

형태        단독 PC. 서버·클라우드·데이터베이스를 쓰지 않는다.
            결과는 파일(parquet · json · csv)로 남는다.
운영체제    Windows 11 Pro (22621)
언어        Python 3.13.7
주요 패키지 pandas · numpy · statsmodels · scipy · matplotlib ·
            shapely(법정동 경계) · PyMuPDF(공고 PDF 판독) ·
            python-pptx(판넬) · markdown(문서)
외부 자료   국토교통부 실거래 상세 자료 오픈 API (활용신청 승인 후 이용)
            서울시보 · 전자관보 공고 원문
문서        LaTeX (XeLaTeX + kotex), Markdown -> HTML
공개        github.com/cosmicpotato2047/capstone-ltz-causal
            cosmicpotato2047.github.io/capstone-ltz-causal/

별도의 기자재나 재료를 쓰지 않았다. 추가 비용은 0원이다.

데이터베이스를 쓰지 않은 것은 설계 판단이다. 자료가 47만 행으로 단일 PC
메모리에 올라가고, 분석이 질의가 아니라 전체 재계산이기 때문이다. 파일로 두면
결과를 바이트 단위로 비교할 수 있어 재현 확인이 간단해진다 — 실제로 새 폴더
새 가상환경에서 돌려 결과 파일 25개가 같은 것을 해시로 확인했다.

이 폴더의 내용
  실행결과_그림/   파이프라인이 만든 그림 전부
  dashboard.html  결과 대시보드. 숫자마다 출처와 재현 명령이 붙어 있다
  결정기록/        설계 판단 31편. 근거와 버린 대안을 함께 적었다
"""


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-zip", action="store_true", help="폴더만 만들고 압축은 안 함")
    a = ap.parse_args()

    for p in (FRONT, INFO, BODY):
        if not p.exists():
            sys.exit(f"없는 파일 — {p.relative_to(io.ROOT)}\n"
                     "  앞 3쪽과 제출정보는 data/forms/ 에 둔다 (git 제외).")
    info = json.loads(INFO.read_text(encoding="utf-8"))
    학번, 이름 = info["학번"], info["이름"]
    stem = f"{색인번호}_{이름}"

    root = OUTDIR / stem
    if root.exists():
        shutil.rmtree(root)
    for f in FOLDERS:
        (root / f).mkdir(parents=True)

    def name(what: str) -> str:
        return f"캡스톤디자인_{what}_{학번}_{이름}.pdf"

    # ── 0 계획서 ────────────────────────────────────────────────────────
    # 안내서가 '수업 발표자료 대체 가능' 이라 했지만, 주제제안서가 본래의
    # 계획서다. Markdown 을 같은 렌더러로 HTML 로 만든 뒤 인쇄한다.
    if PROPOSAL.exists():
        html = PROPOSAL.with_suffix(".html")
        subprocess.run([sys.executable, str(io.ROOT / "scripts" / "render_report.py"),
                        str(PROPOSAL.relative_to(io.ROOT))],
                       cwd=io.ROOT, capture_output=True)
        if html.exists():
            html_to_pdf(html, root / FOLDERS[0] / name("계획서"))
        else:
            note("계획서 — proposal.html 을 만들지 못했다")
    else:
        note("계획서 — docs/proposal.md 가 없다")

    # ── 1 보고서 본문 ───────────────────────────────────────────────────
    merge_pdf([FRONT, BODY], root / FOLDERS[1] / name("보고서본문"),
              {"title": "캡스톤디자인 결과보고서 — 서울 토지거래허가구역 인과효과 추정",
               "author": 이름, "subject": "2026학년도 2학기 캡스톤디자인(2)",
               "keywords": "토지거래허가구역, 이중차분, 사건연구, 미시 패널"})

    # ── 2 실행파일 ──────────────────────────────────────────────────────
    d2 = root / FOLDERS[2]
    (d2 / "실행방법.txt").write_text(RUN_TXT, encoding="utf-8")
    for f in ("run_all.py", "requirements.txt", ".env.example", "README.md"):
        if (io.ROOT / f).exists():
            shutil.copy2(io.ROOT / f, d2 / f)

    # ── 3 운영환경·기타 ─────────────────────────────────────────────────
    d3 = root / FOLDERS[3]
    (d3 / "운영환경.txt").write_text(ENV_TXT, encoding="utf-8")
    nfig = copytree(io.FIGURES, d3 / "실행결과_그림")
    dash = io.ROOT / "output" / "dashboard.html"
    if dash.exists():
        shutil.copy2(dash, d3 / "dashboard.html")
    ndec = copytree(io.ROOT / "docs" / "decisions", d3 / "결정기록")

    # ── 4 소스 ──────────────────────────────────────────────────────────
    d4 = root / FOLDERS[4]
    nsrc = copytree(io.ROOT / "scripts", d4 / "scripts")
    for f in ("run_all.py", "requirements.txt", "README.md", ".env.example"):
        if (io.ROOT / f).exists():
            shutil.copy2(io.ROOT / f, d4 / f)
    # 보고서도 소스에서 생성된다. PDF 와 보조 파일은 뺀다.
    npap = copytree(io.ROOT / "paper", d4 / "paper_보고서소스",
                    skip_suffix=(".pdf", ".aux", ".log", ".out", ".toc",
                                 ".bbl", ".blg", ".lof", ".lot", ".synctex.gz"))

    # ── 5 요약보고서(판넬) ──────────────────────────────────────────────
    if PANEL.exists():
        pptx_to_pdf(PANEL, root / FOLDERS[5] / name("요약보고서"))
        shutil.copy2(PANEL, root / FOLDERS[5] / f"요약보고서_판넬_{학번}_{이름}.pptx")
    else:
        note("요약보고서 — 판넬 pptx 가 없다 (scripts/build_panel.py)")

    # ── 6 발표자료 ──────────────────────────────────────────────────────
    if DECK.exists():
        html_to_pdf(DECK, root / FOLDERS[6] / name("발표자료"))
    else:
        note("발표자료 — 슬라이드 HTML 이 없다")

    # ── 묶기 ────────────────────────────────────────────────────────────
    print()
    total = 0
    for f in FOLDERS:
        files = [p for p in (root / f).rglob("*") if p.is_file()]
        size = sum(p.stat().st_size for p in files)
        total += size
        mark = " " if files else "<- 비었다"
        print(f"  {f:<28} 파일 {len(files):>4}개 {size / 1e6:>7.1f}MB {mark}")
    print(f"  {'합계':<28}           {total / 1e6:>7.1f}MB")
    print(f"  그림 {nfig} · 결정기록 {ndec} · 소스 {nsrc} · 보고서소스 {npap}")

    if not a.no_zip:
        z = OUTDIR / f"{stem}.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for p in sorted(root.rglob("*")):
                if p.is_file():
                    zf.write(p, p.relative_to(root.parent))
        print(f"\n  {z.relative_to(io.ROOT)}  {z.stat().st_size / 1e6:.1f}MB")

    if skipped:
        print(f"\n  빠진 것 {len(skipped)}개 — 위의 [건너뜀] 을 보라")
        sys.exit(1)
    print("\n  빠진 것 없음.")


if __name__ == "__main__":
    main()
