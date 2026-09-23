"""전체 파이프라인 진입점.

    python run_all.py                 # 자료 정제부터 문서까지 (수집 제외)
    python run_all.py --collect       # 01 수집부터 전부
    python run_all.py --only 06g      # 한 단계만
    python run_all.py --from 06       # 이 단계부터 끝까지
    python run_all.py --group 분석     # 한 묶음만
    python run_all.py --list          # 단계 목록만 출력

원자료 스냅샷 시점: 2026-09-03. 재현 절차는 README.md 참조.

순서는 의존 관계를 따른다.

  자료    실거래 원자료(data/raw) -> 정제 표 두 장 (분석 8개구 / 서울 전역)
  정책    공고 PDF -> 본문 텍스트 -> 구역 표 -> 법정동별 지정 구간·패널
  분석    사건연구 -> 사전추세·대출 -> 가격지수 -> 위약·합성대조군
          -> 적은 클러스터 추론 -> 오염 제외 -> 문턱 민감도 -> 면적 면제
  재현    선행연구 두 편
  시각화  지정 이력 지도·히트맵, 가격 지도, 단지 목록
  문서    주차 보고서 HTML, 문서 수치 검증

빠진 것: `scripts/extract_seoulsibo.py` 는 서울시보 원본(2.9GB, 저장소 밖)이
있어야 돈다. 그 결과인 발췌 PDF(data/policy/notices/seoul/)는 저장소에 있으므로
이후 단계는 원본 없이 재현된다.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (단계, 묶음, 스크립트, 인자, 설명)
STEPS: list[tuple[str, str, str, list[str], str]] = [
    ("01", "수집", "scripts/01_collect.py", [], "실거래 자료 수집 (수십 분)"),

    ("02", "자료", "scripts/02_build.py", [], "정제 — 분석 8개 자치구"),
    ("02s", "자료", "scripts/02_build.py", ["--scope", "seoul"], "정제 — 서울 25개 자치구"),

    ("P1", "정책", "scripts/build_policy_timeline.py", [], "공고 PDF -> 본문 텍스트·지정 이력"),
    ("P2", "정책", "scripts/parse_notices.py", [], "국토부 공고 목록"),
    ("P3", "정책", "scripts/parse_seoulsibo.py", [], "서울시보 공고 목록"),
    ("P4", "정책", "scripts/extract_zones.py", [], "구역 표 추출"),
    ("P5", "정책", "scripts/build_ltz_spells.py", [], "법정동별 지정 구간·패널"),
    ("P6", "정책", "scripts/check_contamination.py", [], "통제군 오염 점검 (0009)"),
    ("P7", "정책", "scripts/rank_candidates.py", [], "처치군 후보 순위 (0012)"),

    ("04", "분석", "scripts/04_event_study.py", [], "2020.6 지정 사건연구"),
    ("06d", "분석", "scripts/06d_pretrend.py", [], "사전 추세 진단 (0014)"),
    ("06c", "분석", "scripts/06c_loan.py", [], "대출 규제 — 쏠림·삼중차분 (0015)"),
    ("07", "분석", "scripts/07_repeat_sales.py", [], "반복매매 가격지수 (0018)"),
    ("13", "분석", "scripts/13_placebo.py", [], "위약 검정 (0016)"),
    ("17", "분석", "scripts/17_scm.py", [], "합성대조군 (0017)"),
    ("06", "분석", "scripts/06_inference.py", [], "적은 클러스터 추론 (0019)"),
    ("13b", "분석", "scripts/13b_contamination.py", [], "오염 동 제외 (0020)"),
    ("06f", "분석", "scripts/06f_threshold.py", [], "노출도 문턱 민감도 (0021)"),
    ("06g", "분석", "scripts/06g_area_exemption.py", [], "면적 면제 검증 (0022)"),
    ("09", "분석", "scripts/09_release_2025.py", [], "2025 해제·재지정 (0023)"),

    ("R1", "재현", "scripts/replicate/jeon2025.py", [], "선행연구 재현 — 전상현 외 2025"),
    ("R2", "재현", "scripts/replicate/lee2025.py", [], "선행연구 재현 — 이한웅·이춘원 2025"),

    ("V1", "시각화", "scripts/plot_ltz_heatmap.py", [], "지정 이력 히트맵"),
    ("V2", "시각화", "scripts/plot_ltz_map.py", [], "지정 이력 지도 (시점 슬라이더)"),
    ("V3", "시각화", "scripts/plot_price_map.py", [], "25개 자치구 가격 지도"),
    ("V4", "시각화", "scripts/build_apt_directory.py", [], "서울 아파트 단지 목록"),
]


def report_steps() -> list[tuple[str, str, str, list[str], str]]:
    """주차 보고서는 파일이 생길 때마다 늘어나므로 그때그때 찾는다."""
    out = []
    for md in sorted((ROOT / "docs" / "reports").glob("week-*.md")):
        rel = md.relative_to(ROOT).as_posix()
        out.append((f"D{md.stem[-2:]}", "문서", "scripts/render_report.py", [rel],
                    f"보고서 HTML — {md.stem}"))
    out.append(("C1", "문서", "scripts/check_numbers.py", [], "문서 수치 검증 (0007)"))
    return out


def run(step: tuple) -> bool:
    key, _, path, args, label = step
    cmd = " ".join([path, *args])
    print(f"\n{'=' * 72}\n▶ [{key}] {cmd}  —  {label}\n{'=' * 72}", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, str(ROOT / path), *args], cwd=ROOT)
    ok = r.returncode == 0
    print(f"[{'OK' if ok else 'FAIL'}] [{key}] {cmd} ({time.time() - t0:.1f}초)", flush=True)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collect", action="store_true", help="01 수집부터 실행")
    ap.add_argument("--only", help="이 단계 하나만")
    ap.add_argument("--from", dest="start", help="이 단계부터 끝까지")
    ap.add_argument("--group", help="이 묶음만 (자료·정책·분석·재현·시각화·문서)")
    ap.add_argument("--list", action="store_true", help="단계 목록만 출력")
    a = ap.parse_args()

    steps = STEPS + report_steps()
    if not a.collect:
        steps = [s for s in steps if s[0] != "01"]

    if a.list:
        for key, grp, path, args, label in steps:
            print(f"  {key:<5}{grp:<6}{' '.join([path, *args]):<52}{label}")
        return

    if a.only:
        steps = [s for s in steps if s[0] == a.only]
    elif a.start:
        keys = [s[0] for s in steps]
        if a.start not in keys:
            sys.exit(f"없는 단계: {a.start}. --list 로 확인")
        steps = steps[keys.index(a.start):]
    if a.group:
        steps = [s for s in steps if s[1] == a.group]
    if not steps:
        sys.exit("실행할 단계가 없다. --list 로 확인")

    # 뒤 단계가 앞 단계 산출물을 읽으므로 하나라도 실패하면 거기서 멈춘다
    t0 = time.time()
    for i, s in enumerate(steps, 1):
        if not run(s):
            sys.exit(f"\n[{s[0]}] 에서 중단 ({i}/{len(steps)}단계)")
    print(f"\n{'=' * 72}\n{len(steps)}단계 전부 성공 · {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
