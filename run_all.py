"""전체 파이프라인 진입점.

    python run_all.py              # 02 부터 (수집은 오래 걸려 기본 제외)
    python run_all.py --collect    # 01 수집부터 전부
    python run_all.py --only 04    # 특정 단계만

재현 절차는 README.md 참조. 원자료 스냅샷 시점: 2026-09-03.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    ("01", "scripts/01_collect.py", "실거래 자료 수집 (수십 분 소요)"),
    ("02", "scripts/02_build.py", "정제 및 검증"),
    ("04", "scripts/04_event_study.py", "2020.6 지정 event study"),
]
EXTRA = [
    ("R1", "scripts/replicate/jeon2025.py", "선행연구 재현 (전상현 외 2025)"),
    ("R2", "scripts/replicate/lee2025.py", "선행연구 재현 (이한웅·이춘원 2025)"),
    ("C1", "scripts/check_numbers.py", "문서 수치 검증 (결정기록 0007)"),
]


def run(path: str, label: str) -> bool:
    print(f"\n{'=' * 70}\n▶ {path}  —  {label}\n{'=' * 70}")
    t0 = time.time()
    r = subprocess.run([sys.executable, str(ROOT / path)], cwd=ROOT)
    ok = r.returncode == 0
    print(f"[{'OK' if ok else 'FAIL'}] {path} ({time.time() - t0:.1f}초)")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true", help="01 수집 단계 포함")
    ap.add_argument("--only", help="특정 단계 번호만 실행 (예: 04)")
    ap.add_argument("--extra", action="store_true", help="재현 스크립트도 실행")
    args = ap.parse_args()

    steps = STEPS + (EXTRA if args.extra else [])
    if args.only:
        steps = [s for s in steps if s[0] == args.only]
        if not steps:
            sys.exit(f"단계 '{args.only}' 를 찾을 수 없습니다.")
    elif not args.collect:
        steps = [s for s in steps if s[0] != "01"]

    for num, path, label in steps:
        if not run(path, label):
            sys.exit(f"\n{num} 단계에서 중단되었습니다.")
    print("\n전체 완료. 결과는 output/ 참조.")


if __name__ == "__main__":
    main()
