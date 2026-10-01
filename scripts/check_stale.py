"""산출물 가운데 이번 실행에서 다시 만들어지지 않은 것을 찾는다.

    python run_all.py && python scripts/check_stale.py --since 1200
    python scripts/check_stale.py --since 86400        # 하루 안에 안 바뀐 것

**왜 필요한가.** 전체 재실행(`run_all.py`)은 **다시 만들어지는 파일만** 잡는다.
스크립트가 더 이상 쓰지 않는 산출물은 아무도 건드리지 않으므로 `git status` 에도
안 뜨고 영원히 낡은 채로 남는다.

2026-09-30 에 실제로 그랬다. `output/results/es_price.csv`·`es_volume.csv` 가
2026-09-03 파일이었는데, `04_event_study.py` 가 그 사이 JSON 저장으로 바뀌면서
아무도 만들지 않게 됐다. 그래서 0013(사건시점 정렬) 수정 **이전** 숫자가
석 달 가까이 남아 있었다. 전체 재실행을 두 번 했는데도 못 잡았다.

이 점검은 그 반대로 본다 — 바뀐 것이 아니라 **안 바뀐 것**을 센다.

**전체 재실행 직후에만 뜻이 있다.** 기준이 '지금부터 거슬러 몇 초'이므로
시간이 지난 뒤 돌리면 멀쩡한 파일까지 낡은 것으로 센다. `run_all.py` 의
마지막 단계(C2)로 등록해 두었으니 보통은 따로 돌릴 일이 없다.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import io  # noqa: E402

io.setup_stdout()

# 파이프라인이 만드는 곳. 원자료와 공고 원문은 손으로 넣는 것이라 뺀다.
WATCH = [
    ("output/results", "*.json"),
    ("output/results", "*.csv"),
    ("output/figures", "*.png"),
    ("data/processed", "*"),
    ("data/policy", "*.csv"),
]
# 파이프라인이 만들지 **않는 것이 맞는** 파일. 앞부분이 맞으면 건너뛴다.
# 여기 적을 때는 왜 파이프라인 밖인지 한 줄로 남긴다.
SKIP = (
    ("data/policy/loan_events.csv", "손으로 정리한 표 (0015)"),
    ("data/policy/regulated_area_events.csv", "손으로 정리한 표"),
    ("data/policy/notices/manifest.csv", "공고 원문 보관 대장. 손으로 적는다 (README)"),
    ("data/policy/notices/seoulsibo_pages.csv",
     "extract_seoulsibo.py 산출. 서울시보 원본 2.9GB 가 있어야 돈다"),
    ("output/results/explore_loan_heterogeneity.json",
     "scripts/explore/ 의 탐색 산출. 파이프라인에 넣지 않는다"),
    ("output/figures/notice_maps/",
     "render_notice_pages.py 로 사람이 한 번 뽑아 보관하는 증거물 (0010)"),
)


# index.html 이 가리켜야 하는 읽을거리. 만들어 놓고 링크를 안 걸면 아무도
# 못 본다 — 2026-10-02 발표 자료가 실제로 그랬다. 파일이 없어지는 것이 아니라
# **가리키는 데가 없어지는** 종류의 고아다.
LINKED = ["docs/slides/*.html", "docs/reports/*.html", "docs/*.html"]


def check_linked() -> list[str]:
    """만들었는데 index.html 이 가리키지 않는 문서를 찾는다."""
    idx = io.ROOT / "index.html"
    print("\n" + "=" * 78)
    print("index.html 이 안 가리키는 문서")
    print("=" * 78)
    if not idx.exists():
        print("  index.html 이 없다. 건너뛴다.")
        return []
    text = idx.read_text(encoding="utf-8")
    missing = []
    for pat in LINKED:
        for f in sorted(io.ROOT.glob(pat)):
            rel = f.relative_to(io.ROOT).as_posix()
            if rel not in text:
                missing.append(rel)
    if not missing:
        print("  없음. 읽을거리가 전부 걸려 있다.")
        return []
    for rel in missing:
        print(f"  [확인] {rel}")
    print("\n  만들었으면 index.html 에 카드를 추가한다.")
    return missing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", type=float, default=1800,
                    help="이 초 안에 안 바뀐 파일을 낡은 것으로 본다 (기본 1800)")
    ap.add_argument("--strict", action="store_true",
                    help="낡은 산출물이 있으면 1 로 끝낸다 (기본은 알리기만)")
    a = ap.parse_args()

    cut = time.time() - a.since
    fresh, stale = [], []
    for d, pat in WATCH:
        base = io.ROOT / d
        if not base.exists():
            continue
        for f in sorted(base.rglob(pat)):
            if not f.is_file():
                continue
            rel = f.relative_to(io.ROOT).as_posix()
            if any(rel.startswith(k) for k, _ in SKIP):
                continue
            (fresh if f.stat().st_mtime >= cut else stale).append(
                (rel, time.strftime("%Y-%m-%d %H:%M",
                                    time.localtime(f.stat().st_mtime))))

    print("=" * 78)
    print(f"산출물 신선도 — 최근 {a.since / 60:.0f}분 기준")
    print("=" * 78)
    print(f"  다시 만들어진 파일 {len(fresh)}개 · 안 만들어진 파일 {len(stale)}개")
    if stale:
        print("\n  이번 실행에서 아무도 만들지 않은 파일")
        print(f"  {'파일':<52}{'마지막 수정':>18}")
        for rel, when in stale:
            print(f"  {rel:<52}{when:>18}")
        print("\n  확인할 것 — 아래 셋 중 하나다.")
        print("   1. 만드는 스크립트가 run_all.py 에 빠졌다        -> STEPS 에 추가")
        print("   2. 아무도 안 만드는 고아 파일이다                -> 지운다")
        print("   3. 손으로 관리하는 파일이다                      -> SKIP 에 적는다")
    else:
        print("\n  낡은 산출물 없음. 모든 파일이 이번 실행에서 다시 만들어졌다.")

    unlinked = check_linked()
    if a.strict and (stale or unlinked):
        sys.exit(1)


if __name__ == "__main__":
    main()
