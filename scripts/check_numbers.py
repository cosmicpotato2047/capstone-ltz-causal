"""문서에 적힌 수치가 분석 결과와 어긋나는지 검사한다.

    python scripts/check_numbers.py

배경은 결정기록 0007. 요약하면 두 가지를 구분한다.

  A. 조건이 달라서 숫자가 다른 것    -> 둘 다 사실. 검사 대상 아님
  B. 조건은 같은데 옛 값이 남은 것    -> 오류. 이 스크립트가 잡는다

따라서 이 스크립트는 "모든 문서를 최신 숫자로 맞추는" 도구가 아니다.
주차 보고서 같은 동결 문서는 그 시점의 기록이므로 갱신하지 않는다.

검사 항목
  1. 살아있는 문서에 폐기된 값이 남아 있는가          -> 오류
  2. 동결 문서에 스냅샷 표기가 있는가                  -> 경고
  3. 감시 대상 수치가 어느 문서에 어떤 값으로 있는가   -> 보고

현재 값은 output/results/*.json 에서 읽는다. 이 스크립트에 하드코딩하면
같은 문제가 반복되기 때문이다. 폐기된 값 목록만 손으로 관리한다(추가 전용).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import io  # noqa: E402

io.setup_stdout()

# --- 감시 대상 --------------------------------------------------------------
# current  : output/results/<파일>.json 의 키 경로
# retired  : 과거에 쓰였으나 지금은 조건이 달라진 값 (추가 전용, 지우지 않음)
WATCHED = {
    "분석 표본": {
        "source": ("02_build", ["표본"]),
        "retired": {472812: "신고지연 절단 적용 전",
                    321785: "6개 자치구만 수집됐을 때"},
    },
    "단지 수": {
        "source": ("02_build", ["단지"]),
        "retired": {2613: "6개 자치구만 수집됐을 때"},
    },
    "반복매매 쌍": {
        "source": ("02_build", ["반복매매쌍"]),
        "retired": {253921: "6개 자치구만 수집됐을 때"},
    },
    "취소거래 제외": {
        "source": ("02_build", ["제외", "해제(취소)된 거래"]),
        "retired": {4285: "6개 자치구만 수집됐을 때"},
    },
}

# --- 문서 분류 --------------------------------------------------------------
LIVING = ["docs/backlog.md", "README.md", "docs/glossary.md", "docs/proposal.md"]
FROZEN_GLOBS = ["docs/reports/*.md", "docs/slides/*.html"]
SNAPSHOT_RE = re.compile(r"스냅샷|수집 ?시점|20\d\d-\d\d-\d\d 수집")


# 정수가 아닌 값(효과 크기 등)은 JSON 에서 끌어올 수 없어 따로 둔다.
# 값 자체보다 **폐기된 표현이 살아있는 문서에 남아 있는지**가 중요하다.
RETIRED_TEXT = {
    "-59%": "거래량 k=0. 달력 월 정렬 + 빈칸 방치 (결정기록 0013 이전)",
    "−59%": "같음",
    "-58.7%": "같음",
    "−58.7%": "같음",
}


def load_current() -> dict[str, int]:
    out = {}
    for name, spec in WATCHED.items():
        if "source" not in spec:          # 수동 관리 항목은 건너뛴다
            continue
        f, path = spec["source"]
        p = io.RESULTS / f"{f}.json"
        if not p.exists():
            sys.exit(f"{p} 가 없습니다. scripts/02_build.py 를 먼저 실행하세요.")
        v = json.loads(p.read_text(encoding="utf-8"))
        for k in path:
            v = v[k]
        out[name] = int(v)
    return out


def numbers_in(text: str) -> set[int]:
    """1,234 또는 1234 형태의 정수를 모두 뽑는다."""
    return {int(m.replace(",", "")) for m in re.findall(r"\d{1,3}(?:,\d{3})+|\d{4,}", text)}


def check_retired_text() -> int:
    """폐기된 표현이 살아있는 문서에 남아 있는가."""
    bad = 0
    print()
    print("=" * 74)
    print("4) 폐기된 표현이 살아있는 문서에 남아 있는가")
    print("=" * 74)
    for rel in LIVING:
        p = io.ROOT / rel
        if not p.exists():
            continue
        for ln in p.read_text(encoding="utf-8").splitlines():
            # 옛 값임을 명시한 줄은 통과시킨다. '-59% -> -76%' 처럼 변경
            # 이력을 적는 것은 정당하고, 오히려 남겨야 한다.
            # 예외어는 좁게 둔다. '이전'은 '지정 이전'에도 걸려 못 쓴다.
            if any(w in ln for w in ("→", "->", "종전", "폐기", "0013")):
                continue
            for expr, why in RETIRED_TEXT.items():
                if expr in ln:
                    print(f"  [오류] {rel} 에 '{expr}' — {why}")
                    print(f"         {ln.strip()[:78]}")
                    bad += 1
    if not bad:
        print("  없음")
    return bad


def main() -> None:
    cur = load_current()
    errors, warns = [], []

    print("=" * 74)
    print("현재 값  (output/results/)")
    print("=" * 74)
    for k, v in cur.items():
        print(f"  {k:<14} {v:>12,}")

    print("\n" + "=" * 74)
    print("1) 살아있는 문서 — 폐기된 값이 남아 있는가")
    print("=" * 74)
    for rel in LIVING:
        p = io.ROOT / rel
        if not p.exists():
            continue
        nums = numbers_in(p.read_text(encoding="utf-8"))
        bad = []
        for name, spec in WATCHED.items():
            for old, why in spec["retired"].items():
                if old in nums:
                    bad.append(f"{name} {old:,} ({why}) -> 현재 {cur[name]:,}")
        if bad:
            print(f"  [오류] {rel}")
            for b in bad:
                print(f"         {b}")
            errors += bad
        else:
            print(f"  [정상] {rel}")

    print("\n" + "=" * 74)
    print("2) 동결 문서 — 조건(스냅샷) 표기가 있는가")
    print("=" * 74)
    frozen = [p for g in FROZEN_GLOBS for p in sorted(io.ROOT.glob(g))]
    for p in frozen:
        rel = p.relative_to(io.ROOT).as_posix()
        text = p.read_text(encoding="utf-8")
        if SNAPSHOT_RE.search(text):
            print(f"  [정상] {rel}")
        else:
            print(f"  [경고] {rel} — 스냅샷/수집시점 표기 없음")
            warns.append(rel)

    print("\n" + "=" * 74)
    print("3) 감시 수치가 어느 문서에 있는가")
    print("=" * 74)
    all_docs = [io.ROOT / r for r in LIVING if (io.ROOT / r).exists()] + frozen
    for name, spec in WATCHED.items():
        print(f"\n  {name}  (현재 {cur[name]:,})")
        for p in all_docs:
            nums = numbers_in(p.read_text(encoding="utf-8"))
            rel = p.relative_to(io.ROOT).as_posix()
            if cur[name] in nums:
                print(f"    현재값  {rel}")
            for old in spec["retired"]:
                if old in nums:
                    tag = "폐기값" if rel in LIVING else "옛값(동결문서)"
                    print(f"    {tag}  {rel}  <- {old:,}")

    print("\n" + "=" * 74)
    if errors:
        print(f"실패 — 살아있는 문서에 폐기된 값 {len(errors)}건")
        print("동결 문서(주차 보고서·발표자료)의 옛값은 그 시점의 기록이므로 고치지 않는다.")
        sys.exit(1)
    if check_retired_text():
        sys.exit(1)
    print(f"통과 — 살아있는 문서에 폐기된 값 없음" + (f" (경고 {len(warns)}건)" if warns else ""))


if __name__ == "__main__":
    main()
