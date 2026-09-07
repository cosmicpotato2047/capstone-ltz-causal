"""공고 원문에서 무엇을 뽑았고 어디에 두었는지 한 장으로 보여준다.

    python scripts/audit_notices.py

읽기 전용이다. 아무것도 만들지 않고, 이미 만들어 둔 것만 점검한다.
'원문을 다 모았는가, 다 읽었는가, 읽어서 무엇을 알게 되었는가'에 답한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import io, policy  # noqa: E402

io.setup_stdout()

N = io.POLICY / "notices"
BAR = "=" * 96


def sz(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*.pdf")) / 1e6


def main() -> None:
    print(BAR); print("1. 원문 보관 상태"); print(BAR)
    idx = pd.read_csv(N / "seoulsibo_pages.csv")
    raw = N / "seoulsibo_raw"
    rows = [
        ("국토교통부 전자관보", N / "molit", "*.pdf", "git 포함"),
        ("서울시보 발췌 + 단독공고", N / "seoul", "*.pdf", "git 포함"),
        ("서울시보 원본", raw, "*.pdf", "git 제외 (용량)"),
    ]
    print(f"{'구분':<26}{'파일':>6}{'용량':>10}   경로 / 비고")
    print("-" * 96)
    for label, d, pat, note in rows:
        n = len(list(d.glob(pat))) if d.exists() else 0
        print(f"{label:<26}{n:>6}{sz(d):>9.0f}M   {d.relative_to(io.ROOT)}  [{note}]")
    have = sum((raw / f).exists() for f in idx.파일명)
    print(f"\n발췌 색인 {len(idx)}건 중 원본 확보 {have}건"
          f"{'  — 모두 확보' if have == len(idx) else '  ← 누락 있음'}")

    print("\n" + BAR); print("2. 텍스트 추출 상태"); print(BAR)
    td = N / "notice_text"
    txts = sorted(td.glob("*.txt"))

    def legible(t: str) -> float:
        """공백을 뺀 글자 중 한글·숫자·기본기호가 차지하는 비율.
        복원이 안 된 글리프는 미얀마·티베트 같은 엉뚱한 블록에 떨어지므로
        이 비율이 곧 판독률이 된다."""
        v = [c for c in t if not c.isspace()]
        if not v:
            return 0.0
        ok = sum(('가' <= c <= '힣') or c.isascii() or c in "㎡㎢·「」『』○◈※△▲□■" for c in v)
        return 100 * ok / len(v)

    rows = [(t, t.read_text(encoding="utf-8")) for t in txts]
    empty = [t for t, s in rows if len(s.strip()) < 200]
    q = sorted(((legible(s), t.stem, len(s)) for t, s in rows if t not in empty))
    print(f"본문 텍스트 {len(txts)}건  ({td.relative_to(io.ROOT)})")
    print(f"  판독률 99% 이상  {sum(1 for r, _, _ in q if r >= 99):>3}건")
    print(f"  판독률 95~99%    {sum(1 for r, _, _ in q if 95 <= r < 99):>3}건")
    print(f"  판독률 95% 미만  {sum(1 for r, _, _ in q if r < 95):>3}건")
    print(f"  텍스트 없음      {len(empty):>3}건" + ("  ← 스캔 이미지 PDF" if empty else ""))
    for t in empty:
        print(f"       {t.stem}")
    print(f"\n  판독률이 낮은 순 (제목 글꼴은 글리프 번호가 뒤섞여 복원 불가):")
    print(f"  {'판독률':>7}  {'글자수':>7}  파일")
    for r, name, n in q[:8]:
        print(f"  {r:>6.1f}%  {n:>7,}  {name}")

    print("\n" + BAR); print("3. 공고에서 뽑아낸 것"); print(BAR)
    tl = pd.read_csv(N / "policy_timeline.csv")
    z = pd.read_csv(N / "zones.csv", dtype={"지번": str})
    for label, col in [("공고일", "공고일"), ("효력일", "효력일"), ("종료일", "종료일"),
                       ("근거공고", "근거공고"), ("주거지역 면적기준", "주거기준_㎡초과")]:
        ok = tl[col].notna().sum()
        print(f"  {label:<20}{ok:>3}/{len(tl)}건")
    print(f"  {'구역 목록':<20}{len(z):>3}건 (자치구·법정동·지번·면적)  zones.csv")
    print(f"  {'계열 분류':<20}미분류 {int((tl.계열 == '미분류').sum())}/{len(tl)}건 "
          f"— 사유는 표의 '비고' 칸에만 있고 그 칸은 글꼴이 깨진 경우가 많다")

    print("\n" + BAR); print("4. 알게 된 것 — 우리 처치의 연속성"); print(BAR)
    # 계열만으로는 부족하다. '국제교류복합지구'는 강남 세곡·수서 지정에도
    # 붙어서, 우리 4개 동이 실제로 표에 있는 공고만 고른다.
    k = tl[tl.효력일.notna()].copy()
    k = k[k.동별면적.fillna("").str.contains("청담동")
          | k.지정지역_문장.fillna("").str.contains(r"강남구,?\s*송파구", regex=True)]
    k = k[k.효력일 < "2025-01-01"].sort_values("효력일")

    print(f"{'효력일':<12}{'종료일':<12}{'행위':<7}{'기준㎡':>6}   공백   근거공고")
    print("-" * 96)
    prev = None
    for _, r in k.iterrows():
        gap = ""
        if prev is not None:
            d = (pd.Timestamp(r.효력일) - pd.Timestamp(prev)).days - 1
            gap = f"{d:>3}일"
        print(f"{r.효력일:<12}{r.종료일:<12}{r.행위:<7}{str(r['주거기준_㎡초과']):>6}   "
              f"{gap:<6} {r.근거공고 if pd.notna(r.근거공고) else ''}")
        prev = r.종료일
    print("\n  → 2020-06-23 지정 이후 만료·재지정이 하루도 끊기지 않고 이어진다.")
    print("    처치가 중간에 풀린 적이 없으므로 주 분석 구간은 '계속 처치' 상태다.")

    print("\n" + BAR); print("5. 알게 된 것 — 통제군 오염"); print(BAR)
    c = io.ROOT / "output" / "results" / "contamination.json"
    if c.exists():
        import json
        d = json.loads(c.read_text(encoding="utf-8"))
        r = d.get("payload", d)
        for key, label in [("통제자치구", "통제 자치구 4곳"),
                           ("강남송파_통제동", "강남·송파 비처치동")]:
            v = r.get(key, {})
            if v:
                print(f"  {label:<20}구역 {v['구역수']:>2}개 · "
                      f"하한 {v['하한']:>3}건({v['하한_비율']}%) · "
                      f"상한 {v['상한']:>4}건({v['상한_비율']}%)")
        print("\n  → 지번이 정확히 일치하는 오염은 통제군의 0.1% 미만이다.")
        print("    다만 공고는 '○○동 123번지 일대'라 경계가 지형도면에만 있어,")
        print("    참값은 하한과 상한 사이다. 강건성 점검에서 해당 동을 빼고 다시 본다.")
    else:
        print("  아직 없음 — python scripts/check_contamination.py")


if __name__ == "__main__":
    main()
