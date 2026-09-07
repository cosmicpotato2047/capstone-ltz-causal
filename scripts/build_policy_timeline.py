"""공고 원문에서 처치 이력을 구조화해 하나의 표로 만든다.

    python scripts/build_policy_timeline.py

입력  data/policy/notices/{molit,seoul}/*.pdf
출력  data/policy/notices/policy_timeline.csv   처치 이력 (분석에 쓰는 것)
      data/policy/notices/notice_text/*.txt     공고별 본문 (검증용)

시보 지면의 표는 글리프 번호로 저장된 서브셋 글꼴을 써서 그냥 뽑으면 깨진다.
lib.pdftext 가 글꼴별 오프셋을 찾아 되돌린 뒤 문장과 표에서 함께 뽑는다.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import io, policy  # noqa: E402
from lib.pdftext import text_of  # noqa: E402

io.setup_stdout()

NOTICES = io.POLICY / "notices"
TEXTDIR = NOTICES / "notice_text"

DONG = re.compile(r"([가-힣]+동\d?가?)\s*\(\s*([\d,\.]+)\s*㎢\s*\)")
# '재지정기간 : 당초 공고 …기간 만료 후, 2021년 6월 23일부터' 형태도 잡는다
PERIOD = re.compile(r"(?:재)?지정\s*기간\s*:?\s*(?:[\s\S]{0,80}?)"
                    r"(20\d\d)\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일"
                    r"\s*부터\s*(20\d\d)\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일")
BASIS = re.compile(r"공고\s*제?\s*(20\d\d)\s*-\s*(\d+)\s*호")
PUBDATE = re.compile(r"(20\d\d)\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일\s*\n?\s*(?:서\s*울\s*특\s*별\s*시\s*장|국토교통부\s*장관)")
AREA_RULE = re.compile(r"주거지역\s*(\d+)\s*㎡\s*초과")
SCOPE = re.compile(r"지정지역\s*:?\s*([^\n]{0,120})")
TOTAL = re.compile(r"([\d,]+(?:\.\d+)?)\s*㎢")

ACT = [("해제", "해제"), ("조정", "조정"), ("재지정", "재지정"), ("지정", "지정")]



def act_of(t: str, name: str) -> str:
    head = t[:800] + name
    for label, kw in ACT:
        if kw in head:
            return label
    return ""


def main() -> None:
    TEXTDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for sub, issuer in (("molit", "국토교통부"), ("seoul", "서울특별시")):
        d = NOTICES / sub
        if not d.exists():
            continue
        for f in sorted(d.glob("*.pdf")):
            t = text_of(f)
            (TEXTDIR / f"{f.stem}.txt").write_text(t, encoding="utf-8")

            per = PERIOD.search(t)
            eff = exp = ""
            if per:
                g = per.groups()
                eff = f"{g[0]}-{int(g[1]):02d}-{int(g[2]):02d}"
                exp = f"{g[3]}-{int(g[4]):02d}-{int(g[5]):02d}"
            pub = PUBDATE.search(t)
            pubd = (f"{pub.group(1)}-{int(pub.group(2)):02d}-{int(pub.group(3)):02d}"
                    if pub else "")
            basis = BASIS.findall(t)
            rule = AREA_RULE.search(t)
            scope = SCOPE.search(t)
            dongs = DONG.findall(t)

            rows.append({
                "기관": issuer,
                "파일": f.name,
                "공고일": pubd,
                "효력일": eff,
                "종료일": exp,
                "계열": policy.series_of(t),
                "행위": act_of(t, f.name),
                "근거공고": "|".join(f"{a}-{b}" for a, b in basis[:3]),
                "지정지역_문장": (scope.group(1).strip()[:70] if scope else ""),
                "동별면적": "|".join(f"{d}:{a}" for d, a in dongs[:12]),
                "주거기준_㎡초과": rule.group(1) if rule else "",
                "텍스트길이": len(t),
            })

    df = pd.DataFrame(rows).sort_values(["효력일", "공고일"], ascending=False)
    out = NOTICES / "policy_timeline.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print("=" * 112)
    print("우리 처치군 계열만 — 국제교류복합지구 · 주요재건축단지 · 강남3구용산 · 용산정비창")
    print("=" * 112)
    key = df[df.계열.str.contains("국제교류복합지구|주요재건축단지|강남3구용산|용산정비창", na=False)]
    key = key.sort_values("효력일")
    print(f"{'효력일':<12}{'종료일':<12}{'행위':<7}{'기준':>4}  {'계열':<26}동별 면적 / 근거")
    print("-" * 112)
    for _, r in key.iterrows():
        info = r.동별면적 or r.지정지역_문장
        print(f"{r.효력일:<12}{r.종료일:<12}{r.행위:<7}{r['주거기준_㎡초과']:>4}  "
              f"{r.계열[:24]:<26}{str(info)[:52]}")
    print(f"\n전체 {len(df)}건 저장: {out.name}")
    print(f"본문 텍스트 {len(list(TEXTDIR.glob('*.txt')))}건: notice_text/")


if __name__ == "__main__":
    main()
