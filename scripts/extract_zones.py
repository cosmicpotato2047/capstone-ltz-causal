"""공고 본문에서 구역 목록(자치구·법정동·지번·면적)을 뽑는다.

    python scripts/extract_zones.py

입력  data/policy/notices/notice_text/*.txt   (build_policy_timeline.py 가 생성)
출력  data/policy/notices/zones.csv

공고의 구역 표는 세로로 풀려 나온다. 자치구가 한 줄, 구역명이 한 줄,
위치가 한 줄, 면적이 한 줄이다. 자치구는 여러 구역에 걸쳐 반복되지 않을
때가 있으므로 '현재 자치구'를 들고 가며 훑는다.

    성동구
    금호23
    금호4가동 1109번지 일대
    30,706

위치 줄에 자치구가 다시 붙는 공고도 있고(강동구 천호동 467-61번지 일대),
국제교류복합지구 공고처럼 '대치동(3.53㎢)' 형태로 ㎢ 를 쓰는 공고도 있다.

용도: 통제군 오염 점검. 정비사업·재건축 구역이 우리 통제 자치구·법정동에
걸리는지, 걸린다면 언제부터인지 확인한다.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import io, policy  # noqa: E402

io.setup_stdout()

NOTICES = io.POLICY / "notices"
TD = NOTICES / "notice_text"
GU_LIST = ("종로구 중구 용산구 성동구 광진구 동대문구 중랑구 성북구 강북구 도봉구 "
           "노원구 은평구 서대문구 마포구 양천구 강서구 구로구 금천구 영등포구 "
           "동작구 관악구 서초구 강남구 송파구 강동구").split()
GU_ALT = "|".join(GU_LIST)
GU_ONLY = re.compile(rf"^\s*({GU_ALT})\s*$")
# 위치 줄: [자치구] 동/가 지번 [번지] 일대
# 신월7동 · 금호4가동 · 금호동4가 · 충정로3가 · 본동 — 표기가 제각각이다
DONG = (r"(?:[가-힣]{1,5}\d?가동|[가-힣]{1,5}\d?동\d?가|[가-힣]{1,5}\d?동"
        r"|[가-힣]{1,5}로\d?가|[가-힣]{1,5}\d?가)")
LOC = re.compile(rf"^\s*(?:({GU_ALT})\s+)?({DONG})\s+(\d+(?:-\d+)?)\s*번지?\s*(?:일대)?\s*$")
# 면적: 33,000 (㎡) 또는 3.53 (㎢)
M2 = re.compile(r"^\s*([\d]{1,3}(?:,\d{3})+)\s*(?:㎡)?\s*$")
KM2 = re.compile(r"^\s*(\d+\.\d{1,4})\s*(?:㎢)?\s*$")
# 국제교류복합지구식: 대치동(3.53㎢)
INLINE = re.compile(rf"({DONG})\s*\(\s*([\d.,]+)\s*㎢\s*\)")
# '압구정동' 또는 '성수동1가, 성수동2가' 처럼 지번 없이 동만 적은 줄
DONGS_ONLY = re.compile(rf"^\s*{DONG}(?:\s*,\s*{DONG})*\s*$")

# '재지정기간 : 당초 공고 …기간 만료 후, 2021년 6월 23일부터' 형태도 잡는다
PERIOD = re.compile(r"(?:재)?지정\s*기간\s*:?\s*(?:[\s\S]{0,80}?)"
                    r"(20\d\d)\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일"
                    r"\s*부터\s*(20\d\d)\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일")



# 자치구 상태를 풀어야 하는 줄.
#  - 서울 밖(경기도 하남시 교산동 …): 서울 자치구명이 아니라서 GU_ONLY 에 안 걸리고,
#    그대로 두면 직전 자치구가 남의 동네를 삼킨다.
#  - 지형도면 붙임: '청담동(2.3㎢)' 같은 캡션이 본문 표와 떨어진 곳에 다시 나온다.
RESET = re.compile(r"^\s*(?:[가-힣]{2,5}(?:특별시|광역시|특별자치시|특별자치도|도)"
                   r"|[가-힣]{2,5}(?:시|군)|붙\s*임\s*\d*|별\s*첨\s*\d*"
                   r"|[^\n]{0,20}지형도면[^\n]{0,20})\s*$")
# 공고가 행정동 이름을 쓰는 경우가 있다. '금호4가동'(행정동) -> '금호동4가'(법정동).
HJ_GA = re.compile(r"^([가-힣]{1,4})(\d)가동$")


def to_beopjeong(dong: str) -> str:
    """행정동 표기를 법정동 표기로 되돌린다. 해당 없으면 그대로."""
    m = HJ_GA.match(dong)
    return f"{m.group(1)}동{m.group(2)}가" if m else dong


def area_of(ln: str) -> float | None:
    """면적 줄을 ㎢ 로 읽는다. 공고는 ㎡(33,000)와 ㎢(3.53)를 섞어 쓴다."""
    a, k = M2.match(ln), KM2.match(ln)
    if a:
        return float(a.group(1).replace(",", "")) / 1e6
    return float(k.group(1)) if k else None


def period_of(t: str) -> tuple[str, str]:
    m = PERIOD.search(t)
    if not m:
        return "", ""
    g = m.groups()
    return (f"{g[0]}-{int(g[1]):02d}-{int(g[2]):02d}",
            f"{g[3]}-{int(g[4]):02d}-{int(g[5]):02d}")


def main() -> None:
    rows = []
    for f in sorted(TD.glob("*.txt")):
        raw = f.read_text(encoding="utf-8")
        ser, (eff, exp) = policy.series_of(raw), period_of(raw)
        base = {"파일": f.stem, "계열": ser, "효력일": eff, "종료일": exp}
        lines = [re.sub(r"[ \t]+", " ", x).strip() for x in raw.splitlines()]

        cur_gu, pending = "", None
        for n, ln in enumerate(lines):
            g = GU_ONLY.match(ln)
            if g:
                cur_gu, pending = g.group(1), None
                continue
            if RESET.match(ln):        # 서울 밖으로 넘어갔거나 붙임이 시작됐다
                cur_gu, pending = "", None
                continue
            if pending is not None:                 # 직전 위치 줄의 면적을 받는다
                a = area_of(ln)
                if a is not None:
                    pending["면적_km2"] = a
                rows.append(pending)
                pending = None
                if a is not None:
                    continue
            m = LOC.match(ln)
            if m:
                gu, dong, jibun = m.groups()
                if gu:
                    cur_gu = gu
                if cur_gu:
                    pending = {**base, "자치구": cur_gu, "법정동": to_beopjeong(dong),
                               "지번": jibun, "면적_km2": None, "면적공유": False}
                continue
            # 지번 없이 동만 적는 공고도 있다 — '압구정동 / 1,149,476 / 압구정 아파트지구'.
            # 바로 다음 줄이 면적일 때만 인정해 오탐을 막는다.
            d = DONGS_ONLY.match(ln)
            if d and cur_gu:
                nxt = next((x for x in lines[n+1:n+3] if x), "")
                a = area_of(nxt)
                if a is not None:
                    dongs = [x.strip() for x in d.group(0).split(",") if x.strip()]
                    for dong in dongs:
                        rows.append({**base, "자치구": cur_gu, "법정동": to_beopjeong(dong),
                                     "지번": "", "면적_km2": a,
                                     "면적공유": len(dongs) > 1})
                    continue
            for dong, km2 in INLINE.findall(ln):    # 대치동(3.53㎢)
                if cur_gu:
                    rows.append({**base, "자치구": cur_gu, "법정동": to_beopjeong(dong),
                                 "지번": "", "면적공유": False,
                                 "면적_km2": float(km2.replace(",", ""))})
        if pending:
            rows.append(pending)

    if not rows:
        sys.exit("구역을 찾지 못했습니다.")
    df = (pd.DataFrame(rows)
          .drop_duplicates(subset=["파일", "자치구", "법정동", "지번"])
          .sort_values(["효력일", "자치구", "법정동"])
          .reset_index(drop=True))
    out = NOTICES / "zones.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"구역 {len(df):,}건 · 파일 {df.파일.nunique()}개 · 저장 {out.name}\n")

    print("=" * 88)
    print("계열별 구역 수")
    print("=" * 88)
    for s, n in df.계열.value_counts().items():
        print(f"  {s[:52]:<54}{n:>6,}")

    # --- 통제군 오염 점검 --------------------------------------------------
    reno = df[df.계열.str.contains("신속통합기획|공공재개발|모아타운|주요재건축단지",
                                   na=False)].copy()
    ctrl = [policy.SGG_NAME[c] for c in policy.CONTROL_SGG]
    win_lo = "2017-06-23"
    win_hi = policy.EVENTS["E2020"]["effect"].replace(year=2023).strftime("%Y-%m-%d")

    print("\n" + "=" * 88)
    print(f"통제 자치구 오염 — 주 분석 구간({win_lo} ~ {win_hi}) 안에 효력이 시작된 구역")
    print("=" * 88)
    hit = reno[reno.자치구.isin(ctrl) & reno.효력일.between(win_lo, win_hi)]
    if hit.empty:
        print("  없음")
    else:
        for (gu, dong, jibun), g in hit.groupby(["자치구", "법정동", "지번"]):
            print(f"  {gu} {dong} {jibun:<10} 최초 효력 {g.효력일.min()} "
                  f"({g.면적_km2.dropna().max() or 0:.4f}㎢, 공고 {len(g)}건)")

    print("\n" + "=" * 88)
    print("주 분석 통제동 — 강남·송파의 비처치 법정동에 걸린 구역 (구간 무관)")
    print("=" * 88)
    h2 = reno[reno.자치구.isin(["강남구", "송파구"]) &
              ~reno.법정동.isin(policy.TREATED_2020)]
    if h2.empty:
        print("  없음")
    else:
        for (dong, jibun), g in h2.groupby(["법정동", "지번"]):
            print(f"  {g.자치구.iloc[0]} {dong} {jibun:<10} 최초 효력 {g.효력일.min()}"
                  f"  [{g.계열.iloc[0][:28]}]")


if __name__ == "__main__":
    main()
