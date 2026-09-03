"""이한웅·이춘원(2025) 사양 복제 및 진단.

대한부동산학회지 43(2), 161-180.
논문 사양: 처치 = 잠실동, 통제 = 송파구 나머지 동. 정책 전후 3년.
㎡당 매매가에 대해 각 집단 내부의 사전-사후 t-검정을 보고한다.

논문 보고 (표):
  ±3개월  잠실 2,310만 -> 2,690만 (+16.5%, t=5.11)  비잠실 1,400 -> 1,460 (+4.3%, t=2.91)
  ±6개월  잠실 2,290    -> 2,510    ( +9.6%, t=4.74)  비잠실 1,360 -> 1,530 (+12.5%, t=11.8)
  ±12개월 잠실 2,210    -> 2,630    (+19.0%, t=14.1)  비잠실 1,310 -> 1,610 (+22.9%, t=29.9)
  거래량  ±3개월 -87.4% / ±6개월 -69.2% / ±12개월 -73.4%

핵심: 보고된 t-검정은 각 집단 **내부**의 사전-사후 비교이며 이중차분 계수가 아니다.
논문 표의 수치로 차이의 차이를 계산하면 ±6개월부터 부호가 음(-)으로 바뀐다.

    python scripts/replicate/lee2025.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import econ, io, policy  # noqa: E402

io.setup_stdout()

E = policy.EVENTS["E2020"]["effect"]
WINDOWS = (3, 6, 12, 24, 36)
PAPER = {3: (16.5, 4.3), 6: (9.6, 12.5), 12: (19.0, 22.9)}   # (잠실 %, 비잠실 %)


def window(months: int) -> pd.DataFrame:
    df = io.load_trades()
    d = df[df.sgg_nm == "송파구"].copy()
    d = d[(d.deal_date >= E - pd.DateOffset(months=months))
          & (d.deal_date < E + pd.DateOffset(months=months))]
    d["T"] = d.umdNm.eq("잠실동")
    d["P"] = d.deal_date >= E
    return d


def main() -> None:
    out = {}

    print("=" * 92)
    print("1) 논문 표 재현 — 집단별 사전·사후 ㎡당 평균가와 차이의 차이")
    print("=" * 92)
    print(f"{'기간':<9}{'잠실 사전':>10}{'사후':>9}{'증감':>8}"
          f"{'비잠실 사전':>12}{'사후':>9}{'증감':>8}{'DID':>10}{'논문 DID':>11}")
    for m in WINDOWS:
        d = window(m)
        g = d.groupby(["T", "P"]).price_per_m2.mean().unstack()
        t_ch = 100 * (g.loc[True, True] / g.loc[True, False] - 1)
        c_ch = 100 * (g.loc[False, True] / g.loc[False, False] - 1)
        paper = f"{PAPER[m][0]-PAPER[m][1]:+.1f}%p" if m in PAPER else "—"
        print(f"±{m:<8}{g.loc[True,False]:>10.0f}{g.loc[True,True]:>9.0f}{t_ch:>+7.1f}%"
              f"{g.loc[False,False]:>12.0f}{g.loc[False,True]:>9.0f}{c_ch:>+7.1f}%"
              f"{t_ch-c_ch:>+9.1f}%p{paper:>11}")
        out[f"w{m}_did_pp"] = round(t_ch - c_ch, 2)

    print("\n" + "=" * 92)
    print("2) 같은 구간을 이중차분으로 추정 — 통제 유무에 따른 차이")
    print("=" * 92)
    print(f"{'기간':<9}{'단순(통제 없음)':>18}{'단지·월 고정효과':>20}{'N':>10}")
    for m in WINDOWS:
        d = window(m)
        d["Tf"] = d["T"].astype(float); d["Pf"] = d["P"].astype(float)
        d["TP"] = d.Tf * d.Pf
        d["ym_s"] = d.ym.astype(str); d["log_area"] = np.log(d.area_m2)
        X = ["TP", "floor_no", "age_at_deal", "log_area"]
        r0 = sm.OLS(d.log_price_per_m2, sm.add_constant(d[["Tf", "Pf", "TP"]])).fit(
            cov_type="cluster", cov_kwds={"groups": d.umd_full_cd})
        dm = econ.demean(pd.concat([d[["log_price_per_m2"]], d[X]], axis=1),
                         d[["aptSeq", "ym_s"]])
        r1 = sm.OLS(dm.log_price_per_m2, dm[X]).fit(
            cov_type="cluster", cov_kwds={"groups": d.umd_full_cd})
        f = lambda r: (f"{100*(np.exp(r.params['TP'])-1):+.1f}%"
                       + ("*" if r.pvalues["TP"] < .05 else " "))
        print(f"±{m:<8}{f(r0):>18}{f(r1):>20}{len(d):>10,}")
        out[f"w{m}_simple"] = round(float(r0.params["TP"]), 4)
        out[f"w{m}_unitfe"] = round(float(r1.params["TP"]), 4)

    print("\n" + "=" * 92)
    print("3) ±3개월 구간의 표본 구성 변화 — 양(+)의 추정치가 어디서 오는가")
    print("=" * 92)
    d = window(3)
    d = d[d["T"]]
    for p, lab in ((False, "사전"), (True, "사후")):
        x = d[d.P == p]
        print(f"  {lab} {len(x):>4}건: 평균면적 {x.area_m2.mean():5.1f}㎡ | "
              f"평균연식 {x.age_at_deal.mean():4.1f}년 | 거래단지 {x.aptSeq.nunique():>3}개")
    a, b = d[~d.P], d[d.P]
    print(f"\n  거래량 {100*(len(b)/len(a)-1):+.1f}% / 면적 {b.area_m2.mean()-a.area_m2.mean():+.1f}㎡ "
          f"/ 연식 {b.age_at_deal.mean()-a.age_at_deal.mean():+.1f}년")
    print("  -> 지정 직후 거래가 급감하며 팔리는 물건의 구성이 바뀐다.")
    print("     단지 고정효과를 넣으면 ±3개월의 양(+)의 추정치가 사실상 0으로 소멸한다.")

    io.save_result("replicate_lee2025", out)


if __name__ == "__main__":
    main()
