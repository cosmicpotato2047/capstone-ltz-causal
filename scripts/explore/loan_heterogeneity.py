"""대출 노출도에 따른 토허제 효과 이질성 — 예비 분석.

배경: 1주차 발표에서 "대출을 고려해야 하지 않느냐"는 질문을 받았다.
두 가지 해석이 가능하다.

  (a) 교란변수 — 같은 시기 대출규제가 토허제 효과와 섞였다
  (b) 효과 이질성 — 대출 조건에 따라 토허제 효과가 다르다

2019-12-16 시행된 15억 초과 주택담보대출 금지는 명목상 전국 동일이지만
가격 수준에 따라 차등이므로, 고가 물건이 많은 처치동에 실질적으로 더 세게 걸린다.
(2019년 거래 기준 15억 초과 비중: 처치동 79.7% vs 통제동 32.8%)

이 스크립트는 (a)를 검정하고 (b)를 측정한다.

    python scripts/explore/loan_heterogeneity.py

주의: 예비 분석이다. 확정하려면 삼중차분으로 다시 짜야 한다 (백로그 6c).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import econ, io, policy  # noqa: E402

io.setup_stdout()

RULE_DATE = pd.Timestamp("2019-12-16")   # 15억 초과 주담대 금지
THRESHOLD = 15.0                          # 억원
SGG = ["강남구", "송파구"]


def prepare() -> pd.DataFrame:
    df = io.load_trades()
    d = df[df.sgg_nm.isin(SGG)].copy()

    # 규칙 시행 이전 가격으로 (단지·면적) 조합을 분류한다.
    # 사후 가격으로 나누면 처치의 결과를 기준으로 표본을 가르게 된다.
    pre = d[(d.deal_date >= "2019-01-01") & (d.deal_date < RULE_DATE)]
    med = pre.groupby(["aptSeq", "area_m2"]).amount_eok.median()
    d = d.join(med.rename("pre_med"), on=["aptSeq", "area_m2"])
    d = d[d.pre_med.notna()].copy()

    d["over15"] = d.pre_med >= THRESHOLD
    d["kq"] = policy.event_quarter(d.deal_date, "E2020")
    d = d[(d.kq >= -8) & (d.kq <= 8)]
    d["treated"] = policy.assign_2020(d)
    d["ym_s"] = d.ym.astype(str)
    d["log_area"] = np.log(d.area_m2)
    return d


def main() -> None:
    d = prepare()

    print("=" * 78)
    print("1) 15억 룰의 노출도가 처치·통제에 다른가")
    print("=" * 78)
    y19 = d[(d.deal_date >= "2019-01-01") & (d.deal_date < "2020-01-01")]
    for t, lab in ((True, "처치동"), (False, "통제동")):
        x = y19[y19.treated == t]
        print(f"  {lab}: 거래 {len(x):>6,}건 · 15억 초과 비중 {100*(x.amount_eok>=15).mean():>5.1f}%"
              f" · 중위가 {x.amount_eok.median():.1f}억")
    print("  -> 명목상 전국 동일 규제이나 실질 노출도가 크게 다르다")

    print("\n" + "=" * 78)
    print("2) 노출도별 사전추세와 처치효과")
    print("=" * 78)
    out = {}
    for grp, lab in ((True, f"{THRESHOLD:.0f}억 초과 — 대출 전면금지"),
                     (False, f"{THRESHOLD:.0f}억 이하 — 규제 약함")):
        s = d[d.over15 == grp]
        es = econ.event_study(
            s, y="log_price_per_m2", treated="treated", kq="kq",
            fes=["aptSeq", "ym_s"], controls=["floor_no", "age_at_deal", "log_area"],
            cluster="umd_full_cd", kmin=-6, kmax=6)
        e = {int(r.k): r for _, r in es.iterrows()}
        sig = lambda k: "*" if (e[k]["ci_lo"] > 0 or e[k]["ci_hi"] < 0) else " "
        kk = np.array([-4, -3, -2, -1])
        slope, _ = np.polyfit(kk, [e[k]["coef"] for k in kk], 1)

        print(f"\n[{lab}]  거래 {len(s):,}건 (처치 {int(s.treated.sum()):,})")
        print("  사전  " + "  ".join(f"k{k}={e[k]['coef']:+.3f}{sig(k)}" for k in range(-5, 0)))
        print("  사후  " + "  ".join(f"k{k}={e[k]['coef']:+.3f}{sig(k)}" for k in range(0, 5)))
        print(f"  사전추세 기울기 {slope:+.4f}/분기   k=4 효과 "
              f"{100*(np.exp(e[4]['coef'])-1):+.1f}%")
        out[lab] = {"n": int(len(s)), "n_treated": int(s.treated.sum()),
                    "pretrend_slope": round(float(slope), 4),
                    "k4_pct": round(100*(np.exp(e[4]['coef'])-1), 1)}

    print("\n" + "=" * 78)
    print("3) 판정")
    print("=" * 78)
    print("""  (a) 교란 가설: 기각.
      15억 룰이 사전추세의 원인이라면 초과 그룹에서 추세가 가팔라야 하나
      실제로는 이하 그룹이 더 가파르다.

  (b) 효과 이질성: 지지.
      처치효과가 이하 그룹에서 2배 이상 크다. 해석은
      "15억 초과 물건은 이미 대출이 막혀 있어 토허제가 추가로 줄 충격이 작고,
       이하 물건은 대출로 갭투자가 가능했으므로 실거주 의무의 제약이 크다".

  한계: 이하 그룹의 처치 표본이 얇고, 비싼 동네의 저가 물건이라 성격이 특이할 수
      있다. 확정하려면 삼중차분(시간 x 처치동 x 노출도)으로 다시 짜야 한다.""")

    io.save_result("explore_loan_heterogeneity", {
        "규칙": f"{THRESHOLD}억 초과 주담대 금지 ({RULE_DATE.date()})",
        "분류기준": "2019-01-01 ~ 규칙시행 이전 (단지·면적) 중위 거래가",
        "노출도": {"처치동": round(float((y19[y19.treated].amount_eok >= 15).mean()), 3),
                   "통제동": round(float((y19[~y19.treated].amount_eok >= 15).mean()), 3)},
        "그룹별": out,
        "판정": {"교란가설": "기각", "효과이질성": "지지(예비)"},
    })


if __name__ == "__main__":
    main()
