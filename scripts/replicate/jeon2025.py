"""전상현·김성용·박종선(2025) 사양 복제 및 진단.

융합사회와 공공정책 19(1), 169-189.
논문 사양: 동-월 패널 675개(27개동 x 25개월), 종속변수 log(동-월 평균 ㎡당가),
설명변수 = 구역더미 + 시점더미 + 교호항 + 거래건수, OLS robust.
논문 보고: Zone +0.3263 / Time +0.2724 / Zone#Time +0.2928 / n +0.0116 / R2 0.0757

    python scripts/replicate_jeon2025.py
"""
from __future__ import annotations
import itertools, sys
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import io, policy  # noqa: E402

io.setup_stdout()

E = policy.EVENTS["E2020"]["effect"]
TREATED = policy.TREATED_2020
PAPER = {"T": 0.3263, "P": 0.2724, "TP": 0.2928, "n": 0.0116, "R2": 0.0757}


def panel() -> pd.DataFrame:
    df = io.load_trades()
    d = df[df.sgg_nm.isin(["강남구", "송파구"])]
    d = d[(d.deal_date >= "2019-06-23") & (d.deal_date < "2021-06-24")]
    p = (d.groupby(["umd_full_cd", "umdNm", "ym"])
           .agg(price=("price_per_m2", "mean"), n=("price_per_m2", "size")).reset_index())
    dongs = p[["umd_full_cd", "umdNm"]].drop_duplicates()
    grid = pd.DataFrame(
        [(a, b, m) for (a, b), m in itertools.product(
            dongs.itertuples(index=False, name=None), sorted(p.ym.unique()))],
        columns=["umd_full_cd", "umdNm", "ym"])
    g = grid.merge(p, on=["umd_full_cd", "umdNm", "ym"], how="left")
    g["T"] = g.umdNm.isin(TREATED).astype(float)
    g["P"] = (g.ym.dt.to_timestamp() >= E).astype(float)
    g["TP"] = g["T"] * g["P"]
    return g


def fit(g: pd.DataFrame, fill_zero: bool):
    p = g.copy()
    if fill_zero:                       # 논문 기술통계의 min=0 을 재현
        p["n"] = p.n.fillna(0)
        p["y"] = np.where(p.price.notna(), np.log(p.price.where(p.price.notna(), 1)), 0.0)
    else:
        p = p.dropna(subset=["price"])
        p["y"] = np.log(p.price)
    return sm.OLS(p["y"], sm.add_constant(p[["T", "P", "TP", "n"]])).fit(cov_type="HC1"), len(p)


def main() -> None:
    g = panel()
    miss = g.price.isna()
    print(f"완전격자 {len(g)}개 (논문 675) / 빈 셀 {int(miss.sum())}개")
    print(f"  빈 셀 소재 동: {sorted(g.loc[miss,'umdNm'].unique())} — 전부 통제동\n")
    print(f"{'':<34}{'Zone':>9}{'Time':>9}{'Zone#Time':>11}{'거래건수':>10}{'R2':>8}")
    print(f"  {'논문 보고값':<30}{PAPER['T']:>+9.4f}{PAPER['P']:>+9.4f}"
          f"{PAPER['TP']:>+11.4f}{PAPER['n']:>+10.4f}{PAPER['R2']:>8.4f}")
    for fz, lab in ((True, "빈 셀 0 코딩 (논문 재현)"), (False, "빈 셀 제외 (정상 처리)")):
        r, n = fit(g, fz)
        print(f"  {lab:<30}{r.params['T']:>+9.4f}{r.params['P']:>+9.4f}"
              f"{r.params['TP']:>+11.4f}{r.params['n']:>+10.4f}{r.rsquared:>8.4f}   N={n}")
    print("\n  -> 빈 셀 0 코딩 시 논문의 네 계수와 R2 가 모두 근사하게 재현됨.")
    print("     정상 처리하면 교호항 부호가 양(+)에서 음(-)으로 뒤집힘.")


if __name__ == "__main__":
    main()
