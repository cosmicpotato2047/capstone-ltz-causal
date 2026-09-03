"""2020.6.23 토허구역 지정 event study.

가격: 단지(aptSeq) + 계약월 양방향 고정효과 하에서 처치동 x 분기더미 계수를 추정.
거래량: 법정동-월 패널에서 동일 구조.
기준분기 k=-1. 표준오차는 법정동 클러스터.

    python scripts/event_study.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
for f in ("Malgun Gothic", "Batang"):
    if f in {x.name for x in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
EVENT = pd.Timestamp("2020-06-23")
TREATED_DONG = {"삼성동", "청담동", "대치동", "잠실동"}
SGG = ["강남구", "송파구"]          # 같은 구 안에서 비교
KMIN, KMAX = -8, 8                  # 분기, ±24개월
REF = -1


def demean(data: pd.DataFrame, fe: pd.DataFrame,
           tol: float = 1e-9, maxiter: int = 200) -> pd.DataFrame:
    """교대투영으로 다중 고정효과 제거. fe 의 각 열이 하나의 고정효과."""
    out = data.astype(float).copy()
    keys = [fe[c].to_numpy() for c in fe.columns]
    for _ in range(maxiter):
        prev = out.to_numpy(copy=True)
        for k in keys:
            out = out - out.groupby(k).transform("mean")
        if np.nanmax(np.abs(out.to_numpy() - prev)) < tol:
            break
    return out


def event_quarter(dates: pd.Series) -> pd.Series:
    m = (dates.dt.year - EVENT.year) * 12 + (dates.dt.month - EVENT.month)
    return np.floor(m / 3).astype(int)


def run(y: str, df: pd.DataFrame, fes: list[str], extra: list[str],
        cluster: str) -> pd.DataFrame:
    ks = [k for k in range(KMIN, KMAX + 1) if k != REF]
    X = pd.DataFrame({f"k{k}": ((df.kq == k) & df.treated).astype(float) for k in ks},
                     index=df.index)
    for c in extra:
        X[c] = df[c].astype(float)
    dm = demean(pd.concat([df[[y]], X], axis=1), df[fes])
    res = sm.OLS(dm[y], dm[list(X.columns)]).fit(
        cov_type="cluster", cov_kwds={"groups": df[cluster]})
    out = pd.DataFrame({"k": ks,
                        "coef": [res.params[f"k{k}"] for k in ks],
                        "se": [res.bse[f"k{k}"] for k in ks]})
    return pd.concat([out, pd.DataFrame({"k": [REF], "coef": [0.0], "se": [0.0]})]
                     ).sort_values("k").reset_index(drop=True)


def main() -> None:
    df = pd.read_parquet(ROOT / "data/processed/trades.parquet")
    df = df[df.sgg_nm.isin(SGG)].copy()
    df["kq"] = event_quarter(df.deal_date)
    df = df[(df.kq >= KMIN) & (df.kq <= KMAX)]
    df["treated"] = df.umdNm.isin(TREATED_DONG)
    df["log_area"] = np.log(df.area_m2)
    df["ym_s"] = df.ym.astype(str)
    print(f"표본 {len(df):,}건 / 처치 {int(df.treated.sum()):,} 통제 {int((~df.treated).sum()):,}")
    print(f"법정동 클러스터 {df.umd_full_cd.nunique()}개 / 단지 {df.aptSeq.nunique():,}개\n")

    price = run("log_price_per_m2", df, ["aptSeq", "ym_s"],
                ["floor_no", "age_at_deal", "log_area"], "umd_full_cd")

    # 거래량: 법정동-월 패널
    pan = (df.groupby(["umd_full_cd", "ym_s"]).size().rename("n").reset_index())
    meta = df.groupby("umd_full_cd").agg(treated=("treated", "first")).reset_index()
    pan = pan.merge(meta, on="umd_full_cd")
    pan["deal_date"] = pd.PeriodIndex(pan.ym_s, freq="M").to_timestamp()
    pan["kq"] = event_quarter(pan.deal_date)
    pan["log_n"] = np.log(pan.n)
    vol = run("log_n", pan, ["umd_full_cd", "ym_s"], [], "umd_full_cd")

    for nm, r in (("가격 (log ㎡당가)", price), ("거래량 (log 건수)", vol)):
        print(f"--- {nm} ---")
        print("   k   계수      표준오차   95% CI")
        for _, x in r.iterrows():
            lo, hi = x.coef - 1.96 * x.se, x.coef + 1.96 * x.se
            star = " *" if (lo > 0 or hi < 0) and x.k != REF else ""
            print(f"  {int(x.k):>3} {x.coef:+7.4f}  {x.se:7.4f}  [{lo:+6.3f},{hi:+6.3f}]{star}")
        print()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, (nm, r) in zip(axes, (("가격: log ㎡당 거래가", price),
                                   ("거래량: log 월별 건수", vol))):
        ax.errorbar(r.k, r.coef, yerr=1.96 * r.se, fmt="o-", ms=4, lw=1.4,
                    capsize=3, color="#2b6cb0")
        ax.axhline(0, color="#999", lw=.8)
        ax.axvline(-0.5, color="#c53030", ls="--", lw=1.2)
        ax.set_title(nm)
        ax.set_xlabel("지정 시점 기준 분기 (0 = 2020Q3)")
        ax.set_ylabel("처치동 - 통제동")
        ax.grid(alpha=.25)
    fig.suptitle("2020.6.23 토지거래허가구역 지정 — 사전추세 및 처치효과 (강남·송파 내 비교)",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    out = ROOT / "output"; out.mkdir(exist_ok=True)
    fig.savefig(out / "event_study_2020.png", dpi=150, bbox_inches="tight")
    price.to_csv(out / "es_price.csv", index=False)
    vol.to_csv(out / "es_volume.csv", index=False)
    print(f"저장: {out/'event_study_2020.png'}")


if __name__ == "__main__":
    main()
