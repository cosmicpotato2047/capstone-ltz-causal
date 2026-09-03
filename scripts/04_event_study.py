"""[04] 2020.6.23 토허구역 지정 event study.

가격: 단지(aptSeq) x 계약월 양방향 고정효과 하에서 처치동 x 분기더미 계수 추정.
거래량: 법정동-월 패널에서 동일 구조.
기준분기 k=-1. 표준오차는 법정동 클러스터. (결정기록 0001, 0003)

    python scripts/04_event_study.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import econ, io, policy  # noqa: E402

io.setup_stdout()
for f in ("Malgun Gothic", "Batang"):
    if f in {x.name for x in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

EVENT = "E2020"
SGG = ["강남구", "송파구"]      # 같은 구 안에서 비교
KMIN, KMAX = -8, 8


def main() -> None:
    df = io.load_trades()
    df = df[df.sgg_nm.isin(SGG)].copy()
    df["kq"] = policy.event_quarter(df.deal_date, EVENT)
    df = df[(df.kq >= KMIN) & (df.kq <= KMAX)]
    df["treated"] = policy.assign_2020(df)
    df["log_area"] = np.log(df.area_m2)
    df["ym_s"] = df.ym.astype(str)

    print(f"[{policy.EVENTS[EVENT]['label']}]")
    print(f"표본 {len(df):,}건 / 처치 {int(df.treated.sum()):,} "
          f"통제 {int((~df.treated).sum()):,}")
    print(f"법정동 클러스터 {df.umd_full_cd.nunique()}개 / "
          f"단지 {df.aptSeq.nunique():,}개\n")

    price = econ.event_study(
        df, y="log_price_per_m2", treated="treated", kq="kq",
        fes=["aptSeq", "ym_s"], controls=["floor_no", "age_at_deal", "log_area"],
        cluster="umd_full_cd", kmin=KMIN, kmax=KMAX)

    # 거래량: 법정동-월 패널
    pan = df.groupby(["umd_full_cd", "ym_s"]).size().rename("n").reset_index()
    meta = df.groupby("umd_full_cd").agg(treated=("treated", "first")).reset_index()
    pan = pan.merge(meta, on="umd_full_cd")
    pan["deal_date"] = pd.PeriodIndex(pan.ym_s, freq="M").to_timestamp()
    pan["kq"] = policy.event_quarter(pan.deal_date, EVENT)
    pan["log_n"] = np.log(pan.n)
    vol = econ.event_study(
        pan, y="log_n", treated="treated", kq="kq",
        fes=["umd_full_cd", "ym_s"], controls=[], cluster="umd_full_cd",
        kmin=KMIN, kmax=KMAX)

    for nm, r in (("가격 (log ㎡당가)", price), ("거래량 (log 건수)", vol)):
        print(f"--- {nm} ---")
        print("   k   계수      표준오차   95% CI            실효")
        for _, x in r.iterrows():
            sig = " *" if (x.ci_lo > 0 or x.ci_hi < 0) and x.k != -1 else ""
            print(f"  {int(x.k):>3} {x.coef:+7.4f}  {x.se:7.4f}  "
                  f"[{x.ci_lo:+6.3f},{x.ci_hi:+6.3f}]  {x.effect_pct:+6.1f}%{sig}")
        print()

    # 그림
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
    fig.suptitle("2020.6.23 토지거래허가구역 지정 — 사전추세 및 처치효과 "
                 "(강남·송파 내 비교)", y=1.02, fontsize=12)
    fig.tight_layout()
    io.FIGURES.mkdir(parents=True, exist_ok=True)
    out = io.FIGURES / "event_study_2020.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")

    io.save_result("04_event_study", {
        "사건": policy.EVENTS[EVENT]["label"],
        "효력일": str(policy.EVENTS[EVENT]["effect"].date()),
        "표본": int(len(df)), "처치": int(df.treated.sum()),
        "클러스터": int(df.umd_full_cd.nunique()),
        "가격": price.to_dict("records"),
        "거래량": vol.to_dict("records"),
    })
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
