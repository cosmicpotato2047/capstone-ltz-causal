"""[06c] 대출 규제가 토허제 효과에 섞였는가.

    python scripts/06c_loan.py

입력  data/processed/trades_seoul.parquet
      data/policy/loan_events.csv
출력  output/results/06c_loan.json
      output/figures/06c_bunching.png

**왜 보는가.** 2019-12-17 시행된 12·16 대책은 시가 15억 초과 아파트의
주택구입 목적 주담대를 전면 금지했다. 우리 처치군(청담·삼성·대치·잠실)은
거래의 64%가 15억 초과이고 통제군은 3% 미만이다([[0011-통제군-선정]]).
두 정책이 같은 대상에 6개월 간격으로 걸렸으므로, 2020-06-23 지정 효과
안에 대출 규제의 지연 효과가 섞여 있을 수 있다.

두 단계로 본다.

  1. **규제가 실제로 물렸는가** — 15억 문턱 바로 아래에 거래가 쏠리는지.
     규제 시행 전·중·후를 견주면 쏠림이 규제와 함께 나타나고 사라지는지
     보인다 (bunching, Best & Kleven 2018).
  2. **효과가 노출도에 따라 다른가** — 처치군 안에서 15억 초과·이하를
     갈라 삼중차분. 대출 규제가 이미 묶어 둔 고가 주택에서 토허제의
     추가 효과가 작다면, 우리가 재는 것의 일부는 대출 효과다.

**주의.** 대출 규제는 지역이 아니라 **주택 가격**에 걸린다. 그래서 노출도는
거래 시점의 가격이 아니라 **처치 이전에 이미 정해져 있던 값**으로 잡아야
한다. 거래 시점 가격으로 가르면 결과변수로 처치를 정의하는 꼴이 된다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                    # noqa: E402
import pandas as pd                   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import econ, io, policy  # noqa: E402

io.setup_stdout()
for f in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
    if f in {x.name for x in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

# loan_events.csv 에서 확인한 1차 자료 기준 날짜
RULE_ON = pd.Timestamp("2019-12-17")    # 15억 초과 주담대 금지 시행
RULE_OFF = pd.Timestamp("2023-03-02")   # 감독규정 개정으로 한도 규제 폐지
LTV_ON = pd.Timestamp("2019-12-23")     # 9억 초과분 LTV 20%
THRESHOLDS = (9.0, 15.0)                # 억원
BAND = 0.5                              # 문턱 위아래 이 폭으로 견준다


def bunch_ratio(x: pd.Series, thr: float, band: float = BAND) -> tuple[int, int]:
    """문턱 바로 아래와 바로 위의 거래 수."""
    below = int(((x >= thr - band) & (x < thr)).sum())
    above = int(((x >= thr) & (x < thr + band)).sum())
    return below, above


def main() -> None:
    df = pd.read_parquet(io.TRADES_SEOUL)
    df["eok"] = df.amount_manwon / 10000
    out: dict = {}

    print("=" * 76)
    print("1. 규제가 실제로 물렸는가 — 문턱 바로 아래 쏠림")
    print("=" * 76)
    periods = [
        ("규제 이전", "2018-01-01", "2019-12-16"),
        ("15억 금지 중", "2019-12-17", "2023-03-01"),
        ("폐지 이후", "2023-03-02", "2026-05-31"),
    ]
    tab = {}
    for thr in THRESHOLDS:
        print(f"\n  {thr:.0f}억 문턱  (바로 아래 {thr-BAND:.1f}~{thr:.1f} / "
              f"바로 위 {thr:.1f}~{thr+BAND:.1f})")
        tab[thr] = {}
        for lab, a, b in periods:
            s = df[df.deal_date.between(a, b)]
            lo, hi = bunch_ratio(s.eok, thr)
            r = lo / max(hi, 1)
            tab[thr][lab] = {"아래": lo, "위": hi, "비율": round(r, 3)}
            print(f"    {lab:<14}{lo:>7,} / {hi:>7,} = {r:>5.2f}배")
    out["쏠림"] = {str(k): v for k, v in tab.items()}

    print("\n  → 15억 규제가 켜졌을 때만 쏠림이 생기고 폐지하자 사라지면,")
    print("    그 규제가 실제로 거래를 움직였다는 뜻이다.")

    # --- 그림 : 가격 분포와 문턱 ------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    bins = np.arange(6, 25.01, 0.25)
    for ax, (lab, a, b) in zip(axes, periods):
        s = df[df.deal_date.between(a, b)]
        h, _ = np.histogram(s.eok, bins=bins)
        h = h / max(h.sum(), 1) * 100
        ax.bar(bins[:-1], h, width=0.24, color="#2c6e9c", align="edge")
        for thr in THRESHOLDS:
            ax.axvline(thr, color="#c0392b", lw=1.2, ls="--")
            lo_, hi_ = bunch_ratio(s.eok, thr)
            ax.annotate(f"{thr:.0f}억\n아래/위 {lo_/max(hi_,1):.2f}배",
                        xy=(thr + 0.3, h.max() * 0.80), fontsize=8.5,
                        color="#c0392b", va="top")
        ax.set_title(f"{lab}\n({a[:7]} ~ {b[:7]})", fontsize=10)
        ax.set_xlabel("거래가 (억원)")
        ax.set_xlim(6, 25)
    axes[0].set_ylabel("거래 비중 (%)")
    fig.suptitle("15억·9억 문턱 주변 가격 분포 — 규제가 켜진 구간에만 문턱 아래가 부푼다",
                 fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fp = io.FIGURES / "06c_bunching.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)

    # --- 2. 노출도별 효과 --------------------------------------------------
    print("\n" + "=" * 76)
    print("2. 노출도별 효과 — 처치군 안에서 15억 초과·이하를 가른다")
    print("=" * 76)
    g = io.load_trades()
    g = g[g.sgg_nm.isin(["강남구", "송파구"])].copy()
    g["treated"] = policy.assign_2020(g)
    g["em"] = policy.event_month(g.deal_date, "E2020")

    # 노출도는 처치 이전에 정해진 값이어야 한다. 단지별로 12·16 대책
    # 시행 직전 2년의 중위 거래가를 쓴다. 거래 시점 가격으로 가르면
    # 결과변수로 처치를 정의하는 꼴이 된다.
    pre = g[g.deal_date.between(RULE_ON - pd.DateOffset(years=2), RULE_ON)]
    med = pre.groupby("aptSeq").amount_manwon.median() / 10000
    g["base_eok"] = g.aptSeq.map(med)
    g["high"] = g.base_eok > 15.0
    have = g.base_eok.notna()
    print(f"  기준가를 매길 수 있는 거래 {int(have.sum()):,} / {len(g):,}"
          f" ({100*have.mean():.0f}%)")
    gg = g[have].copy()
    print(f"  처치군 중 15억 초과 단지 비중 "
          f"{100*gg[gg.treated].high.mean():.0f}% · "
          f"통제군 {100*gg[~gg.treated].high.mean():.0f}%")

    KMIN, KMAX = -8, 8
    res = {}
    for lab, sub in (("15억 초과 단지", gg[gg.high]), ("15억 이하 단지", gg[~gg.high])):
        meta = sub.groupby("umd_full_cd").agg(treated=("treated", "first")).reset_index()
        p = sub.groupby(["umd_full_cd", "em"]).size().rename("n").reset_index()
        full = pd.MultiIndex.from_product(
            [meta.umd_full_cd, range(KMIN * 3, KMAX * 3 + 3)],
            names=["umd_full_cd", "em"]).to_frame(index=False)
        p = (full.merge(p, on=["umd_full_cd", "em"], how="left")
             .fillna({"n": 0}).merge(meta, on="umd_full_cd"))
        if p.treated.nunique() < 2:
            print(f"  {lab}: 처치·통제가 모두 있지 않아 추정 불가")
            continue
        p["kq"] = np.floor(p.em / 3).astype(int)
        p["em_s"] = p.em.astype(str)
        p["y"] = np.log1p(p.n)
        r = econ.event_study(p, y="y", treated="treated", kq="kq",
                             fes=["umd_full_cd", "em_s"], controls=[],
                             cluster="umd_full_cd", kmin=KMIN, kmax=KMAX).set_index("k")
        res[lab] = {int(k): round(float(100 * np.expm1(r.loc[k, "coef"])), 2)
                    for k in (0, 1, 2, 4)}
        print(f"\n  [{lab}]  법정동 {p.umd_full_cd.nunique()}개")
        for k in (0, 1, 2, 4):
            print(f"    k={k:>2}  {res[lab][k]:>7.1f}%   (p={r.loc[k,'p']:.3f})")
    out["노출도별"] = res


    # --- 3. 삼중차분 — 두 효과의 차이가 유의한가 -------------------------
    print("\n" + "=" * 76)
    print("3. 삼중차분 — 고가와 저가의 효과 차이가 유의한가")
    print("=" * 76)
    print("  2절은 표본을 갈라 각각 추정한 것이라 '차이'의 유의성을 말할 수 없다.")
    print("  한 식에 교차항을 넣어 그 차이를 직접 추정한다.\n")

    # 원자료 평균을 먼저 보여 준다. 사건연구는 기준분기 하나에 기대는데
    # 그 분기가 이상치면(결정기록 0014) 표본을 가를수록 크게 흔들린다.
    # 삼중차분은 사전 구간 전체를 기준으로 삼으므로 원자료와 맞아야 한다.
    print("  원자료 — 법정동×사건월 칸의 평균 거래건수")
    print(f"  {'가격군':<6}{'집단':<6}{'사전':>7}{'사후':>7}{'변화':>9}")
    print("  " + "-" * 36)
    raw = {}
    for hi in (False, True):
        s = gg[(gg.high == hi) & gg.em.between(KMIN * 3, KMAX * 3 + 2)]
        q = s.groupby(["umd_full_cd", "em"]).size().rename("n").reset_index()
        q["treated"] = q.umd_full_cd.map(s.groupby("umd_full_cd").treated.first())
        q["post"] = q.em >= 0
        m = q.groupby(["treated", "post"]).n.mean()
        lab = "고가" if hi else "저가"
        raw[lab] = {}
        for tr in (True, False):
            a, b = m.get((tr, False), np.nan), m.get((tr, True), np.nan)
            raw[lab]["처치" if tr else "통제"] = {
                "사전": round(float(a), 1), "사후": round(float(b), 1),
                "변화_pct": round(float(100 * (b / a - 1)), 1)}
            print(f"  {lab if tr else '':<6}{'처치' if tr else '통제':<6}"
                  f"{a:>7.1f}{b:>7.1f}{100*(b/a-1):>8.0f}%")
    out["원자료평균"] = raw
    print("\n  저가는 처치·통제가 비슷하게 줄고, 고가는 처치만 크게 줄었다.\n")

    meta = (gg.groupby(["umd_full_cd", "high"])
            .agg(treated=("treated", "first")).reset_index())
    cnt = gg.groupby(["umd_full_cd", "high", "em"]).size().rename("n").reset_index()
    full = pd.MultiIndex.from_product(
        [range(KMIN * 3, KMAX * 3 + 3)], names=["em"]).to_frame(index=False)
    full = meta.merge(full, how="cross")
    p3 = (full.merge(cnt, on=["umd_full_cd", "high", "em"], how="left")
          .fillna({"n": 0}))
    p3["kq"] = np.floor(p3.em / 3).astype(int)
    p3["post"] = (p3.kq >= 0).astype(float)
    p3["y"] = np.log1p(p3.n)
    # 셀 식별자. 고정효과는 (법정동×가격군)과 (사건월×가격군)으로 둔다.
    # 그래야 가격군마다 다른 시간 추세가 흡수되고, 남는 것이 삼중차분이다.
    p3["cell"] = p3.umd_full_cd.astype(str) + "_" + p3.high.astype(str)
    p3["time"] = p3.em.astype(str) + "_" + p3.high.astype(str)
    p3["did"] = p3.treated.astype(float) * p3.post
    p3["ddd"] = p3["did"] * p3.high.astype(float)

    dm = econ.demean(p3[["y", "did", "ddd"]], p3[["cell", "time"]])
    import statsmodels.api as sm
    fit = sm.OLS(dm["y"], dm[["did", "ddd"]]).fit(
        cov_type="cluster", cov_kwds={"groups": p3.umd_full_cd})
    b_did, b_ddd = fit.params["did"], fit.params["ddd"]
    print(f"  셀 {len(p3):,}칸 (법정동×가격군 {p3.cell.nunique()} × 사건월 {p3.em.nunique()})")
    print(f"  {'항':<28}{'계수':>9}{'표준오차':>10}{'p':>8}   효과")
    print("  " + "-" * 70)
    print(f"  {'처치×사후 (저가 기준)':<28}{b_did:>9.3f}{fit.bse['did']:>10.3f}"
          f"{fit.pvalues['did']:>8.3f}   {100*np.expm1(b_did):>6.1f}%")
    print(f"  {'처치×사후×고가 (차이)':<28}{b_ddd:>9.3f}{fit.bse['ddd']:>10.3f}"
          f"{fit.pvalues['ddd']:>8.3f}   {100*np.expm1(b_ddd)-0:>6.1f}%p 더")
    print(f"  {'고가 단지의 효과 (합)':<28}{b_did+b_ddd:>9.3f}{'':>10}{'':>8}"
          f"   {100*np.expm1(b_did+b_ddd):>6.1f}%")
    sig = fit.pvalues["ddd"] < 0.05
    print(f"\n  → 차이는 {'유의하다' if sig else '유의하지 않다'}"
          f" (p={fit.pvalues['ddd']:.3f}).")
    if not sig:
        print("    고가에서 더 커 보이지만 그 차이를 통계적으로 단정할 수 없다.")
    out["삼중차분"] = {
        "처치×사후": round(float(b_did), 4),
        "처치×사후×고가": round(float(b_ddd), 4),
        "p_ddd": round(float(fit.pvalues["ddd"]), 4),
        "저가_효과_pct": round(float(100 * np.expm1(b_did)), 2),
        "고가_효과_pct": round(float(100 * np.expm1(b_did + b_ddd)), 2),
        "셀수": int(len(p3)),
    }

    io.save_result("06c_loan", out)
    print(f"\n저장 {fp.relative_to(io.ROOT)} · output/results/06c_loan.json")


if __name__ == "__main__":
    main()
