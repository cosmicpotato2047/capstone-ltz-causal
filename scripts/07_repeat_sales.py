"""[07] 자체 반복매매 가격지수 — 같은 집이 두 번 팔린 기록만으로 짓는다.

    python scripts/07_repeat_sales.py

입력  data/processed/trades.parquet
출력  data/processed/rs_index.csv          지수 (전체·처치동·통제동)
      data/processed/rs_pairs.parquet      쓰인 반복매매 쌍
      output/results/07_repeat_sales.json
      output/figures/07_repeat_sales.png

**왜 만드는가.** 지정 직후 거래가 −71~−80% 줄었다([[0017-합성대조군]]).
거래가 그만큼 줄면 **팔리는 물건의 구성이 바뀐다.** 급한 사람만 팔거나 특정
평형만 남을 수 있다. 그 상태에서 평균 거래가를 견주면 정책 효과인지 구성
변화인지 가를 수 없다 ([[0001-주결과변수-거래량]] 근거 4).

반복매매지수는 **같은 물건이 두 번 팔린 기록만** 쓴다. 물건이 고정되므로
구성 변화에 흔들리지 않는다 (Bailey·Muth·Nourse 1963, Case & Shiller 1987).

## 어떻게 짓는가

같은 물건의 두 거래에 대해

    log(P_t2) − log(P_t1) = β_t2 − β_t1 + 오차

를 세우고, 설명변수를 t2 에 +1, t1 에 −1 인 희소 행렬로 둔다. 추정된 β 를
지수로 환산한다. 기준 시점의 β 는 0 으로 고정한다.

Case & Shiller(1987)의 3단계 가중을 쓴다. 오래 보유한 쌍일수록 가격 변화의
분산이 커지므로, 잔차 제곱을 보유기간에 회귀해 그 역수의 제곱근으로 가중한다.

## 물건을 무엇으로 보는가

실거래 자료에는 호실 번호가 없다. **(단지, 전용면적, 층)** 을 한 물건으로 본다.
한 층에 같은 면적의 호실이 여럿이면 구분되지 않는다 — 월 단위로 보면
5.8%의 칸에 두 건 이상이 들어 있다. 다만 같은 단지·같은 면적·같은 층이면
가격이 거의 같으므로 지수에 주는 편의는 2차적이다.

그래도 짧은 간격의 쌍은 걸러낸다. 보유 **180일 미만**을 빼면 '가격 변화가
정확히 0'인 쌍이 4.1% 에서 1.5% 로 줄어든다. 연율 수익률이 ±50%를 넘는
0.5% 도 뺀다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
import numpy as np                            # noqa: E402
import pandas as pd                           # noqa: E402
from scipy import sparse                      # noqa: E402
from scipy.sparse.linalg import lsqr          # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import io, policy  # noqa: E402

io.setup_stdout()
for f in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
    if f in {x.name for x in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

MIN_HOLD = 180        # 최소 보유일
MAX_ANN = 0.50        # 연율 로그수익률 상한 (절댓값)
BASE = "2017-06"      # 지수 기준 시점 = 100. 주 분석 창의 시작
FREQ = "M"


def make_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """연속 거래 쌍을 만든다."""
    d = df.sort_values("deal_date").copy()
    d["unit"] = (d.aptSeq.astype(str) + "|" + d.area_m2.round(2).astype(str)
                 + "|" + d.floor_no.astype(str))
    for nm, col in (("p", "amount_manwon"), ("t", "deal_date")):
        d["prev_" + nm] = d.groupby("unit")[col].shift()
    p = d[d.prev_p.notna()].copy()
    p["hold"] = (p.deal_date - p.prev_t).dt.days
    p["dlog"] = np.log(p.amount_manwon / p.prev_p)
    p["ann"] = p.dlog / (p.hold.clip(lower=1) / 365.25)
    n0 = len(p)
    p = p[(p.hold >= MIN_HOLD) & (p.ann.abs() <= MAX_ANN)]
    print(f"  연속쌍 {n0:,} → 보유 {MIN_HOLD}일 이상·연율 ±{MAX_ANN:.0%} 이내 "
          f"{len(p):,} ({100*len(p)/n0:.0f}%)")
    return p


def bmn(pairs: pd.DataFrame, periods: pd.PeriodIndex) -> pd.Series:
    """반복매매 회귀. Case-Shiller 3단계 가중."""
    pi = {p: i for i, p in enumerate(periods)}
    i2 = pairs.deal_date.dt.to_period(FREQ).map(pi).to_numpy()
    i1 = pairs.prev_t.dt.to_period(FREQ).map(pi).to_numpy()
    ok = ~(pd.isna(i1) | pd.isna(i2))
    i1, i2 = i1[ok].astype(int), i2[ok].astype(int)
    y = pairs.dlog.to_numpy()[ok]
    hold = pairs.hold.to_numpy()[ok]
    n, T = len(y), len(periods)

    rows = np.repeat(np.arange(n), 2)
    cols = np.empty(2 * n, int)
    cols[0::2], cols[1::2] = i2, i1
    vals = np.empty(2 * n)
    vals[0::2], vals[1::2] = 1.0, -1.0
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(n, T))

    base = pi[pd.Period(BASE, freq=FREQ)]
    keep = [c for c in range(T) if c != base]      # 기준 시점 계수를 0 으로 고정
    Xk = X[:, keep]

    def solve(w=None):
        if w is None:
            b = lsqr(Xk, y, atol=1e-10, btol=1e-10, iter_lim=4000)[0]
        else:
            W = sparse.diags(w)
            b = lsqr(W @ Xk, w * y, atol=1e-10, btol=1e-10, iter_lim=4000)[0]
        full = np.zeros(T)
        full[keep] = b
        return full

    # 1단계 — 가중 없이
    b1 = solve()
    e = y - (X @ b1)
    # 2단계 — 잔차 제곱을 보유기간에 회귀 (분산이 보유기간에 비례한다는 가정)
    A = np.column_stack([np.ones(n), hold / 365.25])
    coef, *_ = np.linalg.lstsq(A, e ** 2, rcond=None)
    var = np.maximum(A @ coef, 1e-6)
    # 3단계 — 분산의 역수의 제곱근으로 가중
    b3 = solve(1.0 / np.sqrt(var))
    print(f"    1단계 잔차 표준편차 {e.std():.4f} · "
          f"보유 1년당 분산 증가 {coef[1]:.5f}")
    return pd.Series(100 * np.exp(b3), index=periods, name="index")


def coverage(pairs: pd.DataFrame, periods: pd.PeriodIndex) -> pd.Series:
    """각 시점에 쌍이 몇 개나 걸리는가. 지수가 얇은 구간을 알려 준다."""
    a = pairs.deal_date.dt.to_period(FREQ).value_counts()
    b = pairs.prev_t.dt.to_period(FREQ).value_counts()
    return (a.add(b, fill_value=0).reindex(periods).fillna(0).astype(int))


def main() -> None:
    df = io.load_trades()
    df = df[df.sgg_nm.isin(["강남구", "송파구"])].copy()
    df["treated"] = policy.assign_2020(df)

    print("=" * 76)
    print("반복매매 쌍 만들기 — 강남·송파")
    print("=" * 76)
    pairs = make_pairs(df)
    pairs["treated"] = pairs.umdNm.isin(policy.TREATED_2020)
    print(f"  처치동 {int(pairs.treated.sum()):,}쌍 · "
          f"통제동 {int((~pairs.treated).sum()):,}쌍")
    print(f"  보유기간 중앙 {pairs.hold.median():.0f}일 · "
          f"단지 {pairs.aptSeq.nunique():,}개")

    lo = min(pairs.prev_t.min(), pairs.deal_date.min()).to_period(FREQ)
    hi = max(pairs.prev_t.max(), pairs.deal_date.max()).to_period(FREQ)
    periods = pd.period_range(lo, hi, freq=FREQ)

    print("\n" + "=" * 76)
    print(f"반복매매 회귀 — {len(periods)}개 시점, 기준 {BASE} = 100")
    print("=" * 76)
    out = {}
    idx = {}
    for lab, sub in (("전체", pairs), ("처치동", pairs[pairs.treated]),
                     ("통제동", pairs[~pairs.treated])):
        print(f"  [{lab}]  {len(sub):,}쌍")
        idx[lab] = bmn(sub, periods)
        cov = coverage(sub, periods)
        idx[lab + "_쌍수"] = cov
        thin = int((cov < 10).sum())
        print(f"    쌍이 10개 미만인 달 {thin}개 / {len(periods)}")

    ix = pd.DataFrame(idx)
    ix.index.name = "연월"
    ix.to_csv(io.PROCESSED / "rs_index.csv", encoding="utf-8-sig")
    pairs.to_parquet(io.PROCESSED / "rs_pairs.parquet", index=False)

    # 주요 시점 값
    marks = ["2017-06", "2019-12", "2020-06", "2021-06", "2022-06",
             "2023-06", "2024-06", "2025-06", "2026-05"]
    print("\n" + "=" * 76)
    print("지수 (기준 2017-06 = 100)")
    print("=" * 76)
    print(f"  {'연월':<10}{'전체':>9}{'처치동':>9}{'통제동':>9}{'처치/통제':>10}")
    rows = []
    for m in marks:
        p = pd.Period(m, freq=FREQ)
        if p not in ix.index:
            continue
        a, b, c = ix.loc[p, "전체"], ix.loc[p, "처치동"], ix.loc[p, "통제동"]
        rows.append({"연월": m, "전체": round(a, 1), "처치동": round(b, 1),
                     "통제동": round(c, 1), "비": round(b / c, 3)})
        print(f"  {m:<10}{a:>9.1f}{b:>9.1f}{c:>9.1f}{b/c:>10.3f}")
    out["지수"] = rows

    # 지정 전후 처치/통제 비
    p_pre = pd.Period("2020-06", freq=FREQ) - 1
    r_pre = ix.loc[p_pre, "처치동"] / ix.loc[p_pre, "통제동"]
    for k, lab in ((12, "1년 뒤"), (24, "2년 뒤"), (36, "3년 뒤")):
        p = pd.Period("2020-06", freq=FREQ) + k
        if p not in ix.index:
            continue
        r = ix.loc[p, "처치동"] / ix.loc[p, "통제동"]
        out[f"처치통제비_{lab}"] = round(100 * (r / r_pre - 1), 2)
        print(f"  지정 직전 대비 {lab}: 처치/통제 비 {100*(r/r_pre-1):+.1f}%")


    # --- 구성 변화를 실제로 걸러냈는가 -------------------------------------
    print("\n" + "=" * 76)
    print("구성 변화 점검 — 중위 거래가 지수와 견준다")
    print("=" * 76)
    print("  중위 거래가는 '그 달에 팔린 물건'의 값이라 구성이 바뀌면 따라 흔들린다.")
    print("  반복매매지수는 같은 물건만 보므로 흔들리지 않아야 한다.\n")

    df["pm"] = df.deal_date.dt.to_period(FREQ)
    base_p = pd.Period(BASE, freq=FREQ)
    comp = {}
    for lab, sub in (("처치동", df[df.treated]), ("통제동", df[~df.treated])):
        med = sub.groupby("pm").amount_manwon.median()
        med = 100 * med / med.get(base_p, np.nan)
        comp[lab] = med.reindex(periods)
        # 그 달 거래의 평균 전용면적 — 구성이 바뀌는지 직접 본다
        comp[lab + "_면적"] = sub.groupby("pm").area_m2.mean().reindex(periods)

    win = [p for p in periods if pd.Period("2019-06", freq=FREQ) <= p
           <= pd.Period("2021-06", freq=FREQ)]
    print(f"  {'연월':<10}{'평균 전용면적(처치동)':>20}{'중위가 지수':>12}{'반복매매지수':>13}")
    for m in ("2019-12", "2020-03", "2020-06", "2020-09", "2020-12", "2021-06"):
        p = pd.Period(m, freq=FREQ)
        if p not in periods:
            continue
        print(f"  {m:<10}{comp['처치동_면적'][p]:>19.1f}㎡"
              f"{comp['처치동'][p]:>12.1f}{ix.loc[p,'처치동']:>13.1f}")

    a = comp["처치동_면적"]
    pre_a = a[[p for p in periods if pd.Period("2019-06", freq=FREQ) <= p
               < pd.Period("2020-06", freq=FREQ)]].mean()
    post_a = a[[p for p in periods if pd.Period("2020-06", freq=FREQ) <= p
                < pd.Period("2021-06", freq=FREQ)]].mean()
    print(f"\n  처치동 평균 전용면적  지정 전 1년 {pre_a:.1f}㎡ → 후 1년 {post_a:.1f}㎡"
          f"  ({100*(post_a/pre_a-1):+.1f}%)")
    print("  → 거래가 70% 줄면서 팔리는 물건의 크기 자체가 바뀌었다.")
    print("    중위 거래가로 가격을 말하면 이 변화가 섞여 들어간다.")
    out["구성변화"] = {"처치동_평균면적_지정전1년": round(float(pre_a), 2),
                       "처치동_평균면적_지정후1년": round(float(post_a), 2),
                       "변화_pct": round(float(100 * (post_a / pre_a - 1)), 2)}

    # --- 그림 --------------------------------------------------------------
    t = ix.index.to_timestamp()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    ax = axes[0]
    ax.plot(t, ix["처치동"], color="#c0392b", lw=1.7, label="처치 4개 동")
    ax.plot(t, ix["통제동"], color="#2980b9", lw=1.7, label="강남·송파 통제동")
    ax.axvline(pd.Timestamp("2020-06-23"), color="#333", lw=1.0, ls="--")
    ax.text(pd.Timestamp("2020-07-15"), ax.get_ylim()[1] * 0.97, " 2020-06-23 지정",
            fontsize=8.5, va="top")
    ax.set_ylabel(f"반복매매지수 ({BASE} = 100)")
    ax.set_title("자체 반복매매 가격지수", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    rel = 100 * (ix["처치동"] / ix["통제동"]) / r_pre
    ax.plot(t, rel, color="#8e44ad", lw=1.7)
    ax.axhline(100, color="#888", lw=0.8)
    ax.axvline(pd.Timestamp("2020-06-23"), color="#333", lw=1.0, ls="--")
    ax.set_ylabel("처치/통제 비 (지정 직전 = 100)")
    ax.set_title("처치동은 통제동 대비 어떻게 움직였나", fontsize=11, loc="left")
    fig.suptitle("반복매매지수 — 같은 집이 두 번 팔린 기록만으로",
                 fontsize=12.5, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fp = io.FIGURES / "07_repeat_sales.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)

    out["쌍"] = {"전체": int(len(pairs)), "처치동": int(pairs.treated.sum()),
                 "통제동": int((~pairs.treated).sum()),
                 "최소보유일": MIN_HOLD, "연율상한": MAX_ANN, "기준": BASE}
    io.save_result("07_repeat_sales", out)
    print(f"\n저장 {fp.relative_to(io.ROOT)}")
    print(f"     data/processed/rs_index.csv · rs_pairs.parquet")


if __name__ == "__main__":
    main()
