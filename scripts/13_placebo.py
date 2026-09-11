"""[13] 위약 검정 — 우리가 재는 것이 정말 처치 효과인가.

    python scripts/13_placebo.py

입력  data/processed/trades.parquet, trades_seoul.parquet
출력  output/results/13_placebo.json
      output/figures/13_placebo.png

세 가지를 본다.

  A. **가짜 사건일** — 지정이 없던 시점을 사건일로 두고 같은 추정을 돌린다.
     진짜 효과라면 가짜 날짜에서는 안 나와야 한다.
  B. **가짜 처치동** — 지정되지 않은 동을 처치군인 척 세운다. 배제 기준은
     결정기록 0012 와 같이 '동 전체 지정'뿐이다. 부분 지정까지 빼면
     남는 동이 넷뿐이라 검정이 성립하지 않는다.
  C. **평균 회귀 점검** — [[0015-대출규제-처리]] 에서 드러난 문제다.
     고가 처치동은 사전에 칸당 42.2건으로 고가 통제동(15.7건)의 2.7배였다.
     '원래 활발하던 곳이 평균으로 돌아온 것'과 처치 효과를 가려야 한다.
     사전 거래량이 비슷한 동끼리 짝지어 다시 추정한다.

A·B 는 '아무 데서나 나오는 값이 아니다'를 보이고, C 는 '하필 활발하던
곳이라서 나온 값이 아니다'를 보인다. 셋 다 통과해야 −53.7% 를 쓸 수 있다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                    # noqa: E402
import pandas as pd                   # noqa: E402
import statsmodels.api as sm          # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import econ, io, policy  # noqa: E402

io.setup_stdout()
for f in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
    if f in {x.name for x in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

EVENT = "E2020"
SGG = ["강남구", "송파구"]
PRE, POST = 24, 24       # 사건월 기준 앞뒤 개월. 가짜 날짜에도 같게 쓴다
REAL = policy.EVENTS[EVENT]["effect"]


def month_index(dates: pd.Series, anchor: pd.Timestamp) -> pd.Series:
    """효력일에 맞춘 경과 개월 (policy.event_month 와 같은 방식)."""
    d = pd.to_datetime(dates) - pd.Timedelta(days=anchor.day - 1)
    return (d.dt.year - anchor.year) * 12 + (d.dt.month - anchor.month)


def did_once(df: pd.DataFrame, anchor: pd.Timestamp, treated_col: str) -> dict:
    """한 사건일에 대해 단일 이중차분 계수를 낸다."""
    d = df.copy()
    d["em"] = month_index(d.deal_date, anchor)
    d = d[d.em.between(-PRE, POST - 1)]
    if d.empty or d[treated_col].nunique() < 2:
        return {}
    meta = d.groupby("umd_full_cd").agg(t=(treated_col, "first")).reset_index()
    cnt = d.groupby(["umd_full_cd", "em"]).size().rename("n").reset_index()
    full = meta.merge(pd.DataFrame({"em": range(-PRE, POST)}), how="cross")
    p = full.merge(cnt, on=["umd_full_cd", "em"], how="left").fillna({"n": 0})
    p["post"] = (p.em >= 0).astype(float)
    p["did"] = p.t.astype(float) * p.post
    p["y"] = np.log1p(p.n)
    p["time"] = p.em.astype(str)
    if p.t.nunique() < 2:
        return {}
    dm = econ.demean(p[["y", "did"]], p[["umd_full_cd", "time"]])
    fit = sm.OLS(dm["y"], dm[["did"]]).fit(
        cov_type="cluster", cov_kwds={"groups": p.umd_full_cd})
    b = float(fit.params["did"])
    return {"coef": b, "se": float(fit.bse["did"]), "p": float(fit.pvalues["did"]),
            "pct": float(100 * np.expm1(b))}


def main() -> None:
    df = io.load_trades()
    df = df[df.sgg_nm.isin(SGG)].copy()
    df["treated"] = policy.assign_2020(df)
    out: dict = {}

    real = did_once(df, REAL, "treated")
    print("=" * 76)
    print("기준 — 진짜 사건일 2020-06-23")
    print("=" * 76)
    print(f"  이중차분 {real['pct']:+.1f}%  (계수 {real['coef']:+.3f}, "
          f"표준오차 {real['se']:.3f}, p={real['p']:.4f})")
    out["실제"] = {k: round(v, 4) for k, v in real.items()}

    # --- A. 가짜 사건일 ----------------------------------------------------
    print("\n" + "=" * 76)
    print("A. 가짜 사건일 — 지정이 없던 날을 사건일로 두면")
    print("=" * 76)
    # 진짜 사건과 창이 겹치지 않도록 24개월 이상 떨어뜨린다
    fakes = [pd.Timestamp(x) for x in
             ("2013-06-23", "2014-06-23", "2015-06-23", "2016-06-23",
              "2017-06-23", "2023-06-23", "2024-06-23")]
    rows = []
    for f in fakes:
        r = did_once(df, f, "treated")
        if not r:
            continue
        rows.append({"사건일": str(f.date()), **{k: round(v, 4) for k, v in r.items()}})
        mark = " *" if r["p"] < 0.05 else ""
        print(f"  {f.date()}   {r['pct']:>+7.1f}%   p={r['p']:.3f}{mark}")
    n_sig = sum(1 for r in rows if r["p"] < 0.05)
    print(f"\n  가짜 {len(rows)}개 중 5% 수준에서 유의 {n_sig}개")
    print(f"  진짜 사건의 계수 {real['coef']:+.3f} 가 가짜들 사이에서 어디쯤인가:")
    fc = sorted(r["coef"] for r in rows)
    print(f"    가짜 계수 범위 [{fc[0]:+.3f}, {fc[-1]:+.3f}]")
    print(f"    진짜가 더 작은가(더 큰 감소인가): "
          f"{'그렇다' if real['coef'] < fc[0] else '아니다'}")
    out["A_가짜사건일"] = rows

    # --- B. 가짜 처치동 ----------------------------------------------------
    print("\n" + "=" * 76)
    print("B. 가짜 처치동 — 지정된 적 없는 동을 처치군인 척 세우면")
    print("=" * 76)
    # 배제 기준은 결정기록 0012 와 같아야 한다. 거기서 통제동을 빼는 기준은
    # '동 전체 지정'뿐이고, 동의 1~3%만 덮는 정비사업 부분 지정은 빼지
    # 않기로 했다. 여기서만 다르게 하면 앞뒤가 맞지 않는다.
    # 다만 압구정동처럼 동 전체가 나중에 지정된 곳은 가짜가 아니므로 뺀다.
    ever = set()
    sp = io.POLICY / "ltz_spells.csv"
    if sp.exists():
        s = pd.read_csv(sp)
        s = s[(s.상태 == "지정") & (s.지정방식 == "동전체") & (s.시작일 < "2025-10-01")]
        ever = set(zip(s.자치구, s.법정동))
    clean = [d for d in sorted(set(df[~df.treated].umdNm.unique()))
             if not any((g, d) in ever for g in SGG)]
    dirty = [d for d in sorted(set(df[~df.treated].umdNm.unique())) if d not in clean]
    print(f"  비처치동 {len(clean)+len(dirty)}개 중 한 번도 지정 안 된 동 {len(clean)}개")
    print(f"  제외 — 나중에 동 전체가 지정된 동 {len(dirty)}개: {', '.join(dirty)}")
    rows_b = []
    for d in clean:
        sub = df[~df.treated & df.umdNm.isin(clean)].copy()   # 오염된 동도 뺀다
        sub["fake"] = sub.umdNm == d
        if sub.fake.sum() < 200:
            continue
        r = did_once(sub, REAL, "fake")
        if not r:
            continue
        rows_b.append({"동": d, **{k: round(v, 4) for k, v in r.items()}})
    rows_b.sort(key=lambda x: x["coef"])
    print(f"  {'법정동':<10}{'효과':>9}{'p':>8}")
    for r in rows_b:
        mark = " *" if r["p"] < 0.05 else ""
        print(f"  {r['동']:<10}{r['pct']:>+8.1f}%{r['p']:>8.3f}{mark}")
    n_sig_b = sum(1 for r in rows_b if r["p"] < 0.05)
    below = sum(1 for r in rows_b if r["coef"] < real["coef"])
    print(f"\n  가짜 처치동 {len(rows_b)}개 중 유의 {n_sig_b}개")
    print(f"  진짜보다 더 크게 감소한 가짜 {below}개 "
          f"→ 경험적 p ≈ {(below+1)/(len(rows_b)+1):.3f}")
    out["B_가짜처치동"] = {"목록": rows_b, "유의": n_sig_b,
                          "경험적p": round((below + 1) / (len(rows_b) + 1), 4)}

    # --- C. 평균 회귀 ------------------------------------------------------
    print("\n" + "=" * 76)
    print("C. 평균 회귀 점검 — 사전 거래량이 비슷한 동끼리만 견주면")
    print("=" * 76)
    d = df.copy()
    d["em"] = month_index(d.deal_date, REAL)
    pre = d[d.em.between(-PRE, -1)]
    vol = pre.groupby("umd_full_cd").size() / PRE      # 사전 월평균 거래
    nm = d.groupby("umd_full_cd").umdNm.first()
    tr = d.groupby("umd_full_cd").treated.first()
    tab = pd.DataFrame({"동": nm, "월평균": vol, "처치": tr}).dropna()
    print(f"  사전 월평균 거래량 — 처치동 {tab[tab.처치].월평균.mean():.1f}건 "
          f"vs 통제동 {tab[~tab.처치].월평균.mean():.1f}건")

    res_c = {}
    for lo, hi, lab in ((0, 1e9, "전체"), (10, 1e9, "월 10건 이상"),
                        (20, 1e9, "월 20건 이상"), (30, 1e9, "월 30건 이상")):
        keep = tab[(tab.월평균 >= lo) & (tab.월평균 < hi)].index
        sub = d[d.umd_full_cd.isin(keep)]
        if sub.treated.nunique() < 2:
            continue
        r = did_once(sub, REAL, "treated")
        if not r:
            continue
        t_ = tab.loc[keep]
        res_c[lab] = {**{k: round(v, 4) for k, v in r.items()},
                      "처치동": int(t_.처치.sum()), "통제동": int((~t_.처치).sum()),
                      "처치_월평균": round(float(t_[t_.처치].월평균.mean()), 1),
                      "통제_월평균": round(float(t_[~t_.처치].월평균.mean()), 1)}
        v = res_c[lab]
        print(f"  {lab:<14}처치 {v['처치동']:>2}동({v['처치_월평균']:>5.1f}건) "
              f"통제 {v['통제동']:>2}동({v['통제_월평균']:>5.1f}건)   "
              f"{v['pct']:>+7.1f}%  p={v['p']:.3f}")
    out["C_평균회귀"] = res_c
    print("\n  사전 거래량을 맞출수록 계수가 0 으로 가면 평균 회귀를 의심해야 한다.")

    # --- 그림 --------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    ax = axes[0]
    xs = [r["coef"] for r in rows]
    ax.scatter(range(len(xs)), xs, color="#7f8c8d", s=36, label="가짜 사건일")
    ax.axhline(real["coef"], color="#c0392b", lw=1.6, label="진짜 2020-06-23")
    ax.axhline(0, color="#bbb", lw=0.8)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r["사건일"][:7] for r in rows], rotation=45, fontsize=8)
    ax.set_ylabel("이중차분 계수 (log1p)")
    ax.set_title("A. 가짜 사건일", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    xb = [r["coef"] for r in rows_b]
    ax.scatter(range(len(xb)), xb, color="#7f8c8d", s=36, label="가짜 처치동")
    ax.axhline(real["coef"], color="#c0392b", lw=1.6, label="진짜 처치동")
    ax.axhline(0, color="#bbb", lw=0.8)
    ax.set_xticks(range(len(rows_b)))
    ax.set_xticklabels([r["동"] for r in rows_b], rotation=60, fontsize=7.5)
    ax.set_title("B. 가짜 처치동", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)
    fig.suptitle("위약 검정 — 진짜 사건이 가짜들 바깥에 있는가",
                 fontsize=12.5, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fp = io.FIGURES / "13_placebo.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)

    io.save_result("13_placebo", out)
    print(f"\n저장 {fp.relative_to(io.ROOT)} · output/results/13_placebo.json")


if __name__ == "__main__":
    main()
