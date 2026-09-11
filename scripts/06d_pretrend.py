"""[06d] 사전 추세 위배를 진단하고, 대응 셋을 나란히 재 본다.

    python scripts/06d_pretrend.py

입력  data/processed/trades.parquet
출력  output/results/06d_pretrend.json
      output/figures/06d_pretrend.png

**문제.** 사건 시점 정렬을 바로잡고(결정기록 0013) 사전 구간을 제대로
출력하니 거래량도 사전 추세를 위배한다. 주 결과변수를 가격에서 거래량으로
바꾼 근거가 '거래량은 위배하지 않는다'였으므로([[0001]]) 그대로 둘 수 없다.

**진단.** 월 단위로 펴 보면 사전 기간의 처치-통제 격차가 기준월(k=-1,
2020년 5월) 대비 내내 음수다. 사전에 추세가 있다기보다 **기준월 하나가
높다**는 뜻이다. 2020년 5월은 코로나 1차 유행이 잦아들며 거래가 반등하고
6·17 대책 발표 직전이라 막차 수요가 몰린 달이다.

그래서 셋을 재고 결정한다.

    A. 기준 기간 — k=-1 한 달 대신 사전 여러 달을 기준으로
    B. 구간 절단 — 12·16 대책(2019-12-16) 이후만 사전으로
    C. 사전 추세 외삽 — 사전 기울기를 사후로 늘려 뺀 값

셋의 답이 같은 방향이면 결론은 안전하다. 갈리면 그 폭이 곧 불확실성이다.
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

EVENT = "E2020"
SGG = ["강남구", "송파구"]
KMIN, KMAX = -8, 8
LOAN = pd.Timestamp("2019-12-16")     # 12·16 대책 (시가 15억 초과 주담대 금지)


def panel(df: pd.DataFrame, mmin: int, mmax: int) -> pd.DataFrame:
    """법정동 × 사건월 거래량 패널. 거래 없는 달은 0으로 채운다 (결정기록 0013)."""
    meta = df.groupby("umd_full_cd").agg(treated=("treated", "first")).reset_index()
    p = df.groupby(["umd_full_cd", "em"]).size().rename("n").reset_index()
    full = pd.MultiIndex.from_product(
        [meta.umd_full_cd, range(mmin, mmax + 1)],
        names=["umd_full_cd", "em"]).to_frame(index=False)
    p = (full.merge(p, on=["umd_full_cd", "em"], how="left")
         .fillna({"n": 0}).merge(meta, on="umd_full_cd"))
    p["kq"] = np.floor(p.em / 3).astype(int)
    p["em_s"] = p.em.astype(str)
    p["y"] = np.log1p(p.n)
    return p


def run(p: pd.DataFrame, kmin: int, kmax: int, ref: int = -1) -> pd.DataFrame:
    return econ.event_study(p, y="y", treated="treated", kq="kq",
                            fes=["umd_full_cd", "em_s"], controls=[],
                            cluster="umd_full_cd", kmin=kmin, kmax=kmax, ref=ref)


def pct(c: float) -> float:
    return 100 * (np.expm1(c))


def main() -> None:
    df = io.load_trades()
    df = df[df.sgg_nm.isin(SGG)].copy()
    df["treated"] = policy.assign_2020(df)
    df["em"] = policy.event_month(df.deal_date, EVENT)
    p = panel(df, KMIN * 3, KMAX * 3 + 2)
    out: dict = {"사건": policy.EVENTS[EVENT]["label"], "기준사양": {}}

    print("=" * 78)
    print("진단 — 사전 기간의 처치·통제 격차가 기준월 대비 어떻게 움직이나")
    print("=" * 78)
    g = p.groupby(["em", "treated"]).y.mean().unstack()
    gap = (g[True] - g[False])
    pre = gap.loc[-24:-2]
    print(f"  k=-1 (기준월)            {gap.loc[-1]:+.3f}")
    print(f"  사전 24개월 평균          {pre.mean():+.3f}   "
          f"(표준편차 {pre.std():.3f})")
    print(f"  차이                    {gap.loc[-1]-pre.mean():+.3f}"
          "   ← 기준월이 이만큼 높다")
    lo = gap.loc[-24:-7].mean()      # 12·16 이전
    hi = gap.loc[-6:-2].mean()       # 12·16 이후 ~ 지정 직전
    print(f"\n  12·16 이전 평균 (k -24~-7)  {lo:+.3f}")
    print(f"  12·16 이후 평균 (k  -6~-2)  {hi:+.3f}   차이 {hi-lo:+.3f}")
    out["진단"] = {"기준월": round(float(gap.loc[-1]), 4),
                   "사전평균": round(float(pre.mean()), 4),
                   "대책이전": round(float(lo), 4), "대책이후": round(float(hi), 4)}

    # --- 사양 A : 기준 분기를 바꿔 본다 ------------------------------------
    print("\n" + "=" * 78)
    print("A. 기준 분기를 바꾸면 k=0 이 얼마나 움직이나")
    print("=" * 78)
    print(f"  {'기준':>6}{'k=0':>10}{'k=1':>10}{'k=4':>10}   사전 유의 개수")
    rows = []
    for ref in (-1, -2, -3, -4, -5, -6):
        r = run(p, KMIN, KMAX, ref=ref).set_index("k")
        npre = int((r.loc[KMIN:-1].p < 0.05).sum())
        rows.append({"기준": ref, "k0": pct(r.loc[0, "coef"]),
                     "k1": pct(r.loc[1, "coef"]), "k4": pct(r.loc[4, "coef"]),
                     "사전유의": npre})
        print(f"  {ref:>6}{rows[-1]['k0']:>9.1f}%{rows[-1]['k1']:>9.1f}%"
              f"{rows[-1]['k4']:>9.1f}%{npre:>12}")
    out["A_기준분기"] = [{k: (round(v, 2) if isinstance(v, float) else v)
                          for k, v in x.items()} for x in rows]

    # --- 사양 B : 12·16 대책 이후만 사전으로 --------------------------------
    print("\n" + "=" * 78)
    print("B. 사전 구간을 자르면")
    print("=" * 78)
    cut_em = int((LOAN.year - 2020) * 12 + LOAN.month - 6)      # 대략의 사건월
    specs = {
        "전체 (k -8~8)": (KMIN, KMAX),
        "12·16 이후만 (k -2~8)": (-2, KMAX),
        "12·16 이전만 사전 (k -8~-3 + 0~8)": None,
    }
    b = {}
    for nm, kk in specs.items():
        if kk is None:
            q = p[(p.kq <= -3) | (p.kq >= 0)]
            r = run(q, KMIN, KMAX, ref=-3).set_index("k")
        else:
            r = run(p, kk[0], kk[1], ref=kk[0]).set_index("k")
        v = pct(r.loc[0, "coef"])
        b[nm] = round(float(v), 2)
        print(f"  {nm:<36}k=0  {v:>7.1f}%   (p={r.loc[0,'p']:.3f})")
    out["B_구간절단"] = b

    # --- 사양 C : 사전 기울기를 빼 준다 ------------------------------------
    print("\n" + "=" * 78)
    print("C. 사전 기울기를 사후로 늘려 뺀다")
    print("=" * 78)
    r = run(p, KMIN, KMAX, ref=-1).set_index("k")
    pre_k = np.arange(KMIN, -1)
    pre_c = r.loc[KMIN:-2, "coef"].values
    slope, intercept = np.polyfit(pre_k, pre_c, 1)
    print(f"  사전 기울기 {slope:+.4f} / 분기  (사전 계수를 직선으로 맞춘 값)")
    adj = {}
    for k in (0, 1, 2, 4, 8):
        raw = r.loc[k, "coef"]
        fit = intercept + slope * k
        adj[k] = {"원": round(float(pct(raw)), 2),
                  "보정": round(float(pct(raw - fit)), 2)}
        print(f"  k={k:>2}  원 {pct(raw):>7.1f}%   기울기 제거 {pct(raw-fit):>7.1f}%")
    out["C_기울기보정"] = {"기울기": round(float(slope), 5), "값": adj}

    # --- 그림 --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for ref, c in ((-1, "#c0392b"), (-4, "#2980b9")):
        rr = run(p, KMIN, KMAX, ref=ref).set_index("k")
        ax.plot(rr.index, [pct(x) for x in rr.coef], "o-", color=c, ms=4,
                label=f"기준 k={ref}")
        ax.fill_between(rr.index, [pct(x) for x in rr.ci_lo],
                        [pct(x) for x in rr.ci_hi], color=c, alpha=0.10)
    ax.axhline(0, color="#888", lw=0.8)
    ax.axvline(-0.5, color="#333", lw=1.0, ls="--")
    ax.text(-0.4, ax.get_ylim()[1] * 0.94, " 2020-06-23 지정", fontsize=9)
    ax.set_xlabel("사건 분기 k (효력일 기준)")
    ax.set_ylabel("거래량 변화 (%)")
    ax.set_title("사전 추세와 기준 분기 — 기준을 어디로 두느냐가 사후 크기를 바꾼다",
                 fontsize=12, loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fp = io.FIGURES / "06d_pretrend.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)

    io.save_result("06d_pretrend", out)
    print(f"\n저장 {fp.relative_to(io.ROOT)} · output/results/06d_pretrend.json")


if __name__ == "__main__":
    main()
