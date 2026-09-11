"""[17] 합성대조군 — 사전 궤적을 맞춘 통제군을 자료가 직접 만든다.

    python scripts/17_scm.py
    python scripts/17_scm.py --pool seoul   # 기증자 풀을 서울 25개구로

입력  data/processed/trades.parquet, trades_seoul.parquet
출력  output/results/17_scm.json
      output/figures/17_scm.png

**왜 하는가.** 위약 검정 C([[0016-위약검정]])에서 사전 거래량을 손으로 맞출수록
효과가 −33.5% 에서 −24.8% 로 줄었다. 평균 회귀가 섞였을 수 있다는 뜻이다.

문턱을 손으로 정하는 방식은 서툴다. 문턱이 임의고, 표본을 버려 검정력도
잃는다. 합성대조군은 **사전 궤적이 처치군과 가장 비슷해지도록 통제군의
가중평균을 자료가 직접 찾는다.** 문턱도 없고 표본도 안 버린다.

    처치 = 청담·삼성·대치·잠실을 합친 하나의 단위
    기증자 = 지정된 적 없는 법정동들
    결과 = log1p(월 거래건수) − 그 단위의 사전 평균  (수준이 아니라 **변화**)
    가중치 = 사전 기간 궤적의 제곱오차를 최소화

수준을 빼는 이유는 처치 단위가 월 210건이고 기증자 동은 중앙값 7건이기
때문이다. 가중치가 음수가 아니고 합이 1이라 볼록결합으로는 그 수준에
닿을 수 없다. 사전 평균을 빼면 모두 0 에서 출발해 모양만 맞추게 된다.

가중치는 음수가 아니고 합이 1이다. 그래서 외삽하지 않는다 — 합성 통제군은
기증자들이 실제로 보인 범위 안에서만 만들어진다.

**추론은 배치 위약(placebo-in-space)으로 한다.** 기증자 하나하나를 처치군인
척 세워 같은 절차를 돌리고, 진짜의 사후/사전 오차비가 가짜들 사이에서
어디쯤인지 본다 (Abadie, Diamond & Hainmueller 2010).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                    # noqa: E402
import pandas as pd                   # noqa: E402
from scipy.optimize import nnls       # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import io, policy  # noqa: E402

io.setup_stdout()
for f in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
    if f in {x.name for x in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

EVENT = "E2020"
PRE, POST = 36, 27        # 사건월 기준. 사후 끝은 식별 구간(2025-10-19) 안이다
MIN_PRE_TRADES = 3        # 기증자 자격: 사전 월평균 최소 거래


def simplex_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """합이 1이고 음수가 아닌 가중치. ||Xw - y|| 를 최소화한다.

    큰 수 M 을 붙인 행을 더해 합=1 제약을 벌점으로 넣는 흔한 방법이다.
    기증자가 수십 개뿐이라 이 정도면 충분하다."""
    M = 1e6
    A = np.vstack([X, M * np.ones((1, X.shape[1]))])
    b = np.concatenate([y, [M]])
    w, _ = nnls(A, b)
    s = w.sum()
    return w / s if s > 0 else w


def build_panel(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """법정동 × 사건월 로그 거래량 표. 행=법정동, 열=사건월."""
    df = df.copy()
    df["em"] = policy.event_month(df.deal_date, EVENT)
    df = df[df.em.between(-PRE, POST - 1)]
    cnt = df.groupby(["key", "em"]).size().rename("n").reset_index()
    full = pd.MultiIndex.from_product(
        [sorted(df.key.unique()), range(-PRE, POST)],
        names=["key", "em"]).to_frame(index=False)
    p = full.merge(cnt, on=["key", "em"], how="left").fillna({"n": 0})
    p["y"] = np.log1p(p.n)
    wide = p.pivot(index="key", columns="em", values="y")
    # **수준을 맞춘다.** 처치 단위는 4개 동을 합친 것이라 월 210건인데
    # 기증자 동은 중앙값 7건이다. 가중치가 음수가 아니고 합이 1이므로
    # 볼록결합으로는 그 수준에 닿을 수 없고, 사전 적합이 실패한다
    # (정규화 전 RMSPE 0.41, 경험적 p 0.57).
    # 단위별 사전 평균을 빼면 모두 0 에서 출발하므로 SCM 이 **궤적의 모양**을
    # 맞춘다. 이중차분의 단위 고정효과와 같은 조작이다.
    pre_cols = [c for c in wide.columns if c < 0]
    # 기증자 자격은 정규화 **전** 실제 거래량으로 본다
    pre_trades = np.expm1(wide[pre_cols]).mean(axis=1)
    base = wide[pre_cols].mean(axis=1)
    wide = wide.sub(base, axis=0)
    return wide, np.arange(-PRE, POST), pre_trades


def fit_one(y_pre, D_pre, y_all, D_all):
    """한 처치 단위에 대해 가중치와 격차를 낸다."""
    w = simplex_weights(D_pre.T, y_pre)
    synth = D_all.T @ w
    gap = y_all - synth
    rmspe_pre = float(np.sqrt(np.mean(gap[:PRE] ** 2)))
    rmspe_post = float(np.sqrt(np.mean(gap[PRE:] ** 2)))
    return w, synth, gap, rmspe_pre, rmspe_post


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", choices=("gangnam", "seoul"), default="seoul",
                    help="기증자 풀. gangnam=강남·송파 비처치동, seoul=서울 전역")
    a = ap.parse_args()

    # 지정된 적 있는 법정동은 기증자에서 뺀다 (동 전체 지정 기준, 결정기록 0012)
    ever = set()
    sp = io.POLICY / "ltz_spells.csv"
    if sp.exists():
        s = pd.read_csv(sp)
        s = s[(s.상태 == "지정") & (s.지정방식 == "동전체") &
              (s.시작일 < "2025-10-01")]
        ever = set(zip(s.자치구, s.법정동))

    if a.pool == "seoul":
        df = pd.read_parquet(io.TRADES_SEOUL)
    else:
        df = io.load_trades()
        df = df[df.sgg_nm.isin(["강남구", "송파구"])]
    df = df.copy()
    df["treated"] = (df.sgg_nm.isin(["강남구", "송파구"]) &
                     df.umdNm.isin(policy.TREATED_2020))
    df["key"] = np.where(df.treated, "처치(4개 동)",
                         df.sgg_nm.astype(str) + " " + df.umdNm.astype(str))

    wide, ems, pre_trades = build_panel(df)
    treat_key = "처치(4개 동)"
    if treat_key not in wide.index:
        sys.exit("처치 단위가 없습니다.")

    # 기증자 자격 — 지정 이력이 없고 사전 거래가 너무 적지 않아야
    pre_cols = [c for c in wide.columns if c < 0]
    donors = []
    for k in wide.index:
        if k == treat_key:
            continue
        gu, _, dong = k.partition(" ")
        if (gu, dong) in ever:
            continue
        if pre_trades[k] < MIN_PRE_TRADES:
            continue
        donors.append(k)
    if len(donors) < 5:
        sys.exit(f"기증자가 {len(donors)}개뿐입니다.")

    Y = wide.loc[treat_key].to_numpy(float)
    D = wide.loc[donors].to_numpy(float)
    w, synth, gap, r_pre, r_post = fit_one(Y[:PRE], D[:, :PRE], Y, D)

    print("=" * 78)
    print(f"합성대조군 — 기증자 풀 {a.pool} · {len(donors)}개 법정동")
    print("=" * 78)
    print(f"  사전 {PRE}개월 적합 오차(RMSPE) {r_pre:.4f}")
    print(f"  사후 {POST}개월 격차 오차(RMSPE) {r_post:.4f}")
    print(f"  비율 {r_post/max(r_pre,1e-9):.2f}배   ← 클수록 사후에만 벌어졌다\n")

    top = sorted(zip(donors, w), key=lambda x: -x[1])[:8]
    print("  가중치 상위")
    for k, v in top:
        if v > 1e-4:
            print(f"    {k:<16}{v:.3f}")

    # 사후 평균 격차를 % 로
    post_gap = float(np.mean(gap[PRE:]))
    p0 = float(np.mean(gap[PRE:PRE + 3]))       # 지정 직후 3개월
    print(f"\n  사후 {POST}개월 평균 격차  {post_gap:+.3f}  → {100*np.expm1(post_gap):+.1f}%")
    print(f"  지정 직후 3개월 격차     {p0:+.3f}  → {100*np.expm1(p0):+.1f}%")

    # --- 배치 위약 ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("배치 위약 — 기증자를 하나씩 처치군인 척 세우면")
    print("=" * 78)
    ratios = []
    for i, k in enumerate(donors):
        others = [j for j in range(len(donors)) if j != i]
        Yi, Di = D[i], D[others]
        _, _, g, rp, rq = fit_one(Yi[:PRE], Di[:, :PRE], Yi, Di)
        if rp < 1e-6:
            continue
        ratios.append({"동": k, "사전": rp, "비율": rq / rp,
                       "사후평균": float(np.mean(g[PRE:])),
                       "직후": float(np.mean(g[PRE:PRE + 3]))})
    real_ratio = r_post / max(r_pre, 1e-9)

    # 사전 적합이 나쁜 가짜는 추론에서 뺀다 (Abadie, Diamond & Hainmueller 2010).
    # 애초에 합성이 안 되는 단위는 사후 격차가 커도 그것이 처치 때문이 아니다.
    # 자의적 선택으로 보이지 않도록 문턱을 바꿔 가며 함께 보고한다.
    print(f"  {'사전적합 문턱':<22}{'가짜 수':>7}{'진짜 순위':>9}{'경험적 p':>10}")
    print("  " + "-" * 50)
    infer = {}
    for mult in (None, 5, 2):
        if mult is None:
            keep, lab = ratios, "전부"
        else:
            keep = [x for x in ratios if x["사전"] <= mult * r_pre]
            lab = f"진짜의 {mult}배 이내"
        if not keep:
            continue
        rank = sum(1 for x in keep if x["비율"] >= real_ratio) + 1
        pv = rank / (len(keep) + 1)
        infer[lab] = {"가짜수": len(keep), "순위": rank, "p": round(pv, 4)}
        print(f"  {lab:<22}{len(keep):>7}{rank:>9}{pv:>10.3f}")

    ratios.sort(key=lambda x: -x["비율"])
    print(f"\n  오차비 상위 (사전 적합 무관)")
    print(f"  {'단위':<18}{'사전':>7}{'오차비':>8}{'사후평균':>10}")
    for x in ratios[:6]:
        flag = "" if x["사전"] <= 2 * r_pre else "   사전적합 나쁨"
        print(f"  {x['동']:<18}{x['사전']:>7.3f}{x['비율']:>8.2f}"
              f"{x['사후평균']:>+10.3f}{flag}")
    print(f"  {'-- 진짜 --':<18}{r_pre:>7.3f}{real_ratio:>8.2f}{post_gap:>+10.3f}")

    # 검정 통계량을 하나 더 본다. 우리는 효과가 지정 직후에 크고 시간이
    # 지나며 옅어진다는 것을 이미 알고 있고, k>=4 는 해석하지 않기로 했다
    # ([[0014-사전추세-위배-대응]]). 그렇다면 사후 27개월 평균보다
    # **직후 3개월 격차**가 이 설계에 맞는 통계량이다.
    print("\n  직후 3개월 격차로 다시 세우면")
    print(f"  {'사전적합 문턱':<22}{'가짜 수':>7}{'진짜 순위':>9}{'경험적 p':>10}")
    print("  " + "-" * 50)
    infer0 = {}
    for mult in (None, 2):
        keep = ratios if mult is None else [x for x in ratios if x["사전"] <= mult * r_pre]
        lab = "전부" if mult is None else f"진짜의 {mult}배 이내"
        if not keep:
            continue
        rank0 = sum(1 for x in keep if x["직후"] <= p0) + 1
        pv = rank0 / (len(keep) + 1)
        infer0[lab] = {"가짜수": len(keep), "순위": rank0, "p": round(pv, 4)}
        print(f"  {lab:<22}{len(keep):>7}{rank0:>9}{pv:>10.3f}")
    worst = sorted(ratios, key=lambda x: x["직후"])[:5]
    print(f"\n  직후 3개월이 가장 크게 벌어진 단위")
    print(f"  {'단위':<18}{'직후3개월':>10}")
    for x in worst:
        print(f"  {x['동']:<18}{100*np.expm1(x['직후']):>+9.1f}%")
    print(f"  {'-- 진짜 --':<18}{100*np.expm1(p0):>+9.1f}%")


    # --- 그림 --------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    ax.plot(ems, Y, color="#c0392b", lw=1.8, label="처치 4개 동")
    ax.plot(ems, synth, color="#2980b9", lw=1.8, ls="--", label="합성 대조군")
    ax.axvline(-0.5, color="#333", lw=1.0, ls=":")
    ax.set_xlabel("사건월 (효력일 기준)")
    ax.set_ylabel("log(1+월 거래건수)")
    ax.set_title("실제와 합성 대조군", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    for x, k in zip(ratios, [r["동"] for r in ratios]):
        pass
    for i, k in enumerate(donors):
        others = [j for j in range(len(donors)) if j != i]
        _, _, g, rp, _ = fit_one(D[i][:PRE], D[others][:, :PRE], D[i], D[others])
        if rp < 2 * r_pre:      # 사전 적합이 나쁜 기증자는 그리지 않는다
            ax.plot(ems, g, color="#c9c9c9", lw=0.8)
    ax.plot(ems, gap, color="#c0392b", lw=2.0, label="처치 4개 동")
    ax.axhline(0, color="#888", lw=0.8)
    ax.axvline(-0.5, color="#333", lw=1.0, ls=":")
    ax.set_xlabel("사건월")
    ax.set_ylabel("실제 - 합성")
    ax.set_title("배치 위약 (회색 = 기증자를 처치군인 척)", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)
    fig.suptitle("합성대조군 — 사전 궤적을 맞춘 뒤 사후에 벌어지는가",
                 fontsize=12.5, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fp = io.FIGURES / "17_scm.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)

    io.save_result("17_scm", {
        "기증자풀": a.pool, "기증자수": len(donors),
        "사전RMSPE": round(r_pre, 4), "사후RMSPE": round(r_post, 4),
        "오차비": round(real_ratio, 3),
        "사후평균격차_pct": round(100 * float(np.expm1(post_gap)), 2),
        "직후3개월_pct": round(100 * float(np.expm1(p0)), 2),
        "추론_사후평균": infer, "추론_직후3개월": infer0,
        "가중치": {k: round(float(v), 4) for k, v in top if v > 1e-4},
    })
    print(f"\n저장 {fp.relative_to(io.ROOT)} · output/results/17_scm.json")


if __name__ == "__main__":
    main()
