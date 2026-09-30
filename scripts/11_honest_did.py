"""[11] Honest DiD — 사전 추세가 사후에도 이어졌다면 효과는 어디까지 남는가.

    python scripts/11_honest_did.py

입력  output/results/04_event_study.json  (사건연구 계수와 표준오차)
출력  output/results/11_honest_did.json, output/figures/11_honest_did.png

**왜 하는가.** 이중차분은 처치가 없었다면 두 집단이 평행하게 움직였을 것이라는
가정에 기댄다. 규제지역 지정은 가격이 급등한 곳을 골라 하므로 이 가정이 위배될
소지가 크고, 실제로 가격 사건연구의 사전 계수가 0 이 아니다. 지금까지 이 문제를
**진단**하고([[0014]]) 기준 분기를 옮겨 보는 것으로 다뤘는데, 그것은 몇 개의
대안 사양을 보는 것이지 위배의 크기를 모수로 놓고 푸는 것이 아니다.

Rambachan & Roth(2023)는 평행추세를 '성립한다/안 한다'로 묻지 않는다. **사후에
남아 있을 수 있는 추세 위배의 크기를 모수 하나로 제한하고, 그 모수를 키워 가며
효과의 구간이 언제 0 을 품는지**를 본다. 그 지점을 붕괴값(breakdown)이라 한다.

## 두 가지 제한

델타(δ)를 '처치가 없었어도 있었을 집단 간 격차'라고 하자. 기준 분기에서 δ = 0 이고,
사전 구간에서는 처치가 없으므로 관측된 계수가 곧 δ 다.

**상대크기 Δ^RM(M̄)** — 사후의 분기간 변화폭이 사전의 최대 변화폭의 M̄ 배를
넘지 않는다.

    |δ_t − δ_(t−1)| ≤ M̄ · D,   D = max over 사전 |δ_s − δ_(s−1)|

그러면 사후 l 분기의 편의는 많아야 (l+1)·M̄·D 다. 이것이 그 제한 아래에서
**정확한 상한**이다(각 증분을 모두 한 방향으로 몰면 달성된다).

**평활성 Δ^SD(M)** — 추세가 직선에서 벗어나는 정도를 제한한다. 사전 계수에
직선을 맞춰 사후로 연장한 것을 δ̂ 로 두고, 이차차분이 M 을 넘지 않으면
벗어남은 많아야 M·(l+1)(l+2)/2 다.

## 무엇을 재나

주장하는 숫자 셋에 각각 건다.

    거래량 직후 3개월   k=0        (−79.8%, 0019)
    거래량 24개월 평균  k=0..7     (−25 ~ −38%, 0016·0020)
    가격 12개월 평균    k=0..3     (−4 ~ −7%, 0025)

표본 오차는 편의 상한과 **더해서** 구간을 낸다. 편의 상한을 아는 값처럼 다루므로
보수적이다(D 자체의 추정 오차를 세지 않는다).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import io  # noqa: E402

io.setup_stdout()
for f in ("Malgun Gothic", "Batang"):
    if f in {x.name for x in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

Z = 1.96
MBAR = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
REF = -1


def pre_max_step(t: pd.DataFrame) -> tuple[float, str]:
    """사전 구간의 최대 분기간 변화폭 D 와 그것이 나온 자리."""
    p = t[t.k <= REF].sort_values("k")
    d = p.coef.diff().abs()
    i = d.idxmax()
    return float(d.loc[i]), f"k={int(p.k.loc[i]) - 1}→{int(p.k.loc[i])}"


def linear_pre(t: pd.DataFrame) -> tuple[float, float]:
    """사전 계수에 1/se² 가중 직선을 맞춘다. 기준 분기는 se=0 이라 뺀다."""
    p = t[(t.k <= REF) & (t.se > 0)]
    w = 1.0 / p.se.to_numpy() ** 2
    x, y = p.k.to_numpy(float), p.coef.to_numpy(float)
    W = w.sum()
    xb, yb = (w * x).sum() / W, (w * y).sum() / W
    slope = (w * (x - xb) * (y - yb)).sum() / (w * (x - xb) ** 2).sum()
    return float(slope), float(yb - slope * xb)


def target(t: pd.DataFrame, ks: list[int]) -> tuple[float, float, np.ndarray]:
    """대상 계수(평균), 보수적 표준오차, 사후 시차 배열."""
    s = t[t.k.isin(ks)].sort_values("k")
    beta = float(s.coef.mean())
    se = float(s.se.mean())          # 완전상관을 가정한 상한
    lags = s.k.to_numpy() - REF - 1  # k=0 이 l=0
    return beta, se, lags


def rm_bound(lags: np.ndarray, mbar: float, D: float) -> float:
    """Δ^RM(M̄) 아래 대상(평균)의 편의 상한."""
    return float(np.mean((lags + 1) * mbar * D))


def sd_bound(lags: np.ndarray, m: float) -> float:
    """Δ^SD(M) 아래 직선 연장에서 벗어나는 폭의 상한."""
    return float(np.mean((lags + 1) * (lags + 2) / 2 * m))


def pct(x: float) -> float:
    return 100 * np.expm1(x)


def breakdown(beta: float, se: float, unit: float, use_se: bool) -> float:
    """구간이 0 을 품기 시작하는 모수 값. unit 은 모수 1 단위당 편의."""
    room = abs(beta) - (Z * se if use_se else 0.0)
    return float(room / unit) if room > 0 and unit > 0 else 0.0


def run(t: pd.DataFrame, ks: list[int], label: str, claim: str) -> dict:
    D, where = pre_max_step(t)
    slope, icpt = linear_pre(t)
    beta, se, lags = target(t, ks)
    drift = float(np.mean([slope * (k - REF) for k in ks]))   # 직선 연장 편의

    print(f"\n{'=' * 88}")
    print(f"{label}  (주장: {claim})")
    print("=" * 88)
    print(f"  대상 계수 {pct(beta):+.1f}%  (분기 {ks[0]}~{ks[-1]} 평균, "
          f"보수적 표준오차 {se:.4f})")
    print(f"  사전 최대 분기간 변화폭 D = {D:.4f} ({where}, {pct(D):+.1f}%p 규모)")
    print(f"  사전 직선 기울기 {slope:+.4f}/분기 → 사후 평균 연장 {pct(drift):+.1f}%")

    print(f"\n  Δ^RM(M̄) — 사후 변화폭이 사전 최대의 M̄ 배까지")
    print(f"  {'M̄':>6}{'편의 상한':>12}{'구간 (점추정)':>22}{'구간 (표본오차 포함)':>26}")
    rows = []
    for m in MBAR:
        b = rm_bound(lags, m, D)
        lo_p, hi_p = pct(beta - b), pct(beta + b)
        lo_c, hi_c = pct(beta - b - Z * se), pct(beta + b + Z * se)
        rows.append({"Mbar": m, "편의상한": round(b, 4),
                     "점추정_lo": round(lo_p, 1), "점추정_hi": round(hi_p, 1),
                     "구간_lo": round(lo_c, 1), "구간_hi": round(hi_c, 1),
                     "0포함": bool(lo_c <= 0 <= hi_c)})
        mark = "  <- 0 을 품는다" if lo_c <= 0 <= hi_c else ""
        print(f"  {m:>6.2f}{b:>12.4f}   [{lo_p:>7.1f}%, {hi_p:>7.1f}%]"
              f"   [{lo_c:>7.1f}%, {hi_c:>7.1f}%]{mark}")

    unit = rm_bound(lags, 1.0, D)
    bd_p = breakdown(beta, se, unit, use_se=False)
    bd_c = breakdown(beta, se, unit, use_se=True)
    print(f"\n  붕괴값  점추정 M̄ = {bd_p:.2f} · 표본오차 포함 M̄ = {bd_c:.2f}")

    print(f"\n  Δ^SD(M) — 직선에서 벗어나는 폭을 분기마다 M 까지")
    print(f"  {'M':>6}{'분기마다':>10}{'편의 상한':>11}{'구간 (표본오차 포함)':>26}")
    sd_rows = []
    grid = [0.0, 0.002, 0.005, 0.01, 0.02, 0.03]
    for m in grid:
        b = sd_bound(lags, m)
        lo_c = pct(beta - drift - b - Z * se)
        hi_c = pct(beta - drift + b + Z * se)
        sd_rows.append({"M": m, "M_pctp": round(pct(m), 2), "편의상한": round(b, 4),
                        "구간_lo": round(lo_c, 1), "구간_hi": round(hi_c, 1),
                        "0포함": bool(lo_c <= 0 <= hi_c)})
        mark = "  <- 0 을 품는다" if lo_c <= 0 <= hi_c else ""
        print(f"  {m:>6.3f}{pct(m):>9.2f}%p{b:>11.4f}   "
              f"[{lo_c:>7.1f}%, {hi_c:>7.1f}%]{mark}")
    unit_sd = sd_bound(lags, 1.0)
    bd_sd = breakaway = breakdown(beta - drift, se, unit_sd, use_se=True)
    print(f"  붕괴값  M = {bd_sd:.4f} = 분기마다 {pct(bd_sd):.2f}%p 씩 휨"
          f"  (직선 연장 편의 {pct(drift):+.1f}% 를 뺀 뒤)")

    return {"대상": label, "주장": claim, "분기": ks,
            "계수": round(beta, 4), "효과_pct": round(pct(beta), 1),
            "표준오차_보수": round(se, 4),
            "D": round(D, 4), "D_위치": where,
            "사전기울기": round(slope, 4), "직선연장_pct": round(pct(drift), 1),
            "RM": rows, "붕괴_Mbar_점추정": round(bd_p, 2),
            "붕괴_Mbar_구간": round(bd_c, 2),
            "SD": sd_rows, "붕괴_M": round(bd_sd, 4),
            "붕괴_M_pctp": round(pct(bd_sd), 2)}


def main() -> None:
    src = json.loads((io.RESULTS / "04_event_study.json").read_text(encoding="utf-8"))
    price = pd.DataFrame(src["가격"])
    vol = pd.DataFrame(src["거래량"])

    print("=" * 88)
    print("Honest DiD — 사전 추세가 사후에도 이어졌다면 (Rambachan & Roth 2023)")
    print("=" * 88)
    print(f"  입력: 04_event_study.json · 표본 {src['표본']:,}건 · "
          f"클러스터 {src['클러스터']}개")
    print("  주의: 편의 상한을 아는 값처럼 다루므로 구간이 보수적이다")

    out = {"입력": "04_event_study.json", "Z": Z, "기준분기": REF,
           "결과": [run(vol, [0], "거래량 — 지정 직후 3개월", "−79.8% (0019)"),
                    run(vol, list(range(0, 8)), "거래량 — 24개월 평균",
                        "−25 ~ −38% (0016·0020)"),
                    run(price, list(range(0, 4)), "가격 — 지정 후 12개월 평균",
                        "−4 ~ −7% (0025)")]}

    # 평균 하나로 묶으면 뒤 분기의 느슨한 상한이 앞 분기까지 끌어내린다.
    # 분기마다 따로 보면 효과가 어디까지 강건한지가 드러난다.
    print("\n" + "=" * 88)
    print("분기별 붕괴값 — 효과는 어느 분기까지 사전 추세를 견디는가")
    print("=" * 88)
    # D 를 사전 최대로 두면 거래량 쪽은 k=−2→−1 한 칸이 D 를 정한다. 그 칸은
    # 0014 가 '기준월 이상치'로 진단한 자리다. 두 번째로 큰 변화폭으로도 함께 낸다.
    def second_step(t: pd.DataFrame) -> float:
        p = t[t.k <= REF].sort_values("k")
        d = p.coef.diff().abs().dropna().sort_values(ascending=False)
        return float(d.iloc[1])

    print(f"  {'k':>3}{'거래량':>9}{'붕괴 M':>8}{'(2위 D)':>9}   |"
          f"{'가격':>8}{'붕괴 M':>8}{'(2위 D)':>9}")
    per = []
    for k in range(0, 8):
        row = {"k": k}
        cells = []
        for t, nm in ((vol, "거래량"), (price, "가격")):
            D, _ = pre_max_step(t)
            D2 = second_step(t)
            b, se, lags = target(t, [k])
            row[nm + "_pct"] = round(pct(b), 1)
            row[nm + "_붕괴M"] = round(breakdown(b, se, rm_bound(lags, 1.0, D), True), 2)
            row[nm + "_붕괴M_2위"] = round(breakdown(b, se, rm_bound(lags, 1.0, D2), True), 2)
            cells.append(f"{row[nm+'_pct']:>8.1f}%{row[nm+'_붕괴M']:>8.2f}"
                         f"{row[nm+'_붕괴M_2위']:>9.2f}")
        per.append(row)
        print(f"  {k:>3}{cells[0]}   |{cells[1]}")
    print("  붕괴 M 이 1 을 넘으면 '사전 최대만큼의 위배가 사후에 그대로 이어져도")
    print("  0 과 구분된다'는 뜻이다")
    print(f"  D 최대 = 거래량 {pre_max_step(vol)[0]:.4f} · 가격 {pre_max_step(price)[0]:.4f}")
    print(f"  D 2위  = 거래량 {second_step(vol):.4f} · 가격 {second_step(price):.4f}")
    out["분기별_붕괴"] = per
    out["D_2위"] = {"거래량": round(second_step(vol), 4),
                    "가격": round(second_step(price), 4)}

    # ==================================================================
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.8))
    ax = axes[0]
    for t, c, lab in ((vol, "#c53030", "거래량"), (price, "#3182ce", "가격")):
        y = 100 * np.expm1(t.coef)
        # 구간은 로그 계수에서 잡고 나서 백분율로 옮긴다. 비대칭이 된다
        lo = 100 * np.expm1(t.coef - Z * t.se)
        hi = 100 * np.expm1(t.coef + Z * t.se)
        ax.errorbar(t.k, y, yerr=[y - lo, hi - y], fmt="o-", ms=4,
                    lw=1.5, capsize=2, color=c, label=lab)
    ax.axhline(0, color="#999", lw=.8)
    ax.axvline(-0.5, color="#1a202c", ls="--", lw=1.2)
    ax.set_title("사건연구 — 사전 계수가 0 이 아니다")
    ax.set_xlabel("지정 기준 분기")
    ax.set_ylabel("효과 (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)

    for ax, res, c in ((axes[1], out["결과"][2], "#3182ce"),
                       (axes[2], out["결과"][0], "#c53030")):
        m = [r["Mbar"] for r in res["RM"]]
        lo = [r["구간_lo"] for r in res["RM"]]
        hi = [r["구간_hi"] for r in res["RM"]]
        ax.fill_between(m, lo, hi, color=c, alpha=.18)
        ax.plot(m, lo, color=c, lw=1.6)
        ax.plot(m, hi, color=c, lw=1.6)
        ax.axhline(0, color="#1a202c", lw=1.0)
        bd = res["붕괴_Mbar_구간"]
        if bd <= max(m):
            ax.axvline(bd, color="#dd6b20", ls="--", lw=1.3)
            ax.annotate(f"붕괴 M = {bd:.2f}", xy=(bd, ax.get_ylim()[1]),
                        xytext=(5, -14), textcoords="offset points",
                        fontsize=9, color="#dd6b20")
        ax.set_title(res["대상"] + f"\n({res['효과_pct']:+.1f}%)", fontsize=11)
        ax.set_xlabel("사후 위배가 사전 최대의 몇 배까지 (M)")
        ax.set_ylabel("강건 구간 (%)")
        ax.grid(alpha=.25)

    fig.suptitle("사전 추세를 감안하면 어디까지 남는가 — 상대크기 제한 아래의 강건 구간", y=1.03)
    fig.tight_layout()
    fp = io.FIGURES / "11_honest_did.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")

    io.save_result("11_honest_did", out)
    print(f"\n저장 {fp} · output/results/11_honest_did.json")


if __name__ == "__main__":
    main()
