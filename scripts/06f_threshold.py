"""[06f] 노출도 문턱 민감도 — 삼중차분의 고가/저가 경계를 옮겨 본다.

06c 의 삼중차분은 단지를 12·16 대책 직전 2년 중위 거래가로 15억 초과/이하로
갈랐다. 15억은 정책(15억 초과 주담대 금지)이 정한 문턱이라 임의로 고른 값은
아니다. 그래도 "15억을 골라서 그 결과가 나온 것 아니냐"는 물음에는 그림으로
답하는 편이 낫다.

경계를 9·12·15·18·20억으로 옮겨가며 같은 삼중차분을 돌리고 계수를 잇는다.

  - 어느 문턱에서나 고가 쪽이 더 크게 줄면: 문턱 선택의 문제가 아니다.
  - 15억에서만 튀면: 15억이라는 정책 문턱에 뭔가 있거나, 우연이다.

9억은 처치 단지의 88% 가 위쪽이라 저가 처치군이 얇다(결정기록 0015).
그 끝에서 추정이 흔들리는 것은 표본 탓이다.

추론은 결정기록 0019 와 같다(WCR 부트스트랩, 무작위화 등꼬리). 기본 표본과
결정기록 0020 의 오염 6개 제외 표본을 나란히 돌린다.

    python scripts/06f_threshold.py
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
from lib import io, policy  # noqa: E402
from lib.inference import (EVENT, HIGH, KMAX, KMIN, RULE_ON, SEED, SGG,  # noqa: E402
                           fit, randomization, spec_ddd, wild_boot)

io.setup_stdout()
for f in ("Malgun Gothic", "Batang"):
    if f in {x.name for x in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

THRESHOLDS = (9.0, 12.0, 15.0, 18.0, 20.0)
B_06F = 4_999
CONTAM = ["압구정동", "거여동", "마천동", "송파동", "신천동", "일원동"]  # 0020


def raw_means(g: pd.DataFrame) -> dict:
    """법정동×사건월 칸의 평균 거래건수, 사전/사후. 06c 3절과 같은 집계."""
    s = g[g.em.between(KMIN * 3, KMAX * 3 + 2)]
    out = {}
    for hi in (False, True):
        q = (s[s.high == hi].groupby(["umd_full_cd", "em"]).size()
             .rename("n").reset_index())
        q["treated"] = q.umd_full_cd.map(s.groupby("umd_full_cd").treated.first())
        q["post"] = q.em >= 0
        m = q.groupby(["treated", "post"]).n.mean()
        lab = "고가" if hi else "저가"
        for tr in (True, False):
            a, b = m.get((tr, False), np.nan), m.get((tr, True), np.nan)
            out[f"{lab}_{'처치' if tr else '통제'}"] = {
                "사전": round(float(a), 1), "사후": round(float(b), 1),
                "변화_pct": round(float(100 * (b / a - 1)), 1)}
    return out


AREA_BANDS = [0, 40, 60, 85, 135, 1000]
AREA_LABS = ["40㎡ 미만", "40~60", "60~85", "85~135", "135 이상"]


def by_area(df: pd.DataFrame) -> pd.DataFrame:
    """전용면적대별 월평균 거래의 사전(24개월)·사후(27개월) 변화. 원자료, 고정효과 없음.

    2020-06 지정의 허가 대상은 주거지역 대지 18㎡ 초과였다(2022-06 부터 6㎡).
    대지지분이 작은 소형은 허가를 받지 않아도 됐다. 대지지분은 자료에 없으므로
    전용면적으로 대신 본다. 탐색용이다.
    """
    d = df[df.em.between(-24, 26)].copy()
    d["post"] = d.em >= 0
    d["면적"] = pd.cut(d.area_m2, AREA_BANDS, labels=AREA_LABS, right=False)
    m = d.groupby(["treated", "면적", "post"], observed=True).size().unstack("post")
    m.columns = ["사전", "사후"]
    m["사전"] /= 24
    m["사후"] /= 27
    m["변화"] = 100 * (m.사후 / m.사전 - 1)
    t, c = m.xs(True, level="treated"), m.xs(False, level="treated")
    return pd.DataFrame({"처치_사전월평균": t.사전.round(1),
                         "처치_변화_pct": t.변화.round(1),
                         "통제_변화_pct": c.변화.round(1),
                         "차이_pp": (t.변화 - c.변화).round(1)})


def main() -> None:
    df = io.load_trades()
    df = df[df.sgg_nm.isin(SGG)].copy()
    df["treated"] = policy.assign_2020(df)
    df["em"] = policy.event_month(df.deal_date, EVENT)

    samples = {"기본": df, "오염 6개 제외": df[~df.umdNm.isin(CONTAM)]}
    rows = []
    for sname, d in samples.items():
        print("=" * 84)
        print(f"{sname}")
        print("=" * 84)
        print(f"  {'문턱':>5}{'처치 단지 저/고':>16}{'저가 효과':>10}{'고가 효과':>10}"
              f"{'차이(로그)':>11}{'WCR':>8}{'무작위':>8}{'고가 사전비':>11}")
        rng = np.random.default_rng(SEED)
        for thr in THRESHOLDS:
            s, g = spec_ddd(d.copy(), high=thr)
            b, *_ = fit(s, s.make(s.treated))
            lo_eff, hi_eff = float(100 * np.expm1(b[0])), float(100 * np.expm1(b[0] + b[1]))
            _, _, _, p_wcr, _ = wild_boot(s, rng, B=B_06F)
            vals, act, N_all = randomization(s)
            N = vals.size
            left = float((vals <= act).mean())
            right = float((vals >= act).mean())
            p_ri = min(1.0, 2 * min(left, right))

            u = g[g.treated].groupby("aptSeq").high.first()
            n_lo, n_hi = int((~u).sum()), int(u.sum())
            rm = raw_means(g)
            ratio = rm["고가_처치"]["사전"] / rm["고가_통제"]["사전"]

            rows.append({"표본": sname, "문턱_억": thr,
                         "처치단지_저가": n_lo, "처치단지_고가": n_hi,
                         "저가효과_pct": round(lo_eff, 1), "고가효과_pct": round(hi_eff, 1),
                         "차이_계수": round(float(b[1]), 4),
                         "WCR_p": round(p_wcr, 4), "RI_p_등꼬리": round(p_ri, 4),
                         "RI_경우의수": N, "RI_식별불가": N_all - N,
                         "고가_사전_처치대통제": round(ratio, 2), "원자료": rm})
            mark = " ←정책" if thr == HIGH else ""
            print(f"  {thr:>4.0f}억{f'{n_lo} / {n_hi}':>16}{lo_eff:>9.1f}%{hi_eff:>9.1f}%"
                  f"{b[1]:>+11.3f}{p_wcr:>8.3f}{p_ri:>8.3f}{ratio:>10.1f}배{mark}")
        print()

    tab = pd.DataFrame(rows)
    print("  '고가 사전비' = 고가 처치 칸의 사전 평균 거래 / 고가 통제 칸의 사전 평균 거래.")
    print("  이 비가 클수록 평균 회귀로 설명될 여지가 크다(결정기록 0015 한계, 0019).")

    print("\n" + "=" * 84)
    print("탐색 — 전용면적대별 원자료 (허가 대상이 대지 18㎡ 초과였다)")
    print("=" * 84)
    area = by_area(df)
    print(area.to_string())
    pre = df[df.deal_date.between(RULE_ON - pd.DateOffset(years=2), RULE_ON)]
    med = pre.groupby("aptSeq").amount_manwon.median() / 10000
    tt = df[df.treated].assign(base=df.aptSeq.map(med)).dropna(subset=["base"])
    size_lo = float(tt[tt.base <= 9].area_m2.median())
    size_hi = float(tt[tt.base > 9].area_m2.median())
    print(f"\n  처치동 9억 이하 단지 거래의 전용면적 중위 {size_lo:.1f}㎡ / 9억 초과 {size_hi:.1f}㎡")
    print("  → 가격 경계를 낮출수록 '저가'가 '소형'과 겹친다.")

    # --- 그림 -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    ax = axes[0]
    base = tab[tab.표본 == "기본"]
    ax.plot(base.문턱_억, base.고가효과_pct, "o-", color="#c53030", lw=1.8, label="고가 단지")
    ax.plot(base.문턱_억, base.저가효과_pct, "s-", color="#2b6cb0", lw=1.8, label="저가 단지")
    for _, r in base.iterrows():
        ax.annotate(f"{int(r.처치단지_저가)}/{int(r.처치단지_고가)}",
                    (r.문턱_억, r.저가효과_pct), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8, color="#555")
    ax.axhline(0, color="#999", lw=.8)
    ax.axvline(HIGH, color="#333", ls="--", lw=1)
    ax.set_xticks(THRESHOLDS)
    ax.set_xticklabels([f"{t:.0f}억" for t in THRESHOLDS])
    ax.set_xlabel("고가/저가 경계 (12·16 직전 2년 단지 중위가)")
    ax.set_ylabel("사후 27개월 평균 효과 (%)")
    ax.set_title("경계별 고가·저가 단지의 효과 (기본 표본)\n숫자는 처치 단지 수 저가/고가")
    ax.legend()
    ax.grid(alpha=.25)

    ax = axes[1]
    for sname, mk, col in (("기본", "o-", "#333"), ("오염 6개 제외", "s--", "#c53030")):
        t = tab[tab.표본 == sname]
        ax.plot(t.문턱_억, t.차이_계수, mk, color=col, lw=1.6, label=sname)
        for _, r in t.iterrows():
            ax.annotate(f"{r.WCR_p:.2f}", (r.문턱_억, r.차이_계수), textcoords="offset points",
                        xytext=(6, 4 if sname == "기본" else -12), fontsize=8, color=col)
    ax.axhline(0, color="#999", lw=.8)
    ax.axvline(HIGH, color="#333", ls="--", lw=1)
    ax.set_xticks(THRESHOLDS)
    ax.set_xticklabels([f"{t:.0f}억" for t in THRESHOLDS])
    ax.set_xlabel("고가/저가 경계")
    ax.set_ylabel("삼중차분 계수 (고가 - 저가, 로그)")
    ax.set_title("경계별 삼중차분 계수 — 숫자는 WCR p")
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    fp = io.FIGURES / "06f_threshold.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")

    io.save_result("06f_threshold", {"문턱_억": list(THRESHOLDS), "정책문턱": HIGH,
                                     "B": B_06F, "seed": SEED, "결과": rows,
                                     "면적대_원자료": area.reset_index().to_dict("records"),
                                     "처치동_전용면적중위_9억이하": round(size_lo, 1),
                                     "처치동_전용면적중위_9억초과": round(size_hi, 1)})
    print(f"\n저장 {fp} · output/results/06f_threshold.json")


if __name__ == "__main__":
    main()
