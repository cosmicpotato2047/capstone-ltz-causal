"""[06] 클러스터가 적을 때의 추론 — 와일드 클러스터 부트스트랩과 무작위화 추론.

주 사양의 표준오차는 법정동 클러스터다. 강남·송파 법정동이 27개인데 그중
처치가 4개뿐이다. 클러스터 강건 표준오차는 클러스터 수가 많다는 가정에 기대고,
특히 **처치 클러스터가 적으면** 표준오차를 작게 잡아 p 값이 과하게 작아진다
(MacKinnon & Webb 2017). 관행 p 가 10^-8 이 나오는 것은 그 증상일 수 있다.

두 방법으로 다시 잰다.

  1. 와일드 클러스터 부트스트랩 (WCR, 귀무가설 부과, Webb 6점 가중)
     법정동마다 잔차에 같은 무작위 가중을 곱해 t 통계량의 분포를 만든다.
  2. 무작위화 추론 (Fisher)
     27개 동 중 4개를 처치로 고르는 모든 경우(C(27,4) = 17,550)를 돌려
     진짜 배정의 추정치가 몇 번째인지 센다. 분포 가정이 없다.

검정하는 숫자는 셋이다.
  - 사건연구 k=0          (04_event_study 와 같은 사양)
  - 24개월 이중차분        (13_placebo 의 '실제' 와 같은 사양)
  - 삼중차분 처치×사후×고가 (06c_loan 과 같은 사양)

패널은 칸마다 사건월을 0 으로 채운 균형 격자라 두 방향 고정효과 제거가 닫힌
식(행 평균·열 평균을 빼고 전체 평균을 더함)으로 정확히 된다. 삼중차분은 시간
고정효과가 가격군별이므로 가격군마다 따로 뺀다. 그래서 수만 번 돌려도 빠르다.
첫머리에서 세 사양 모두 기존 추정과 일치하는지 확인한다.

    python scripts/06_inference.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import econ, io, policy  # noqa: E402

io.setup_stdout()
for f in ("Malgun Gothic", "Batang"):
    if f in {x.name for x in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

from lib.inference import (B, EVENT, KMAX, KMIN, SEED, SGG,  # noqa: E402
                           fit, onehot, randomization, se_j, spec_ddd,
                           spec_did, spec_es, wild_boot)

# --- 본체 ---------------------------------------------------------------------
def main() -> None:
    df = io.load_trades()
    df = df[df.sgg_nm.isin(SGG)].copy()
    df["treated"] = policy.assign_2020(df)
    df["em"] = policy.event_month(df.deal_date, EVENT)
    rng = np.random.default_rng(SEED)

    specs = [spec_es(df), spec_did(df)]
    s_ddd, gg = spec_ddd(df)
    specs.append(s_ddd)

    # ------------------------------------------------------------------
    print("=" * 76)
    print("0. 닫힌 식이 기존 추정과 같은가")
    print("=" * 76)
    ref = {}
    # 사건연구 — econ.event_study
    s = specs[0]
    T = s.y.size // s.G
    pan = pd.DataFrame({"g": s.cl, "em": np.tile(np.arange(KMIN * 3, KMAX * 3 + 3), s.G),
                        "y": s.y, "treated": s.treated[s.cl]})
    pan["kq"] = np.floor(pan.em / 3).astype(int)
    pan["em_s"] = pan.em.astype(str)
    r = econ.event_study(pan, y="y", treated="treated", kq="kq", fes=["g", "em_s"],
                         controls=[], cluster="g", kmin=KMIN, kmax=KMAX).set_index("k")
    ref[s.name] = (r.loc[0, "coef"], r.loc[0, "se"])
    # 이중차분·삼중차분 — econ.demean + statsmodels
    for s in specs[1:]:
        X = s.make(s.treated)
        cols = [f"x{i}" for i in range(X.shape[0])]
        d = pd.DataFrame(X.T, columns=cols).assign(y=s.y, cl=s.cl)
        # 칸 식별자와 시간 식별자를 행 순서에서 복원한다
        cell, time = [], []
        off = 0
        for gi, (a, b) in enumerate(s.groups):
            cell += list(np.repeat(np.arange(off, off + a), b))
            time += [f"{gi}_{t}" for t in np.tile(np.arange(b), a)]
            off += a
        d["cell"], d["time"] = cell, time
        dm = econ.demean(d[["y"] + cols], d[["cell", "time"]])
        f_ = sm.OLS(dm["y"], dm[cols]).fit(cov_type="cluster",
                                             cov_kwds={"groups": d.cl})
        ref[s.name] = (f_.params.iloc[s.j], f_.bse.iloc[s.j])

    for s in specs:
        C = onehot(s.cl, s.G)
        b, A, Xt, u = fit(s, s.make(s.treated))
        se = se_j(s, A, Xt, u, C)
        rb, rs = ref[s.name]
        print(f"  {s.name:<16} 계수 {b[s.j]:+.6f} / 기존 {rb:+.6f}   "
              f"표준오차 {se:.6f} / 기존 {rs:.6f}")
        assert abs(b[s.j] - rb) < 1e-6 and abs(se - rs) < 1e-4, f"{s.name} 어긋남"
    print(f"  → 셋 다 일치. 법정동 {specs[0].G}개 (처치 {int(specs[0].treated.sum())})")

    # ------------------------------------------------------------------
    print("\n" + "=" * 76)
    print(f"1. 와일드 클러스터 부트스트랩 (WCR, Webb 6점, B={B:,})")
    print("=" * 76)
    print(f"  {'사양':<16}{'계수':>9}{'효과':>9}{'관행 p':>10}{'부트 p(양측)':>14}{'부트 p(감소)':>14}")
    wb, boots = {}, {}
    for s in specs:
        coef, t0, p1, p2, ts = wild_boot(s, rng)
        p_conv = float(2 * stats.t.sf(abs(t0), s.G - 1))
        boots[s.name] = (ts, t0)
        wb[s.name] = {"coef": round(coef, 4),
                      "효과_pct": round(float(100 * np.expm1(coef)), 1),
                      "t": round(t0, 3), "관행_p_t(G-1)": round(p_conv, 5),
                      "부트_p_양측": round(p2, 4), "부트_p_감소": round(p1, 4)}
        print(f"  {s.name:<16}{coef:>+9.3f}{100*np.expm1(coef):>8.1f}%{p_conv:>10.4f}"
              f"{p2:>14.4f}{p1:>14.4f}")
    print("\n  관행 p 는 자유도 G-1 의 t 분포로 다시 계산했다(statsmodels 기본은 정규).")
    print("  삼중차분의 '효과' 칸은 고가 단지가 저가보다 더 줄어든 폭이다(차이 계수).")

    # ------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("2. 무작위화 추론 — 27개 동 중 4개를 고르는 모든 경우")
    print("=" * 76)
    ri, dist = {}, {}
    for s in specs:
        vals, act, N_all = randomization(s)
        N = vals.size
        rank = int((vals <= act).sum())
        p1, p2 = rank / N, float((np.abs(vals) >= abs(act)).mean())
        # 가짜 분포가 비대칭이면 절댓값 양측은 긴 꼬리를 과하게 센다.
        # 등꼬리 p = 2 × min(왼쪽, 오른쪽) 도 함께 둔다.
        p_eq = min(1.0, 2 * min(p1, float((vals >= act).mean())))
        dist[s.name] = (vals, act)
        if N < N_all:
            print(f"  ({s.name}: {N_all - N:,}가지는 식별 불가라 뺐다)")
        ri[s.name] = {"경우의수": N, "식별불가_제외": N_all - N,
                      "진짜_계수": round(float(act), 4),
                      "순위_작은쪽부터": rank,
                      "p_감소": round(p1, 5), "p_양측_절댓값": round(p2, 5),
                      "p_양측_등꼬리": round(p_eq, 5)}
        print(f"  {s.name:<16}진짜 {act:+.3f}  순위 {rank:,} / {N:,}  "
              f"p(감소) {p1:.5f}  p(절댓값) {p2:.4f}  p(등꼬리) {p_eq:.4f}")
    print("\n  가정: 귀무가설 아래에서 어느 4개 동이 처치를 받았어도 같았다(교환가능성).")
    print("  실제 지정은 무작위가 아니라 국제교류복합지구 인근이라는 이유가 있었다.")

    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    for col, s in enumerate(specs):
        ts, t0 = boots[s.name]
        ax = axes[0, col]
        ax.hist(ts, bins=80, color="#9aa7b4")
        ax.axvline(t0, color="#c0392b", lw=2)
        ax.set_title(f"{s.name}\n부트스트랩 t 분포 (귀무가설 아래)", fontsize=11)
        ax.text(0.02, 0.95, f"진짜 t = {t0:.2f}\n양측 p = {wb[s.name]['부트_p_양측']:.4f}",
                transform=ax.transAxes, va="top")
        vals, act = dist[s.name]
        ax = axes[1, col]
        ax.hist(vals, bins=80, color="#9aa7b4")
        ax.axvline(act, color="#c0392b", lw=2)
        ax.set_title(f"가짜 배정 {len(vals):,}가지의 계수", fontsize=11)
        ax.set_xlabel("계수 (로그)")
        ax.text(0.02, 0.95, f"진짜 {act:+.3f}\n순위 {ri[s.name]['순위_작은쪽부터']:,}"
                f" / {len(vals):,}", transform=ax.transAxes, va="top")
    fig.suptitle("법정동 27개(처치 4개)에서의 추론 — 빨간 선이 진짜", y=1.0)
    fig.tight_layout()
    io.FIGURES.mkdir(parents=True, exist_ok=True)
    fp = io.FIGURES / "06_inference.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    io.save_result("06_inference", {"클러스터": specs[0].G,
                                    "처치클러스터": int(specs[0].treated.sum()),
                                    "B": B, "seed": SEED, "가중": "Webb 6점",
                                    "와일드부트스트랩": wb, "무작위화추론": ri})
    print(f"\n저장 {fp} · output/results/06_inference.json")



if __name__ == "__main__":
    main()
