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

import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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

EVENT = "E2020"
SGG = ["강남구", "송파구"]
KMIN, KMAX, REF = -8, 8, -1       # 04_event_study 와 같다
PRE, POST = 24, 24                # 13_placebo 와 같다
RULE_ON = pd.Timestamp("2019-12-17")   # 06c_loan 과 같다
HIGH = 15.0
B = 9_999
SEED = 20260918

# Webb(2014) 6점 분포. 클러스터가 적을 때 Rademacher(±1) 보다 낫다.
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5),
                 np.sqrt(0.5), 1.0, np.sqrt(1.5)])


# --- 사양 하나 -----------------------------------------------------------------
@dataclass
class Spec:
    """y, 설계행렬, 고정효과 제거 함수, 클러스터를 한데 묶는다.

    y     (n,)          행 순서는 칸 우선, 사건월 다음
    make  처치 동 집합 -> 설계행렬 (K, n). 무작위화 추론에서 다시 부른다
    M     (..., n) -> (..., n) 고정효과 제거
    cl    (n,) 행마다 클러스터(법정동) 번호
    j     검정할 계수의 위치
    """
    name: str
    y: np.ndarray
    make: Callable[[np.ndarray], np.ndarray]
    M: Callable[[np.ndarray], np.ndarray]
    cl: np.ndarray
    G: int
    treated: np.ndarray       # 법정동별 처치 여부 (G,)
    j: int
    effect: Callable[[np.ndarray], float]   # β -> 보고할 로그 효과
    groups: list[tuple[int, int]]           # M 을 만든 (칸 수, 사건월 수) 묶음


def twoway(shape_groups: list[tuple[int, int]]) -> Callable:
    """(칸 수, 사건월 수) 묶음들이 이어 붙은 행에 대해 묶음마다 두 방향 제거."""
    bounds = np.cumsum([0] + [a * b for a, b in shape_groups])

    def M(v: np.ndarray) -> np.ndarray:
        out = np.empty_like(v, dtype=float)
        for (a, b), s, e in zip(shape_groups, bounds[:-1], bounds[1:]):
            w = v[..., s:e].reshape(*v.shape[:-1], a, b)
            w = (w - w.mean(-1, keepdims=True) - w.mean(-2, keepdims=True)
                 + w.mean((-2, -1), keepdims=True))
            out[..., s:e] = w.reshape(*v.shape[:-1], a * b)
        return out
    return M


# --- 추정 ---------------------------------------------------------------------
def fit(s: Spec, X: np.ndarray):
    Xt = s.M(X).T                                  # (n, K)
    yt = s.M(s.y)
    A = np.linalg.inv(Xt.T @ Xt)
    b = A @ (Xt.T @ yt)
    return b, A, Xt, yt - Xt @ b


def onehot(cl: np.ndarray, G: int) -> np.ndarray:
    C = np.zeros((cl.size, G))
    C[np.arange(cl.size), cl] = 1.0
    return C


def se_j(s: Spec, A, Xt, u, C) -> float:
    """계수 j 의 클러스터 강건 표준오차 (statsmodels 와 같은 소표본 보정)."""
    n, K = Xt.shape
    c = Xt @ A[s.j]
    sc = (c * u) @ C
    return float(np.sqrt(s.G / (s.G - 1) * (n - 1) / (n - K) * (sc ** 2).sum()))


def wild_boot(s: Spec, rng) -> tuple[float, float, float, float, np.ndarray]:
    """β_j = 0 을 부과한 WCR 부트스트랩."""
    X = s.make(s.treated)
    K, n = X.shape
    C = onehot(s.cl, s.G)
    b, A, Xt, u = fit(s, X)
    t0 = b[s.j] / se_j(s, A, Xt, u, C)

    yt = s.M(s.y)
    keep = [i for i in range(K) if i != s.j]
    if keep:
        Xr = Xt[:, keep]
        fit_r = Xr @ np.linalg.solve(Xr.T @ Xr, Xr.T @ yt)
    else:
        fit_r = np.zeros(n)
    er = yt - fit_r

    c = Xt @ A[s.j]
    fac = s.G / (s.G - 1) * (n - 1) / (n - K)
    ts = np.empty(B)
    CH = 500
    for s0 in range(0, B, CH):
        m = min(CH, B - s0)
        w = rng.choice(WEBB, size=(m, s.G))[:, s.cl]          # 동마다 같은 가중
        ystar = fit_r[None] + w * er[None]
        # X̃ 는 이미 고정효과 제거 공간에 있으므로 β* 는 y* 를 다시 빼지 않아도 같다
        bstar = ystar @ Xt @ A                                 # (m, K)
        ustar = s.M(ystar) - bstar @ Xt.T
        sc = (ustar * c[None]) @ C                             # (m, G)
        ts[s0:s0 + m] = bstar[:, s.j] / np.sqrt(fac * (sc ** 2).sum(1))
    p2 = float((np.abs(ts) >= abs(t0)).mean())
    p1 = float((ts <= t0).mean())                              # 감소 방향
    return float(b[s.j]), float(t0), p1, p2, ts


def randomization(s: Spec) -> tuple[np.ndarray, float, int]:
    """처치 동 수만큼을 고르는 모든 조합에 대해 보고할 효과를 낸다."""
    nt = int(s.treated.sum())
    combos = list(itertools.combinations(range(s.G), nt))
    real = tuple(np.flatnonzero(s.treated))
    yt = s.M(s.y)
    vals = np.empty(len(combos))
    for i, cmb in enumerate(combos):
        t_ = np.zeros(s.G, bool)
        t_[list(cmb)] = True
        Xt = s.M(s.make(t_)).T
        XX = Xt.T @ Xt
        # 가짜 동이 모두 한 가격군에만 단지를 가지면 삼중차분 계수가 식별되지
        # 않는다. 그런 배정은 비교에서 뺀다 (몇 개인지 보고한다).
        if np.linalg.matrix_rank(XX) < XX.shape[0]:
            vals[i] = np.nan
            continue
        vals[i] = s.effect(np.linalg.solve(XX, Xt.T @ yt))
    act = vals[combos.index(real)]
    return vals[~np.isnan(vals)], act, len(combos)


# --- 사양 만들기 --------------------------------------------------------------
def spec_es(df: pd.DataFrame) -> Spec:
    lo, hi = KMIN * 3, KMAX * 3 + 2
    codes = sorted(df.umd_full_cd.unique())
    em = np.arange(lo, hi + 1)
    T, G = em.size, len(codes)
    cnt = (df[df.em.between(lo, hi)].groupby(["umd_full_cd", "em"]).size()
           .unstack(fill_value=0).reindex(index=codes, columns=em, fill_value=0))
    tr = df.groupby("umd_full_cd").treated.first().reindex(codes).to_numpy(bool)
    kq = np.floor(em / 3).astype(int)
    ks = [k for k in range(KMIN, KMAX + 1) if k != REF]

    def make(t):
        return np.stack([np.outer(t, kq == k).reshape(-1) for k in ks]).astype(float)
    j = ks.index(0)
    return Spec("사건연구 k=0", np.log1p(cnt.to_numpy(float)).reshape(-1), make,
                twoway([(G, T)]), np.repeat(np.arange(G), T), G, tr, j,
                lambda b: b[j], [(G, T)])


def spec_did(df: pd.DataFrame) -> Spec:
    lo, hi = -PRE, POST - 1
    codes = sorted(df.umd_full_cd.unique())
    em = np.arange(lo, hi + 1)
    T, G = em.size, len(codes)
    cnt = (df[df.em.between(lo, hi)].groupby(["umd_full_cd", "em"]).size()
           .unstack(fill_value=0).reindex(index=codes, columns=em, fill_value=0))
    tr = df.groupby("umd_full_cd").treated.first().reindex(codes).to_numpy(bool)
    post = (em >= 0).astype(float)
    return Spec("24개월 이중차분", np.log1p(cnt.to_numpy(float)).reshape(-1),
                lambda t: np.outer(t, post).reshape(1, -1).astype(float),
                twoway([(G, T)]), np.repeat(np.arange(G), T), G, tr, 0,
                lambda b: b[0], [(G, T)])


def spec_ddd(df: pd.DataFrame) -> tuple[Spec, pd.DataFrame]:
    """06c_loan 3절과 같은 사양. 칸 = (법정동, 가격군)."""
    pre = df[df.deal_date.between(RULE_ON - pd.DateOffset(years=2), RULE_ON)]
    med = pre.groupby("aptSeq").amount_manwon.median() / 10000
    g = df.assign(base=df.aptSeq.map(med))
    g = g[g.base.notna()].copy()
    g["high"] = g.base > HIGH
    lo, hi = KMIN * 3, KMAX * 3 + 2
    em = np.arange(lo, hi + 1)
    T = em.size
    codes = sorted(g.umd_full_cd.unique())
    G = len(codes)
    idx = {c: i for i, c in enumerate(codes)}
    tr = g.groupby("umd_full_cd").treated.first().reindex(codes).to_numpy(bool)
    post = (np.floor(em / 3) >= 0).astype(float)

    ys, cls, groups, hflag = [], [], [], []
    for h in (False, True):
        cells = sorted(g[g.high == h].umd_full_cd.unique())
        cnt = (g[(g.high == h) & g.em.between(lo, hi)]
               .groupby(["umd_full_cd", "em"]).size()
               .unstack(fill_value=0).reindex(index=cells, columns=em, fill_value=0))
        ys.append(np.log1p(cnt.to_numpy(float)).reshape(-1))
        cls.append(np.repeat([idx[c] for c in cells], T))
        groups.append((len(cells), T))
        hflag.append(np.full(len(cells) * T, float(h)))
    y = np.concatenate(ys)
    cl = np.concatenate(cls)
    hf = np.concatenate(hflag)
    pst = np.tile(post, sum(a for a, _ in groups))

    def make(t):
        d = t[cl].astype(float) * pst
        return np.stack([d, d * hf])
    s = Spec("삼중차분 (차이)", y, make, twoway(groups), cl, G, tr, 1,
             lambda b: b[1], groups)
    return s, g


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
