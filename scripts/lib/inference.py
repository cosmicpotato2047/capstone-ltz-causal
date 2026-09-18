"""적은 클러스터 추론의 부품 — 균형 패널 TWFE, WCR 부트스트랩, 무작위화 추론.

`scripts/06_inference.py` 에서 만들고 13b 에서도 쓰려고 옮겼다(결정기록 0019).

사양(Spec)은 y, 설계행렬을 만드는 함수, 고정효과 제거 함수, 클러스터를 묶는다.
패널이 칸 × 사건월 균형 격자라 고정효과 제거가 닫힌 식으로 정확히 된다.
사양 만드는 함수(spec_es/spec_did/spec_ddd)는 거래 표를 받으므로, 표본을
미리 걸러서 넘기면 그 표본으로 같은 사양을 만든다.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

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


def wild_boot(s: Spec, rng, B: int = B) -> tuple[float, float, float, float, np.ndarray]:
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


def spec_did(df: pd.DataFrame, lo: int = -PRE, hi: int = POST - 1,
             name: str = "24개월 이중차분") -> Spec:
    """법정동 × 사건월 이중차분. 창 [lo, hi], 사후는 em >= 0."""
    codes = sorted(df.umd_full_cd.unique())
    em = np.arange(lo, hi + 1)
    T, G = em.size, len(codes)
    cnt = (df[df.em.between(lo, hi)].groupby(["umd_full_cd", "em"]).size()
           .unstack(fill_value=0).reindex(index=codes, columns=em, fill_value=0))
    tr = df.groupby("umd_full_cd").treated.first().reindex(codes).to_numpy(bool)
    post = (em >= 0).astype(float)
    return Spec(name, np.log1p(cnt.to_numpy(float)).reshape(-1),
                lambda t: np.outer(t, post).reshape(1, -1).astype(float),
                twoway([(G, T)]), np.repeat(np.arange(G), T), G, tr, 0,
                lambda b: b[0], [(G, T)])


def spec_triple(g: pd.DataFrame, flag: str, lo: int, hi: int,
                name: str) -> Spec:
    """삼중차분 일반형. 칸 = (법정동, flag 참/거짓), 시간 고정효과는 flag 별.

    g 에는 umd_full_cd, em(사건월), treated, 그리고 bool 열 flag 가 있어야 한다.
    설명변수는 [처치×사후, 처치×사후×flag]. 둘째 계수가 flag 쪽이 더 움직인 폭이다.
    사후는 em >= 0 이다. 창은 [lo, hi] 사건월.
    """
    em = np.arange(lo, hi + 1)
    T = em.size
    codes = sorted(g.umd_full_cd.unique())
    G = len(codes)
    idx = {c: i for i, c in enumerate(codes)}
    tr = g.groupby("umd_full_cd").treated.first().reindex(codes).to_numpy(bool)
    post = (em >= 0).astype(float)

    ys, cls, groups, hflag = [], [], [], []
    for h in (False, True):
        cells = sorted(g[g[flag] == h].umd_full_cd.unique())
        cnt = (g[(g[flag] == h) & g.em.between(lo, hi)]
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
    return Spec(name, y, make, twoway(groups), cl, G, tr, 1, lambda b: b[1], groups)


def spec_ddd(df: pd.DataFrame, high: float = HIGH) -> tuple[Spec, pd.DataFrame]:
    """06c_loan 3절과 같은 사양. 칸 = (법정동, 가격군)."""
    pre = df[df.deal_date.between(RULE_ON - pd.DateOffset(years=2), RULE_ON)]
    med = pre.groupby("aptSeq").amount_manwon.median() / 10000
    g = df.assign(base=df.aptSeq.map(med))
    g = g[g.base.notna()].copy()
    g["high"] = g.base > high
    # 사건분기 기준 사후(floor(em/3) >= 0)는 em >= 0 과 같다
    return spec_triple(g, "high", KMIN * 3, KMAX * 3 + 2, "삼중차분 (차이)"), g
