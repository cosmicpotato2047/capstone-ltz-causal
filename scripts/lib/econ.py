"""계량 추정 헬퍼 — 다중 고정효과 제거와 사건 시점 분석."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def demean(data: pd.DataFrame, fe: pd.DataFrame,
           tol: float = 1e-9, maxiter: int = 200) -> pd.DataFrame:
    """교대투영으로 다중 고정효과를 제거한다 (Frisch-Waugh-Lovell).

    data : 종속변수와 설명변수
    fe   : 각 열이 하나의 고정효과 그룹 키
    """
    out = data.astype(float).copy()
    keys = [fe[c].to_numpy() for c in fe.columns]
    for _ in range(maxiter):
        prev = out.to_numpy(copy=True)
        for k in keys:
            out = out - out.groupby(k).transform("mean")
        if np.nanmax(np.abs(out.to_numpy() - prev)) < tol:
            break
    return out


def event_study(df: pd.DataFrame, y: str, treated: str, kq: str,
                fes: list[str], controls: list[str], cluster: str,
                kmin: int = -8, kmax: int = 8, ref: int = -1) -> pd.DataFrame:
    """처치더미 x 사건분기 더미의 계수를 추정한다. 기준분기 계수는 0으로 고정."""
    ks = [k for k in range(kmin, kmax + 1) if k != ref]
    X = pd.DataFrame(
        {f"k{k}": ((df[kq] == k) & df[treated]).astype(float) for k in ks},
        index=df.index)
    for c in controls:
        X[c] = df[c].astype(float)
    dm = demean(pd.concat([df[[y]], X], axis=1), df[fes])
    res = sm.OLS(dm[y], dm[list(X.columns)]).fit(
        cov_type="cluster", cov_kwds={"groups": df[cluster]})
    out = pd.DataFrame({
        "k": ks,
        "coef": [res.params[f"k{k}"] for k in ks],
        "se": [res.bse[f"k{k}"] for k in ks],
        "p": [res.pvalues[f"k{k}"] for k in ks],
    })
    out = pd.concat([out, pd.DataFrame({"k": [ref], "coef": [0.0],
                                        "se": [0.0], "p": [np.nan]})])
    out = out.sort_values("k").reset_index(drop=True)
    out["ci_lo"] = out.coef - 1.96 * out.se
    out["ci_hi"] = out.coef + 1.96 * out.se
    out["effect_pct"] = 100 * (np.exp(out.coef) - 1)
    return out


def did(df: pd.DataFrame, y: str, interaction: str, X: list[str],
        fes: list[str] | None, cluster: str):
    """단일 이중차분 계수를 추정한다."""
    cols = [interaction] + X
    if fes:
        dm = demean(pd.concat([df[[y]], df[cols]], axis=1), df[fes])
        return sm.OLS(dm[y], dm[cols]).fit(
            cov_type="cluster", cov_kwds={"groups": df[cluster]})
    d = sm.add_constant(df[cols])
    return sm.OLS(df[y], d).fit(cov_type="cluster",
                                cov_kwds={"groups": df[cluster]})


def ppml(y: np.ndarray, X: np.ndarray, cell: np.ndarray,
         time: np.ndarray | None = None, cluster: np.ndarray | None = None,
         tol: float = 1e-10, maxiter: int = 100) -> dict:
    """단위 고정효과를 프로파일로 흡수한 포아송 유사최대우도(PPML).

    거래 건수는 0 이 많고 단지 규모가 크게 다르다. log1p 선형 모형은 칸마다
    같은 무게를 주므로 거래가 드문 작은 단지에 끌려가고, 그래서 집계 배수와
    어긋난다(결정기록 0023). 포아송은 기대 건수에 비례해 가중하므로 계수가
    집계 배수와 같은 이야기를 한다.

    단위 고정효과는 더미로 넣지 않는다. 포아송에서는 주어진 나머지 모수에 대해
    alpha_c = log(sum y) - log(sum exp(eta)) 로 닫힌 식이 있어 정확히 흡수된다.
    시간 고정효과는 개수가 적어 더미로 넣는다.

    y       (n,)   건수
    X       (n,K)  설명변수 (고정효과 제외)
    cell    (n,)   단위 고정효과 코드. 이 코드 안에서 합이 0 인 칸은 정보가
                   없으므로 자동으로 빠진다
    time    (n,)   시간 고정효과 코드
    cluster (n,)   군집. 기본은 cell

    반환: coef, se, z, p, n, K, G, iter
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    cell = pd.factorize(np.asarray(cell))[0]
    cluster = cell if cluster is None else pd.factorize(np.asarray(cluster))[0]

    # 창 안에서 한 건도 없는 칸은 alpha 가 발산한다. 빼고 추정한다.
    tot = np.bincount(cell, weights=y)
    keep = tot[cell] > 0
    y, X = y[keep], X[keep]
    cell = pd.factorize(cell[keep])[0]
    cluster = pd.factorize(cluster[keep])[0]

    Z = X
    if time is not None:
        t = pd.factorize(np.asarray(time)[keep])[0]
        D = np.zeros((t.size, t.max()))
        D[np.arange(t.size)[t > 0], t[t > 0] - 1] = 1.0
        Z = np.hstack([X, D])
        # 시간 고정효과를 묶음별로 따로 둘 때(예: (주, 소형))는 칸 고정효과와
        # 겹치는 더미가 묶음 수만큼 생긴다. 칸평균을 뺀 뒤 순위를 올리는
        # 열만 남긴다. 설명변수(X)는 순서상 먼저라 항상 살아남는다.
        Zc = Z - (pd.DataFrame(Z).groupby(cell).transform("mean").to_numpy())
        Gm = Zc.T @ Zc
        sel: list[int] = []
        for k in range(Z.shape[1]):
            cand = sel + [k]
            if np.linalg.matrix_rank(Gm[np.ix_(cand, cand)], tol=1e-8) == len(cand):
                sel = cand
        Z = Z[:, sel]

    n, K = Z.shape
    C = cell.max() + 1
    logtot = np.log(np.bincount(cell, weights=y))

    def at(th):
        """주어진 theta 에서 칸 효과를 닫힌 식으로 풀고 mu 와 로그우도를 낸다.

        칸 효과를 넣으면 칸마다 sum(mu) = sum(y) 가 되므로 로그우도의
        -sum(mu) 항이 상수가 된다. 남는 것은 sum(y * (alpha + eta)) 다.
        """
        eta = np.clip(Z @ th, -30, 30)
        a = logtot - np.log(np.bincount(cell, weights=np.exp(eta)))
        return np.exp(a[cell] + eta), float(y @ (a[cell] + eta))

    theta = np.zeros(K)
    mu, ll = at(theta)
    Zt = Z.copy()
    for it in range(1, maxiter + 1):
        # mu 가중 칸평균을 뺀 설계행렬 (프로파일 우도의 정보행렬)
        sm_ = np.bincount(cell, weights=mu)
        Zb = np.stack([np.bincount(cell, weights=mu * Z[:, k]) / sm_
                       for k in range(K)], axis=1)
        Zt = Z - Zb[cell]
        g = Zt.T @ (y - mu)
        H = Zt.T @ (mu[:, None] * Zt)
        step = np.linalg.solve(H, g)
        # 감쇠 뉴턴. 소형처럼 칸이 적은 묶음에서 전진폭이 지나치면 발산한다
        for _ in range(30):
            cand = theta + step
            mu_c, ll_c = at(cand)
            if np.isfinite(ll_c) and ll_c >= ll - 1e-12:
                break
            step = step / 2
        else:
            break
        theta, mu, ll = cand, mu_c, ll_c
        if np.max(np.abs(step)) < tol:
            break

    # 수렴한 theta 에서 다시 만들어 샌드위치에 쓴다
    sm_ = np.bincount(cell, weights=mu)
    Zb = np.stack([np.bincount(cell, weights=mu * Z[:, k]) / sm_
                   for k in range(K)], axis=1)
    Zt = Z - Zb[cell]
    H = Zt.T @ (mu[:, None] * Zt)

    G = cluster.max() + 1
    s = np.zeros((G, K))
    np.add.at(s, cluster, Zt * (y - mu)[:, None])
    Hi = np.linalg.inv(H)
    # 흡수한 칸 고정효과도 모수로 세어 자유도를 깎는다 (ppmlhdfe 와 같은 보정)
    fac = G / (G - 1) * (n - 1) / (n - K - C)
    V = fac * Hi @ (s.T @ s) @ Hi
    se = np.sqrt(np.diag(V))
    z = theta[:X.shape[1]] / se[:X.shape[1]]
    from scipy import stats
    return {"coef": theta[:X.shape[1]], "se": se[:X.shape[1]], "z": z,
            "p": 2 * stats.norm.sf(np.abs(z)), "n": int(n), "K": int(K),
            "G": int(G), "iter": it, "칸": int(C)}
