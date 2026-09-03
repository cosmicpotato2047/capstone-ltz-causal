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
