"""[08] 헤도닉 가격지수 — 구성 통제의 사다리와, 0024 가 남긴 가격 숙제.

    python scripts/08_hedonic.py

입력  data/processed/trades.parquet, trades_seoul.parquet, data/processed/rs_index.csv
출력  data/processed/hedonic_index.csv
      output/results/08_hedonic.json, output/figures/08_hedonic.png

**목적 둘.**

(가) 반복매매지수([[0018]])와 대조한다. 반복매매는 같은 집만 보므로 구성 변화에
강하지만 두 번 팔린 집만 쓴다(선택편의). 헤도닉은 모든 거래를 쓰되 관측되는
특성만 통제한다. 둘이 같은 이야기를 하면 서로를 떠받친다.

**구성 통제의 사다리**로 넷을 나란히 짓는다. 아래로 갈수록 더 많은 것을 고정한다.

    중위가              아무것도 고정하지 않는다
    헤도닉 (특성)        면적과 층을 고정한다
    헤도닉 + 단지        거기에 단지의 질까지 고정한다
    반복매매            같은 집만 본다

(나) [[0024]] 가 남긴 숙제. 지정 전 ㎡당가를 맞춘 표본에서 처치동 가격 효과가
−9.7% 에서 −2.1%(비유의)로 약해졌다. 처치 효과인지 가격대 효과인지 가르지
못했다. 여기서 두 가지를 더 해 본다.

    1. 통제를 한 칸씩 늘려 가며 계수가 어디서 움직이는지 본다
    2. **가격 합성대조군** — 사전 궤적이 맞도록 자료가 기증자 동의 가중치를
       직접 고르게 한다. 공통지지가 손으로 구간을 자르는 것이라면 이쪽은
       자료가 고른다. 17_scm 이 거래량에 쓴 방법과 같다.

연식은 넣지 않는다. 단지 고정효과 아래에서 연식은 시간의 일차함수라 월
고정효과와 공선이다.
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
from scipy.optimize import nnls  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import econ, geo, io, policy  # noqa: E402

io.setup_stdout()
for f in ("Malgun Gothic", "Batang"):
    if f in {x.name for x in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

EVENT = "E2020"
E = policy.EVENTS[EVENT]["effect"]
SGG = ["강남구", "송파구"]
SGG_NEAR = ["강동구", "성동구", "광진구", "동작구"]   # 12_spillover 의 R3
BASE = "2017-06"              # 07_repeat_sales 와 같은 기준 시점
IDX_FROM = "2015-01"          # 지수를 그릴 구간
CONTAM = ["압구정동", "거여동", "마천동", "송파동", "신천동", "일원동"]   # 0020
FLOOR_BINS = [0, 2, 5, 10, 15, 200]
FLOOR_LAB = ["1~2층", "3~5층", "6~10층", "11~15층", "16층 이상"]
FLOOR_REF = "6~10층"
PRE_M, POST_M = 36, 27        # 17_scm 과 같은 창
MIN_PRE = 3                   # 기증자 자격: 사전 월평균 최소 거래
N_PLACEBO_MAX = 200           # 배치 위약에 쓸 기증자 수 상한


# --- 헤도닉 -------------------------------------------------------------------
def add_features(d: pd.DataFrame) -> pd.DataFrame:
    """헤도닉 설명변수. 면적은 로그와 그 제곱, 층은 구간 더미."""
    d = d.copy()
    la = np.log(d.area_m2)
    d["면적"] = la
    d["면적2"] = la ** 2
    fb = pd.cut(d.floor_no, FLOOR_BINS, labels=FLOOR_LAB)
    for lab in FLOOR_LAB:
        if lab != FLOOR_REF:
            d[lab] = (fb == lab).astype(float)
    return d


FEATS = ["면적", "면적2"] + [x for x in FLOOR_LAB if x != FLOOR_REF]


def hedonic_index(d: pd.DataFrame, unit_fe: bool) -> pd.Series:
    """월별 지수. 기준 시점 = 100.

    unit_fe=False 면 관측 특성만, True 면 단지 고정효과까지 통제한다.
    """
    d = add_features(d)
    d["ymm"] = d.ym.astype(str)
    months = sorted(d.ymm.unique())
    base = BASE if BASE in months else months[0]
    D = pd.get_dummies(d.ymm, prefix="m", dtype=float).drop(columns=f"m_{base}")
    X = pd.concat([d[FEATS].reset_index(drop=True), D.reset_index(drop=True)], axis=1)
    y = d.log_price_per_m2.reset_index(drop=True)
    if unit_fe:
        dm = econ.demean(pd.concat([y.rename("y"), X], axis=1),
                         d[["aptSeq"]].reset_index(drop=True))
        res = sm.OLS(dm.y, dm[X.columns]).fit()
    else:
        res = sm.OLS(y, sm.add_constant(X)).fit()
    idx = {base: 0.0}
    for m in months:
        if m != base:
            idx[m] = float(res.params[f"m_{m}"])
    s = pd.Series(idx).sort_index()
    return 100 * np.exp(s)


def median_index(d: pd.DataFrame, col: str) -> pd.Series:
    """손대지 않은 중위가 지수. col 이 총액이면 구성 변화가 그대로 들어온다."""
    m = d.groupby(d.ym.astype(str))[col].median()
    return 100 * m / m.loc[BASE]


# --- 합성대조군 ----------------------------------------------------------------
def simplex_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """합이 1이고 음수가 아닌 가중치 (17_scm 과 같다)."""
    M = 1e6
    A = np.vstack([X, M * np.ones((1, X.shape[1]))])
    b = np.concatenate([y, [M]])
    w, _ = nnls(A, b)
    s = w.sum()
    return w / s if s > 0 else w


def main() -> None:
    d8 = io.load_trades()
    gs = d8[d8.sgg_nm.isin(SGG)].copy()
    gs["처치"] = policy.assign_2020(gs)
    treated = gs[gs.처치]
    control = gs[~gs.처치 & ~gs.umdNm.isin(CONTAM)]
    out: dict = {}

    # ==================================================================
    print("=" * 88)
    print("1. 구성 통제의 사다리 — 같은 자료로 지수 넷을 짓는다")
    print("=" * 88)
    rs = pd.read_csv(io.PROCESSED / "rs_index.csv", encoding="utf-8-sig")
    rs = rs.set_index("연월")
    idx = {}
    for name, sub in (("처치동", treated), ("통제동", control)):
        idx[(name, "중위가(총액)")] = median_index(sub, "amount_manwon")
        idx[(name, "중위가(㎡당)")] = median_index(sub, "price_per_m2")
        idx[(name, "헤도닉")] = hedonic_index(sub, unit_fe=False)
        idx[(name, "헤도닉+단지")] = hedonic_index(sub, unit_fe=True)
    idx[("처치동", "반복매매")] = rs.처치동
    idx[("통제동", "반복매매")] = rs.통제동
    tab = pd.DataFrame(idx)
    tab.index.name = "연월"
    tab = tab.loc[tab.index >= IDX_FROM]
    tab.to_csv(io.PROCESSED / "hedonic_index.csv", encoding="utf-8-sig")

    KINDS = ["중위가(총액)", "중위가(㎡당)", "헤도닉", "헤도닉+단지", "반복매매"]
    picks = ["2020-06", "2020-09", "2021-06", "2022-06"]
    print(f"  기준 {BASE} = 100 · 처치동")
    print(f"  {'시점':<10}" + "".join(f"{k:>14}" for k in KINDS))
    rows = []
    for m in picks:
        vals = [float(tab[("처치동", k)].get(m, np.nan)) for k in KINDS]
        rows.append({"시점": m, **dict(zip(KINDS, [round(v, 1) for v in vals]))})
        print(f"  {m:<10}" + "".join(f"{v:>13.1f}" for v in vals))
    out["지수_처치동"] = rows
    g = tab.loc["2020-09"]
    print(f"\n  0018 이 짚은 2020-09 (처치동 월평균 전용면적이 103.4㎡ 로 튄 달)")
    for k in KINDS:
        print(f"    {k:<14}{g[('처치동', k)]:>8.1f}")
    gap_all = g[("처치동", "중위가(총액)")] - g[("처치동", "반복매매")]
    gap_m2 = g[("처치동", "중위가(총액)")] - g[("처치동", "중위가(㎡당)")]
    print(f"  총액 중위가와 반복매매의 간격 {gap_all:+.1f}포인트 중, ㎡당가로")
    print(f"  바꾸는 것만으로 {gap_m2:+.1f}포인트가 걷힌다")
    print("  -> 중위가의 부풀림은 대부분 '큰 물건이 팔렸다' 하나였다")

    # 지정 후 12개월 평균으로 사다리를 요약한다
    print(f"\n  지정 후 12개월 평균 (처치동 − 통제동, {BASE} 기준 지수의 차이)")
    print(f"  {'':<14}{'사전 12개월':>13}{'사후 12개월':>13}{'변화':>10}")
    lad = []
    pre_m = [str(p) for p in pd.period_range(E - pd.DateOffset(months=12), periods=12, freq="M")]
    post_m = [str(p) for p in pd.period_range(E, periods=12, freq="M")]
    for k in KINDS:
        t = tab[("처치동", k)]
        c = tab[("통제동", k)]
        a = float((t.reindex(pre_m) / c.reindex(pre_m)).mean())
        b = float((t.reindex(post_m) / c.reindex(post_m)).mean())
        lad.append({"지수": k, "사전비": round(a, 4), "사후비": round(b, 4),
                    "변화_pct": round(100 * (b / a - 1), 1)})
        print(f"  {k:<14}{a:>13.3f}{b:>13.3f}{100 * (b / a - 1):>9.1f}%")
    out["지수_사다리"] = lad
    print("  네 지수가 같은 방향이면 구성 변화가 결론을 만들지 않았다는 뜻이다")

    # ==================================================================
    print("\n" + "=" * 88)
    print("2. 통제를 한 칸씩 늘린다 — 0024 의 숙제")
    print("=" * 88)
    w = pd.concat([treated, control]).copy()
    w["em"] = policy.event_month(w.deal_date, EVENT)
    w = w[w.em.between(-12, 11)].copy()
    w = add_features(w)
    w["d"] = w.처치.astype(float) * (w.em >= 0)
    w["ems"] = w.em.astype(str)
    pre24 = d8[d8.deal_date.between(E - pd.DateOffset(months=24), E)]
    w["기저"] = w.aptSeq.map(pre24.groupby("aptSeq").price_per_m2.median())
    q = w[w.처치].기저
    blo, bhi = float(q.quantile(.05)), float(q.quantile(.95))

    def did(s: pd.DataFrame, feats: bool) -> tuple[float, float, int]:
        cols = ["d"] + (FEATS if feats else [])
        dm = econ.demean(pd.concat([s[["log_price_per_m2"]], s[cols]], axis=1),
                         s[["aptSeq", "ems"]])
        f = sm.OLS(dm.log_price_per_m2, dm[cols]).fit(
            cov_type="cluster", cov_kwds={"groups": s.umd_full_cd})
        return (float(100 * np.expm1(f.params.d)), float(f.pvalues.d), len(s))

    # 통제군을 서울 17개 구로 바꾼 표본 — 0024 의 사다리와 잇는다
    sl0 = pd.read_parquet(io.TRADES_SEOUL)
    sl0["em"] = policy.event_month(sl0.deal_date, EVENT)
    sp0 = pd.read_csv(io.POLICY / "ltz_spells.csv", encoding="utf-8-sig")
    sp0["시작일"] = pd.to_datetime(sp0.시작일)
    sp0["종료일"] = pd.to_datetime(sp0.종료일)
    lo0, hi0 = E - pd.DateOffset(months=12), E + pd.DateOffset(months=12)
    bad0 = (set(sp0[(sp0.상태 == "지정") & (sp0.시작일 <= hi0)
                    & (sp0.종료일 >= lo0)].법정동) - policy.TREATED_2020)
    # 12_spillover 의 R4 와 같은 집단이다 — 강남·송파와 인접 4개 구를 뺀 19개 구.
    # 서초·용산이 여기 들어간다. 2020 년에는 미처치이고 처치동과 맞닿지도 않아
    # 통제군으로 쓸 수 있으며, 가격대가 비슷해 공통지지에서 중요하다.
    far = sl0[~sl0.sgg_nm.isin(SGG + SGG_NEAR) & ~sl0.umdNm.isin(bad0)].copy()
    far["처치"] = False
    ws = pd.concat([treated.assign(em=policy.event_month(treated.deal_date, EVENT)),
                    far]).copy()
    ws = ws[ws.em.between(-12, 11)].copy()
    ws = add_features(ws)
    ws["d"] = ws.처치.astype(float) * (ws.em >= 0)
    ws["ems"] = ws.em.astype(str)
    pre24s = sl0[sl0.deal_date.between(E - pd.DateOffset(months=24), E)]
    ws["기저"] = ws.aptSeq.map(pre24s.groupby("aptSeq").price_per_m2.median())

    print(f"  {'통제군':<18}{'사양':<26}{'가격효과':>10}{'p':>8}{'거래':>9}")
    steps = []
    grids = [
        ("강남·송파 통제동", w, blo, bhi),
        ("서울 나머지 19개 구", ws, blo, bhi),
    ]
    for cname, s0, lo_, hi_ in grids:
        band0 = s0[(s0.기저 >= lo_) & (s0.기저 <= hi_)]
        for slab, s1, feats in (
                ("① 단지·월 고정효과만", s0, False),
                ("② + 면적·층 (헤도닉)", s0, True),
                ("③ ① + 가격대 공통지지", band0, False),
                ("④ ② + 가격대 공통지지", band0, True)):
            e, p, n = did(s1, feats)
            steps.append({"통제군": cname, "사양": slab, "효과_pct": round(e, 1),
                          "p": round(p, 4), "거래": n})
            print(f"  {cname:<18}{slab:<26}{e:>9.1f}%{p:>8.3f}{n:>9,d}")
        print()
    print(f"  공통지지 구간 {blo:,.0f}~{bhi:,.0f} 만원/㎡ (처치동 단지의 5~95%)")
    print("  가격대를 맞추는 것이 결과를 크게 바꾸는 쪽은 서울 통제군이다 —")
    print("  강남·송파 통제동은 이미 비슷한 가격대라 바뀔 것이 없다")
    out["통제_사다리"] = steps
    out["공통지지_구간_만원"] = [round(blo), round(bhi)]

    # ==================================================================
    print("\n" + "=" * 88)
    print("3. 가격 합성대조군 — 기증자 가중치를 자료가 고른다")
    print("=" * 88)
    sl = sl0[sl0.em.between(-PRE_M, POST_M - 1)].copy()
    sp = sp0
    lo, hi = E - pd.DateOffset(months=PRE_M), E + pd.DateOffset(months=POST_M)
    bad = (set(sp[(sp.상태 == "지정") & (sp.시작일 <= hi) & (sp.종료일 >= lo)].법정동)
           - policy.TREATED_2020)

    # 결과변수: 단지 고정효과와 특성을 걷어낸 가격 잔차의 동·월 평균
    sl = add_features(sl)
    dm_s = econ.demean(pd.concat([sl[["log_price_per_m2"]], sl[FEATS]], axis=1),
                       sl[["aptSeq"]])
    beta = sm.OLS(dm_s.log_price_per_m2, dm_s[FEATS]).fit().params
    sl["resid"] = dm_s.log_price_per_m2 - dm_s[FEATS].to_numpy() @ beta.to_numpy()

    sl["처치"] = sl.umdNm.isin(policy.TREATED_2020) & sl.sgg_nm.isin(SGG)
    sl["key"] = np.where(sl.처치, "처치동", sl.sgg_nm + " " + sl.umdNm)
    # 맞닿은 동은 기증자에서 뺀다. [[0024]] 에서 그 동들의 가격이 17개 구 대비
    # −5.0% 움직였다. 파급을 받은 동을 통제군에 넣으면 효과가 0 쪽으로 끌린다.
    adj = geo.touching(policy.TREATED_2020, set(SGG))
    bad = bad | adj
    ok = ~sl.umdNm.isin(bad) | sl.처치
    sl = sl[ok]
    print(f"  기증자에서 뺀 동: 창 안 지정 {len(bad - adj)}개 + 맞닿은 동 {len(adj)}개")

    cell = sl.groupby(["key", "em"]).agg(y=("resid", "mean"), n=("resid", "size"))
    wide_y = cell.y.unstack()
    wide_n = cell.n.unstack().fillna(0)
    pre_cols = [c for c in wide_y.columns if c < 0]
    eligible = (wide_n[pre_cols].mean(axis=1) >= MIN_PRE) & wide_y.notna().all(axis=1)
    wide_y = wide_y[eligible]
    base_lv = wide_y[pre_cols].mean(axis=1)
    wide_y = wide_y.sub(base_lv, axis=0)      # 수준이 아니라 변화를 맞춘다

    if "처치동" not in wide_y.index:
        sys.exit("처치동이 기증자 자격에서 빠졌다 — 창이나 문턱을 확인할 것")
    donors = [k for k in wide_y.index if k != "처치동"]
    Y = wide_y.loc["처치동"].to_numpy()
    D = wide_y.loc[donors].to_numpy()
    npre = len(pre_cols)
    wts = simplex_weights(D[:, :npre].T, Y[:npre])
    synth = D.T @ wts
    gap = Y - synth
    rm_pre = float(np.sqrt(np.mean(gap[:npre] ** 2)))
    rm_post = float(np.sqrt(np.mean(gap[npre:] ** 2)))
    eff12 = float(100 * np.expm1(gap[npre:npre + 12].mean()))
    eff_all = float(100 * np.expm1(gap[npre:].mean()))

    print(f"  기증자 {len(donors)}개 법정동 · 사전 RMSPE {rm_pre:.4f} · "
          f"사후 {rm_post:.4f} (비 {rm_post / rm_pre:.2f})")
    top = sorted(zip(donors, wts), key=lambda x: -x[1])[:6]
    print("  가중치 상위: " + " · ".join(f"{k} {v:.2f}" for k, v in top if v > 0.005))
    print(f"  지정 후 12개월 평균 격차 {eff12:+.1f}% · 사후 전체 {eff_all:+.1f}%")

    # 배치 위약 — 기증자 하나씩 처치로 놓고 같은 계산
    ratios = []
    for i, k in enumerate(donors[:N_PLACEBO_MAX]):
        others = [j for j in range(len(donors)) if j != i]
        yk, Dk = D[i], D[others]
        wk = simplex_weights(Dk[:, :npre].T, yk[:npre])
        gk = yk - Dk.T @ wk
        a = np.sqrt(np.mean(gk[:npre] ** 2))
        b = np.sqrt(np.mean(gk[npre:] ** 2))
        if a > 0:
            ratios.append((k, float(b / a)))
    act = rm_post / rm_pre
    worse = sum(1 for _, r in ratios if r >= act)
    p_emp = (worse + 1) / (len(ratios) + 1)
    print(f"  배치 위약 {len(ratios)}개 중 오차비가 더 큰 것 {worse}개 "
          f"-> 경험적 p = {p_emp:.3f}")
    out["가격_SCM"] = {"기증자수": len(donors), "사전RMSPE": round(rm_pre, 4),
                       "사후RMSPE": round(rm_post, 4), "오차비": round(act, 2),
                       "사후12개월_pct": round(eff12, 1),
                       "사후전체_pct": round(eff_all, 1),
                       "위약수": len(ratios), "경험적p": round(p_emp, 4),
                       "가중치": {k: round(float(v), 4) for k, v in top if v > 0.005}}

    # ==================================================================
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.8))
    ax = axes[0]
    cols = {"중위가(총액)": "#cbd5e0", "중위가(㎡당)": "#a0aec0",
            "헤도닉": "#dd6b20", "헤도닉+단지": "#3182ce", "반복매매": "#c53030"}
    x = pd.PeriodIndex(tab.index, freq="M").to_timestamp()
    for k, c in cols.items():
        ax.plot(x, tab[("처치동", k)], color=c, lw=1.6, label=k)
    ax.axvline(E, color="#1a202c", ls="--", lw=1.2)
    ax.set_title(f"처치동 가격지수 넷 ({BASE} = 100)")
    ax.set_ylabel("지수")
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)

    ax = axes[1]
    # 월별 비는 거래가 적은 달에 크게 튄다. 3개월 이동평균으로 고르고
    # 사건 앞뒤 3년만 본다
    lo_x, hi_x = pd.Timestamp("2017-07"), pd.Timestamp("2023-06")
    for k, c in cols.items():
        r = tab[("처치동", k)] / tab[("통제동", k)]
        r = 100 * r / r.loc[str(pd.Period(E, "M"))]
        r = r.rolling(3, center=True).mean()
        m = (x >= lo_x) & (x <= hi_x)
        ax.plot(x[m], r[m], color=c, lw=1.7, label=k)
    ax.axvline(E, color="#1a202c", ls="--", lw=1.2)
    ax.axhline(100, color="#999", lw=.8)
    ax.set_title("처치동 ÷ 통제동 (지정 시점 = 100, 3개월 이동평균)")
    ax.set_ylabel("상대 지수")
    ax.grid(alpha=.25)

    ax = axes[2]
    em = np.arange(-PRE_M, POST_M)
    ax.plot(em, 100 * np.expm1(Y), color="#c53030", lw=1.8, label="처치동")
    ax.plot(em, 100 * np.expm1(synth), color="#3182ce", lw=1.6, ls="--",
            label="합성대조군")
    ax.axvline(-0.5, color="#1a202c", ls="--", lw=1.2)
    ax.axhline(0, color="#999", lw=.8)
    ax.set_title("가격 합성대조군 — 특성·단지를 걷어낸 잔차")
    ax.set_xlabel("지정 기준 사건월")
    ax.set_ylabel("사전 평균 대비 (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)

    fig.suptitle("헤도닉 지수와 가격 효과 — 구성과 가격대를 걷어내면 무엇이 남는가",
                 y=1.03)
    fig.tight_layout()
    fp = io.FIGURES / "08_hedonic.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")

    out.update({"기준시점": BASE, "층구간": FLOOR_LAB, "기준층": FLOOR_REF,
                "창_SCM": [-PRE_M, POST_M - 1]})
    io.save_result("08_hedonic", out)
    print(f"\n저장 {fp} · output/results/08_hedonic.json · "
          f"data/processed/hedonic_index.csv")


if __name__ == "__main__":
    main()
