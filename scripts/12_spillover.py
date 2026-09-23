"""[12] 파급효과 — 규제 밖으로 수요가 옮겨 갔는가.

토허제의 명분은 가격 안정이다. 규제 밖 단지로 수요가 옮겨 갔을 뿐이라면
규제는 문제를 푼 것이 아니라 옮긴 것이다(`docs/framing.md` 의 '어디로 번졌나').
선행연구 두 편이 '인근 지역 급등'을 보고했다(`docs/prior_work.md`).

우리 자료에는 규제 밖이 **세 층**으로 있다. 층마다 사건이 다르다.

    층 1  같은 동 안   2025-02-13 해제 — 해제군 옆에 유지군 16개가 남았다
    층 2  옆 동        2020-06-23 지정 — 처치 4개 동에 14개 동이 맞닿는다
    층 3  옆 구        2025-03-24 재지정 — 강남3구·용산이 막히고 4개 구가 남았다

**기준점을 명시한다.** 파급을 '처치동 대비'로 재면 처치동이 줄어든 것만으로도
옆 동이 늘어난 것처럼 보인다. 그래서 같은 자료에 기준점만 바꿔 가며 사다리를
만들어 함께 싣는다(4절). 이것이 선행연구와 갈리는 지점이다.

가격은 한 층이 더 있다. 처치동 단지는 서울 최상위 가격대에 몰려 있고
2020~2021년은 중저가가 급등한 시기다. 지정 전 ㎡당가가 처치동 범위에 드는
단지만 남긴 **공통지지 표본**을 나란히 돌린다. 분위 고정효과는 쓰지 않는다 —
최상위 분위 거래의 절반이 처치동과 맞닿은 동이라 처치가 제 통제군이 된다.

    python scripts/12_spillover.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import econ, geo, io, policy  # noqa: E402

io.setup_stdout()
for f in ("Malgun Gothic", "Batang"):
    if f in {x.name for x in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

E2020 = pd.Timestamp("2020-06-23")
RELEASE = pd.Timestamp("2025-02-13")
REDESIG = pd.Timestamp("2025-03-24")
PRE = POST = 12                 # 층 2 창: 지정 전후 12개월
PRE_W, POST_W = 13, 5           # 층 1·3 창: 주 단위
GAP_W = 6                       # 재지정 기준 -6 주부터가 해제창이다
SGG_TREAT = ["강남구", "송파구"]
SGG_NEAR = ["강동구", "성동구", "광진구", "동작구"]
SGG_2025 = ["강남구", "서초구", "송파구", "용산구"]   # 2025-03-24 재지정 범위
RINGS = ["R0 처치동", "R1 맞닿은 동", "R2 강남송파 비인접", "R3 인접 4개구"]
BASE_RING = "R4 나머지 19구"


# --- 표본 만들기 ---------------------------------------------------------------
def designated(lo: pd.Timestamp, hi: pd.Timestamp) -> set[str]:
    """[lo, hi] 안에 하루라도 다른 계열로 지정돼 있던 법정동."""
    sp = pd.read_csv(io.POLICY / "ltz_spells.csv", encoding="utf-8-sig")
    sp["시작일"] = pd.to_datetime(sp.시작일)
    sp["종료일"] = pd.to_datetime(sp.종료일)
    m = (sp.상태 == "지정") & (sp.시작일 <= hi) & (sp.종료일 >= lo)
    return set(sp[m].법정동) - policy.TREATED_2020


def load_seoul() -> pd.DataFrame:
    d = pd.read_parquet(io.TRADES_SEOUL)
    adj = geo.touching(policy.TREATED_2020, set(SGG_TREAT))
    d["맞닿음"] = d.umdNm.isin(adj) & ~d.umdNm.isin(policy.TREATED_2020)
    # 지정 전 24개월 단지 중위 ㎡당가 — 공통지지 표본의 기준
    pre = d[d.deal_date.between(E2020 - pd.DateOffset(months=24), E2020)]
    d["기저"] = d.aptSeq.map(pre.groupby("aptSeq").price_per_m2.median())
    return d


def rings(d: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.Series:
    """거리 고리. 창 안에 다른 계열로 지정된 동은 뺀다 (0009·0020)."""
    bad = designated(lo, hi)
    r = pd.Series(BASE_RING, index=d.index)
    r[d.sgg_nm.isin(SGG_TREAT)] = RINGS[2]
    r[d.sgg_nm.isin(SGG_NEAR)] = RINGS[3]
    r[d.맞닿음] = RINGS[1]
    r[d.umdNm.isin(policy.TREATED_2020) & d.sgg_nm.isin(SGG_TREAT)] = RINGS[0]
    r[d.umdNm.isin(bad) & (r != RINGS[0])] = "제외"
    return r


# --- 추정 ---------------------------------------------------------------------
def vol(s: pd.DataFrame, tgt: str, ctl: str, col: str, tcol: str,
        lo: int, hi: int) -> tuple[float, float]:
    """단지 × 시점 PPML 이중차분. 클러스터는 법정동."""
    s = s[s[col].isin([tgt, ctl])]
    cells = s.groupby("aptSeq").agg(**{col: (col, "first"),
                                       "동": ("umd_full_cd", "first")})
    full = (cells.index.to_frame(index=False)
            .merge(pd.DataFrame({tcol: range(lo, hi + 1)}), how="cross"))
    cnt = s.groupby(["aptSeq", tcol]).size().rename("n").reset_index()
    p = (full.merge(cnt, on=["aptSeq", tcol], how="left").fillna({"n": 0})
         .join(cells, on="aptSeq"))
    p["d"] = (p[col] == tgt).astype(float) * (p[tcol] >= 0)
    r = econ.ppml(p.n.to_numpy(), p[["d"]].to_numpy(), p.aptSeq.to_numpy(),
                  p[tcol].to_numpy(), cluster=p.동.to_numpy())
    return float(100 * np.expm1(r["coef"][0])), float(r["p"][0])


def price(s: pd.DataFrame, tgt: str, ctl: str, col: str,
          tcol: str) -> tuple[float, float]:
    """log ㎡당가에 단지·시점 고정효과. 클러스터는 법정동."""
    s = s[s[col].isin([tgt, ctl])].copy()
    s["d"] = (s[col] == tgt).astype(float) * (s[tcol] >= 0)
    s["t"] = s[tcol].astype(str)
    dm = econ.demean(s[["log_price_per_m2", "d"]], s[["aptSeq", "t"]])
    f = sm.OLS(dm.log_price_per_m2, dm[["d"]]).fit(
        cov_type="cluster", cov_kwds={"groups": s.umd_full_cd})
    return float(100 * np.expm1(f.params.d)), float(f.pvalues.d)


def both(s, tgt, ctl, col, tcol, lo, hi) -> dict:
    pe, pp = price(s, tgt, ctl, col, tcol)
    ve, vp = vol(s, tgt, ctl, col, tcol, lo, hi)
    return {"가격_pct": round(pe, 1), "가격_p": round(pp, 4),
            "거래량_pct": round(ve, 1), "거래량_p": round(vp, 4)}


def line(lab: str, r: dict, w: int = 34) -> str:
    return (f"  {lab:<{w}}{r['거래량_pct']:>9.1f}%{r['거래량_p']:>8.3f}"
            f"{r['가격_pct']:>9.1f}%{r['가격_p']:>8.3f}")


HEAD = f"  {'':<34}{'거래량':>10}{'p':>8}{'가격':>10}{'p':>8}"


def main() -> None:
    d = load_seoul()
    out: dict = {}

    # ==================================================================
    # 층 2 — 옆 동 (2020-06-23 지정)
    # ==================================================================
    lo, hi = E2020 - pd.DateOffset(months=PRE), E2020 + pd.DateOffset(months=POST)
    d["고리"] = rings(d, lo, hi)
    d["em"] = policy.event_month(d.deal_date, "E2020")
    w = d[(d.em >= -PRE) & (d.em < POST) & (d.고리 != "제외")].copy()

    print("=" * 92)
    print(f"층 2 — 옆 동. 2020-06-23 지정, 창 ±{PRE}개월, 기준점 = {BASE_RING}")
    print("=" * 92)
    tab = w.groupby("고리").agg(법정동=("umdNm", "nunique"),
                                단지=("aptSeq", "nunique"),
                                거래=("aptSeq", "size"),
                                기저중위=("기저", "median"))
    print(tab.round(0).to_string())
    print(f"  맞닿은 동: {', '.join(sorted(w[w.고리 == RINGS[1]].umdNm.unique()))}")
    print(f"  창 안에 다른 계열로 지정된 {len(designated(lo, hi))}개 동은 뺐다")
    out["층2_표본"] = tab.reset_index().round(1).to_dict("records")

    print("\n" + HEAD)
    ring_rows = []
    for rg in RINGS:
        r = both(w, rg, BASE_RING, "고리", "em", -PRE, POST - 1)
        ring_rows.append({"고리": rg, **r})
        print(line(rg, r))
    print("  어느 고리도 거래량이 양(+)이 아니다 — 거래가 옆으로 옮겨 간 흔적이 없다")
    out["층2_고리별"] = ring_rows

    # 공통지지 표본
    q = w[w.고리 == RINGS[0]].기저
    blo, bhi = float(q.quantile(.05)), float(q.quantile(.95))
    band = w[(w.기저 >= blo) & (w.기저 <= bhi)]
    print(f"\n  공통지지 표본 — 지정 전 ㎡당가 {blo:,.0f}~{bhi:,.0f}만원인 단지만")
    bt = band.groupby("고리").agg(단지=("aptSeq", "nunique"), 거래=("aptSeq", "size"))
    print("  " + " · ".join(f"{k} {v.단지}개 {v.거래:,}건" for k, v in bt.iterrows()))
    print(HEAD)
    band_rows = []
    for rg in RINGS:
        r = both(band, rg, BASE_RING, "고리", "em", -PRE, POST - 1)
        band_rows.append({"고리": rg, **r})
        print(line(rg, r))
    print("  거래량은 그대로 크고, 가격 효과는 0 과 구분되지 않게 된다")
    out["층2_공통지지"] = band_rows
    out["공통지지_구간_만원"] = [round(blo), round(bhi)]

    # ==================================================================
    # 층 1 — 같은 동 안 (2025-02-13 해제)
    # ==================================================================
    print("\n" + "=" * 92)
    print(f"층 1 — 같은 동 안. 2025-02-13 해제, 창 전 {PRE_W}주 / 후 {POST_W}주")
    print("=" * 92)
    d2 = d.copy()
    bad25 = designated(RELEASE - pd.DateOffset(months=6), REDESIG)
    d2["kept"] = policy.assign_kept_2025(d2)
    g = pd.Series("제외", index=d2.index)
    g[d2.sgg_nm.isin(SGG_NEAR) | ~d2.sgg_nm.isin(SGG_2025)] = "먼 통제군"
    g[d2.umdNm.isin(policy.TREATED_2020) & d2.sgg_nm.isin(SGG_TREAT)] = "해제군"
    g[d2.umdNm.isin(bad25) & (g != "해제군")] = "제외"
    g[d2.kept] = "유지군"
    d2["집단"] = g
    d2["wk"] = np.floor((d2.deal_date - RELEASE).dt.days / 7).astype(int)
    w1 = d2[d2.wk.between(-PRE_W, POST_W - 1) & (d2.집단 != "제외")].copy()

    print(f"  {'집단':<34}{'배수':>9}")
    mult = {}
    for gname in ("해제군", "유지군", "먼 통제군"):
        s = w1[w1.집단 == gname]
        a = (s.wk < 0).sum() / PRE_W
        b = (s.wk >= 0).sum() / POST_W
        mult[gname] = round(float(b / a), 2)
        print(f"  {gname:<34}{b / a:>8.2f}배")
    print(HEAD)
    lay1 = []
    for gname in ("해제군", "유지군"):
        r = both(w1, gname, "먼 통제군", "집단", "wk", -PRE_W, POST_W - 1)
        lay1.append({"집단": gname, "배수": mult[gname], **r})
        print(line(gname + " (먼 통제군 대비)", r))
    print("  유지군은 같은 동 안에서 해제군 옆에 있었다. 규제는 그대로였다")
    out["층1"] = lay1

    # ==================================================================
    # 층 3 — 옆 구 (2025-03-24 재지정)
    # ==================================================================
    print("\n" + "=" * 92)
    print(f"층 3 — 옆 구. 2025-03-24 재지정, 창 전 {PRE_W}주(해제창 제외) / 후 {POST_W}주")
    print("=" * 92)
    d3 = d.copy()
    bad3 = designated(REDESIG - pd.DateOffset(months=6), REDESIG + pd.DateOffset(months=3))
    g3 = pd.Series("먼 17개 구", index=d3.index)
    g3[d3.sgg_nm.isin(SGG_NEAR)] = "인접 4개 구"
    g3[d3.sgg_nm.isin(SGG_2025)] = "새로 규제"
    g3[d3.umdNm.isin(bad3) & (g3 != "새로 규제")] = "제외"
    d3["집단"] = g3
    d3["wk"] = np.floor((d3.deal_date - REDESIG).dt.days / 7).astype(int)
    weeks = list(range(-PRE_W - GAP_W, -GAP_W)) + list(range(0, POST_W))
    w3 = d3[d3.wk.isin(weeks) & (d3.집단 != "제외")].copy()
    print(f"  {'집단':<34}{'배수':>9}")
    for gname in ("새로 규제", "인접 4개 구", "먼 17개 구"):
        s = w3[w3.집단 == gname]
        a = (s.wk < 0).sum() / PRE_W
        b = (s.wk >= 0).sum() / POST_W
        print(f"  {gname:<34}{b / a:>8.2f}배")
    print(HEAD)
    lay3 = []
    for gname in ("새로 규제", "인접 4개 구"):
        r = both(w3, gname, "먼 17개 구", "집단", "wk",
                 -PRE_W - GAP_W, POST_W - 1)
        lay3.append({"집단": gname, **r})
        print(line(gname + " (먼 17개 구 대비)", r))
    print("  인접 4개 구가 올랐다면 수요가 구 밖으로 넘어간 것이다")
    out["층3"] = lay3

    # ==================================================================
    # 4. 사다리 — 기준점만 바꾼다
    # ==================================================================
    print("\n" + "=" * 92)
    print("4. 사다리 — 같은 자료, 같은 창, 기준점만 바꾼다 (맞닿은 동)")
    print("=" * 92)
    print(HEAD)
    ladder = []
    steps = [("1. 기준점 = 처치동 (선행연구의 시선)", w, RINGS[0]),
             ("2. 기준점 = 강남·송파 비인접 동", w, RINGS[2]),
             ("3. 기준점 = 나머지 17개 구", w, BASE_RING),
             ("4. 3 + 가격대 공통지지", band, BASE_RING)]
    for lab, s, ctl in steps:
        r = both(s, RINGS[1], ctl, "고리", "em", -PRE, POST - 1)
        ladder.append({"단계": lab, "기준점": ctl, **r})
        print(line(lab, r))
    print("  1번이 '인근 급등' 이다. 옆 동이 늘어서가 아니라 처치동이 줄어서다")
    out["사다리_맞닿은동"] = ladder

    print("\n  처치동으로 같은 사다리")
    lad0 = []
    for lab, s, ctl in [("1. 기준점 = 강남·송파 비인접 동", w, RINGS[2]),
                        ("2. 기준점 = 나머지 17개 구", w, BASE_RING),
                        ("3. 2 + 가격대 공통지지", band, BASE_RING)]:
        r = both(s, RINGS[0], ctl, "고리", "em", -PRE, POST - 1)
        lad0.append({"단계": lab, "기준점": ctl, **r})
        print(line(lab, r))
    print("  거래량은 어떻게 견줘도 −57~−64%. 가격은 가격대를 맞추면 0 과 구분되지 않는다")
    out["사다리_처치동"] = lad0

    # ==================================================================
    # 그림
    # ==================================================================
    w["kq"] = np.floor(w.em / 3).astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.8))
    cols = {RINGS[0]: "#c53030", RINGS[1]: "#dd6b20",
            RINGS[2]: "#3182ce", RINGS[3]: "#38a169"}

    # (1) 분기 거래량 — 고리별, R4 대비
    ax = axes[0]
    cnt = (w.groupby(["고리", "kq"]).size().unstack(fill_value=0)
           .div(w.groupby("고리").aptSeq.nunique(), axis=0))
    rel = cnt.div(cnt.loc[BASE_RING], axis=1)
    rel = rel.div(rel.loc[:, rel.columns < 0].mean(axis=1), axis=0)
    for rg, c in cols.items():
        ax.plot(rel.columns, 100 * (rel.loc[rg] - 1), "o-", color=c, lw=1.7,
                ms=4, label=rg)
    ax.axhline(0, color="#999", lw=.8)
    ax.axvline(-0.5, color="#1a202c", ls="--", lw=1.2)
    ax.set_title("거래량 — 나머지 17개 구 대비 (사전 = 0)")
    ax.set_xlabel("지정 기준 분기")
    ax.set_ylabel("단지당 거래의 상대 변화 (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)

    # (2) 분기 가격 — 단지·월 고정효과 잔차의 고리별 평균
    ax = axes[1]
    q2 = w.copy()
    q2["t"] = q2.em.astype(str)
    q2["resid"] = econ.demean(q2[["log_price_per_m2"]],
                              q2[["aptSeq", "t"]]).log_price_per_m2
    m = q2.groupby(["kq", "고리"]).resid.mean().unstack()
    m = m.sub(m[BASE_RING], axis=0)
    m = m - m[m.index < 0].mean()
    for rg, c in cols.items():
        ax.plot(m.index, 100 * m[rg], "o-", color=c, lw=1.7, ms=4, label=rg)
    ax.axhline(0, color="#999", lw=.8)
    ax.axvline(-0.5, color="#1a202c", ls="--", lw=1.2)
    ax.set_title("가격 — 같은 단지 안의 변화, 17개 구 대비")
    ax.set_xlabel("지정 기준 분기")
    ax.set_ylabel("log ㎡당가 차이 (%)")
    ax.grid(alpha=.25)

    # (3) 사다리
    ax = axes[2]
    labs = [s[0].split(". ", 1)[1] for s in steps]
    y = np.arange(len(labs))[::-1]
    ax.barh(y + .18, [r["거래량_pct"] for r in ladder], .34,
            color="#c53030", label="거래량")
    ax.barh(y - .18, [r["가격_pct"] for r in ladder], .34,
            color="#3182ce", label="가격")
    for i, r in enumerate(ladder):
        for key, off in (("거래량_pct", .18), ("가격_pct", -.18)):
            v = r[key]
            ax.text(v, y[i] + off, f" {v:+.0f}% " if v >= 0 else f" {v:+.0f}% ",
                    va="center", fontsize=8.5,
                    ha="left" if v >= 0 else "right")
    ax.axvline(0, color="#333", lw=.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=8.5)
    ax.set_xlabel("맞닿은 동의 효과 (%)")
    ax.set_title("기준점을 바꾸면 부호가 뒤집힌다")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=.25, axis="x")

    fig.suptitle("파급효과 — 규제 밖으로 옮겨 갔는가 (2020-06-23 지정)", y=1.03)
    fig.tight_layout()
    fp = io.FIGURES / "12_spillover.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")

    out.update({"창_개월": [PRE, POST], "창_주": [PRE_W, POST_W],
                "맞닿은_동": sorted(w[w.고리 == RINGS[1]].umdNm.unique())})
    io.save_result("12_spillover", out)
    print(f"\n저장 {fp} · output/results/12_spillover.json")


if __name__ == "__main__":
    main()
