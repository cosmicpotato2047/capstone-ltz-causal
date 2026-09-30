"""[5b] 사양 곡선 — 설계 선택을 전부 조합하면 결론이 어디서 흔들리는가.

    python scripts/05b_spec_curve.py

입력  data/processed/trades.parquet, trades_seoul.parquet
출력  output/results/05b_spec_curve.json, output/figures/05b_spec_curve.png

**왜 하는가.** 제안서의 RQ4 — "선행연구의 상반된 결론은 실질적 차이인가 설계
선택의 산물인가". 지금까지 이 질문에 두 번 답했다. 세 논문의 지역 정의를 우리
자료에 그대로 적용해 전부 음수임을 보였고([선행연구 정리](../docs/prior_work.md)),
기준점만 바꿔도 부호가 뒤집힌다는 것을 보였다([[0024]]).

둘 다 한 번에 한 축만 움직였다. 사양 곡선은 **합리적인 선택을 모두 조합해
한 그림에 놓는다**(Simonsohn·Simmons·Nelson 2020). 어느 조합에서도 결론이 같으면
그 결론은 설계 선택의 산물이 아니다. 갈리면 **어느 축이 갈랐는지**가 보인다.

## 무엇을 움직이나

거래량 192 사양 = 처치군 2 × 통제군 4 × 창 3 × 사건정렬 2 × 빈칸 2 × 척도 2
가격  144 사양 = 처치군 2 × 통제군 3 × 고정효과 2 × 특성 2 × 공통지지 2 × 창 3

축은 전부 이 연구가 실제로 내린 결정에서 왔다.

    처치군    4개 동([[0012]]) / 잠실동만(이한웅·이춘원 2025 의 정의)
    통제군    강남·송파 비처치 / 오염 6개 제외([[0020]]) / 인접 4개 구 / 서울 19개 구
    사건정렬  효력일 기준 사건월([[0013]]) / 달력 월(0013 이전 방식)
    빈칸      거래 없는 달을 0 으로 채움([[0013]]) / 안 채움
    척도      log1p 선형 / PPML([[0023]])
    고정효과  법정동 / 단지([[0003]])
    특성      없음 / 면적·층([[0025]])
    공통지지  전체 / 지정 전 ㎡당가를 처치동 범위로([[0024]])

## 무엇을 재나

    거래량  지정 직후 3개월 (표본 = 사건월 −W ~ 2)
    가격    지정 후 12개월 평균 (표본 = 사건월 −W ~ 11)

표준오차는 법정동 군집 관행값이다. [[0019]] 가 보인 대로 처치 동이 적은 사양에서는
이 p 가 너무 작다. **곡선은 점추정을 보는 그림이고, 구간은 참고로만 읽는다.**
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import econ, io, policy  # noqa: E402

io.setup_stdout()
for f in ("Malgun Gothic", "Batang"):
    if f in {x.name for x in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

E = policy.EVENTS["E2020"]["effect"]
SGG_T = ["강남구", "송파구"]
SGG_NEAR = ["강동구", "성동구", "광진구", "동작구"]
CONTAM = ["압구정동", "거여동", "마천동", "송파동", "신천동", "일원동"]   # 0020
WINDOWS = [12, 24, 36]
POST_V, POST_P = 3, 12       # 거래량 직후 3개월 · 가격 12개월
FLOOR_BINS = [0, 2, 5, 10, 15, 200]
FLOOR_LAB = ["1~2층", "3~5층", "6~10층", "11~15층", "16층 이상"]
FEATS = ["면적", "면적2"] + [x for x in FLOOR_LAB if x != "6~10층"]
Z = 1.96


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    la = np.log(d.area_m2)
    d["면적"], d["면적2"] = la, la ** 2
    fb = pd.cut(d.floor_no, FLOOR_BINS, labels=FLOOR_LAB)
    for lab in FLOOR_LAB:
        if lab != "6~10층":
            d[lab] = (fb == lab).astype(float)
    return d


def load() -> pd.DataFrame:
    """서울 전역 + 처치·통제 표식 + 사건월 두 가지 + 기저 ㎡당가."""
    d = pd.read_parquet(io.TRADES_SEOUL)
    d["em_eff"] = policy.event_month(d.deal_date, "E2020")           # 효력일 기준
    ym = d.deal_date.dt.year * 12 + d.deal_date.dt.month
    d["em_cal"] = ym - (E.year * 12 + E.month)                       # 달력 월
    pre = d[d.deal_date.between(E - pd.DateOffset(months=24), E)]
    d["기저"] = d.aptSeq.map(pre.groupby("aptSeq").price_per_m2.median())
    d["강남송파"] = d.sgg_nm.isin(SGG_T)
    d["인접구"] = d.sgg_nm.isin(SGG_NEAR)
    d["먼구"] = ~d.강남송파 & ~d.인접구
    return add_features(d)


TREAT = {"4개 동": policy.TREATED_2020, "잠실동만": {"잠실동"}}
CTRL_V = {
    "강남·송파 비처치": lambda d, t: d.강남송파 & ~d.umdNm.isin(t),
    "오염 6개 제외": lambda d, t: d.강남송파 & ~d.umdNm.isin(t) & ~d.umdNm.isin(CONTAM),
    "인접 4개 구": lambda d, t: d.인접구,
    "서울 19개 구": lambda d, t: d.먼구,
}
CTRL_P = {k: CTRL_V[k] for k in ("강남·송파 비처치", "오염 6개 제외", "서울 19개 구")}


# --- 거래량 -------------------------------------------------------------------
def vol_spec(d, tname, cname, W, align, fill, scale) -> tuple[float, float, int]:
    t = TREAT[tname]
    em = d["em_eff"] if align == "효력일" else d["em_cal"]
    keep = (d.umdNm.isin(t) & d.강남송파) | CTRL_V[cname](d, t)
    s = d[keep & em.between(-W, POST_V - 1)].copy()
    s["em"] = em[s.index]
    s["tr"] = s.umdNm.isin(t) & s.강남송파
    cnt = s.groupby(["umd_full_cd", "em"]).size().rename("n").reset_index()
    if fill == "0 채움":
        codes = sorted(s.umd_full_cd.unique())
        grid = pd.MultiIndex.from_product([codes, range(-W, POST_V)],
                                          names=["umd_full_cd", "em"]).to_frame(index=False)
        cnt = grid.merge(cnt, on=["umd_full_cd", "em"], how="left").fillna({"n": 0})
    cnt["tr"] = cnt.umd_full_cd.map(s.groupby("umd_full_cd").tr.first())
    cnt["d"] = (cnt.tr & (cnt.em >= 0)).astype(float)
    cnt["ems"] = cnt.em.astype(str)
    G = cnt.umd_full_cd.nunique()
    if scale == "PPML":
        r = econ.ppml(cnt.n.to_numpy(), cnt[["d"]].to_numpy(),
                      cnt.umd_full_cd.to_numpy(), cnt.em.to_numpy())
        return float(r["coef"][0]), float(r["se"][0]), G
    cnt["y"] = np.log1p(cnt.n)
    dm = econ.demean(cnt[["y", "d"]], cnt[["umd_full_cd", "ems"]])
    f = sm.OLS(dm.y, dm[["d"]]).fit(cov_type="cluster",
                                    cov_kwds={"groups": cnt.umd_full_cd})
    return float(f.params.d), float(f.bse.d), G


# --- 가격 ---------------------------------------------------------------------
def price_spec(d, tname, cname, W, fe, feat, band) -> tuple[float, float, int]:
    t = TREAT[tname]
    keep = (d.umdNm.isin(t) & d.강남송파) | CTRL_P[cname](d, t)
    s = d[keep & d.em_eff.between(-W, POST_P - 1)].copy()
    s["tr"] = s.umdNm.isin(t) & s.강남송파
    if band == "가격대 맞춤":
        q = s[s.tr].기저
        lo, hi = q.quantile(.05), q.quantile(.95)
        s = s[s.기저.between(lo, hi)]
    if s.tr.sum() < 50 or (~s.tr).sum() < 50:
        return np.nan, np.nan, 0
    s["d"] = (s.tr & (s.em_eff >= 0)).astype(float)
    s["ems"] = s.em_eff.astype(str)
    unit = "aptSeq" if fe == "단지" else "umd_full_cd"
    cols = ["d"] + (FEATS if feat == "면적·층" else [])
    dm = econ.demean(pd.concat([s[["log_price_per_m2"]], s[cols]], axis=1),
                     s[[unit, "ems"]])
    f = sm.OLS(dm.log_price_per_m2, dm[cols]).fit(
        cov_type="cluster", cov_kwds={"groups": s.umd_full_cd})
    return float(f.params.d), float(f.bse.d), int(s.umd_full_cd.nunique())


# --- 곡선 ---------------------------------------------------------------------
def curve(rows: list[dict], axes_names: list[str], title: str, ax_top, ax_bot,
          color: str) -> dict:
    r = pd.DataFrame(rows).dropna(subset=["coef"]).sort_values("coef")
    r["pct"] = 100 * np.expm1(r.coef)
    r["lo"] = 100 * np.expm1(r.coef - Z * r.se)
    r["hi"] = 100 * np.expm1(r.coef + Z * r.se)
    r = r.reset_index(drop=True)
    x = np.arange(len(r))
    sig = (r.hi < 0) | (r.lo > 0)
    ax_top.vlines(x, r.lo, r.hi, color=color, alpha=.22, lw=1.2)
    ax_top.scatter(x[sig], r.pct[sig], s=9, color=color, zorder=3, label="유의")
    ax_top.scatter(x[~sig], r.pct[~sig], s=9, facecolors="none",
                   edgecolors=color, zorder=3, label="비유의")
    ax_top.axhline(0, color="#1a202c", lw=1.0)
    ax_top.set_title(title)
    ax_top.set_ylabel("효과 (%)")
    ax_top.legend(fontsize=8, loc="lower right")
    ax_top.grid(alpha=.22)

    labels, ys = [], []
    for a in axes_names:
        for lvl in sorted(r[a].unique()):
            labels.append(f"{a}: {lvl}")
            ys.append((r[a] == lvl).to_numpy())
    for i, m in enumerate(ys):
        ax_bot.scatter(x[m], np.full(m.sum(), len(ys) - 1 - i), s=3,
                       color="#2d3748")
    ax_bot.set_yticks(range(len(ys)))
    ax_bot.set_yticklabels(labels[::-1], fontsize=7)
    ax_bot.set_xlabel("사양 (효과 크기 순)")
    ax_bot.set_xlim(ax_top.get_xlim())
    ax_bot.grid(alpha=.15, axis="x")
    return {"사양수": int(len(r)),
            "중위_pct": round(float(r.pct.median()), 1),
            "최소_pct": round(float(r.pct.min()), 1),
            "최대_pct": round(float(r.pct.max()), 1),
            "음수_비율": round(float(100 * (r.coef < 0).mean()), 1),
            "유의음수_비율": round(float(100 * ((r.hi < 0)).mean()), 1),
            "표": r.drop(columns=["coef", "se"]).round(2).to_dict("records")}


def main() -> None:
    d = load()
    out: dict = {}

    print("=" * 88)
    print("1. 거래량 — 지정 직후 3개월")
    print("=" * 88)
    vaxes = ["처치군", "통제군", "창", "사건정렬", "빈칸", "척도"]
    vrows = []
    for tn, cn, W, al, fi, sc in itertools.product(
            TREAT, CTRL_V, WINDOWS, ("효력일", "달력 월"),
            ("0 채움", "안 채움"), ("log1p", "PPML")):
        b, se, G = vol_spec(d, tn, cn, W, al, fi, sc)
        vrows.append({"처치군": tn, "통제군": cn, "창": f"±{W}개월", "사건정렬": al,
                      "빈칸": fi, "척도": sc, "coef": b, "se": se, "동": G})
    print(f"  {len(vrows)} 사양")

    print("\n" + "=" * 88)
    print("2. 가격 — 지정 후 12개월 평균")
    print("=" * 88)
    paxes = ["처치군", "통제군", "고정효과", "특성", "공통지지", "창"]
    prows = []
    for tn, cn, W, fe, ft, bd in itertools.product(
            TREAT, CTRL_P, WINDOWS, ("법정동", "단지"),
            ("없음", "면적·층"), ("전체", "가격대 맞춤")):
        b, se, G = price_spec(d, tn, cn, W, fe, ft, bd)
        prows.append({"처치군": tn, "통제군": cn, "창": f"±{W}개월", "고정효과": fe,
                      "특성": ft, "공통지지": bd, "coef": b, "se": se, "동": G})
    print(f"  {len(prows)} 사양")

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25], hspace=.06, wspace=.22)
    a1, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    b1, b2 = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    out["거래량"] = curve(vrows, vaxes, "거래량 — 지정 직후 3개월", a1, a2, "#c53030")
    out["가격"] = curve(prows, paxes, "가격 — 지정 후 12개월 평균", b1, b2, "#3182ce")

    for nm, s in (("거래량", out["거래량"]), ("가격", out["가격"])):
        print(f"\n  [{nm}] {s['사양수']} 사양 · 중위 {s['중위_pct']:+.1f}% · "
              f"범위 {s['최소_pct']:+.1f}% ~ {s['최대_pct']:+.1f}%")
        print(f"    음수 {s['음수_비율']}% · 유의하게 음수 {s['유의음수_비율']}%")

    # 어느 축이 결과를 가르나 — 축별 중위값의 폭
    print("\n" + "=" * 88)
    print("3. 어느 축이 결과를 가르나 (축을 한 값에 고정했을 때 중위값)")
    print("=" * 88)
    infl = {}
    for nm, rows, axes_ in (("거래량", vrows, vaxes), ("가격", prows, paxes)):
        r = pd.DataFrame(rows).dropna(subset=["coef"])
        r["pct"] = 100 * np.expm1(r.coef)
        print(f"\n  [{nm}]")
        items = []
        for a in axes_:
            m = r.groupby(a).pct.median()
            items.append((a, float(m.max() - m.min()), m))
        for a, spread, m in sorted(items, key=lambda x: -x[1]):
            detail = " · ".join(f"{k} {v:+.1f}%" for k, v in m.items())
            print(f"  {a:<8} 폭 {spread:>6.1f}%p   {detail}")
        infl[nm] = [{"축": a, "폭_pct포인트": round(sp, 1),
                     "값별_중위": {k: round(float(v), 1) for k, v in m.items()}}
                    for a, sp, m in sorted(items, key=lambda x: -x[1])]
    out["축별_영향"] = infl

    fig.suptitle("사양 곡선 — 합리적인 선택을 모두 조합하면 (2020-06-23 지정)", y=.94)
    fp = io.FIGURES / "05b_spec_curve.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    io.save_result("05b_spec_curve", out)
    print(f"\n저장 {fp} · output/results/05b_spec_curve.json")


if __name__ == "__main__":
    main()
