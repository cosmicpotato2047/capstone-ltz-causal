"""[09] 2025년 처치 역전 — 2월 해제와 3월 재지정.

2025-02-13 서울시는 국제교류복합지구 토허구역을 대부분 해제하면서 재건축
추진 단지 16개(공고 14개 항목, 결정기록 0008)만 남겼다. 39일 뒤인
2025-03-24 강남3구·용산 전 아파트를 다시 묶었다. 처치가 1 -> 0 -> 1 로
뒤집힌 드문 구간이다.

**주 분석은 재지정이다.** 원래 계획은 해제를 하이라이트로 두고 지역과
재건축 여부를 축으로 삼중차분하는 것이었으나, 해제 구간에는 움직이지 않은
비교군이 없다. 규제와 무관한 4개 구도 같은 기간 3.3배 올랐다. 반면 재지정
때는 변하지 않은 집단이 둘 있다 (결정기록 0023).

    유지군      16개 단지. 2020년부터 계속 규제. 재지정으로 바뀐 것이 없다
    비처치 4개구 강동·성동·광진·동작. 2025-10-20 까지 지정되지 않는다

세 번째 축은 재건축 여부가 아니라 **전용면적**으로 둔다. 유지군 16개에는
40㎡ 미만 거래가 0건이라 재건축 축을 세울 수 없고, 면적은 0022 가 세워 둔
사전 예측을 검정하기 때문이다.

    2020-06-23 지정은 주거지역 대지 18㎡ 초과만 허가 대상이라 소형이 빠져나갔다
    2025-03-24 재지정은 6㎡ 초과라 소형도 허가 대상이다 -> 소형도 같이 줄어야 한다

추정량은 포아송 유사최대우도(PPML)다. 해제군 단지의 거래 중위가 2년에 2건,
유지군은 28건이라 log1p 선형 모형은 작은 단지에 끌려간다. 두 값을 함께 싣는다.
클러스터는 법정동이 아니라 단지라 이번에는 적은 클러스터 문제가 없다(0019).

    python scripts/09_release_2025.py
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
from lib import econ, io, policy  # noqa: E402
from lib.inference import SEED, Spec, fit, twoway, wild_boot  # noqa: E402

io.setup_stdout()
for f in ("Malgun Gothic", "Batang"):
    if f in {x.name for x in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

RELEASE = pd.Timestamp("2025-02-13")    # 부분 해제 효력
ANNOUNCE = pd.Timestamp("2025-03-19")   # 재지정 발표
REDESIG = pd.Timestamp("2025-03-24")    # 재지정 효력
E2020 = pd.Timestamp("2020-06-23")      # 최초 지정 효력

SMALL = 40.0            # 소형 기준 전용면적 (0021·0022 와 같다)
POST_W = 5              # 사후 5주 = 35일
POST_LONG = 12          # 강건성. 2025-06-27 대출규제 전에 끝난다
PRE_NEAR = 5            # 기저 (가) 직전 5주 = 해제창. 과열 상태다
PRE_NORM = 13           # 기저 (나) 해제 전 13주 = 정상
GAP = 6                 # 재지정 기준 -6 주부터가 해제창이다
CTRL_SGG = ["강동구", "성동구", "광진구", "동작구"]
B_09 = 4_999
N_FAKE = 2_000          # 가짜 유지군 무작위 배정 횟수
OLD_YEAR = 1990         # 유지군 16개는 모두 1978~1986년 준공


# --- 집단 만들기 ---------------------------------------------------------------
def designated(at: pd.Timestamp) -> set[str]:
    """그 날짜에 이미 토허구역이던 법정동 (일부만 지정된 동도 포함)."""
    sp = pd.read_csv(io.POLICY / "ltz_spells.csv", encoding="utf-8-sig")
    sp["시작일"] = pd.to_datetime(sp.시작일)
    sp["종료일"] = pd.to_datetime(sp.종료일)
    m = (sp.상태 == "지정") & (sp.시작일 <= at) & (sp.종료일 >= at)
    return set(sp[m].법정동)


def make_groups(d: pd.DataFrame, at: pd.Timestamp) -> pd.DataFrame:
    """at 시점의 규제 상태로 다섯 집단을 나눈다."""
    already = designated(at) - policy.TREATED_2020
    d = d.copy()
    d["kept"] = policy.assign_kept_2025(d)
    in_tr = d.umdNm.isin(policy.TREATED_2020)
    gangnam = d.sgg_nm.isin(["강남구", "송파구"])
    g = pd.Series("제외", index=d.index)
    g[gangnam & ~in_tr] = "강남송파기타"
    g[in_tr] = "해제군"
    g[d.kept] = "유지군"
    g[d.sgg_nm.isin(CTRL_SGG)] = "통제구"
    g[d.umdNm.isin(already)] = "제외"      # 다른 계열로 이미 지정된 동
    g[d.kept] = "유지군"                   # 유지군은 동 단위 제외에 걸리지 않는다
    d["그룹"] = g
    d["소형"] = d.area_m2 < SMALL
    return d


def wk_of(dates: pd.Series, anchor: pd.Timestamp) -> pd.Series:
    """효력일 기준 주. 0 = 효력일이 든 7일."""
    return np.floor((pd.to_datetime(dates) - anchor).dt.days / 7).astype(int)


def panel(sub: pd.DataFrame, anchor: pd.Timestamp, weeks: list[int],
          cellcols: list[str]) -> pd.DataFrame:
    """칸 × 주 균형 격자. 칸은 창 안에 거래가 한 번이라도 있는 조합만 만든다."""
    s = sub.copy()
    s["wk"] = wk_of(s.deal_date, anchor)
    w = s[s.wk.isin(weeks)]
    cells = w.groupby(cellcols, as_index=False).size().drop(columns="size")
    full = cells.merge(pd.DataFrame({"wk": weeks}), how="cross")
    cnt = w.groupby(cellcols + ["wk"]).size().rename("n").reset_index()
    p = full.merge(cnt, on=cellcols + ["wk"], how="left").fillna({"n": 0})
    p["cell"] = p[cellcols].astype(str).agg("|".join, axis=1)
    return p.sort_values(["cell", "wk"]).reset_index(drop=True)


# --- 추정 ---------------------------------------------------------------------
def log1p_wcr(p: pd.DataFrame, rng) -> tuple[float, float]:
    """같은 격자에 log1p 선형 이중차분. 적은 클러스터 부트스트랩 p 를 함께."""
    cells = p.cell.unique()
    T = p.wk.nunique()
    G = len(cells)
    tr = p.groupby("cell").d_treat.first().reindex(cells).to_numpy(bool)
    post = (p.wk.to_numpy()[:T] >= 0).astype(float)
    s = Spec("log1p", np.log1p(p.n.to_numpy(float)),
             lambda t: np.outer(t, post).reshape(1, -1).astype(float),
             twoway([(G, T)]), np.repeat(np.arange(G), T), G, tr, 0,
             lambda b: b[0], [(G, T)])
    b, *_ = fit(s, s.make(s.treated))
    _, _, _, p2, _ = wild_boot(s, rng, B=B_09)
    return float(100 * np.expm1(b[0])), float(p2)


def did_ppml(sub, anchor, weeks, treat_groups, rng=None) -> dict:
    """PPML 이중차분. 처치 = treat_groups 에 든 집단."""
    p = panel(sub, anchor, weeks, ["aptSeq", "그룹"])
    p["d_treat"] = p.그룹.isin(treat_groups)
    p["d"] = p.d_treat.astype(float) * (p.wk >= 0)
    r = econ.ppml(p.n.to_numpy(), p[["d"]].to_numpy(),
                  p.aptSeq.to_numpy(), p.wk.to_numpy())
    out = {"PPML_pct": round(float(100 * np.expm1(r["coef"][0])), 1),
           "PPML_p": round(float(r["p"][0]), 4),
           "단지": int(p.aptSeq.nunique()),
           "처치단지": int(p[p.d_treat].aptSeq.nunique()),
           "거래": int(p.n.sum())}
    if rng is not None:
        e, pw = log1p_wcr(p, rng)
        out["log1p_pct"] = round(e, 1)
        out["log1p_WCR_p"] = round(pw, 4)
    return out


def ddd_size(sub, anchor, weeks, treat_groups) -> dict:
    """크기 삼중차분. 칸 = (단지, 소형), 시간 고정효과 = (주, 소형)."""
    p = panel(sub, anchor, weeks, ["aptSeq", "그룹", "소형"])
    p["d_treat"] = p.그룹.isin(treat_groups)
    post = (p.wk >= 0).astype(float)
    X = np.column_stack([p.d_treat.astype(float) * post,
                         p.d_treat.astype(float) * post * p.소형.astype(float)])
    tcode = p.wk.astype(str) + "|" + p.소형.astype(str)
    r = econ.ppml(p.n.to_numpy(), X, p.cell.to_numpy(), tcode.to_numpy(),
                  cluster=p.aptSeq.to_numpy())
    b = r["coef"]
    w = p[p.d_treat]
    return {"대형_pct": round(float(100 * np.expm1(b[0])), 1),
            "소형_pct": round(float(100 * np.expm1(b[0] + b[1])), 1),
            "차이_계수": round(float(b[1]), 4),
            "차이_p": round(float(r["p"][1]), 4),
            "처치_소형_거래": int(w[w.소형].n.sum()),
            "처치_대형_거래": int(w[~w.소형].n.sum())}


def ratio(sub: pd.DataFrame, anchor: pd.Timestamp,
          pre: list[int], post: list[int]) -> float:
    """하루 평균 거래의 사후/사전 배수."""
    s = sub.assign(wk=wk_of(sub.deal_date, anchor))
    a = (s.wk.isin(pre)).sum() / len(pre)
    b = (s.wk.isin(post)).sum() / len(post)
    return float(b / a) if a else float("nan")


def main() -> None:
    base = io.load_trades()
    d = make_groups(base, REDESIG - pd.Timedelta(days=1))
    d = d[d.그룹 != "제외"]
    new_reg = ["해제군", "강남송파기타"]       # 재지정으로 새로 규제된 집단

    near = list(range(-PRE_NEAR, 0)) + list(range(0, POST_W))
    norm = list(range(-PRE_NORM - GAP, -GAP)) + list(range(0, POST_W))
    longw = list(range(-PRE_NORM - GAP, -GAP)) + list(range(0, POST_LONG))

    # ==================================================================
    print("=" * 86)
    print("집단 (2025-03-23 기준 규제 상태)")
    print("=" * 86)
    tab = (d[d.deal_date >= "2024-01-01"]
           .groupby("그룹").agg(법정동=("umdNm", "nunique"),
                                단지=("aptSeq", "nunique"),
                                거래_2024이후=("aptSeq", "size")))
    print(tab.to_string())
    print("  해제군 = 처치 4개 동의 유지군 아닌 단지 · 강남송파기타 = 두 구의 나머지 법정동")
    print("  통제구 = 강동·성동·광진·동작 (2025-10-20 까지 미지정)")
    print("  제외 = 다른 계열로 이미 지정돼 있던 동 (0009·0020)")
    out = {"집단": tab.reset_index().to_dict("records")}

    # ==================================================================
    print("\n" + "=" * 86)
    print("1. 원자료 — 주별 거래 건수 (0주 = 해제 2025-02-13)")
    print("=" * 86)
    w = d.assign(wk=wk_of(d.deal_date, RELEASE))
    wt = (w[w.wk.between(-13, 19)].groupby(["wk", "그룹"]).size()
          .unstack(fill_value=0))
    wt.insert(0, "주시작", [str((RELEASE + pd.Timedelta(days=7 * k)).date())
                            for k in wt.index])
    print(wt.to_string())
    print("  5주(3/20~3/26)에 막차, 6주부터 재지정 효력. 11주(5/1)부터 회복이 시작된다")
    out["주별_거래"] = wt.reset_index().to_dict("records")

    # ==================================================================
    print("\n" + "=" * 86)
    print("2. 재지정 (2025-03-24) — 주 결과")
    print("=" * 86)
    print(f"  {'처치':<14}{'통제':<14}{'기저':<12}{'PPML':>9}{'p':>8}"
          f"{'log1p':>9}{'WCR p':>8}{'처치단지':>7}")
    rows = []
    combos = [("해제군", ["해제군"], "유지군", ["유지군"]),
              ("해제군", ["해제군"], "통제구", ["통제구"]),
              ("강남송파 전체", new_reg, "유지군", ["유지군"]),
              ("강남송파 전체", new_reg, "통제구", ["통제구"])]
    for tlab, tg, clab, cg in combos:
        for blab, weeks in (("해제창 5주", near), ("정상 13주", norm),
                            ("정상+사후12주", longw)):
            rng = np.random.default_rng(SEED)
            sub = d[d.그룹.isin(tg + cg)]
            r = did_ppml(sub, REDESIG, weeks, tg, rng)
            rows.append({"처치": tlab, "통제": clab, "기저": blab, **r})
            star = "  <-주" if (blab == "정상 13주" and clab == "유지군"
                                and tlab == "해제군") else ""
            print(f"  {tlab:<14}{clab:<14}{blab:<12}{r['PPML_pct']:>8.1f}%"
                  f"{r['PPML_p']:>8.3f}{r['log1p_pct']:>8.1f}%"
                  f"{r['log1p_WCR_p']:>8.3f}{r['처치단지']:>7d}{star}")
    print("  사후 12주로 늘리면 효과가 크게 줄어든다. 충격이 사라졌기 때문이다 —")
    print("  거래는 6주쯤 얼어 있다가 5월부터 돌아온다. '지속'이 아니라 '직후'의 숫자다")
    out["재지정_효과"] = rows

    # ==================================================================
    print("\n" + "=" * 86)
    print("3. 0022 의 사전 예측 — 허가 면적 기준이 바뀌면 소형도 줄어야 한다")
    print("=" * 86)
    print(f"  {'사건':<26}{'허가대상':<16}{'40㎡ 이상':>10}{'40㎡ 미만':>10}"
          f"{'차이':>9}{'p':>8}")
    size_rows = []
    # 2020 지정 — 같은 통제군(비처치 4개구)으로 같은 사양을 건다
    d20 = make_groups(base, E2020 - pd.Timedelta(days=1))
    d20 = d20[d20.그룹 != "제외"]
    d20.loc[d20.그룹 == "유지군", "그룹"] = "해제군"   # 2020 에는 아직 유지군이 없다
    w20 = list(range(-PRE_NORM, 0)) + list(range(0, POST_W))
    for lab, sub, anchor, weeks, tg, rule in (
            ("2020-06-23 지정", d20[d20.그룹.isin(["해제군", "통제구"])],
             E2020, w20, ["해제군"], "대지 18㎡ 초과"),
            ("2025-03-24 재지정", d[d.그룹.isin(new_reg + ["통제구"])],
             REDESIG, norm, new_reg, "대지 6㎡ 초과")):
        r = ddd_size(sub, anchor, weeks, tg)
        size_rows.append({"사건": lab, "허가대상": rule, **r})
        print(f"  {lab:<26}{rule:<16}{r['대형_pct']:>9.1f}%{r['소형_pct']:>9.1f}%"
              f"{r['차이_계수']:>+9.3f}{r['차이_p']:>8.3f}")
        print(f"  {'':<42}(처치 거래 대형 {r['처치_대형_거래']}건 · "
              f"소형 {r['처치_소형_거래']}건)")
    out["크기_삼중차분"] = size_rows
    print("  차이 계수 > 0 = 소형이 덜 줄었다. 2020 에는 크고 2025 에는 0 이어야 한다")

    # ==================================================================
    print("\n" + "=" * 86)
    print("4. 막차 — 재지정 발표(3/19)와 효력(3/24) 사이 나흘")
    print("=" * 86)
    rush = []
    for g in ["해제군", "강남송파기타", "유지군", "통제구"]:
        s = d[d.그룹 == g]
        a = s.deal_date.between("2025-03-15", "2025-03-18").sum()
        b = s.deal_date.between("2025-03-20", "2025-03-23").sum()
        c = s.deal_date.between("2025-03-24", "2025-03-27").sum()
        rush.append({"집단": g, "발표전_나흘": int(a), "발표후_나흘": int(b),
                     "효력후_나흘": int(c),
                     "막차배수": round(b / a, 2) if a else None})
        print(f"  {g:<14} 3/15~18 {a:>4}건 -> 3/20~23 {b:>4}건 "
              f"-> 3/24~27 {c:>4}건")
    out["막차"] = rush

    # ==================================================================
    print("\n" + "=" * 86)
    print("5. 해제 (2025-02-13) — 방향만. 크기는 단정하지 않는다")
    print("=" * 86)
    pre13 = list(range(-13, 0))
    post5 = list(range(0, 5))
    print(f"  {'집단':<14}{'규제 변화':<14}{'배수':>7}")
    rel = []
    for g, chg in (("해제군", "규제 -> 무규제"), ("유지군", "없음 (규제)"),
                   ("강남송파기타", "없음 (무규제)"), ("통제구", "없음 (무규제)")):
        rt = ratio(d[d.그룹 == g], RELEASE, pre13, post5)
        rel.append({"집단": g, "변화": chg, "배수": round(rt, 2)})
        print(f"  {g:<14}{chg:<14}{rt:>6.2f}배")
    print("  규제와 무관한 통제구도 3배 넘게 올랐다 -> 해제 고유 효과를 분리할 수 없다")
    print(f"\n  {'통제군':<14}{'PPML':>9}{'p':>8}")
    for clab in ("유지군", "강남송파기타", "통제구"):
        sub = d[d.그룹.isin(["해제군", clab])]
        r = did_ppml(sub, RELEASE, pre13 + post5, ["해제군"])
        rel.append({"통제군": clab, **r})
        print(f"  {clab:<14}{r['PPML_pct']:>8.1f}%{r['PPML_p']:>8.3f}")
    out["해제_참고"] = rel

    # ==================================================================
    print("\n" + "=" * 86)
    print("6. 위약")
    print("=" * 86)
    print("  (가) 달력 위약 — 같은 사양을 앞선 해의 3월 24일에 건다")
    cal = []
    for y in (2021, 2022, 2023, 2024, 2025):
        a = pd.Timestamp(f"{y}-03-24")
        sub = d[d.그룹.isin(["해제군", "유지군"])]
        r = did_ppml(sub, a, norm, ["해제군"])
        cal.append({"연도": y, **r})
        print(f"    {y}  {r['PPML_pct']:>8.1f}%  p={r['PPML_p']:.3f}"
              + ("   <- 실제" if y == 2025 else ""))
    print("    2021·2022 는 그 해에 실제 정책이 있어 0 이 나올 이유가 없다"
          " (2·4 대책, 금리 인상)")
    out["위약_달력"] = cal

    print(f"\n  (나) 통제군 바꿔치기 — 유지군 16개 대신 통제구에서 16개를"
          f" {N_FAKE:,}번 무작위로 뽑아 통제군으로 쓴다")
    print("      유지군은 무작위로 정해진 집단이 아니라 서울시가 재건축 추진을 보고")
    print("      고른 단지다. 다른 통제군을 썼다면 답이 달랐을지 본다.")
    print(f"      {'기증자 풀':<28}{'단지':>5}{'가짜 중위':>10}{'2.5%':>9}{'97.5%':>9}"
          f"{'실제 위치':>9}")
    win = d.assign(wk=wk_of(d.deal_date, REDESIG))
    win = win[win.wk.isin(norm)]
    rt = ratio(d[d.그룹 == "해제군"], REDESIG, norm[:PRE_NORM], norm[PRE_NORM:])
    act = 100 * (rt / ratio(d[d.그룹 == "유지군"], REDESIG,
                            norm[:PRE_NORM], norm[PRE_NORM:]) - 1)
    swap = []
    for plab, mask, need in (("통제구 전체", pd.Series(True, index=win.index), 5),
                             (f"통제구 {OLD_YEAR}년 이전 준공",
                              win.build_year <= OLD_YEAR, 3)):
        rng = np.random.default_rng(SEED)
        cnt = win[(win.그룹 == "통제구") & mask].groupby("aptSeq").size()
        donors = cnt[cnt >= need].index.to_numpy()
        if len(donors) < 20:
            print(f"      {plab:<28}{len(donors):>5}   기증자가 적어 건너뛴다")
            continue
        pool = d[d.aptSeq.isin(donors)]
        vals = np.empty(N_FAKE)
        for i in range(N_FAKE):
            pick = rng.choice(donors, size=16, replace=False)
            rc = ratio(pool[pool.aptSeq.isin(pick)], REDESIG,
                       norm[:PRE_NORM], norm[PRE_NORM:])
            vals[i] = 100 * (rt / rc - 1)
        vals = vals[np.isfinite(vals)]
        lo, hi = np.percentile(vals, [2.5, 97.5])
        pct = float((vals <= act).mean())
        swap.append({"기증자_풀": plab, "기증자_단지": int(len(donors)),
                     "가짜_중위_pct": round(float(np.median(vals)), 1),
                     "가짜_2.5_pct": round(float(lo), 1),
                     "가짜_97.5_pct": round(float(hi), 1),
                     "실제_pct": round(act, 1), "실제_백분위": round(pct, 3)})
        print(f"      {plab:<28}{len(donors):>5}{np.median(vals):>9.1f}%"
              f"{lo:>8.1f}%{hi:>8.1f}%{pct*100:>8.0f}%")
    print(f"      실제(유지군) {act:.1f}%. 가짜 통제군은 더 큰 감소를 낸다 —")
    print("      유지군을 쓰는 쪽이 보수적이다. 통제군 선택이 결론을 만들지 않았다")
    out["통제군_바꿔치기"] = swap

    # ==================================================================
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.8))
    ax = axes[0]
    base13 = wt.loc[-13:-1]
    for g, col in (("해제군", "#c53030"), ("유지군", "#2b6cb0"),
                   ("강남송파기타", "#718096"), ("통제구", "#38a169")):
        if g not in wt.columns:
            continue
        ax.plot(wt.index, 100 * wt[g] / base13[g].mean(), "o-", color=col,
                lw=1.6, ms=3.5, label=g)
    ax.axvline(-0.5, color="#c53030", ls="--", lw=1.2)
    ax.axvline(5.5, color="#1a202c", ls="--", lw=1.2)
    ax.text(-0.4, ax.get_ylim()[1] * .95, "해제 2/13", fontsize=8, color="#c53030")
    ax.text(5.6, ax.get_ylim()[1] * .95, "재지정 3/24", fontsize=8, color="#1a202c")
    ax.axhline(100, color="#999", lw=.8)
    ax.set_title("주별 거래 (해제 전 13주 평균 = 100)")
    ax.set_xlabel("해제일 기준 주")
    ax.set_ylabel("지수")
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)

    ax = axes[1]
    days = pd.date_range("2025-03-10", "2025-04-06")
    for g, col in (("해제군", "#c53030"), ("유지군", "#2b6cb0")):
        s = d[d.그룹 == g]
        c = s.groupby(s.deal_date.dt.normalize()).size().reindex(days, fill_value=0)
        ax.plot(days, c.values, "o-", color=col, lw=1.5, ms=3.5, label=g)
    ax.axvline(ANNOUNCE, color="#dd6b20", ls="--", lw=1.2)
    ax.axvline(REDESIG, color="#1a202c", ls="--", lw=1.2)
    ax.annotate("발표 3/19", xy=(ANNOUNCE, ax.get_ylim()[1] * .92),
                xytext=(-4, 0), textcoords="offset points", ha="right",
                fontsize=8, color="#dd6b20")
    ax.annotate("효력 3/24", xy=(REDESIG, ax.get_ylim()[1] * .74),
                xytext=(4, 0), textcoords="offset points", fontsize=8)
    ax.set_title("발표와 효력 사이 나흘 — 막차와 절벽")
    ax.set_ylabel("하루 거래 건수")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)

    ax = axes[2]
    lab = [r["사건"] for r in size_rows]
    xx = np.arange(2)
    ax.bar(xx - .18, [r["대형_pct"] for r in size_rows], .34,
           color="#718096", label=f"{SMALL:.0f}㎡ 이상")
    ax.bar(xx + .18, [r["소형_pct"] for r in size_rows], .34,
           color="#c53030", label=f"{SMALL:.0f}㎡ 미만")
    for i, r in enumerate(size_rows):
        ax.text(i - .18, r["대형_pct"], f"{r['대형_pct']:.0f}%", ha="center",
                va="top", fontsize=9)
        ax.text(i + .18, r["소형_pct"], f"{r['소형_pct']:.0f}%", ha="center",
                va="top", fontsize=9)
    ax.axhline(0, color="#333", lw=.8)
    ax.set_xticks(xx)
    ax.set_xticklabels(["2020 지정\n(허가 대지 18㎡ 초과)",
                        "2025 재지정\n(허가 대지 6㎡ 초과)"], fontsize=9)
    ax.set_ylabel("통제구 대비 거래 변화 (%)")
    ax.set_title("면적 기준이 바뀌자 소형도 같이 줄었다")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=.25, axis="y")

    fig.suptitle("2025년 처치 역전 — 해제(2/13)와 재지정(3/24)", y=1.03)
    fig.tight_layout()
    fp = io.FIGURES / "09_release_2025.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")

    out.update({"해제일": str(RELEASE.date()), "발표일": str(ANNOUNCE.date()),
                "재지정일": str(REDESIG.date()), "소형기준_㎡": SMALL,
                "사후_주": POST_W, "B": B_09, "seed": SEED})
    io.save_result("09_release_2025", out)
    print(f"\n저장 {fp} · output/results/09_release_2025.json")


if __name__ == "__main__":
    main()
