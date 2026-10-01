"""[18] 취소거래의 시점별 분포 — 규제 직후에 몰리는가.

    python scripts/18_cancellations.py

[[0004-표본-정제-규칙]] 이 `cdealType='O'` 인 해제(취소) 거래를 표본에서
빼면서 숙제를 남겼다. "취소거래의 시점별 분포 자체가 별도 분석 가치가 있다.
규제 직후에 몰린다면 그 자체가 정책 반응이다." 그 숙제가 백로그 18 이다.

**원자료를 다시 읽는다.** 정제 표(trades.parquet)는 취소를 이미 빼 버렸으므로
여기서는 02_build 의 정제를 쓰되 **취소 필터만 끄고** 읽는다.

**취소일(cdealDay)이 있다.** 0004 를 쓸 때는 취소 여부만 본다고 생각했는데
원자료에 취소 일자가 'YY.MM.DD' 로 들어 있다. 그래서 계약일 기준 분포와
**취소일 기준 분포**를 따로 볼 수 있고, 계약에서 취소까지의 지연도 잴 수 있다.

**관측 구간이 2020-02 부터다.** 그 전은 취소가 없었던 것이 아니라 자료에
없다. 2020-06 지정 사건의 사전 구간이 다섯 달뿐이므로 지정 사건의 전후
비교는 약하고, 2025 년 두 사건(해제·재지정)은 양쪽이 다 덮인다.
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
from lib import econ, io, policy  # noqa: E402

for f in ("Malgun Gothic", "Batang"):
    if f in {x.name for x in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

importlib_build = Path(__file__).resolve().parent / "02_build.py"
io.setup_stdout()

E2020 = policy.EVENTS["E2020"]["effect"]
E2025R = policy.EVENTS["E2025FEB"]["effect"]      # 2025-02-13 부분 해제
E2025E = policy.EVENTS["E2025MAR"]["effect"]      # 2025-03-24 확대 재지정
FLAG_START = pd.Timestamp("2020-02-01")          # 취소 플래그가 처음 나타난 달


def load_with_cancelled() -> pd.DataFrame:
    """02_build 의 정제를 그대로 쓰되 취소 필터만 끈다."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("build2", importlib_build)
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)

    raw = b.load_raw(policy.ALL_SGG)
    raw["cancelled"] = raw["cdealType"].fillna("").str.strip().eq("O")
    # 취소일. 'YY.MM.DD' 로 들어온다. 취소가 아닌 행은 비어 있다.
    raw["cancel_date"] = pd.to_datetime(
        raw["cdealDay"].astype(str).str.strip().replace({"": None, "nan": None}),
        format="%y.%m.%d", errors="coerce")
    # 취소 필터를 무력화하고, 플래그는 extra_keep 으로 들고 나온다.
    # build() 가 정렬·재색인을 하므로 바깥에서 색인으로 되붙일 수 없다.
    raw["cdealType"] = ""
    df, _ = b.build(raw, extra_keep=("cancelled", "cancel_date"))
    df["cancelled"] = df.cancelled.fillna(False).astype(bool)
    df["treated"] = policy.assign_2020(df)
    return df


def rate(d: pd.DataFrame) -> float:
    return float(100 * d.cancelled.mean()) if len(d) else float("nan")


def did_rate(df: pd.DataFrame, event: pd.Timestamp, months: int,
             treated_col: str = "treated") -> dict:
    """취소율의 전후·처치/통제 2x2. 비율이므로 %p 로 적는다."""
    w = df[(df.deal_date >= event - pd.DateOffset(months=months))
           & (df.deal_date < event + pd.DateOffset(months=months))].copy()
    w["post"] = w.deal_date >= event
    g = w.groupby([treated_col, "post"]).agg(
        n=("cancelled", "size"), r=("cancelled", lambda s: 100 * s.mean()))
    try:
        t = g.loc[(True, True), "r"] - g.loc[(True, False), "r"]
        c = g.loc[(False, True), "r"] - g.loc[(False, False), "r"]
    except KeyError:
        return {}
    return {
        "창_개월": months,
        "처치_전": round(float(g.loc[(True, False), "r"]), 2),
        "처치_후": round(float(g.loc[(True, True), "r"]), 2),
        "통제_전": round(float(g.loc[(False, False), "r"]), 2),
        "통제_후": round(float(g.loc[(False, True), "r"]), 2),
        "처치_변화pp": round(float(t), 2),
        "통제_변화pp": round(float(c), 2),
        "DID_pp": round(float(t - c), 2),
        "표본": {f"{k[0]}_{k[1]}": int(v) for k, v in g.n.items()},
    }


def main() -> None:
    df = load_with_cancelled()
    obs = df[df.deal_date >= FLAG_START]
    out: dict = {}

    print("=" * 78)
    print("취소거래 — 무엇을 볼 수 있고 무엇을 볼 수 없는가")
    print("=" * 78)
    first = df[df.cancelled].deal_date.min()
    print(f"  전체 표본 {len(df):,}건 중 취소 {int(df.cancelled.sum()):,}건 "
          f"({rate(df):.2f}%)")
    print(f"  취소 플래그가 처음 나타나는 계약일: {first.date()}")
    print(f"  관측 구간 표본 {len(obs):,}건 중 취소 {int(obs.cancelled.sum()):,}건 "
          f"({rate(obs):.2f}%)")
    print(f"  -> 2020-02 이전은 취소가 없었던 것이 아니라 자료에 없다.")
    out["관측"] = {"전체": len(df), "전체_취소": int(df.cancelled.sum()),
                  "전체_취소율": round(rate(df), 3),
                  "플래그_첫_계약일": str(first.date()),
                  "관측구간": len(obs), "관측구간_취소": int(obs.cancelled.sum()),
                  "관측구간_취소율": round(rate(obs), 3)}

    # --- 1. 계약에서 취소까지 ---
    print("\n" + "=" * 78)
    print("1) 계약에서 취소까지 얼마나 걸리는가")
    print("=" * 78)
    c = obs[obs.cancelled & obs.cancel_date.notna()].copy()
    c["lag_d"] = (c.cancel_date - c.deal_date).dt.days
    ok = c[c.lag_d.between(0, 1000)]
    q = ok.lag_d.quantile([0.25, 0.5, 0.75, 0.9]).round(0)
    print(f"  취소일을 읽은 건수 {len(c):,} / 취소 {int(obs.cancelled.sum()):,}")
    print(f"  지연 일수  중위 {q[0.5]:.0f}일 · 사분위 {q[0.25]:.0f}~{q[0.75]:.0f}일 "
          f"· 상위 10% {q[0.9]:.0f}일")
    print(f"  30일 안에 취소된 비율 {100 * (ok.lag_d <= 30).mean():.1f}%")
    out["지연"] = {"건수": int(len(c)), "중위일": float(q[0.5]),
                  "p25": float(q[0.25]), "p75": float(q[0.75]),
                  "p90": float(q[0.9]),
                  "30일내_비율": round(float(100 * (ok.lag_d <= 30).mean()), 1)}

    # --- 2. 지정 직후에 몰리는가 (세 사건) ---
    print("\n" + "=" * 78)
    print("2) 규제 직후에 몰리는가 — 취소율의 이중차분 (%p)")
    print("=" * 78)
    print("  처치군 = 2020.6 지정 4개 법정동, 통제군 = 강동·성동·광진·동작")
    print(f"  {'사건':<22}{'창':>5}{'처치 전':>9}{'후':>8}{'통제 전':>9}{'후':>8}{'DID':>9}")
    ev = {}
    for name, e in (("2020-06-23 지정", E2020),
                    ("2025-02-13 부분해제", E2025R),
                    ("2025-03-24 재지정", E2025E)):
        rows = []
        for m in (3, 6, 12):
            # 2020 사건은 사전이 2020-02 부터뿐이라 3개월 창만 뜻이 있다
            if e == E2020 and m > 6:
                continue
            r = did_rate(df[df.deal_date >= FLAG_START], e, m)
            if not r:
                continue
            rows.append(r)
            print(f"  {name:<22}{m:>4}월{r['처치_전']:>9.2f}{r['처치_후']:>8.2f}"
                  f"{r['통제_전']:>9.2f}{r['통제_후']:>8.2f}{r['DID_pp']:>+9.2f}")
        ev[name] = rows
    out["이중차분"] = ev

    # --- 3. 취소일 기준 분포 — 막차 계약이 취소됐나 ---
    print("\n" + "=" * 78)
    print("3) 2020 년 막차 계약은 취소됐는가")
    print("=" * 78)
    # 공고(2020-06-18)부터 효력(06-23) 사이 닷새가 '막차' 구간이다
    rush = df[(df.deal_date >= "2020-06-18") & (df.deal_date < "2020-06-23")]
    before = df[(df.deal_date >= "2020-05-18") & (df.deal_date < "2020-06-18")]
    after = df[(df.deal_date >= "2020-06-23") & (df.deal_date < "2020-07-23")]
    print(f"  {'구간':<26}{'거래':>8}{'취소':>7}{'취소율':>9}")
    seg = {}
    for nm, d in (("막차 (06-18 ~ 06-22)", rush),
                  ("그 직전 한 달", before), ("지정 후 한 달", after)):
        for who, dd in (("처치", d[d.treated]), ("통제", d[~d.treated])):
            print(f"  {nm + ' · ' + who:<26}{len(dd):>8,}{int(dd.cancelled.sum()):>7,}"
                  f"{rate(dd):>8.2f}%")
            seg[f"{nm}|{who}"] = {"거래": len(dd), "취소": int(dd.cancelled.sum()),
                                  "취소율": round(rate(dd), 2)}
    out["막차"] = seg

    # --- 4. 취소된 거래는 어떤 거래인가 ---
    print("\n" + "=" * 78)
    print("4) 취소된 거래는 비싼 거래인가 (허위신고 가설)")
    print("=" * 78)
    print("  단지·계약월을 고정하고 취소 여부에 대한 log ㎡당가 회귀")
    w = obs[obs.deal_date < policy.IDENTIFICATION_END].copy()
    w["ym_s"] = w.ym.astype(str)
    w["is_c"] = w.cancelled.astype(float)
    dm = econ.demean(w[["log_price_per_m2", "is_c"]], w[["aptSeq", "ym_s"]])
    r = sm.OLS(dm.log_price_per_m2, dm[["is_c"]]).fit(
        cov_type="cluster", cov_kwds={"groups": w.umd_full_cd})
    b, se, p = float(r.params.is_c), float(r.bse.is_c), float(r.pvalues.is_c)
    print(f"  취소 더미 계수 {b:+.4f} (표준오차 {se:.4f}, p={p:.3f}) "
          f"-> {100 * (np.exp(b) - 1):+.1f}%")
    print(f"  같은 단지·같은 달의 정상 거래보다 "
          f"{'비싸다' if b > 0 else '싸다'}")
    out["가격"] = {"계수": round(b, 4), "표준오차": round(se, 4),
                  "p": round(p, 4), "효과pct": round(100 * (np.exp(b) - 1), 2),
                  "표본": len(w)}

    # --- 5. 주 결과가 달라지는가 ---
    print("\n" + "=" * 78)
    print("5) 취소를 넣으면 주 결과가 달라지는가 (0004 한계의 크기)")
    print("=" * 78)
    print("  2020.6 지정 거래량 사건연구. 04_event_study.py 와 같은 사양")
    es = {}
    for label, d in (("취소 제외 (현 사양)", df[~df.cancelled]),
                     ("취소 포함", df)):
        # 04_event_study.py 와 같은 표본이어야 비교가 된다 — 강남·송파 안에서만
        x = d[d.sgg_nm.isin(["강남구", "송파구"])].copy()
        x["kq"] = policy.event_quarter(x.deal_date, "E2020")
        x = x[(x.kq >= -8) & (x.kq <= 8)]
        x["em"] = policy.event_month(x.deal_date, "E2020")
        x["treated"] = policy.assign_2020(x)
        pan = x.groupby(["umd_full_cd", "em"]).size().rename("n").reset_index()
        meta = x.groupby("umd_full_cd").agg(treated=("treated", "first")).reset_index()
        full = pd.MultiIndex.from_product(
            [meta.umd_full_cd, range(int(pan.em.min()), int(pan.em.max()) + 1)],
            names=["umd_full_cd", "em"]).to_frame(index=False)
        pan = full.merge(pan, on=["umd_full_cd", "em"], how="left").merge(
            meta, on="umd_full_cd")
        pan["n"] = pan.n.fillna(0)
        pan["kq"] = np.floor(pan.em / 3).astype(int)
        pan["em_s"] = pan.em.astype(str)
        pan["log_n"] = np.log1p(pan.n)
        res = econ.event_study(pan, y="log_n", treated="treated", kq="kq",
                               fes=["umd_full_cd", "em_s"], controls=[],
                               cluster="umd_full_cd", kmin=-8, kmax=8)
        es[label] = {int(r.k): round(float(r.effect_pct), 2)
                     for _, r in res.iterrows()}
        print(f"  {label:<22}k=0 {es[label][0]:+7.1f}%   "
              f"k=1 {es[label][1]:+7.1f}%   k=4 {es[label][4]:+7.1f}%")
    gap = es["취소 포함"][0] - es["취소 제외 (현 사양)"][0]
    print(f"\n  k=0 차이 {gap:+.1f}%p")
    out["사건연구"] = {"값": es, "k0_차이pp": round(gap, 2)}

    # --- 그림 ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    m = (obs[obs.deal_date < policy.IDENTIFICATION_END]
         .assign(mm=lambda d: d.deal_date.dt.to_period("M"))
         .groupby(["mm", "treated"]).cancelled.agg(["size", "mean"]).reset_index())
    m = m[m["size"] >= 20]
    ax = axes[0]
    for t, lab, col in ((True, "처치군 (4개 동)", "#c53030"),
                        (False, "통제군 (4개 구)", "#2b6cb0")):
        s = m[m.treated == t]
        ax.plot(s.mm.dt.to_timestamp(), 100 * s["mean"], "o-", ms=3, lw=1.3,
                color=col, label=lab)
    for e, lab in ((E2020, "2020.6 지정"), (E2025R, "2025.2 해제"),
                   (E2025E, "2025.3 재지정")):
        ax.axvline(e, color="#555", ls="--", lw=0.9)
        ax.text(e, ax.get_ylim()[1], lab, fontsize=7, rotation=90,
                va="top", ha="right", color="#555")
    ax.set_title("계약월별 취소율 (거래 20건 이상인 달)")
    ax.set_ylabel("취소율 (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ks = sorted(es["취소 제외 (현 사양)"])
    for lab, col in (("취소 제외 (현 사양)", "#2b6cb0"), ("취소 포함", "#dd6b20")):
        ax.plot(ks, [es[lab][k] for k in ks], "o-", ms=4, lw=1.4,
                color=col, label=lab)
    ax.axhline(0, color="#999", lw=0.8)
    ax.axvline(-0.5, color="#555", ls="--", lw=0.9)
    ax.set_title("거래량 사건연구 — 취소 포함 여부")
    ax.set_xlabel("사건 분기 (k)")
    ax.set_ylabel("효과 (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    p = io.FIGURES / "18_cancellations.png"
    fig.savefig(p, dpi=150)
    print(f"\n그림 저장: {p}")

    io.save_result("18_cancellations", out)


if __name__ == "__main__":
    main()
