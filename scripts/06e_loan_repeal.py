"""[6e] 15억 금지가 풀렸을 때 고가 거래는 돌아오는가 — 12·16 의 거울상.

    python scripts/06e_loan_repeal.py

입력  data/processed/trades_seoul.parquet, data/policy/ltz_spells.csv
출력  output/results/06e_loan_repeal.json, output/figures/06e_loan_repeal.png

**무엇을 묻나.** 2019-12-17 에 시가 15억 초과 아파트의 주택구입 목적 주담대가
전면 금지됐고, **2022-12-01 에 그 금지가 폐지**됐다(LTV 50% 허용). 규제를 걸었을
때 고가 거래가 줄었다면 풀었을 때는 돌아와야 한다. 같은 방향이면 [[0015]] 의
해석(대출이 고가 거래를 실제로 막았다)이 크게 단단해진다.

[[0015]] 는 이 날을 **쏠림 구간의 끝**으로만 썼다. 15억 문턱 아래/위 비율이
0.96 → 1.60 → 1.09 로 움직였다는 것이다. 여기서는 폐지일을 **사건으로** 건다.

**이 사건은 깨끗하지 않다.** 셋을 미리 적어 둔다.

    1. 2022-12-01 은 15억 금지와 9억 꺾임을 **동시에** 풀었다. 9억을 대조 문턱으로
       쓸 수 없다([[0021]]).
    2. 한 달 뒤 2023-01-03 대책이 강남3구·용산을 빼고 규제지역을 전부 해제했다.
       저가 지역에 더 유리한 충격이라 고가/저가 비교를 교란한다.
    3. 2022년 말은 금리 급등으로 전국 거래가 얼어붙던 때다. '규제를 풀어도
       안 돌아왔다'가 나와도 금리 탓인지 가릴 수 없다.

그래서 **창을 셋으로 두고** 2023-01-03 이 언제 들어오는지 보이고, **문턱을
9~20억으로 옮겨** 효과가 15억에서만 끊기는지(규제 탓) 가격대 전반에 걸리는지
(금리 탓) 본다. 설계는 [[0021]] 의 문턱 곡선과 같다.
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

io.setup_stdout()
for f in ("Malgun Gothic", "Batang"):
    if f in {x.name for x in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

RULE_ON = pd.Timestamp("2019-12-17")    # 15억 초과 주담대 금지
RULE_OFF = pd.Timestamp("2022-12-01")   # 그 금지의 폐지
JAN3 = pd.Timestamp("2023-01-03")       # 규제지역 대거 해제 — 교란
BAND = 0.5                              # 문턱 위아래 폭 (억)
THR = 15.0
THRS = (9.0, 12.0, 15.0, 18.0, 20.0)
WINDOWS = (1, 3, 12)                    # 전후 개월
BASE_M = 24                             # 고가/저가 배정에 쓸 사전 개월


def bunch_ratio(x: pd.Series, thr: float, band: float = BAND) -> float:
    below = ((x >= thr - band) & (x < thr)).sum()
    above = ((x >= thr) & (x < thr + band)).sum()
    return float(below / above) if above else np.nan


def designated(lo: pd.Timestamp, hi: pd.Timestamp) -> set[str]:
    sp = pd.read_csv(io.POLICY / "ltz_spells.csv", encoding="utf-8-sig")
    sp["시작일"] = pd.to_datetime(sp.시작일)
    sp["종료일"] = pd.to_datetime(sp.종료일)
    m = (sp.상태 == "지정") & (sp.시작일 <= hi) & (sp.종료일 >= lo)
    return set(sp[m].법정동)


def did(d: pd.DataFrame, anchor: pd.Timestamp, months: int,
        thr: float) -> tuple[float, float, int, int]:
    """단지 × 월 PPML. 처치 = 사전 중위 거래가가 문턱을 넘는 단지.

    문턱 배정은 **사건 이전** 거래로만 한다. 결과로 처치를 정하지 않기 위해서다.
    """
    lo = anchor - pd.DateOffset(months=months)
    hi = anchor + pd.DateOffset(months=months)
    pre = d[d.deal_date.between(anchor - pd.DateOffset(months=BASE_M), anchor)]
    med = pre.groupby("aptSeq").amount_eok.median()
    s = d[d.deal_date.between(lo, hi)].copy()
    s["기저"] = s.aptSeq.map(med)
    s = s[s.기저.notna()]
    s["고가"] = s.기저 > thr
    s["m"] = (s.deal_date.dt.year * 12 + s.deal_date.dt.month
              - anchor.year * 12 - anchor.month)
    cells = s.groupby("aptSeq").agg(고가=("고가", "first"),
                                    동=("umd_full_cd", "first"))
    ms = sorted(s.m.unique())
    grid = (cells.index.to_frame(index=False)
            .merge(pd.DataFrame({"m": ms}), how="cross"))
    cnt = s.groupby(["aptSeq", "m"]).size().rename("n").reset_index()
    p = (grid.merge(cnt, on=["aptSeq", "m"], how="left").fillna({"n": 0})
         .join(cells, on="aptSeq"))
    p["d"] = (p.고가 & (p.m >= 0)).astype(float)
    r = econ.ppml(p.n.to_numpy(), p[["d"]].to_numpy(), p.aptSeq.to_numpy(),
                  p.m.to_numpy(), cluster=p.동.to_numpy())
    return (float(100 * np.expm1(r["coef"][0])), float(r["p"][0]),
            int(cells.고가.sum()), int((~cells.고가).sum()))


def main() -> None:
    d = pd.read_parquet(io.TRADES_SEOUL)
    out: dict = {}

    # ==================================================================
    print("=" * 88)
    print("1. 15억 문턱의 쏠림 — 규제가 켜지고 꺼질 때")
    print("=" * 88)
    w = d[d.deal_date.between("2018-01-01", "2024-12-31")].copy()
    w["q"] = w.deal_date.dt.to_period("Q")
    bq = w.groupby("q").amount_eok.apply(lambda x: bunch_ratio(x, THR))
    seg = [("규제 이전", "2018-01-01", "2019-12-16"),
           ("금지 기간", "2019-12-17", "2022-11-30"),
           ("폐지 이후", "2022-12-01", "2026-05-31")]
    print(f"  {'구간':<14}{'아래/위':>9}{'거래(아래+위)':>14}")
    seg_rows = []
    for lab, a, b in seg:
        x = d[d.deal_date.between(a, b)].amount_eok
        n = int(((x >= THR - BAND) & (x < THR + BAND)).sum())
        r = bunch_ratio(x, THR)
        seg_rows.append({"구간": lab, "비율": round(r, 2), "거래": n})
        print(f"  {lab:<14}{r:>9.2f}{n:>14,d}")
    print("  0015 가 쓴 것과 같다. 여기서는 분기별 궤적을 함께 본다")
    out["쏠림_구간"] = seg_rows
    out["쏠림_분기"] = [{"분기": str(k), "비율": (round(float(v), 3)
                         if np.isfinite(v) else None)} for k, v in bq.items()]

    # ==================================================================
    print("\n" + "=" * 88)
    print("2. 폐지를 사건으로 — 고가 단지의 거래가 돌아오는가")
    print("=" * 88)
    bad = designated(RULE_OFF - pd.DateOffset(months=12),
                     RULE_OFF + pd.DateOffset(months=12))
    clean = d[~d.umdNm.isin(bad)]
    print(f"  창 안에 토허구역이던 법정동 {len(bad)}개를 뺀 표본도 함께 본다")
    print(f"\n  {'표본':<16}{'창':>7}{'효과':>10}{'p':>8}{'고가단지':>9}{'저가단지':>9}")
    rows = []
    for sname, s in (("서울 전역", d), ("토허구역 동 제외", clean)):
        for mth in WINDOWS:
            e, pv, nh, nl = did(s, RULE_OFF, mth, THR)
            rows.append({"표본": sname, "창_개월": mth, "효과_pct": round(e, 1),
                         "p": round(pv, 4), "고가단지": nh, "저가단지": nl})
            mark = "  <- 2023-01-03 들어옴" if mth > 1 else "  (1·3 대책 전)"
            print(f"  {sname:<16}{mth:>6}개월{e:>9.1f}%{pv:>8.3f}"
                  f"{nh:>9,d}{nl:>9,d}{mark}")
    out["폐지_DID"] = rows

    # ==================================================================
    print("\n" + "=" * 88)
    print("3. 거울상 — 같은 사양을 금지 시행일(2019-12-17)에도 건다")
    print("=" * 88)
    print(f"  {'사건':<22}{'창':>7}{'효과':>10}{'p':>8}")
    mir = []
    for lab, anchor in (("2019-12-17 금지", RULE_ON), ("2022-12-01 폐지", RULE_OFF)):
        for mth in WINDOWS:
            e, pv, _, _ = did(clean, anchor, mth, THR)
            mir.append({"사건": lab, "창_개월": mth, "효과_pct": round(e, 1),
                        "p": round(pv, 4)})
            print(f"  {lab:<22}{mth:>6}개월{e:>9.1f}%{pv:>8.3f}")
    print("  금지가 음수이고 폐지가 양수면 거울상이다")
    out["거울상"] = mir

    # ==================================================================
    print("\n" + "=" * 88)
    print("4. 문턱을 옮긴다 — 15억에서만 끊기는가, 가격대 전반인가")
    print("=" * 88)
    print(f"  {'문턱':>6}{'폐지 효과':>11}{'p':>8}{'금지 효과':>11}{'p':>8}")
    thr_rows = []
    for t in THRS:
        e1, p1, _, _ = did(clean, RULE_OFF, 3, t)
        e2, p2, _, _ = did(clean, RULE_ON, 3, t)
        thr_rows.append({"문턱_억": t, "폐지_pct": round(e1, 1), "폐지_p": round(p1, 4),
                         "금지_pct": round(e2, 1), "금지_p": round(p2, 4)})
        print(f"  {t:>5.0f}억{e1:>10.1f}%{p1:>8.3f}{e2:>10.1f}%{p2:>8.3f}"
              + ("  ←규제 문턱" if t == THR else ""))
    print("  15억에만 계단이 있으면 규제 탓, 매끈하면 가격대 전반(금리) 탓이다")
    out["문턱곡선"] = thr_rows

    # ==================================================================
    print("\n" + "=" * 88)
    print("5. 위약 — 아무 일도 없던 12월 1일")
    print("=" * 88)
    pl = []
    for y in (2020, 2021, 2023):
        a = pd.Timestamp(f"{y}-12-01")
        e, pv, _, _ = did(clean, a, 3, THR)
        pl.append({"날짜": str(a.date()), "효과_pct": round(e, 1), "p": round(pv, 4)})
        print(f"  {a.date()}  {e:>7.1f}%  p={pv:.3f}")
    print(f"  (실제 2022-12-01 은 위 2번 표의 3개월 창 값)")
    out["위약"] = pl

    # ==================================================================
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.8))
    ax = axes[0]
    x = bq.index.to_timestamp()
    ax.plot(x, bq.values, "o-", color="#3182ce", lw=1.6, ms=4)
    ax.axhline(1.0, color="#999", lw=.9)
    ax.axvline(RULE_ON, color="#c53030", ls="--", lw=1.3)
    ax.axvline(RULE_OFF, color="#2f855a", ls="--", lw=1.3)
    ax.text(RULE_ON, ax.get_ylim()[1] * .97, " 금지 2019-12", fontsize=8,
            color="#c53030", va="top")
    ax.text(RULE_OFF, ax.get_ylim()[1] * .9, " 폐지 2022-12", fontsize=8,
            color="#2f855a", va="top")
    ax.set_title("15억 문턱의 쏠림 — 아래 거래 ÷ 위 거래")
    ax.set_ylabel("배")
    ax.grid(alpha=.25)

    ax = axes[1]
    for lab, c in (("2019-12-17 금지", "#c53030"), ("2022-12-01 폐지", "#2f855a")):
        s = [r for r in mir if r["사건"] == lab]
        ax.plot([r["창_개월"] for r in s], [r["효과_pct"] for r in s], "o-",
                color=c, lw=1.8, label=lab)
    ax.axhline(0, color="#1a202c", lw=.9)
    ax.set_xscale("log")
    ax.set_xticks(list(WINDOWS))
    ax.set_xticklabels([f"±{m}개월" for m in WINDOWS])
    ax.set_title("거울상 — 고가 단지 거래, 저가 대비")
    ax.set_ylabel("효과 (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)

    ax = axes[2]
    tt = [r["문턱_억"] for r in thr_rows]
    ax.plot(tt, [r["폐지_pct"] for r in thr_rows], "o-", color="#2f855a",
            lw=1.8, label="2022-12 폐지")
    ax.plot(tt, [r["금지_pct"] for r in thr_rows], "o-", color="#c53030",
            lw=1.8, label="2019-12 금지")
    ax.axvline(THR, color="#1a202c", ls="--", lw=1.2)
    ax.axhline(0, color="#999", lw=.9)
    ax.set_title("문턱을 옮기면 — 15억에 계단이 있는가")
    ax.set_xlabel("고가/저가 경계 (억)")
    ax.set_ylabel("효과 (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)

    fig.suptitle("15억 금지의 폐지(2022-12-01) — 12·16 의 거울상인가", y=1.03)
    fig.tight_layout()
    fp = io.FIGURES / "06e_loan_repeal.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")

    out.update({"문턱_억": THR, "폭_억": BAND, "창_개월": list(WINDOWS),
                "교란_1·3대책": str(JAN3.date())})
    io.save_result("06e_loan_repeal", out)
    print(f"\n저장 {fp} · output/results/06e_loan_repeal.json")


if __name__ == "__main__":
    main()
