"""[21] 2021.4 B그룹 — 효과가 강남·송파 네 동 밖에서도 나오는가.

    python scripts/21_b_group.py

**왜 하나.** 제안서가 차별점으로 내건 처치 구조는 **2020 / 2021 / 2025 다중
시점**인데 2021 이 비어 있었다. 그리고 선행연구의 공통 한계로 "2021년 4월의
2차 지정을 반영하지 않았다"고 적어 놓고 우리도 쓴 적이 없다. 지금까지
2021-04-27 은 **통제군을 오염시키는 존재로만** 등장했다([[0009]]·[[0020]]).

동시에 이 연구의 가장 큰 약점 — **처치 동이 넷뿐** — 을 메우는 자리다.
17,550가지 중 1위는 "그 네 동이 우연이 아니다"일 뿐 "다른 동에서도 그렇다"가
아니다.

**설계의 핵심은 대칭이다.** B그룹 숫자를 A그룹 숫자 옆에 그냥 놓으면 안 된다.
A 는 같은 구 안(강남·송파)을 통제군으로 썼는데 B 는 그럴 수 없다 — 목동이
속한 양천구에 쓸 수 있는 비지정 동이 **0개**다. 그래서

  1. 두 사건을 **같은 통제군 정의(서울 전역 비지정)로 다시** 잰다
  2. 두 패널을 **쌓아 차이를 직접** 추정한다(stacked DID). 두 숫자를 눈으로
     견주면 '차이가 유의한가'를 말할 수 없다 — [[0015-대출규제-처리]] 에서
     같은 실수를 하고 정정했다

A 의 주 사양(강남·송파 내부, −79.8%)은 그대로 둔다. 여기 재추정은 비교용이다.
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
from lib import econ, geo, inference, io, policy  # noqa: E402

for f in ("Malgun Gothic", "Batang"):
    if f in {x.name for x in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False
io.setup_stdout()

W = 12                       # 창 ±12개월. 두 사건 공통
KMIN, KMAX, REF = -4, 3, -1  # 분기
MIN_DEALS = 30               # 창 안 거래가 이 수 미만인 동은 뺀다
DRAWS = 5_000                # 무작위화 표본 (전수는 10^10 가지)
SMALL = 40.0                 # 소형 경계 (0022 와 같다)

EVENTS = {"E2020": policy.TREATED_2020, "E2021": policy.TREATED_2021}


# --- 표본 ------------------------------------------------------------------
def window(df: pd.DataFrame, ev: str) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    e = policy.EVENTS[ev]["effect"]
    lo, hi = e - pd.DateOffset(months=W), e + pd.DateOffset(months=W)
    w = df[(df.deal_date >= lo) & (df.deal_date < hi)].copy()
    w["em"] = policy.event_month(w.deal_date, ev)
    w["kq"] = np.floor(w.em / 3).astype(int)
    return w, lo, hi


def contaminated(lo, hi) -> set[str]:
    """창 안에 **어떤 계열로든** 지정 구간이 걸친 법정동."""
    sp = pd.read_csv(io.ROOT / "data/policy/ltz_spells.csv")
    s = pd.to_datetime(sp.시작일, errors="coerce")
    e = pd.to_datetime(sp.종료일, errors="coerce")
    return set(sp[(s < hi) & (e >= lo)].법정동)


def panel(w: pd.DataFrame, treated: set[str], drop: set[str]) -> pd.DataFrame:
    """처치동은 남기고, 통제 후보에서 drop 과 거래가 적은 동을 뺀다."""
    g = w.copy()
    g["treated"] = g.umdNm.isin(treated)
    big = g.groupby("umd_full_cd").size()
    g = g[g.umd_full_cd.isin(big[big >= MIN_DEALS].index)]
    return g[g.treated | ~g.umdNm.isin(drop)]


def es(g: pd.DataFrame) -> pd.DataFrame:
    """법정동 × 사건월 패널의 사건연구."""
    meta = g.groupby("umd_full_cd").agg(treated=("treated", "first")).reset_index()
    cnt = g.groupby(["umd_full_cd", "em"]).size().rename("n").reset_index()
    full = pd.MultiIndex.from_product(
        [meta.umd_full_cd, range(KMIN * 3, KMAX * 3 + 3)],
        names=["umd_full_cd", "em"]).to_frame(index=False)
    p = full.merge(cnt, on=["umd_full_cd", "em"], how="left").merge(meta, on="umd_full_cd")
    p["n"] = p.n.fillna(0)
    p["kq"] = np.floor(p.em / 3).astype(int)
    p["em_s"] = p.em.astype(str)
    p["log_n"] = np.log1p(p.n)
    return econ.event_study(p, y="log_n", treated="treated", kq="kq",
                            fes=["umd_full_cd", "em_s"], controls=[],
                            cluster="umd_full_cd", kmin=KMIN, kmax=KMAX, ref=REF)


def did(g: pd.DataFrame, label: str) -> dict:
    """창 전체 이중차분 + WCR + 무작위화."""
    s = inference.spec_did(g, lo=KMIN * 3, hi=KMAX * 3 + 2, name=label)
    b, _t, p1, p2, _ = inference.wild_boot(s, np.random.default_rng(inference.SEED))
    vals, act, n = inference.randomization(s, max_draws=DRAWS)
    rank = int((vals <= act).sum())
    return {"이름": label, "동": int(s.G), "처치동": int(s.treated.sum()),
            "계수": round(float(b), 4), "효과_pct": round(100 * (np.exp(b) - 1), 1),
            "WCR_단측": round(p1, 4), "WCR_양측": round(p2, 4),
            "무작위_순위": rank, "무작위_표본": int(n),
            "무작위_p": round(rank / n, 4)}


def main() -> None:
    df = pd.read_parquet(io.PROCESSED / "trades_seoul.parquet")
    out: dict = {"창_개월": W, "최소거래": MIN_DEALS, "무작위_표본": DRAWS}

    w21, lo21, hi21 = window(df, "E2021")
    w20, lo20, hi20 = window(df, "E2020")
    dirty21, dirty20 = contaminated(lo21, hi21), contaminated(lo20, hi20)
    ring20 = geo.touching(policy.TREATED_2020, {"강남구", "송파구"})
    T20, T21 = policy.TREATED_2020, policy.TREATED_2021

    print("=" * 78)
    print("1) 표본과 통제군")
    print("=" * 78)
    print(f"  B 창 {lo21.date()} ~ {hi21.date()} · 창 안 지정 이력 {len(dirty21)}개 동")
    print(f"  A그룹 맞닿은 동 {len(ring20)}개 · 그중 B그룹 처치동: "
          f"{sorted(T21 & ring20)}")
    print("  -> B그룹 5개 중 3개가 2020 지정의 파급을 받은 동이다. 사전 구간에")
    print("     그 파급이 들어 있으므로 처치군을 둘로 나눠 본다.\n")
    out["A그룹_맞닿은동"] = sorted(ring20)
    out["B그룹중_A파급"] = sorted(T21 & ring20)

    # 두 사건이 서로의 통제군을 오염시키지 않게 상대 처치동을 뺀다
    drop21 = dirty21 | T20 | ring20
    drop20 = dirty20 | T21

    g21 = panel(w21, T21, drop21)
    g20 = panel(w20, T20, drop20)
    for nm, g in (("B (2021.4)", g21), ("A (2020.6)", g20)):
        m = g.groupby("umd_full_cd").treated.first()
        print(f"  {nm}: 처치 {int(m.sum())}개 동 · 통제 {int((~m).sum())}개 동 · "
              f"거래 {len(g):,}건")

    # --- 2) 사건연구 -------------------------------------------------------
    print("\n" + "=" * 78)
    print("2) 사건연구 — 사전 구간이 0 근처인가")
    print("=" * 78)
    ev = {}
    for nm, g in (("B 전체 5개 동", g21),
                  ("B A파급 제외 (목동·여의도)", panel(w21, T21 - ring20, drop21 | (T21 & ring20))),
                  ("A 2020 (같은 통제군)", g20)):
        r = es(g)
        ev[nm] = {int(x.k): round(float(x.effect_pct), 1) for _, x in r.iterrows()}
        pre = r[r.k < REF]
        bad = int(((pre.ci_lo > 0) | (pre.ci_hi < 0)).sum())
        print(f"  {nm:<26}" + " ".join(f"{int(x.k):>+3}:{x.effect_pct:>+6.1f}"
                                       for _, x in r.iterrows()))
        print(f"  {'':<26}사전 {len(pre)}칸 중 유의 {bad}칸")
    out["사건연구"] = ev

    # --- 3) 통제군 사다리 ---------------------------------------------------
    print("\n" + "=" * 78)
    print("3) 통제군 사다리 — 기준점을 바꿔도 같은가")
    print("=" * 78)
    pre_m2 = (w21[w21.em < 0].groupby("umd_full_cd").price_per_m2.median())
    rng_t = pre_m2[pre_m2.index.isin(
        w21[w21.umdNm.isin(T21)].umd_full_cd.unique())]
    band = (rng_t.min(), rng_t.max())
    common = set(pre_m2[(pre_m2 >= band[0]) & (pre_m2 <= band[1])].index)
    print(f"  공통지지 띠: 지정 전 ㎡당 중위가 {band[0]:,.0f} ~ {band[1]:,.0f} 만원")

    ladder = []
    for label, g in (
        ("① 서울 전역 비지정", g21),
        ("② ① + 가격대 공통지지", g21[g21.treated | g21.umd_full_cd.isin(common)]),
        ("③ 오염 미제외", panel(w21, T21, T20 | ring20)),
        ("④ A그룹 맞닿은 동 포함", panel(w21, T21, dirty21 | T20)),
    ):
        ladder.append(did(g, label))
        r = ladder[-1]
        print(f"  {label:<24}{r['효과_pct']:>+7.1f}%  동 {r['동']:>3}  "
              f"WCR {r['WCR_양측']:.3f}  무작위 {r['무작위_순위']}/{r['무작위_표본']}")
    out["사다리"] = ladder

    # --- 4) 두 사건 대칭 ----------------------------------------------------
    print("\n" + "=" * 78)
    print("4) 두 사건을 같은 자로 — 같은 통제군 정의, 같은 창, 같은 추정량")
    print("=" * 78)
    sym = []
    for label, g in (("A 2020.6 (처치 4개 동)", g20),
                     ("B 2021.4 (처치 5개 동)", g21),
                     ("B 2021.4 (목동·여의도만)",
                      panel(w21, T21 - ring20, drop21 | (T21 & ring20)))):
        sym.append(did(g, label))
        r = sym[-1]
        print(f"  {label:<24}{r['효과_pct']:>+7.1f}%  처치 {r['처치동']}개 동  "
              f"WCR {r['WCR_양측']:.3f}  무작위 p {r['무작위_p']:.4f}")
    out["대칭"] = sym
    print("\n  주의 — 위는 **창 전체(±12개월) 이중차분**이다. 늘 인용하는 −79.8% 는")
    print("  **지정 직후 3개월(사건연구 k=0)** 이라 재는 것이 다르다. 섞어 쓰면 안 된다.")
    print(f"  같은 자리끼리 보면 k=0 이 A {ev['A 2020 (같은 통제군)'][0]:+.1f}% 대 "
          f"B {ev['B 전체 5개 동'][0]:+.1f}% 이고,")
    print("  A 를 강남·송파 내부 통제로 재면 −79.8% 다. 통제군을 서울 전역으로")
    print("  바꿔도 거의 그대로다 — [[0028]] 이 통제군 축을 2.3%p 로 잰 것과 맞는다.")

    # --- 4b) 가격 ----------------------------------------------------------
    print("\n" + "=" * 78)
    print("4b) 가격 — 부 결과")
    print("=" * 78)
    pr = {}
    for label, g in (("A 2020.6", g20), ("B 2021.4", g21)):
        x = g.copy()
        x["ym_s"] = x.ym.astype(str)
        x["log_area"] = np.log(x.area_m2)
        r = econ.event_study(x[(x.kq >= KMIN) & (x.kq <= KMAX)],
                             y="log_price_per_m2", treated="treated", kq="kq",
                             fes=["aptSeq", "ym_s"], controls=["floor_no", "log_area"],
                             cluster="umd_full_cd", kmin=KMIN, kmax=KMAX, ref=REF)
        pr[label] = {int(v.k): round(float(v.effect_pct), 2) for _, v in r.iterrows()}
        print(f"  {label:<10}" + " ".join(f"{int(v.k):>+3}:{v.effect_pct:>+6.1f}"
                                          for _, v in r.iterrows()))
    out["가격"] = pr
    print("  B 는 사전이 깨끗한 대신 **사후 후반이 양수**다. 2021-10 ~ 2022-04 는")
    print("  서울 가격 정점이고 B그룹은 재건축 단지라 기대가 더 실렸을 수 있다.")

    # --- 5) 쌓아서 차이를 직접 ---------------------------------------------
    print("\n" + "=" * 78)
    print("5) 쌓아서 차이를 직접 추정한다 (stacked DID)")
    print("=" * 78)
    a = g20[["umd_full_cd", "em", "treated"]].copy(); a["isB"] = False
    b = g21[["umd_full_cd", "em", "treated"]].copy(); b["isB"] = True
    st = pd.concat([a, b], ignore_index=True)
    st = st[st.em.between(KMIN * 3, KMAX * 3 + 2)]
    s = inference.spec_triple(st, "isB", KMIN * 3, KMAX * 3 + 2, "B − A")
    coef, _t, p1, p2, _ = inference.wild_boot(s, np.random.default_rng(inference.SEED))
    vals, act, n = inference.randomization(s, max_draws=DRAWS)
    rank = int((vals <= act).sum())
    print(f"  B − A 차이 계수 {coef:+.4f}  ->  {100 * (np.exp(coef) - 1):+.1f}%")
    print(f"  WCR 양측 {p2:.3f} · 무작위 {rank}/{n} (p {rank / n:.4f})")
    print("  차이가 0 과 구분되지 않으면 **B 가 A 와 다르다고 할 수 없다** 는 뜻이고,")
    print("  그것이 외적 타당성의 답이다.")
    out["stacked"] = {"계수": round(float(coef), 4),
                      "효과_pct": round(100 * (np.exp(coef) - 1), 1),
                      "WCR_양측": round(p2, 4), "WCR_단측": round(p1, 4),
                      "무작위_순위": rank, "무작위_표본": int(n),
                      "동": int(s.G)}

    # --- 6) 면적 경계 민감도 -----------------------------------------------
    print("\n" + "=" * 78)
    print("6) 면적 — 허가 기준이 2020 과 같은 18㎡ 다. 소형이 덜 줄었는가")
    print("=" * 78)
    t21 = w21[w21.umdNm.isin(T21)]
    print(f"  B그룹 처치동 소형 거래: " + " · ".join(
        f"{c:g}㎡ 미만 {int((t21.area_m2 < c).sum())}건" for c in (30, 40, 50, 60)))
    area = []
    for c in (30, 40, 50, 60):
        gg = g21.copy()
        gg["small"] = gg.area_m2 < c
        if gg[gg.treated & gg.small].umd_full_cd.nunique() < 3:
            area.append({"경계": c, "비고": "처치동 소형 표본 부족"})
            print(f"  {c:g}㎡  — 처치동 소형이 3개 동 미만이라 추정하지 않는다")
            continue
        sp = inference.spec_triple(gg, "small", KMIN * 3, KMAX * 3 + 2, f"{c:g}㎡")
        bb, _t, _p1, pp, _ = inference.wild_boot(sp, np.random.default_rng(inference.SEED))
        area.append({"경계": c, "차이_계수": round(float(bb), 4),
                     "차이_pct": round(100 * (np.exp(bb) - 1), 1),
                     "WCR_양측": round(pp, 4)})
        print(f"  {c:g}㎡  소형이 더 움직인 폭 {100 * (np.exp(bb) - 1):+7.1f}%  "
              f"WCR {pp:.3f}")
    out["면적"] = area

    # --- 7) 기준분기 민감도 -------------------------------------------------
    print("\n" + "=" * 78)
    print("7) 기준분기를 옮기면 (0014 와 같은 점검)")
    print("=" * 78)
    base = es(g21)
    c0 = float(base[base.k == 0].coef.iloc[0])
    refs = {}
    for r0 in (-1, -2, -3, -4):
        cr = float(base[base.k == r0].coef.iloc[0])
        refs[r0] = round(100 * (np.exp(c0 - cr) - 1), 1)
        print(f"  기준 k={r0:<3} k=0 효과 {refs[r0]:+7.1f}%")
    out["기준분기"] = refs

    # --- 그림 ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    for nm, col in (("B 전체 5개 동", "#c53030"),
                    ("B A파급 제외 (목동·여의도)", "#dd6b20"),
                    ("A 2020 (같은 통제군)", "#2b6cb0")):
        ks = sorted(ev[nm]); ax.plot(ks, [ev[nm][k] for k in ks], "o-",
                                     ms=4, lw=1.4, color=col, label=nm)
    ax.axhline(0, color="#999", lw=.8); ax.axvline(-.5, color="#555", ls="--", lw=.9)
    ax.set_title("사건연구 — 같은 통제군 정의로 두 사건")
    ax.set_xlabel("사건 분기 (k)"); ax.set_ylabel("효과 (%)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)

    ax = axes[1]
    nm = [r["이름"] for r in sym][::-1]
    val = [r["효과_pct"] for r in sym][::-1]
    ax.barh(nm, val, color=["#2b6cb0" if "A " in n else "#c53030" for n in nm])
    for i, v in enumerate(val):
        ax.text(v / 2, i, f"{v:+.1f}%", va="center", ha="center",
                color="white", fontsize=11, fontweight="bold")
    ax.axvline(0, color="#555", lw=.8)
    ax.set_title("같은 자로 잰 두 사건 (창 전체 이중차분)")
    ax.set_xlabel("효과 (%)"); ax.grid(alpha=.3, axis="x")

    fig.tight_layout()
    p = io.FIGURES / "21_b_group.png"
    fig.savefig(p, dpi=150)
    print(f"\n그림 저장: {p}")
    io.save_result("21_b_group", out)


if __name__ == "__main__":
    main()
