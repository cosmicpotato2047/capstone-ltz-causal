"""[10] 처치 역전 추정량 — 이중차분 계수에 음의 가중치가 붙는가, 부과와 해제는 대칭인가.

    python scripts/10_dcdh.py

입력  data/processed/trades_seoul.parquet, data/policy/ltz_spells.csv
출력  output/results/10_dcdh.json, output/figures/10_dcdh.png

**왜 필요한가.** de Chaisemartin & D'Haultfœuille(2020, AER)은 두 방향 고정효과
이중차분 계수가 개별 칸의 처치효과를 가중평균한 것인데 **그 가중치가 음수일 수
있다**는 것을 보였다. 효과가 칸마다 다르면, 모든 칸에서 효과가 음수인데도 계수가
양수로 나올 수 있다. 처치가 켜졌다 꺼지는 구간에서 특히 그렇다.

제안서는 2025 구간(1 → 0 → 1)에 이 추정량을 쓰겠다고 적었다. 지금까지는 평범한
이중차분을 썼으므로([[0023]]) 두 가지를 확인한다.

    1. 우리 계수에 음의 가중치가 얼마나 붙어 있나 — 진단
    2. 켤 때(부과)와 끌 때(해제)를 나눠 재면 대칭인가 — 제안서의 세 번째 기여

**DID_M 이 하는 일.** 어느 시점 t 에서 처치가 새로 켜진 집단은 그때 처치가
없던 채로 머문 집단과, 처치가 꺼진 집단은 처치를 유지한 집단과 견준다. 두 방향
고정효과처럼 모든 칸을 한 회귀에 넣지 않으므로 음의 가중치가 생기지 않는다.

    부과 DID(+,t) = (새로 켜진 집단의 변화) − (계속 꺼져 있던 집단의 변화)
    해제 DID(−,t) = (계속 켜져 있던 집단의 변화) − (꺼진 집단의 변화)

둘 다 '처치가 있을 때 결과가 얼마나 낮은가'를 잰다. 부호가 같은 방향이다.

**전환 주간(5주차)은 뺀다.** 2025-03-19 발표와 03-24 효력 사이 나흘에 막차가
몰린 주다([[0023]]). 그 주를 기준으로 삼으면 부풀려진 값에서 떨어지는 것을
재게 된다. 실제로 그 주를 기준에 두면 위약이 +19.5% 로 나온다.
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

RELEASE = pd.Timestamp("2025-02-13")
REDESIG = pd.Timestamp("2025-03-24")
SGG_T = ["강남구", "송파구"]
SGG_2025 = ["강남구", "서초구", "송파구", "용산구"]
SGG_NEAR = ["강동구", "성동구", "광진구", "동작구"]
WK_LO, WK_HI = -14, 10
WK_RUSH = 5                 # 막차 주간. 기준에서 뺀다
LAGS = 5                    # 시차 0~4주
B = 999
SEED = 20260930
KMIN, KMAX = -8, 8          # 04_event_study 와 같다


# --- 가중치 진단 ---------------------------------------------------------------
def fe_weights(p: pd.DataFrame, gcol: str, tcol: str) -> dict:
    """두 방향 고정효과 계수와 그 계수가 처치 칸에 주는 가중치.

    D 를 집단·시점 고정효과에 회귀한 잔차 ε 가 가중치의 부호를 정한다
    (dCDH 2020 정리 1). 처치 칸에서 ε < 0 이면 그 칸의 효과가 계수에 **거꾸로**
    들어간다.
    """
    dm = econ.demean(p[["y", "D"]], p[[gcol, tcol]])
    beta = float(np.linalg.lstsq(dm[["D"]].to_numpy(), dm.y.to_numpy(),
                                 rcond=None)[0][0])
    eps = dm.D.to_numpy()
    tr = p.D.to_numpy() == 1
    w = eps * tr
    w = w / w.sum()
    neg = w < 0
    return {"계수": round(beta, 4), "효과_pct": round(float(100 * np.expm1(beta)), 1),
            "처치칸": int(tr.sum()), "음가중_칸": int(neg.sum()),
            "음가중_비율": round(float(100 * neg.sum() / tr.sum()), 1),
            "음가중_합": round(float(w[neg].sum()), 4),
            "양가중_합": round(float(w[~neg].sum()), 4), "_w": w, "_tr": tr}


# --- DID_M --------------------------------------------------------------------
def didm(Y: np.ndarray, D: np.ndarray, t: int, lag: int,
         wt: np.ndarray | None = None) -> dict:
    """전환 시점 t 에서 시차 lag 의 DID_M 두 성분.

    Y, D 는 (단지 × 주) 배열이고 열 번호가 곧 주 인덱스다. 기준은 t−1 이다.
    wt 를 주면 단지마다 그 무게로 평균한다(거래가 많은 단지를 무겁게).
    """
    if wt is None:
        wt = np.ones(Y.shape[0])
    d0, d1 = D[:, t - 1], D[:, t]
    dy = Y[:, t + lag] - Y[:, t - 1]
    ok = np.isfinite(dy)

    def m(mask):
        mm = mask & ok
        return float(np.average(dy[mm], weights=wt[mm])) if mm.sum() else np.nan

    si, so = (d0 == 0) & (d1 == 1), (d0 == 1) & (d1 == 0)
    s0, s1 = (d0 == 0) & (d1 == 0), (d0 == 1) & (d1 == 1)
    out = {}
    if si.sum() and s0.sum():
        out["부과"] = (m(si) - m(s0), int(si.sum()), int(s0.sum()))
    if so.sum() and s1.sum():
        out["해제"] = (m(s1) - m(so), int(so.sum()), int(s1.sum()))
    return out


def boot(Y, D, grp, t, lag, wt, rng, key) -> np.ndarray:
    """단지를 집단 안에서 복원추출해 DID_M 을 다시 낸다."""
    idx_by = {g: np.flatnonzero(grp == g) for g in np.unique(grp)}
    vals = np.empty(B)
    for b in range(B):
        take = np.concatenate([rng.choice(v, size=v.size, replace=True)
                               for v in idx_by.values()])
        r = didm(Y[take], D[take], t, lag, wt[take])
        vals[b] = r.get(key, (np.nan,))[0]
    return vals[np.isfinite(vals)]


def pct(x) -> float:
    return float(100 * np.expm1(x))


def main() -> None:
    rng = np.random.default_rng(SEED)
    out: dict = {}

    # ==================================================================
    print("=" * 88)
    print("1. 주 분석(2020 지정)에는 음의 가중치가 있는가")
    print("=" * 88)
    d8 = io.load_trades()
    g8 = d8[d8.sgg_nm.isin(SGG_T)].copy()
    g8["treated"] = policy.assign_2020(g8)
    g8["em"] = policy.event_month(g8.deal_date, "E2020")
    lo, hi = KMIN * 3, KMAX * 3 + 2
    codes = sorted(g8.umd_full_cd.unique())
    cnt = (g8[g8.em.between(lo, hi)].groupby(["umd_full_cd", "em"]).size()
           .rename("n").reset_index())
    fullm = pd.MultiIndex.from_product([codes, range(lo, hi + 1)],
                                       names=["umd_full_cd", "em"]).to_frame(index=False)
    pm = fullm.merge(cnt, on=["umd_full_cd", "em"], how="left").fillna({"n": 0})
    pm["treated"] = pm.umd_full_cd.map(g8.groupby("umd_full_cd").treated.first())
    pm["D"] = (pm.treated & (pm.em >= 0)).astype(int)
    pm["y"] = np.log1p(pm.n)
    pm["ts"] = pm.em.astype(str)
    w20 = fe_weights(pm, "umd_full_cd", "ts")
    print(f"  법정동 {len(codes)}개 × 사건월 {hi - lo + 1}개 · "
          f"두 방향 고정효과 계수 {w20['효과_pct']:+.1f}%")
    print(f"  처치 칸 {w20['처치칸']:,}개 중 음의 가중치 {w20['음가중_칸']}개 "
          f"({w20['음가중_비율']}%) · 합 {w20['음가중_합']:+.4f}")
    print("  처치가 한 번 켜지고 유지되는 구간이라 음의 가중치가 생길 자리가 없다")
    out["주분석_가중치"] = {k: v for k, v in w20.items() if not k.startswith("_")}

    # ==================================================================
    print("\n" + "=" * 88)
    print("2. 2025 구간 — 처치가 1 → 0 → 1 로 뒤집힌다")
    print("=" * 88)
    sp = pd.read_csv(io.POLICY / "ltz_spells.csv", encoding="utf-8-sig")
    sp["시작일"] = pd.to_datetime(sp.시작일)
    sp["종료일"] = pd.to_datetime(sp.종료일)
    at = REDESIG - pd.Timedelta(days=1)
    already = (set(sp[(sp.상태 == "지정") & (sp.시작일 <= at)
                      & (sp.종료일 >= at)].법정동) - policy.TREATED_2020)

    d = pd.read_parquet(io.TRADES_SEOUL)
    d["kept"] = policy.assign_kept_2025(d)
    g = pd.Series("먼 구", index=d.index)
    g[d.sgg_nm.isin(SGG_NEAR)] = "인접 4개 구"
    g[d.sgg_nm.isin(SGG_2025)] = "새로 규제"
    g[d.umdNm.isin(policy.TREATED_2020) & d.sgg_nm.isin(SGG_T)] = "해제군"
    g[d.umdNm.isin(already) & (g != "해제군")] = "제외"
    g[d.kept] = "유지군"
    d["집단"] = g
    d["wk"] = np.floor((d.deal_date - RELEASE).dt.days / 7).astype(int)
    w = d[(d.집단 != "제외") & d.wk.between(WK_LO, WK_HI)].copy()

    weeks = [k for k in range(WK_LO, WK_HI + 1) if k != WK_RUSH]
    cells = w.groupby(["aptSeq", "wk"]).size().rename("n").reset_index()
    meta = w.groupby("aptSeq").집단.first().rename("집단").reset_index()
    full = meta.merge(pd.DataFrame({"wk": weeks}), how="cross")
    p = full.merge(cells, on=["aptSeq", "wk"], how="left").fillna({"n": 0})

    def status(grp_: str, wk: int) -> int:
        if grp_ == "유지군":
            return 1
        if grp_ == "해제군":
            return 0 if 0 <= wk <= WK_RUSH else 1
        if grp_ == "새로 규제":
            return 1 if wk > WK_RUSH else 0
        return 0

    p["D"] = [status(a, b) for a, b in zip(p.집단, p.wk)]
    p["y"] = np.log1p(p.n)
    p = p.sort_values(["aptSeq", "wk"]).reset_index(drop=True)

    tab = p.groupby("집단").agg(단지=("aptSeq", "nunique"), 거래=("n", "sum"))
    tab["규제"] = [("계속 없음" if k in ("먼 구", "인접 4개 구") else
                    "계속 있음" if k == "유지군" else
                    "1 → 0 → 1" if k == "해제군" else "0 → 1") for k in tab.index]
    print(tab.to_string())
    print(f"  창 {WK_LO}~{WK_HI}주 · 막차 주간({WK_RUSH}주, 3/20~3/26)은 뺐다")
    out["집단"] = tab.reset_index().to_dict("records")

    # ==================================================================
    print("\n" + "=" * 88)
    print("3. 이 구간의 두 방향 고정효과에는 음의 가중치가 붙는가")
    print("=" * 88)
    p["ts"] = p.wk.astype(str)
    w25 = fe_weights(p, "aptSeq", "ts")
    print(f"  두 방향 고정효과 계수 {w25['효과_pct']:+.1f}%")
    print(f"  처치 칸 {w25['처치칸']:,}개 중 음의 가중치 {w25['음가중_칸']:,}개 "
          f"({w25['음가중_비율']}%)")
    print(f"  음의 가중치 합 {w25['음가중_합']:+.4f} · 양의 합 {w25['양가중_합']:+.4f}")
    tw = p[w25["_tr"]].assign(w=w25["_w"][w25["_tr"]])
    print(f"\n  {'집단':<14}{'가중치 합':>11}{'최소':>11}{'최대':>11}")
    for k, v in tw.groupby("집단").w.agg(["sum", "min", "max"]).iterrows():
        print(f"  {k:<14}{v['sum']:>11.4f}{v['min']:>11.6f}{v['max']:>11.6f}")
    print("  계속 처치인 유지군의 가중치 합이 0 이다 — 두 방향 고정효과는")
    print("  '항상 처치인 집단'을 사실상 쓰지 않는다 (dCDH 가 지적한 자리)")
    out["2025_가중치"] = {k: v for k, v in w25.items() if not k.startswith("_")}

    # ==================================================================
    print("\n" + "=" * 88)
    print("4. DID_M — 부과와 해제를 나눠 잰다")
    print("=" * 88)
    Y = p.pivot(index="aptSeq", columns="wk", values="y")
    D = p.pivot(index="aptSeq", columns="wk", values="D")
    cols = list(Y.columns)
    grp = p.groupby("aptSeq").집단.first().reindex(Y.index).to_numpy()
    base = (p[p.wk < 0].groupby("aptSeq").n.sum().reindex(Y.index).fillna(0)
            .to_numpy() + 1.0)          # 거래 가중: 사전 거래 + 1
    Ya, Da = Y.to_numpy(), D.to_numpy()
    t_rel, t_red = cols.index(0), cols.index(WK_RUSH + 1)

    N = p.pivot(index="aptSeq", columns="wk", values="n").to_numpy()

    def agg_ratio(t: int, lag: int, key: str,
                  only: np.ndarray | None = None) -> float:
        """집계 배수 비 — 건수를 그대로 더해서 낸다. PPML 과 같은 무게다."""
        d0, d1 = Da[:, t - 1], Da[:, t]
        si, so = (d0 == 0) & (d1 == 1), (d0 == 1) & (d1 == 0)
        s0, s1 = (d0 == 0) & (d1 == 0), (d0 == 1) & (d1 == 1)
        if only is not None:
            si, so = si & only, so & only
        A, B_ = (si, s0) if key == "부과" else (s1, so)
        ra = N[A, t + lag].sum() / max(N[A, t - 1].sum(), 1e-9)
        rb = N[B_, t + lag].sum() / max(N[B_, t - 1].sum(), 1e-9)
        return float(np.log(ra / rb)) if ra > 0 and rb > 0 else np.nan

    # 막차 주간을 뺐으므로 재지정 쪽은 기준(4주)과 첫 사후(6주) 사이가 2주다.
    # 해제 쪽은 1주다. 시차 번호가 아니라 **기준에서 지난 주 수**로 맞춰야 한다.
    gap = {t_rel: cols[t_rel] - cols[t_rel - 1], t_red: cols[t_red] - cols[t_red - 1]}
    rows = []
    for tlab, t, key in (("해제 (2/13)", t_rel, "해제"),
                         ("재지정 (3/24)", t_red, "부과")):
        print(f"\n  {tlab}  — {key}  (기준 {cols[t-1]}주, 첫 사후 {cols[t]}주)")
        print(f"  {'경과':>5}{'단순평균':>11}{'95% 구간':>21}"
              f"{'거래 가중':>11}{'95% 구간':>21}{'집계 배수':>11}")
        for l in range(LAGS):
            if t + l >= len(cols):
                continue
            r1 = didm(Ya, Da, t, l)
            r2 = didm(Ya, Da, t, l, base)
            if key not in r1:
                continue
            elapsed = cols[t + l] - cols[t - 1]
            b1 = boot(Ya, Da, grp, t, l, np.ones(len(Ya)), rng, key)
            b2 = boot(Ya, Da, grp, t, l, base, rng, key)
            c1 = np.percentile(b1, [2.5, 97.5])
            c2 = np.percentile(b2, [2.5, 97.5])
            ag = agg_ratio(t, l, key)
            rows.append({"전환": tlab, "성분": key, "시차_주": l,
                         "경과_주": int(elapsed),
                         "단순_pct": round(pct(r1[key][0]), 1),
                         "단순_lo": round(pct(c1[0]), 1), "단순_hi": round(pct(c1[1]), 1),
                         "가중_pct": round(pct(r2[key][0]), 1),
                         "가중_lo": round(pct(c2[0]), 1), "가중_hi": round(pct(c2[1]), 1),
                         "집계_pct": round(pct(ag), 1),
                         "전환_단지": r1[key][1], "비교_단지": r1[key][2]})
            print(f"  {elapsed:>4}주{pct(r1[key][0]):>10.1f}%"
                  f"  [{pct(c1[0]):>6.1f}%,{pct(c1[1]):>6.1f}%]"
                  f"{pct(r2[key][0]):>10.1f}%"
                  f"  [{pct(c2[0]):>6.1f}%,{pct(c2[1]):>6.1f}%]"
                  f"{pct(ag):>10.1f}%")
        n = rows[-1]
        print(f"  (전환 단지 {n['전환_단지']}개 · 비교 단지 {n['비교_단지']}개)")
    out["DID_M"] = rows

    # 재부과(해제군)와 최초 부과(새로 규제)를 나눠 본다
    print("\n  재지정을 둘로 나누면 — 규제를 39일 겪어 본 쪽과 처음 겪는 쪽")
    print(f"  {'경과':>5}{'재부과 (해제군)':>18}{'최초 부과 (새로 규제)':>22}")
    split = []
    idx_rel = grp == "해제군"
    idx_new = grp == "새로 규제"
    for l in range(LAGS):
        if t_red + l >= len(cols):
            continue
        e = cols[t_red + l] - cols[t_red - 1]
        a = agg_ratio(t_red, l, "부과", idx_rel)
        b2_ = agg_ratio(t_red, l, "부과", idx_new)
        split.append({"경과_주": int(e), "재부과_pct": round(pct(a), 1),
                      "최초부과_pct": round(pct(b2_), 1)})
        print(f"  {e:>4}주{pct(a):>17.1f}%{pct(b2_):>21.1f}%")
    print("  (집계 배수. 두 집단 모두 계속 꺼져 있던 집단과 견준다)")
    out["부과_분해"] = split

    # ==================================================================
    print("\n" + "=" * 88)
    print("5. 위약 — 전환 직전 한 주에 같은 계산을 건다")
    print("=" * 88)
    pl = []
    for tlab, t, key in (("해제 (2/13)", t_rel, "해제"),
                         ("재지정 (3/24)", t_red, "부과")):
        d0, d1 = Da[:, t - 1], Da[:, t]
        dy = Ya[:, t - 1] - Ya[:, t - 2]
        si, so = (d0 == 0) & (d1 == 1), (d0 == 1) & (d1 == 0)
        s0, s1 = (d0 == 0) & (d1 == 0), (d0 == 1) & (d1 == 1)
        v = (dy[si].mean() - dy[s0].mean()) if key == "부과" else \
            (dy[s1].mean() - dy[so].mean())
        pl.append({"전환": tlab, "성분": key, "위약_pct": round(pct(v), 1)})
        print(f"  {tlab:<16}{key}  {pct(v):+7.1f}%")
    print("  막차 주간을 뺐으므로 재지정 위약이 +19.5% 에서 내려와야 한다")
    out["위약"] = pl

    # ==================================================================
    print("\n" + "=" * 88)
    print("6. 부과와 해제는 대칭인가")
    print("=" * 88)
    # **기준에서 지난 주 수**로 맞춰야 한다. 시차 번호로 맞추면 부과 쪽은
    # 2주, 해제 쪽은 1주짜리 변화를 나란히 놓게 된다(막차 주간을 뺐기 때문).
    sym = []
    print(f"  {'경과':>5}{'부과':>22}{'해제':>22}{'차이':>10}   (거래 가중)")
    for e in sorted({r["경과_주"] for r in rows}):
        a = [r for r in rows if r["성분"] == "부과" and r["경과_주"] == e]
        b = [r for r in rows if r["성분"] == "해제" and r["경과_주"] == e]
        if not (a and b):
            continue
        ov = not (a[0]["가중_hi"] < b[0]["가중_lo"] or b[0]["가중_hi"] < a[0]["가중_lo"])
        sym.append({"경과_주": e, "부과_pct": a[0]["가중_pct"],
                    "해제_pct": b[0]["가중_pct"],
                    "차이_pct포인트": round(a[0]["가중_pct"] - b[0]["가중_pct"], 1),
                    "구간_겹침": bool(ov)})
        print(f"  {e:>4}주{a[0]['가중_pct']:>9.1f}%"
              f"  [{a[0]['가중_lo']:>6.1f},{a[0]['가중_hi']:>6.1f}]"
              f"{b[0]['가중_pct']:>9.1f}%"
              f"  [{b[0]['가중_lo']:>6.1f},{b[0]['가중_hi']:>6.1f}]"
              f"{a[0]['가중_pct'] - b[0]['가중_pct']:>9.1f}%p"
              + ("  구간 겹침" if ov else "  안 겹침"))
    out["대칭성"] = sym

    # ==================================================================
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.8))
    ax = axes[0]
    ww = w25["_w"][w25["_tr"]]
    ax.hist(ww * 1e4, bins=60, color="#3182ce", alpha=.85)
    ax.axvline(0, color="#c53030", lw=1.4)
    ax.set_title(f"처치 칸의 가중치 — 음수 {w25['음가중_비율']}% (합 {w25['음가중_합']:+.3f})")
    ax.set_xlabel("가중치 (1만분의 1 단위)")
    ax.set_ylabel("칸 수")
    ax.grid(alpha=.25)

    ax = axes[1]
    for key, c in (("부과", "#c53030"), ("해제", "#3182ce")):
        s = [r for r in rows if r["성분"] == key]
        if not s:
            continue
        x = [r["경과_주"] for r in s]
        ax.plot(x, [r["가중_pct"] for r in s], "o-", color=c, lw=1.8, label=key)
        ax.fill_between(x, [r["가중_lo"] for r in s], [r["가중_hi"] for r in s],
                        color=c, alpha=.15)
    ax.axhline(0, color="#1a202c", lw=.9)
    ax.set_title("DID_M — 규제가 있을 때 거래가 얼마나 낮은가")
    ax.set_xlabel("기준에서 지난 주")
    ax.set_ylabel("효과 (%)")
    ax.legend(fontsize=9)
    ax.grid(alpha=.25)

    ax = axes[2]
    wt = (w.groupby(["집단", "wk"]).size().unstack(fill_value=0)
          .div(w.groupby("집단").aptSeq.nunique(), axis=0))
    for k, c in (("해제군", "#c53030"), ("유지군", "#3182ce"),
                 ("새로 규제", "#718096"), ("인접 4개 구", "#38a169")):
        if k in wt.index:
            ax.plot(wt.columns, wt.loc[k], "o-", color=c, lw=1.5, ms=3.5, label=k)
    ax.axvspan(-0.5, WK_RUSH + 0.5, color="#fbd38d", alpha=.35)
    ax.text(2.5, ax.get_ylim()[1] * .92, "해제 구간", fontsize=9, ha="center")
    ax.set_title("단지당 주간 거래 — 노란 구간이 규제가 풀린 39일")
    ax.set_xlabel("해제일 기준 주")
    ax.set_ylabel("단지당 거래 건수")
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)

    fig.suptitle("처치 역전 — 음의 가중치 진단과 부과·해제 분리 (dCDH 2020)", y=1.03)
    fig.tight_layout()
    fp = io.FIGURES / "10_dcdh.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")

    out.update({"창_주": [WK_LO, WK_HI], "막차주간": WK_RUSH, "B": B, "seed": SEED})
    io.save_result("10_dcdh", out)
    print(f"\n저장 {fp} · output/results/10_dcdh.json")


if __name__ == "__main__":
    main()
