"""[06g] 면적 면제 검증 — 이질성은 대출인가 허가 면적 기준인가.

결정기록 0021 에서 '저가 단지'가 사실상 '소형 단지'였고, 원자료로 40㎡ 미만만
거래가 줄지 않았다는 것을 봤다. 2020-06 지정의 허가 대상은 주거지역 대지 18㎡
초과였다. 대지지분이 작은 소형은 허가 없이 거래할 수 있었다.

두 가설을 가른다.

  대출   15억 초과는 이미 대출이 막혀 있었으므로 규제가 가격 수준에 따라 다르게 물렸다
  면제   대지 18㎡ 이하는 허가 대상이 아니었으므로 규제가 크기에 따라 다르게 물렸다

(가) 2020-06-23 지정 — 소형 대 나머지 삼중차분
     그리고 소형을 뺀 표본에서 15억 가격 삼중차분이 남는지 본다.
(나) 2022-06-23 기준 강화(18㎡ → 6㎡, 서울특별시공고 제2022-1795호)
     처치동 소형만 새로 허가 대상이 된다. 면제가 원인이면 이때 처치동 소형이
     줄어야 한다. 대출로는 이 시점을 설명할 수 없다.
     사후는 2022-11-22 까지 다섯 달로 끊는다. 2022-12-01 15억 금지 해제가
     대형 쪽을 움직이기 때문이다.
(다) 위약 — 2021-06-23 재지정은 18㎡ 를 그대로 두었다. 소형에 바뀐 것이 없으므로
     (나)와 같은 사양을 이 날짜에 걸면 0 이어야 한다.

소형 기준은 전용면적 40㎡ 미만이다. 대지지분은 자료에 없다. 40 은 0021 의
탐색을 보고 고른 값이므로 30·50·60㎡ 로도 돌려 함께 보고한다.

    python scripts/06g_area_exemption.py
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
from lib.inference import (EVENT, KMAX, KMIN, SEED, SGG,  # noqa: E402
                           fit, randomization, spec_ddd, spec_did, spec_triple,
                           wild_boot)

io.setup_stdout()
for f in ("Malgun Gothic", "Batang"):
    if f in {x.name for x in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

CUT = 40.0
CUTS = (30.0, 40.0, 50.0, 60.0)
B_06G = 4_999
CONTAM = ["압구정동", "거여동", "마천동", "송파동", "신천동", "일원동"]  # 0020
STRENGTHEN = pd.Timestamp("2022-06-23")     # 18㎡ -> 6㎡
PLACEBO = pd.Timestamp("2021-06-23")        # 18㎡ 그대로 재지정
PRE_B, POST_B = 12, 5                        # (나)·(다) 창: 사건월 -12 ~ 4


def month_index(dates: pd.Series, anchor: pd.Timestamp) -> pd.Series:
    """효력일에 맞춘 경과 개월 (policy.event_month 와 같은 방식)."""
    d = pd.to_datetime(dates) - pd.Timedelta(days=anchor.day - 1)
    return (d.dt.year - anchor.year) * 12 + (d.dt.month - anchor.month)


def run(s, rng) -> dict:
    """계수와 두 추론. b0 = 나머지(대형) 효과, b1 = 소형이 더 움직인 폭."""
    b, *_ = fit(s, s.make(s.treated))
    _, _, _, p_wcr, _ = wild_boot(s, rng, B=B_06G)
    vals, act, N_all = randomization(s)
    left, right = float((vals <= act).mean()), float((vals >= act).mean())
    return {"대형효과_pct": round(float(100 * np.expm1(b[0])), 1),
            "소형효과_pct": round(float(100 * np.expm1(b[0] + b[1])), 1),
            "차이_계수": round(float(b[1]), 4),
            "WCR_p": round(p_wcr, 4),
            "RI_p_등꼬리": round(min(1.0, 2 * min(left, right)), 4),
            "RI_경우의수": int(vals.size), "RI_식별불가": int(N_all - vals.size)}


def line(lab: str, r: dict) -> str:
    return (f"  {lab:<22}{r['대형효과_pct']:>9.1f}%{r['소형효과_pct']:>9.1f}%"
            f"{r['차이_계수']:>+10.3f}{r['WCR_p']:>8.3f}{r['RI_p_등꼬리']:>8.3f}")


HEAD = (f"  {'':<22}{'나머지':>10}{'소형':>10}{'차이':>10}{'WCR':>8}{'무작위':>8}")


def quarterly_triple(d: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    """분기별 (처치 소형 − 통제 소형) − (처치 나머지 − 통제 나머지), 사전 평균 = 0."""
    q = d[d.em.between(lo, hi)]
    cnt = (q.groupby(["umd_full_cd", "small", "em"]).size().rename("n").reset_index())
    cells = q.groupby(["umd_full_cd", "small"]).treated.first().reset_index()
    full = cells.merge(pd.DataFrame({"em": range(lo, hi + 1)}), how="cross")
    full = full.merge(cnt, on=["umd_full_cd", "small", "em"], how="left").fillna({"n": 0})
    full["y"] = np.log1p(full.n)
    full["kq"] = np.floor(full.em / 3).astype(int)
    m = full.groupby(["kq", "treated", "small"]).y.mean().unstack(["treated", "small"])
    tri = (m[(True, True)] - m[(False, True)]) - (m[(True, False)] - m[(False, False)])
    tri = tri - tri[tri.index < 0].mean()
    return tri.rename("삼중차").reset_index()


def quarterly_did(d: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    """크기별 분기 (처치 − 통제) 평균 log1p 거래, 사전 평균 = 0. 원자료."""
    q = d[d.em.between(lo, hi)]
    out = []
    for small, sub in q.groupby("small"):
        cnt = sub.groupby(["umd_full_cd", "em"]).size().rename("n").reset_index()
        cells = sub.groupby("umd_full_cd").treated.first().reset_index()
        full = cells.merge(pd.DataFrame({"em": range(lo, hi + 1)}), how="cross")
        full = full.merge(cnt, on=["umd_full_cd", "em"], how="left").fillna({"n": 0})
        full["y"] = np.log1p(full.n)
        full["kq"] = np.floor(full.em / 3).astype(int)
        m = full.groupby(["kq", "treated"]).y.mean().unstack("treated")
        gap = m[True] - m[False]
        gap = gap - gap[gap.index < 0].mean()
        out.append(pd.DataFrame({"kq": gap.index, "small": small, "격차": gap.values}))
    return pd.concat(out)


def did_gap(sub: pd.DataFrame, lo: int, hi: int, drop: tuple = ()) -> tuple[float, float]:
    """사건월 일부를 뺄 수 있는 이중차분 (강건성 점검용, 관행 p)."""
    codes = sorted(sub.umd_full_cd.unique())
    ems = [e for e in range(lo, hi + 1) if e not in drop]
    full = pd.MultiIndex.from_product([codes, ems], names=["c", "em"]).to_frame(index=False)
    cnt = (sub[sub.em.between(lo, hi)].groupby(["umd_full_cd", "em"]).size()
           .rename("n").reset_index().rename(columns={"umd_full_cd": "c"}))
    p = full.merge(cnt, on=["c", "em"], how="left").fillna({"n": 0})
    p["t"] = p.c.map(sub.groupby("umd_full_cd").treated.first()).astype(float)
    p["d"] = p.t * (p.em >= 0)
    p["y"] = np.log1p(p.n)
    p["em_s"] = p.em.astype(str)
    dm = econ.demean(p[["y", "d"]], p[["c", "em_s"]])
    f = sm.OLS(dm.y, dm[["d"]]).fit(cov_type="cluster", cov_kwds={"groups": p.c})
    return float(100 * np.expm1(f.params.d)), float(f.pvalues.d)


def main() -> None:
    base = io.load_trades()
    base = base[base.sgg_nm.isin(SGG)].copy()
    base["treated"] = policy.assign_2020(base)
    samples = {"기본": base, "오염 6개 제외": base[~base.umdNm.isin(CONTAM)]}
    out: dict = {"소형기준_㎡": CUT}

    # ==================================================================
    print("=" * 84)
    print("(가) 2020-06-23 지정 — 소형 대 나머지")
    print("=" * 84)
    print(HEAD)
    A = []
    for sname, d0 in samples.items():
        rng = np.random.default_rng(SEED)
        d = d0.copy()
        d["em"] = policy.event_month(d.deal_date, EVENT)
        for cut in CUTS:
            d["small"] = d.area_m2 < cut
            r = run(spec_triple(d, "small", KMIN * 3, KMAX * 3 + 2, "소형"), rng)
            A.append({"표본": sname, "기준_㎡": cut, **r})
            print(line(f"{sname} · {cut:.0f}㎡ 미만", r) + ("  ←주" if cut == CUT else ""))
    out["가_2020_소형대나머지"] = A

    print("\n  소형을 뺀 표본에서 15억 가격 삼중차분이 남는가")
    print(f"  {'':<22}{'15억이하':>10}{'15억초과':>10}{'차이':>10}{'WCR':>8}{'무작위':>8}")
    A2 = []
    for sname, d0 in samples.items():
        rng = np.random.default_rng(SEED)
        for lab, dd in (("전체", d0), (f"{CUT:.0f}㎡ 이상만", d0[d0.area_m2 >= CUT])):
            dd = dd.copy()
            dd["em"] = policy.event_month(dd.deal_date, EVENT)
            s, _ = spec_ddd(dd)
            r = run(s, rng)
            A2.append({"표본": sname, "범위": lab, **r})
            print(line(f"{sname} · {lab}", r))
    out["가_가격삼중차분_소형제외"] = A2
    print("  (여기서 '나머지'는 15억 이하, '소형'은 15억 초과 자리에 찍힌다)")

    # ==================================================================
    for tag, anchor, key in (("(나) 2022-06-23 기준 강화 18㎡ → 6㎡", STRENGTHEN, "나_2022_기준강화"),
                             ("(다) 위약 — 2021-06-23 재지정 (18㎡ 그대로)", PLACEBO, "다_2021_위약")):
        print("\n" + "=" * 84)
        print(f"{tag}   창: 사건월 {-PRE_B} ~ {POST_B - 1}")
        print("=" * 84)
        print(HEAD)
        rows = []
        for sname, d0 in samples.items():
            rng = np.random.default_rng(SEED)
            d = d0.copy()
            d["em"] = month_index(d.deal_date, anchor)
            for cut in CUTS:
                d["small"] = d.area_m2 < cut
                s = spec_triple(d, "small", -PRE_B, POST_B - 1, "소형")
                r = run(s, rng)
                w = d[d.em.between(-PRE_B, POST_B - 1)]
                n_ts = int((w.treated & w.small).sum())
                rows.append({"표본": sname, "기준_㎡": cut, "처치_소형_거래": n_ts, **r})
                print(line(f"{sname} · {cut:.0f}㎡ 미만", r)
                      + f"  (처치 소형 {n_ts}건)" + ("  ←주" if cut == CUT else ""))
        out[key] = rows

    # ==================================================================
    # 삼중차분은 처치동 대형을 비교 기준으로 쓴다. 그런데 2020 지정 뒤 처치동
    # 대형은 폭락했다가 회복 중이었다. 그 회복이 (나)와 (다)에 모두 들어간다.
    # 대형을 빼고 소형끼리(처치동 소형 대 통제동 소형), 대형끼리 따로 본다.
    print("\n" + "=" * 84)
    print("(나')(다') 크기별 이중차분 — 처치동 대 통제동, 같은 크기끼리")
    print("=" * 84)
    print(f"  {'':<34}{'효과':>9}{'WCR':>8}{'무작위':>8}")
    SPLIT = []
    for tag, anchor_d in (("2022-06-23 강화", STRENGTHEN), ("2021-06-23 위약", PLACEBO)):
        for sname in ("기본", "오염 6개 제외"):
            rng = np.random.default_rng(SEED)
            d = samples[sname].copy()
            d["em"] = month_index(d.deal_date, anchor_d)
            for grp, sub in ((f"{CUT:.0f}㎡ 미만", d[d.area_m2 < CUT]),
                             (f"{CUT:.0f}㎡ 이상", d[d.area_m2 >= CUT])):
                s = spec_did(sub, lo=-PRE_B, hi=POST_B - 1, name=grp)
                b, *_ = fit(s, s.make(s.treated))
                _, _, _, p_wcr, _ = wild_boot(s, rng, B=B_06G)
                vals, act, _ = randomization(s)
                l_, r_ = float((vals <= act).mean()), float((vals >= act).mean())
                p_ri = min(1.0, 2 * min(l_, r_))
                eff = float(100 * np.expm1(b[0]))
                SPLIT.append({"사건": tag, "표본": sname, "크기": grp,
                              "효과_pct": round(eff, 1), "WCR_p": round(p_wcr, 4),
                              "RI_p_등꼬리": round(p_ri, 4), "법정동": s.G})
                print(f"  {tag} · {sname} · {grp:<10}{eff:>8.1f}%{p_wcr:>8.3f}{p_ri:>8.3f}")
        print()
    out["크기별_이중차분"] = SPLIT

    # 처치동 소형은 창 안에 130건뿐이라 분기마다 흔들린다. 효력 직전 분기가
    # 튀었으므로 그 분기를 빼거나 사전 창을 늘려도 같은지 본다.
    print(f"  강건성 — {CUT:.0f}㎡ 미만끼리, 관행 p")
    ROB = []
    for tag, anchor_d in (("2022-06-23 강화", STRENGTHEN), ("2021-06-23 위약", PLACEBO)):
        for sname in ("기본", "오염 6개 제외"):
            d = samples[sname].copy()
            d["em"] = month_index(d.deal_date, anchor_d)
            sub = d[d.area_m2 < CUT]
            r = {"사건": tag, "표본": sname}
            for lab, lo, drop in (("기본창", -PRE_B, ()), ("직전분기뺌", -PRE_B, (-3, -2, -1)),
                                  ("사전24개월", -24, ())):
                e, p_ = did_gap(sub, lo, POST_B - 1, drop)
                r[lab] = {"효과_pct": round(e, 1), "관행_p": round(p_, 4)}
            ROB.append(r)
            print(f"  {tag} · {sname:<8}" + "".join(
                f"  {k} {v['효과_pct']:+6.1f}% ({v['관행_p']:.3f})"
                for k, v in r.items() if isinstance(v, dict)))
    out["소형_강건성"] = ROB

    # ==================================================================
    # 그림 — (나)와 (다)의 분기별 삼중차
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    d = samples["오염 6개 제외"].copy()
    d["em"] = policy.event_month(d.deal_date, EVENT)
    d["small"] = d.area_m2 < CUT
    tri = quarterly_triple(d, KMIN * 3, KMAX * 3 + 2)
    ax = axes[0]
    ax.plot(tri.kq, 100 * np.expm1(tri.삼중차), "o-", color="#2b6cb0", lw=1.6)
    ax.axhline(0, color="#999", lw=.8)
    ax.axvline(-0.5, color="#c53030", ls="--", lw=1.2)
    ax.set_title("(가) 2020-06-23 지정 — 소형이 나머지보다 더 움직인 폭")
    ax.set_xlabel("지정 시점 기준 분기")
    ax.grid(alpha=.25)
    for ax, anchor_d, ttl in ((axes[1], STRENGTHEN, "(나) 2022-06-23 기준 강화 18→6㎡"),
                              (axes[2], PLACEBO, "(다) 위약 2021-06-23 (18㎡ 그대로)")):
        dd = samples["오염 6개 제외"].copy()
        dd["em"] = month_index(dd.deal_date, anchor_d)
        dd["small"] = dd.area_m2 < CUT
        g2 = quarterly_did(dd, -PRE_B, POST_B - 1)
        for small, col, lab in ((True, "#c53030", f"소형 {CUT:.0f}㎡ 미만"),
                                (False, "#718096", f"{CUT:.0f}㎡ 이상")):
            t = g2[g2.small == small]
            ax.plot(t.kq, 100 * np.expm1(t.격차), "o-", color=col, lw=1.6, label=lab)
        ax.axhline(0, color="#999", lw=.8)
        ax.axvline(-0.5, color="#333", ls="--", lw=1.2)
        ax.set_title(ttl + " — 크기별 처치 대 통제")
        ax.set_xlabel("효력일 기준 분기")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=.25)
    axes[0].set_ylabel(f"소형({CUT:.0f}㎡ 미만)이 나머지보다\n처치동에서 더 움직인 폭 (%)")
    fig.suptitle("면적 면제 검증 — 오염 6개 제외 표본, 원자료 분기 평균 (사전 평균 = 0)", y=1.02)
    fig.tight_layout()
    fp = io.FIGURES / "06g_area_exemption.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")

    out.update({"B": B_06G, "seed": SEED, "기준강화일": str(STRENGTHEN.date()),
                "위약일": str(PLACEBO.date()), "창_나다": [-PRE_B, POST_B - 1]})
    io.save_result("06g_area_exemption", out)
    print(f"\n저장 {fp} · output/results/06g_area_exemption.json")


if __name__ == "__main__":
    main()
