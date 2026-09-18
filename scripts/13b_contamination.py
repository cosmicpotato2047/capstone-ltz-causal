"""[13b] 오염 동 제외 강건성 — 통제군에서 나중에 지정된 동을 뺀다.

결정기록 0009 는 통제동 중 여섯 곳이 분석 창 안에서 토지거래허가구역으로
지정된 사실을 찾았다. 통제군이 일부 처치를 받으면 처치·통제 격차가 줄어
추정치가 0 쪽으로 끌린다(감쇠). 방향이 유리하다고 넘기지 않고 뺀 사양을
함께 보고하기로 했다.

  압구정동   2021-04-27~  동 전체   (주요재건축단지 — 압구정 아파트지구)
  거여동     2021-04-04~  혼합
  마천동     2022-01-02~  혼합
  송파동     2022-01-29~  혼합
  신천동     2022-01-29~  혼합
  일원동     2022-01-29~  일부

이 목록은 `data/policy/ltz_spells.csv` 에서 매번 다시 뽑는다. 판독이 늘면
목록도 따라 바뀌어야 하기 때문이다(0009 는 판독 336건 시절에 만들었다).

세 사양을 견준다.
  기본          27개 동
  압구정 제외    26개 동 — 동 전체가 지정된 유일한 곳
  오염 6개 제외  21개 동 — 창 안에 어떤 방식으로든 지정된 통제동 전부

각각 사건연구 k=0, 24개월 이중차분, 삼중차분을 재고, 결정기록 0019 의
추론(WCR 부트스트랩, 무작위화)을 붙인다. 통제 동이 줄면 클러스터가 더
적어지므로 관행 p 는 더 믿기 어렵다.

    python scripts/13b_contamination.py
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
from lib import io, policy  # noqa: E402
from lib.inference import (EVENT, KMAX, KMIN, REF, SEED, SGG,  # noqa: E402
                           fit, randomization, spec_ddd, spec_did, spec_es,
                           wild_boot)

io.setup_stdout()
for f in ("Malgun Gothic", "Batang"):
    if f in {x.name for x in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

# 사건연구 창과 같다: 사건월 -24 ~ 26
WIN_LO = policy.EVENTS[EVENT]["effect"] - pd.DateOffset(months=24)
WIN_HI = policy.EVENTS[EVENT]["effect"] + pd.DateOffset(months=27) - pd.Timedelta(days=1)
B_13B = 4_999        # 사양이 아홉 개라 06 의 절반으로 둔다


def contaminated(df: pd.DataFrame) -> pd.DataFrame:
    """분석 창 안에서 한 번이라도 지정된 통제동과 그 시작일."""
    sp = pd.read_csv(io.ROOT / "data/policy/ltz_spells.csv",
                     parse_dates=["시작일", "종료일"])
    sp = sp[sp.자치구.isin(SGG) & (sp.상태 == "지정")
            & ~sp.법정동.isin(policy.TREATED_2020)]
    end = sp.종료일.fillna(pd.Timestamp("2099-12-31"))
    sp = sp[(sp.시작일 <= WIN_HI) & (end >= WIN_LO)]
    out = (sp.groupby(["자치구", "법정동"])
           .agg(시작=("시작일", "min"),
                방식=("지정방식", lambda s: "/".join(sorted(set(s)))))
           .reset_index())
    out["사건월"] = policy.event_month(out.시작, EVENT).astype(int)
    n = (df[~df.treated].groupby("umdNm").size().rename("거래"))
    return out.merge(n, left_on="법정동", right_index=True, how="left")


def main() -> None:
    df = io.load_trades()
    df = df[df.sgg_nm.isin(SGG)].copy()
    df["treated"] = policy.assign_2020(df)
    df["em"] = policy.event_month(df.deal_date, EVENT)

    print("=" * 76)
    print("0. 창 안에서 지정된 통제동 (ltz_spells.csv 에서 다시 뽑음)")
    print("=" * 76)
    bad = contaminated(df)
    print(bad.sort_values("시작").to_string(index=False))
    first = int(bad.사건월.min())
    print(f"\n  가장 이른 오염이 사건월 {first} 이다. 지정 직후(0~2)에는 통제군 오염이 없다.")
    print("  따라서 k=0 이 움직인다면 오염 때문이 아니라 통제군 구성이 바뀌어서다.")

    samples = {
        "기본": df,
        "압구정 제외": df[df.umdNm != "압구정동"],
        "오염 6개 제외": df[~df.umdNm.isin(bad.법정동)],
    }

    rows, profiles = [], {}
    for sname, d in samples.items():
        d = d.copy()
        specs = [spec_es(d), spec_did(d)]
        s3, _ = spec_ddd(d)
        specs.append(s3)
        G, nt = specs[0].G, int(specs[0].treated.sum())
        print("\n" + "=" * 76)
        print(f"{sname} — 법정동 {G}개 (처치 {nt}, 통제 {G - nt})")
        print("=" * 76)
        print(f"  {'사양':<16}{'계수':>9}{'효과':>9}{'WCR 양측':>11}"
              f"{'무작위 등꼬리':>14}{'순위':>18}")
        rng = np.random.default_rng(SEED)
        for s in specs:
            coef, t0, p1, p2, _ = wild_boot(s, rng, B=B_13B)
            vals, act, N_all = randomization(s)
            N = vals.size
            rank = int((vals <= act).sum())
            p_eq = min(1.0, 2 * min(rank / N, float((vals >= act).mean())))
            eff = float(100 * np.expm1(coef))
            rows.append({"표본": sname, "법정동": G, "사양": s.name,
                         "계수": round(coef, 4), "효과_pct": round(eff, 1),
                         "WCR_p_양측": round(p2, 4), "RI_p_등꼬리": round(p_eq, 4),
                         "RI_순위": rank, "RI_경우의수": N})
            print(f"  {s.name:<16}{coef:>+9.3f}{eff:>8.1f}%{p2:>11.4f}"
                  f"{p_eq:>14.4f}{f'{rank:,} / {N:,}':>18}")
        # 사건연구 전체 궤적 (그림용)
        s = specs[0]
        b, *_ = fit(s, s.make(s.treated))
        ks = [k for k in range(KMIN, KMAX + 1) if k != REF]
        profiles[sname] = dict(zip(ks, b))

    tab = pd.DataFrame(rows)

    print("\n" + "=" * 76)
    print("요약 — 기본 대비 변화")
    print("=" * 76)
    piv = tab.pivot(index="사양", columns="표본", values="효과_pct")[list(samples)]
    print(piv.to_string())
    print("\n  24개월 이중차분이 더 음(-)으로 가면 오염이 효과를 깎고 있었다는 뜻이다(감쇠).")

    # --- 그림 -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 4.8))
    colors = {"기본": "#2b6cb0", "압구정 제외": "#dd6b20", "오염 6개 제외": "#c53030"}
    for i, (sname, prof) in enumerate(profiles.items()):
        ks = sorted(list(prof) + [REF])
        ys = [100 * np.expm1(prof.get(k, 0.0)) for k in ks]
        ax.plot(np.array(ks) + (i - 1) * 0.08, ys, "o-", ms=4, lw=1.4,
                color=colors[sname], label=sname)
    # 같은 분기에 지정된 동은 이름을 한 줄로 묶는다
    bad["kq"] = np.floor(bad.사건월 / 3).astype(int)
    for kq, grp in bad.groupby("kq"):
        ax.axvline(kq, color="#999", ls=":", lw=0.9)
        ax.text(kq + 0.08, 0.97, "\n".join(grp.법정동), transform=ax.get_xaxis_transform(),
                fontsize=8, color="#555", va="top")
    ax.axhline(0, color="#999", lw=.8)
    ax.axvline(-0.5, color="#333", ls="--", lw=1)
    ax.set_xlabel("지정 시점 기준 분기 (0 = 2020-06-23 ~ 09-22)")
    ax.set_ylabel("통제동 대비 처치동 거래량 (%)")
    ax.set_title("통제군에서 오염 동을 뺄 때 사건연구 궤적 — 점선은 각 동의 지정 시작")
    ax.legend(loc="lower left")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fp = io.FIGURES / "13b_contamination.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")

    io.save_result("13b_contamination", {
        "오염동": [{"자치구": r.자치구, "법정동": r.법정동, "시작": str(r.시작.date()),
                    "사건월": int(r.사건월), "방식": r.방식, "거래": int(r.거래)}
                   for r in bad.itertuples()],
        "B": B_13B, "seed": SEED,
        "결과": rows,
    })
    print(f"\n저장 {fp} · output/results/13b_contamination.json")


if __name__ == "__main__":
    main()
