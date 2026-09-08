"""처치군 후보를 인과분석 적합도로 줄 세운다.

    python scripts/rank_candidates.py [--pre 36] [--post 12]

입력  data/policy/ltz_panel.csv           (build_ltz_panel.py)
      data/processed/trades*.parquet
출력  output/results/candidates.json

**후보 하나 = 같은 자치구에서 같은 달에 함께 켜진 법정동 묶음**이다.
공고 하나가 여러 자치구를 건드리면 자치구별로 쪼갠다. 통제군을 같은
자치구 안에서 찾을 것이기 때문이다.

왜 이런 기준들인가:

  통제군    같은 자치구에 그때 미지정인 동이 남아야 비교가 성립한다.
            남지 않으면 (2025.10 서울 전역처럼) 반사실을 만들 수 없다.
  사전청결   처치 전 창에 다른 지정이 없어야 '처치 전'이 진짜 처치 전이다.
  사후길이   효과가 나타나고 추정될 시간.
  표본       거래가 적으면 표준오차가 커서 아무 말도 못 한다.
  15억차     2019-12 시행 15억 초과 주담대 금지 노출도. 처치군과 통제군이
            크게 다르면 토허제 효과와 대출 효과가 분리되지 않는다 (0011).
  역전       해제 후 재지정이 있으면 비흡수 처치로 우리 차별점이 된다 (0005).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import io, policy  # noqa: E402

io.setup_stdout()


def load_trades():
    """서울 25개구 자료가 있으면 그것을, 없으면 분석 8개구를 쓴다."""
    p = io.TRADES_SEOUL if io.TRADES_SEOUL.exists() else io.TRADES
    df = pd.read_parquet(p)
    return df, p.name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", type=int, default=36, help="사전 창(개월)")
    ap.add_argument("--post", type=int, default=12, help="필요한 최소 사후 개월")
    a = ap.parse_args()

    pan = pd.read_csv(io.POLICY / "ltz_panel.csv")
    pan["m"] = pd.PeriodIndex(pan.연월, freq="M")
    df, src = load_trades()
    df["m"] = df.deal_date.dt.to_period("M")
    have_gu = set(df.sgg_nm.unique())

    # 동별 지정 시작 달 = 지정인데 직전 달이 지정이 아니었던 달
    on = pan[pan.상태 == "지정"].copy()
    key = ["자치구", "법정동"]
    starts = []
    for k, g in on.groupby(key):
        ms = sorted(g.m)
        prev = None
        for x in ms:
            if prev is None or (x - prev).n > 1:
                starts.append({"자치구": k[0], "법정동": k[1], "시작": x,
                               "계열": g[g.m == x].계열.iloc[0]})
            prev = x
    S = pd.DataFrame(starts)

    # 동별 지정 개월 집합 (사전 청결·사후 길이 판정용)
    onset = {k: set(g.m) for k, g in on.groupby(key)}
    allset = {k: set(g.m) for k, g in pan.groupby(key)}   # 지정+불명

    rows = []
    for (gu, start), g in S.groupby(["자치구", "시작"]):
        dongs = sorted(g.법정동)
        # 사전 청결 — 처치 전 pre 개월 동안 어느 동도 지정/불명이 아니어야
        pre_ms = {start - i for i in range(1, a.pre + 1)}
        dirty = sum(bool(allset.get((gu, d), set()) & pre_ms) for d in dongs)
        # 사후 길이 — 모든 동이 연속 지정된 개월 수 (최솟값)
        def run(d):
            s, n = onset.get((gu, d), set()), 0
            while start + n in s:
                n += 1
            return n
        post = min(run(d) for d in dongs)
        # 같은 자치구의 통제 후보 — 그 시점에 지정/불명이 아닌 동
        win = {start + i for i in range(-a.pre, post)}
        gu_dongs = set(df[df.sgg_nm == gu].umdNm.unique()) if gu in have_gu else set()
        ctrl = [d for d in gu_dongs - set(dongs)
                if not (allset.get((gu, d), set()) & win)]

        if gu in have_gu:
            t = df[(df.sgg_nm == gu) & df.umdNm.isin(dongs)]
            c = df[(df.sgg_nm == gu) & df.umdNm.isin(ctrl)]
            pre_win = {start - i for i in range(1, a.pre + 1)}
            tp, cp = t[t.m.isin(pre_win)], c[c.m.isin(pre_win)]
            n_t, n_c = len(tp), len(cp)
            e_t = 100 * (tp.amount_manwon > 150000).mean() if n_t else float("nan")
            e_c = 100 * (cp.amount_manwon > 150000).mean() if n_c else float("nan")
        else:
            n_t = n_c = 0
            e_t = e_c = float("nan")

        rows.append({
            "자치구": gu, "시작": str(start), "처치동수": len(dongs),
            "처치동": ",".join(dongs)[:38], "계열": g.계열.iloc[0][:18],
            "사전오염동": dirty, "사후개월": post,
            "통제동수": len(ctrl), "처치거래": n_t, "통제거래": n_c,
            "15억_처치": e_t, "15억_통제": e_c,
            "15억차": abs(e_t - e_c) if n_t and n_c else float("nan"),
            "자료있음": gu in have_gu,
        })

    R = pd.DataFrame(rows)
    ok = R[(R.사전오염동 == 0) & (R.사후개월 >= a.post) & (R.통제동수 > 0) &
           (R.처치거래 > 0)].copy()
    ok = ok.sort_values(["처치거래"], ascending=False)

    print(f"거래 자료: {src} (자치구 {len(have_gu)}개)")
    print(f"후보 {len(R)}개 중 조건 통과 {len(ok)}개  "
          f"[사전 {a.pre}개월 무오염 · 사후 {a.post}개월 이상 · 통제동 존재]\n")
    print(f"{'시작':<9}{'자치구':<7}{'처치동':<28}{'사후':>4}{'통제동':>5}"
          f"{'처치거래':>8}{'통제거래':>8}{'15억차':>7}  계열")
    print("-" * 110)
    for _, r in ok.iterrows():
        print(f"{r.시작:<9}{r.자치구:<7}{r.처치동[:26]:<28}{r.사후개월:>4}"
              f"{r.통제동수:>5}{r.처치거래:>8,}{r.통제거래:>8,}{r['15억차']:>7.0f}"
              f"  {r.계열}")

    fail = R[~R.index.isin(ok.index) & R.자료있음]
    print(f"\n탈락 {len(fail)}개 — 사유")
    for reason, n in [("사전 창에 다른 지정", int((fail.사전오염동 > 0).sum())),
                      (f"사후 {a.post}개월 미만", int((fail.사후개월 < a.post).sum())),
                      ("같은 구에 통제동 없음", int((fail.통제동수 == 0).sum())),
                      ("처치동 거래 0", int((fail.처치거래 == 0).sum()))]:
        print(f"  {reason:<24}{n:>3}개")
    print(f"\n자료 없는 자치구의 후보 {int((~R.자료있음).sum())}개는 판정 보류")

    io.save_result("candidates", {
        "거래자료": src, "사전창": a.pre, "최소사후": a.post,
        "후보수": len(R), "통과": len(ok),
        "상위": ok.head(10).drop(columns=["자료있음"]).to_dict("records"),
    })


if __name__ == "__main__":
    main()
