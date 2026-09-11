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


def _cell(v) -> str:
    """마크다운 표 칸. 계열의 '|' 가 칸 구분자와 충돌하므로 바꿔 준다."""
    return str(v).replace("|", " · ")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", type=int, default=36, help="사전 창(개월)")
    ap.add_argument("--post", type=int, default=12, help="필요한 최소 사후 개월")
    ap.add_argument("--minobs", type=int, default=500,
                    help="처치·통제 중 적은 쪽의 최소 거래 수")
    a = ap.parse_args()

    pan = pd.read_csv(io.POLICY / "ltz_panel.csv")
    z = pd.read_csv(io.POLICY / "notices" / "zones.csv", dtype={"지번": str})
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
    # 같은 달에 성격이 다른 지정이 겹칠 수 있다. 2021-04-04 공공재개발과
    # 2021-04-27 주요재건축단지가 그렇다. 계열까지 나눠야 한 후보가
    # 두 정책의 혼합이 되지 않는다.
    for (gu, start, ser), g in S.groupby(["자치구", "시작", "계열"]):
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

        # 지정 방식 — 처치가 법정동을 얼마나 덮는가.
        # 공고가 지번 없이 동 이름만 적으면 동 전체 또는 지구 단위다 (덮음).
        # 지번을 적으면 그 일대만이라 동의 1~3%다. 처치를 법정동 단위로
        # 주는 우리 설계에서는 후자가 측정오차가 되어 추정치를 0으로 끈다.
        # 그 후보가 시작된 달의 공고만 본다. 같은 동이 몇 해 뒤 다른 공고에
        # 지번 단위로 다시 나오면, 그것까지 세어 2020년 후보가 '혼합'으로
        # 잘못 분류됐다.
        zz = z[(z.자치구 == gu) & z.법정동.isin(dongs) &
               (pd.PeriodIndex(pd.to_datetime(z.효력일), freq="M") == start)]
        no_j = zz.지번.isna() | (zz.지번.astype(str).str.strip() == "")
        방식 = "동전체" if no_j.all() else ("일부" if not no_j.any() else "혼합")

        rows.append({
            "자치구": gu, "시작": str(start), "처치동수": len(dongs),
            "처치동": ",".join(dongs)[:38], "계열": ser[:20],
            "지정방식": 방식,
            "사전오염동": dirty, "사후개월": post,
            "통제동수": len(ctrl), "처치거래": n_t, "통제거래": n_c,
            "유효표본": min(n_t, n_c),
            "15억_처치": e_t, "15억_통제": e_c,
            "15억차": abs(e_t - e_c) if n_t and n_c else float("nan"),
            "자료있음": gu in have_gu,
        })

    R = pd.DataFrame(rows)
    # 통제군이 형해화된 후보를 거른다. 통제 거래가 몇십 건이면 비교가 아니다.
    ok = R[(R.사전오염동 == 0) & (R.사후개월 >= a.post) & (R.통제동수 >= 2) &
           (R.유효표본 >= a.minobs)].copy()
    # 지정방식을 첫 열쇠로 둔다. 구역이 동 전체냐 일부냐가 처치 배정의
    # 정확도를 가르고, 그것이 다른 무엇보다 앞선다 (결정기록 0012).
    ok["_방식"] = ok.지정방식.map({"동전체": 0, "혼합": 1, "일부": 2}).fillna(3)
    ok = ok.sort_values(["_방식", "사후개월", "유효표본"],
                        ascending=[True, False, False]).drop(columns=["_방식"])

    print(f"거래 자료: {src} (자치구 {len(have_gu)}개)")
    print(f"후보 {len(R)}개 중 조건 통과 {len(ok)}개  "
          f"[사전 {a.pre}개월 무오염 · 사후 {a.post}개월 이상 · 통제동 존재]\n")
    print(f"{'시작':<9}{'자치구':<7}{'처치동':<28}{'사후':>4}{'통제동':>5}"
          f"{'처치거래':>8}{'통제거래':>8}{'15억차':>7}{'방식':>6}  계열")
    print("-" * 118)
    for _, r in ok.iterrows():
        print(f"{r.시작:<9}{r.자치구:<7}{r.처치동[:26]:<28}{r.사후개월:>4}"
              f"{r.통제동수:>5}{r.처치거래:>8,}{r.통제거래:>8,}{r['15억차']:>7.0f}"
              f"{r.지정방식:>6}  {r.계열}")

    fail = R[~R.index.isin(ok.index) & R.자료있음]
    print(f"\n탈락 {len(fail)}개 — 사유")
    for reason, n in [("사전 창에 다른 지정", int((fail.사전오염동 > 0).sum())),
                      (f"사후 {a.post}개월 미만", int((fail.사후개월 < a.post).sum())),
                      ("통제동 2개 미만", int((fail.통제동수 < 2).sum())),
                      (f"유효표본 {a.minobs}건 미만", int((fail.유효표본 < a.minobs).sum())),
                      ]:
        print(f"  {reason:<24}{n:>3}개")
    print(f"\n자료 없는 자치구의 후보 {int((~R.자료있음).sum())}개는 판정 보류")

    # 표를 손으로 옮기면 반드시 어긋난다 (결정기록 0007). 문서로 직접 쓴다.
    md = io.ROOT / "docs" / "design" / "처치군-후보표.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    L = ["# 처치군 후보 전수표", "",
         "이 문서는 `scripts/rank_candidates.py` 가 생성한다. **직접 고치지 말 것.**",
         "판단과 결정은 [결정기록 0012](../decisions/0012-처치군-우선순위.md) 에 있다.", "",
         f"- 거래 자료: `{src}` (자치구 {len(have_gu)}개)",
         f"- 후보 {len(R)}개 · 조건 통과 {len(ok)}개",
         f"- 조건: 사전 {a.pre}개월 무오염 · 사후 {a.post}개월 이상 · "
         f"통제동 2개 이상 · 유효표본 {a.minobs}건 이상", "",
         "## 통과한 후보", "",
         "| 순 | 시작 | 자치구 | 처치동 | 계열 | 지정방식 | 사후 | 통제동 | 처치거래 | 통제거래 | 유효표본 | 15억차 |",
         "|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for i, (_, r) in enumerate(ok.iterrows(), 1):
        L.append(f"| {i} | {r.시작} | {r.자치구} | {_cell(r.처치동)} | {_cell(r.계열)} | "
                 f"**{r.지정방식}** | {r.사후개월} | {r.통제동수} | {r.처치거래:,} | "
                 f"{r.통제거래:,} | {r.유효표본:,} | {r['15억차']:.0f}%p |")
    fail = R[~R.index.isin(ok.index)]
    L += ["", f"## 탈락한 후보 {len(fail)}개 — 사유별", "",
          "한 후보가 여러 사유에 걸릴 수 있어 합이 전체보다 크다.", "",
          "| 사유 | 개수 |", "|---|---:|"]
    for reason, n in [("사전 창에 다른 지정이 있었다", int((fail.사전오염동 > 0).sum())),
                      (f"사후 {a.post}개월 미만", int((fail.사후개월 < a.post).sum())),
                      ("같은 자치구에 통제동이 2개 미만", int((fail.통제동수 < 2).sum())),
                      (f"유효표본 {a.minobs}건 미만", int((fail.유효표본 < a.minobs).sum()))]:
        L.append(f"| {reason} | {n} |")
    L += ["", "## 칸 뜻", "",
          "| 칸 | 뜻 |", "|---|---|",
          "| 후보 | 같은 자치구에서, 같은 달에, 같은 계열로 함께 지정된 법정동 묶음 |",
          "| 지정방식 | 공고가 지번 없이 동 이름만 적으면 `동전체`, 지번을 적으면 `일부` |",
          "| 사후 | 지정이 끊기지 않고 이어진 개월 수 |",
          "| 유효표본 | 처치·통제 중 **적은 쪽**의 처치 전 36개월 거래 수 |",
          "| 15억차 | 처치군과 통제군의 15억 초과 거래 비율 차이(%p). "
          "2019-12 대출 규제 노출도가 얼마나 다른가 |", ""]
    md.write_text("\n".join(L), encoding="utf-8")
    print(f"\n표 저장: {md.relative_to(io.ROOT)}")

    io.save_result("candidates", {
        "거래자료": src, "사전창": a.pre, "최소사후": a.post,
        "후보수": len(R), "통과": len(ok),
        "상위": ok.head(10).drop(columns=["자료있음"]).to_dict("records"),
    })


if __name__ == "__main__":
    main()
