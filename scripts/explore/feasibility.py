"""주제 확정 전 관문: 처치/통제 표본이 실제로 충분한지 실측한다.

    python scripts/build_dataset.py && python scripts/feasibility.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import io, policy  # noqa: E402

io.setup_stdout()


# 토허구역 주요 이벤트 (data/policy/ltz_events_DRAFT.csv 와 동기화 필요, 고시 원문 검증 전)
EVENTS = [
    ("2020-06-23 지정", "2020-06-23", {"강남구": ["삼성동", "대치동", "청담동"],
                                        "송파구": ["잠실동"]}),
    ("2021-04-27 지정", "2021-04-27", {"강남구": ["압구정동"],
                                        "성동구": ["성수동1가"]}),
    ("2025-02-12 해제", "2025-02-12", {"강남구": ["삼성동", "대치동", "청담동"],
                                        "송파구": ["잠실동"]}),
]
WINDOWS = (12, 24)


def window_counts(df: pd.DataFrame, date: str, treated: dict, months: int) -> pd.DataFrame:
    d = pd.Timestamp(date)
    lo, hi = d - pd.DateOffset(months=months), d + pd.DateOffset(months=months)
    sub = df[(df.deal_date >= lo) & (df.deal_date < hi)].copy()
    sub = sub[sub.sgg_nm.isin(treated)]
    sub["treated"] = [row.umdNm in treated.get(row.sgg_nm, []) for row in sub.itertuples()]
    sub["post"] = sub.deal_date >= d

    g = (sub.groupby(["sgg_nm", "treated", "post"]).size()
         .unstack("post", fill_value=0).rename(columns={False: "사전", True: "사후"}))
    g.index = g.index.set_levels(["통제동", "처치동"], level="treated")
    return g


def main() -> None:
    df = io.load_trades()

    print("=" * 66)
    print("A. 수집 현황")
    print("=" * 66)
    cov = df.groupby("sgg_nm").agg(거래=("amount_eok", "size"),
                                    시작=("deal_date", "min"), 종료=("deal_date", "max"))
    cov["시작"] = cov["시작"].dt.date
    cov["종료"] = cov["종료"].dt.date
    print(cov.to_string())

    print()
    print("=" * 66)
    print("B. 이벤트별 처치/통제 표본  (DID 성립 여부의 1차 관문)")
    print("=" * 66)
    for label, date, treated in EVENTS:
        print(f"\n[{label}]  처치동: "
              + ", ".join(f"{k} {'/'.join(v)}" for k, v in treated.items()))
        for m in WINDOWS:
            g = window_counts(df, date, treated, m)
            if g.empty:
                print(f"  ±{m}개월: 표본 없음")
                continue
            print(f"  ±{m}개월")
            print("    " + g.to_string().replace("\n", "\n    "))

    print()
    print("=" * 66)
    print("C. 반복매매 쌍  (자체 가격지수 구축 가능성)")
    print("=" * 66)
    key = ["aptSeq", "area_m2", "floor_no"]
    rep = df.groupby(key).size()
    print(f"  동일 (단지·면적·층) 조합: {len(rep):,}개")
    print(f"  2회 이상 거래된 조합    : {(rep >= 2).sum():,}개")
    print(f"  그로부터 만들 수 있는 쌍: {(rep - 1).clip(lower=0).sum():,}쌍")

    print()
    print("=" * 66)
    print("D. 처치동 월별 거래 밀도  (사건 전후 추세를 그릴 수 있는가)")
    print("=" * 66)
    focus = df[df.umdNm.isin(["삼성동", "대치동", "청담동", "잠실동", "압구정동"])]
    if not focus.empty:
        t = (focus.assign(연=focus.deal_date.dt.year)
             .pivot_table(index="연", columns="umdNm", values="amount_eok", aggfunc="size")
             .fillna(0).astype(int))
        print(t.loc[t.index >= 2018].to_string())


if __name__ == "__main__":
    main()
