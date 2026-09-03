"""[02] 원자료(JSON)를 분석용 parquet 한 장으로 정제.

    python scripts/02_build.py   ->  data/processed/trades.parquet

정제 규칙은 결정기록 0004 참조.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import io, policy  # noqa: E402

io.setup_stdout()

# 데이터 검증 기준. 파이프라인이 조용히 깨지는 것을 막는다.
EXPECT_MIN_ROWS = 300_000
EXPECT_MIN_APTS = 2_000
EXPECT_START = pd.Timestamp("2006-12-31")


def load_raw() -> pd.DataFrame:
    frames = []
    for f in sorted(io.RAW.rglob("*.json")):
        rows = json.loads(f.read_text(encoding="utf-8"))
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        sys.exit("data/raw 가 비어 있습니다. scripts/01_collect.py 를 먼저 실행하세요.")
    return pd.concat(frames, ignore_index=True)


def build(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    n0 = len(df)

    df["amount_manwon"] = (df["dealAmount"].astype(str)
                           .str.replace(",", "", regex=False).str.strip()
                           .replace("", np.nan).astype(float))
    df["amount_eok"] = df["amount_manwon"] / 10_000

    df["deal_date"] = pd.to_datetime(
        df["dealYear"].astype(str).str.zfill(4)
        + df["dealMonth"].astype(str).str.zfill(2)
        + df["dealDay"].astype(str).str.zfill(2),
        format="%Y%m%d", errors="coerce")
    df["ym"] = df["deal_date"].dt.to_period("M")

    for c in ("excluUseAr", "floor", "buildYear"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.rename(columns={"excluUseAr": "area_m2", "floor": "floor_no",
                            "buildYear": "build_year"})

    df["price_per_m2"] = df["amount_manwon"] / df["area_m2"]
    df["log_price_per_m2"] = np.log(df["price_per_m2"])
    df["sgg_nm"] = df["sggCd"].map(policy.SGG_NAME).fillna(df["sggCd"])
    df["umd_full_cd"] = df["sggCd"].astype(str) + df["umdCd"].astype(str)
    df["age_at_deal"] = df["dealYear"].astype(int) - df["build_year"]

    # --- 표본 필터 (결정기록 0004) ---
    flags = {}
    cancelled = df["cdealType"].fillna("").str.strip().eq("O")
    flags["해제(취소)된 거래"] = int(cancelled.sum())
    df = df[~cancelled]

    bad = df["deal_date"].isna() | df["amount_manwon"].isna() | df["area_m2"].isna()
    flags["날짜/금액/면적 결측"] = int(bad.sum())
    df = df[~bad]

    nonpos = (df["area_m2"] <= 0) | (df["amount_manwon"] <= 0)
    flags["면적/금액 <= 0"] = int(nonpos.sum())
    df = df[~nonpos]

    late = df["deal_date"] > policy.SAMPLE_END
    flags[f"신고지연 절단 (> {policy.SAMPLE_END.date()})"] = int(late.sum())
    df = df[~late]

    print(f"원자료 {n0:,}건")
    for k, v in flags.items():
        print(f"  제외 - {k}: {v:,}건")
    print(f"분석표본 {len(df):,}건")

    keep = ["deal_date", "ym", "sggCd", "sgg_nm", "umdCd", "umd_full_cd", "umdNm",
            "aptSeq", "aptNm", "aptDong", "jibun", "bonbun", "bubun",
            "roadNmCd", "roadNmBonbun", "roadNmBubun",
            "area_m2", "floor_no", "build_year", "age_at_deal",
            "amount_manwon", "amount_eok", "price_per_m2", "log_price_per_m2",
            "dealingGbn", "slerGbn", "buyerGbn", "landLeaseholdGbn", "rgstDate"]
    out = (df[[c for c in keep if c in df.columns]]
           .sort_values("deal_date").reset_index(drop=True))
    return out, flags


def validate(df: pd.DataFrame) -> None:
    """데이터가 조용히 깨지는 것을 막는 최소 검증."""
    problems = []
    if len(df) < EXPECT_MIN_ROWS:
        problems.append(f"표본 {len(df):,}건 < 기대 {EXPECT_MIN_ROWS:,}건")
    if df.aptSeq.nunique() < EXPECT_MIN_APTS:
        problems.append(f"단지 {df.aptSeq.nunique():,}개 < 기대 {EXPECT_MIN_APTS:,}개")
    if df.deal_date.min() > EXPECT_START:
        problems.append(f"시작일 {df.deal_date.min().date()} 가 예상보다 늦음")
    for c in ("deal_date", "amount_manwon", "area_m2", "aptSeq", "umd_full_cd"):
        if df[c].isna().any():
            problems.append(f"{c} 에 결측 {int(df[c].isna().sum()):,}건")
    if (df.price_per_m2 <= 0).any():
        problems.append("price_per_m2 에 0 이하 존재")
    if problems:
        print("\n[검증 실패]")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("[검증 통과]")


def main() -> None:
    io.PROCESSED.mkdir(parents=True, exist_ok=True)
    df, flags = build(load_raw())
    validate(df)
    df.to_parquet(io.TRADES, index=False)

    print(f"\n저장: {io.TRADES}")
    print(f"기간: {df.deal_date.min().date()} ~ {df.deal_date.max().date()}")
    print(f"자치구 {df.sgg_nm.nunique()}개 / 법정동 {df.umd_full_cd.nunique()}개 "
          f"/ 단지 {df.aptSeq.nunique():,}개")

    io.save_result("02_build", {
        "수집시점": "2026-09-03",
        "표본": int(len(df)),
        "단지": int(df.aptSeq.nunique()),
        "법정동": int(df.umd_full_cd.nunique()),
        "자치구": int(df.sgg_nm.nunique()),
        "시작": str(df.deal_date.min().date()),
        "종료": str(df.deal_date.max().date()),
        "제외": flags,
    })


if __name__ == "__main__":
    main()
