"""캐시된 원자료(JSON)를 분석용 parquet 한 장으로 정제한다.

    python scripts/build_dataset.py
    -> data/processed/trades.parquet
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

SGG_NAME = {
    "11170": "용산구", "11200": "성동구", "11215": "광진구", "11590": "동작구",
    "11650": "서초구", "11680": "강남구", "11710": "송파구", "11740": "강동구",
}


def load_raw() -> pd.DataFrame:
    frames = []
    for f in sorted(RAW.rglob("*.json")):
        rows = json.loads(f.read_text(encoding="utf-8"))
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        sys.exit("data/raw 에 자료가 없습니다. 먼저 collect.py 를 실행하세요.")
    return pd.concat(frames, ignore_index=True)


def build(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)

    # 거래금액: "120,000" (만원) -> 숫자
    df["amount_manwon"] = (
        df["dealAmount"].astype(str).str.replace(",", "", regex=False)
        .str.strip().replace("", np.nan).astype(float)
    )
    df["amount_eok"] = df["amount_manwon"] / 10_000

    # 계약일
    df["deal_date"] = pd.to_datetime(
        df["dealYear"].astype(str).str.zfill(4)
        + df["dealMonth"].astype(str).str.zfill(2)
        + df["dealDay"].astype(str).str.zfill(2),
        format="%Y%m%d", errors="coerce",
    )
    df["ym"] = df["deal_date"].dt.to_period("M")

    for c in ("excluUseAr", "floor", "buildYear"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.rename(columns={"excluUseAr": "area_m2", "floor": "floor_no",
                            "buildYear": "build_year"})

    # 단가: ㎡당 만원 (헤도닉/지수의 기본 결과변수)
    df["price_per_m2"] = df["amount_manwon"] / df["area_m2"]
    df["log_price_per_m2"] = np.log(df["price_per_m2"])

    df["sgg_nm"] = df["sggCd"].map(SGG_NAME).fillna(df["sggCd"])
    # 법정동 10자리 코드 (처치 배정 키)
    df["umd_full_cd"] = df["sggCd"].astype(str) + df["umdCd"].astype(str)
    df["age_at_deal"] = df["dealYear"].astype(int) - df["build_year"]

    # --- 표본 필터 ---
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
    return df[[c for c in keep if c in df.columns]].sort_values("deal_date").reset_index(drop=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = build(load_raw())
    p = OUT / "trades.parquet"
    df.to_parquet(p, index=False)
    print(f"\n저장: {p}")
    print(f"기간: {df['deal_date'].min().date()} ~ {df['deal_date'].max().date()}")
    print(f"자치구: {df['sgg_nm'].nunique()}개 / 법정동: {df['umd_full_cd'].nunique()}개 "
          f"/ 단지(aptSeq): {df['aptSeq'].nunique():,}개")


if __name__ == "__main__":
    main()
