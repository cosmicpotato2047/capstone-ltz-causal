"""서울 아파트 단지 목록을 한 장의 HTML 로 만든다.

    python scripts/build_apt_directory.py

입력  data/processed/trades_seoul.parquet
      data/policy/ltz_spells.csv
출력  output/apt_directory.html

**이것은 아파트 대장이 아니다.** 실거래 매매 기록에서 만든 것이므로
'2006년 이후 매매가 한 번이라도 있었던 단지'의 목록이다. 거래가 없었던
단지는 나오지 않는다. 또 실거래 자료에는 세대수 칸이 없어서 규모는
거래 건수와 최고층으로 짐작할 수밖에 없다.

단지 9천여 개를 한 번에 그리면 느리므로, 자치구·법정동을 열 때 그 안을
그린다. 검색은 전체를 훑는다.
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

TEMPLATE = Path(__file__).resolve().parent / "templates" / "apt_directory.html"
AREA_LO, AREA_HI = 72.0, 90.0        # 전용 84㎡ 로 보는 범위
RECENT_M = 6                         # 자치구 시세는 최근 몇 개월로 볼 것인가


def ltz_by_dong() -> dict[tuple[str, str], dict]:
    """법정동별 토허 지정 이력 요약. 없으면 빈 칸."""
    p = io.POLICY / "ltz_spells.csv"
    if not p.exists():
        return {}
    s = pd.read_csv(p)
    s = s[s.상태 == "지정"]
    out = {}
    for (gu, dong), g in s.groupby(["자치구", "법정동"]):
        g = g.sort_values("시작일")
        w = g[g.지정방식 == "동전체"]
        # 2025-10-20 서울 전역 지정 때문에 '동 전체 지정 이력 있음'은 340개 동
        # 모두에 붙는다. 그것만 보여주면 신호가 되지 않으므로 **언제 처음**
        # 동 전체가 지정됐는지를 보여준다. 2020~2021년이면 우리 처치군 계열이고
        # 2025-10 이면 전역 지정에 휩쓸린 것이다.
        first = w.시작일.min() if len(w) else None
        ser = sorted({x for c in (w if len(w) else g).계열.dropna()
                      for x in str(c).split("|") if x})
        out[(gu, dong)] = {
            "whole": bool(len(w)), "from": first or g.시작일.min(),
            "part_from": g.시작일.min(),
            "early": bool(first and first < "2025-10-01"),
            "n": int(len(g)), "ser": "·".join(ser)[:40],
        }
    return out


def main() -> None:
    df = pd.read_parquet(io.TRADES_SEOUL)
    df = df.sort_values("deal_date")
    ltz = ltz_by_dong()

    a = df.groupby("aptSeq").agg(
        gu=("sgg_nm", "first"), dong=("umdNm", "first"), nm=("aptNm", "first"),
        jb=("jibun", "first"), yr=("build_year", "median"),
        fl=("floor_no", "max"), nar=("area_m2", "nunique"),
        n=("deal_date", "size"),
        first=("deal_date", "min"), last=("deal_date", "max"),
        hi=("amount_manwon", "max"),
    ).reset_index()

    last = df.groupby("aptSeq").tail(1).set_index("aptSeq")
    a["last_amt"] = a.aptSeq.map(last.amount_manwon)
    a["last_area"] = a.aptSeq.map(last.area_m2)
    # 주력 면적 — 가장 많이 거래된 전용면적
    main_area = (df.groupby(["aptSeq", "area_m2"]).size().rename("k").reset_index()
                 .sort_values("k").groupby("aptSeq").tail(1).set_index("aptSeq"))
    a["m_area"] = a.aptSeq.map(main_area.area_m2)
    # 84㎡ 중위가. 단지별은 20년 전체(거래가 드물어 최근으로 자르면 비어 버린다),
    # 자치구별은 최근 6개월이다. 기간이 다르므로 화면에 그렇게 적는다.
    n84 = df[df.area_m2.between(AREA_LO, AREA_HI)]
    a["med84"] = a.aptSeq.map(n84.groupby("aptSeq").amount_manwon.median())
    cut = (df.deal_date.max().to_period("M") - (RECENT_M - 1)).to_timestamp()
    n84r = n84[n84.deal_date >= cut]

    def eok(v):
        return None if pd.isna(v) else round(float(v) / 10000, 1)

    gus = []
    for gu, gg in a.groupby("gu"):
        dongs = []
        for dong, dd in gg.groupby("dong"):
            dd = dd.sort_values("n", ascending=False)
            # 단지 9천 개에 키 이름을 매번 붙이면 파일이 1.4MB 가 된다.
            # 배열로 보내고 자리 뜻은 FIELDS 로 알린다.
            apts = [[
                r.nm, str(r.jb), None if pd.isna(r.yr) else int(r.yr),
                None if pd.isna(r.fl) else int(r.fl),
                int(r.nar), int(r.n),
                r.first.strftime("%Y-%m"), r.last.strftime("%Y-%m"),
                eok(r.last_amt),
                None if pd.isna(r.last_area) else round(float(r.last_area)),
                None if pd.isna(r.m_area) else round(float(r.m_area)),
                eok(r.hi), eok(r.med84),
            ] for r in dd.itertuples()]
            z = ltz.get((gu, dong))
            dongs.append({"nm": dong, "na": len(apts), "nd": int(dd.n.sum()),
                          "ltz": z, "apts": apts})
        dongs.sort(key=lambda x: -x["nd"])
        m = n84r[n84r.sgg_nm == gu].amount_manwon.median()
        gus.append({"nm": gu, "na": int(gg.aptSeq.nunique()), "nd": int(gg.n.sum()),
                    "m84": eok(m), "dongs": dongs})
    gus.sort(key=lambda x: -x["nd"])

    payload = {
        "fields": ["nm", "jb", "yr", "fl", "na", "n", "f", "l",
                   "la", "lz", "ma", "hi", "m84"],
        "gus": gus,
        "total": {"apt": int(a.aptSeq.nunique()), "deal": int(a.n.sum()),
                  "dong": int(a.groupby(["gu", "dong"]).ngroups),
                  "from": df.deal_date.min().strftime("%Y-%m"),
                  "to": df.deal_date.max().strftime("%Y-%m"),
                  "recent": cut.strftime("%Y-%m")},
        "treated": sorted(policy.TREATED_2020),
    }
    if not TEMPLATE.exists():
        sys.exit(f"{TEMPLATE} 가 없습니다.")
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "__DATA__", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    out = io.ROOT / "output" / "apt_directory.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")

    print(f"저장 {out.relative_to(io.ROOT)}  ({out.stat().st_size/1e6:.2f}MB)")
    print(f"  단지 {payload['total']['apt']:,}개 · 법정동 {payload['total']['dong']}개 "
          f"· 자치구 {len(gus)}개")
    print(f"  거래 {payload['total']['deal']:,}건 "
          f"({payload['total']['from']} ~ {payload['total']['to']})")
    print(f"  토허 이력이 붙은 법정동 {sum(1 for g in gus for d in g['dongs'] if d['ltz'])}개")


if __name__ == "__main__":
    main()
