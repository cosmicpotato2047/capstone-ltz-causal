"""통제군 오염 점검 — 우리 처치가 아닌 토허구역이 통제군에 걸리는가.

    python scripts/check_contamination.py

입력  data/policy/notices/zones.csv   (extract_zones.py 가 생성)
      data/processed/trades.parquet
출력  output/results/contamination.json

서울시는 국제교류복합지구(우리 처치) 말고도 정비사업 후보지 등 여러 구역을
같은 법조항으로 토허구역에 넣는다. 그 구역이 통제군에 걸리면 통제군도 일부
처치를 받은 셈이 되어 추정치가 0쪽으로 끌린다.

공고는 구역을 '○○동 123번지 일대'로 적는다. '일대'의 경계는 지형도면에만
있으므로 거래 자료만으로는 정확히 가를 수 없다. 그래서 두 값을 함께 낸다.

    하한  공고에 적힌 지번과 정확히 일치하는 거래
    상한  그 법정동 전체의 거래

참값은 둘 사이에 있고, 상한이 이미 작다면 더 좁힐 필요가 없다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import io, policy  # noqa: E402

io.setup_stdout()

# 주 분석 구간 — 2020.6 지정 전후 ±36개월 (결정기록 0002)
WIN_LO = pd.Timestamp("2017-06-23")
WIN_HI = pd.Timestamp("2023-06-23")


def main() -> None:
    z = pd.read_csv(io.POLICY / "notices" / "zones.csv", dtype={"지번": str})
    df = io.load_trades()
    df["gu"] = df.sggCd.astype(str)
    df["jibun"] = df.jibun.astype(str).str.strip()

    main_win = df[(df.deal_date >= WIN_LO) & (df.deal_date < WIN_HI)]
    gu_of = {v: k for k, v in policy.SGG_NAME.items()}

    # 우리 처치 자체인 구역은 오염이 아니다
    z = z[~((z.자치구.isin(["강남구", "송파구"])) & (z.법정동.isin(policy.TREATED_2020)))]
    z = z[z.자치구.isin(policy.SGG_NAME.values())]
    z = z[z.효력일.notna() & (z.효력일 != "")]
    z = z[(z.효력일 >= WIN_LO.strftime("%Y-%m-%d")) & (z.효력일 < WIN_HI.strftime("%Y-%m-%d"))]

    rows = []
    for (gu, dong, jibun), g in z.groupby(["자치구", "법정동", "지번"], dropna=False):
        eff = pd.Timestamp(g.효력일.min())
        sub = main_win[(main_win.gu == gu_of[gu]) & (main_win.umdNm == dong) &
                       (main_win.deal_date >= eff)]
        has_j = isinstance(jibun, str) and jibun.strip() != ""
        rows.append({
            "자치구": gu, "법정동": dong, "지번": jibun if has_j else "",
            "효력일": eff.date().isoformat(),
            "구역유형": "지번" if has_j else "구역",
            "계열": g.계열.iloc[0], "면적_km2": g.면적_km2.dropna().max(),
            # 지번을 안 적고 '○○ 아파트지구'처럼 구역으로 지정한 공고는
            # 거래를 지번으로 가를 수 없다. 하한을 0으로 적으면 거짓이 된다.
            "하한_지번일치": int((sub.jibun == str(jibun)).sum()) if has_j else None,
            "상한_법정동": int(len(sub)),
        })
    r = pd.DataFrame(rows).sort_values(["자치구", "효력일"])

    ctrl_gu = [policy.SGG_NAME[c] for c in policy.CONTROL_SGG]
    ctrl_n = len(main_win[main_win.gu.isin(policy.CONTROL_SGG)])
    # 강남·송파의 비처치 법정동 = 주 분석의 실질 통제동
    gs = main_win[main_win.gu.isin(["11680", "11710"]) &
                  ~main_win.umdNm.isin(policy.TREATED_2020)]

    def block(title: str, sel: pd.DataFrame, denom: int, denom_label: str) -> dict:
        print("\n" + "=" * 96)
        print(title)
        print("=" * 96)
        if sel.empty:
            print("  걸리는 구역 없음")
            return {"구역수": 0, "하한": 0, "상한": 0, "분모": denom}
        print(f"  {'법정동':<10}{'지번':<10}{'효력일':<12}{'면적㎢':>8}"
              f"{'하한':>7}{'상한':>9}   계열")
        for _, x in sel.iterrows():
            a = f"{x.면적_km2:.4f}" if pd.notna(x.면적_km2) else "  -   "
            lo = "  -" if pd.isna(x.하한_지번일치) else f"{int(x.하한_지번일치)}"
            print(f"  {x.법정동:<10}{str(x.지번):<10}{x.효력일:<12}{a:>8}"
                  f"{lo:>7}{x.상한_법정동:>9}   {str(x.계열)[:26]}")
        lo = int(sel.하한_지번일치.fillna(0).sum())
        # 상한은 법정동 단위다. 한 동에 구역이 둘이면 그 동을 두 번 세면 안 된다.
        uniq = sel.sort_values("효력일").drop_duplicates(subset=["자치구", "법정동"])
        hi = int(uniq.상한_법정동.sum())
        nz = int((sel.구역유형 == "구역").sum())
        print(f"\n  {denom_label} {denom:,}건 대비")
        print(f"    하한(지번 일치)   {lo:>7,}건  {100*lo/denom:6.3f}%"
              + (f"   ※ 지번 없는 구역 {nz}개는 하한을 낼 수 없어 빠졌다" if nz else ""))
        print(f"    상한(법정동 전체) {hi:>7,}건  {100*hi/denom:6.3f}%   "
              f"[{len(uniq)}개 법정동]")
        return {"구역수": len(sel), "지번없는구역": nz, "하한": lo, "상한": hi,
                "분모": denom, "하한_비율": round(100*lo/denom, 4),
                "상한_비율": round(100*hi/denom, 4)}

    res = {
        "구간": [WIN_LO.date().isoformat(), WIN_HI.date().isoformat()],
        "통제자치구": block(
            "통제 자치구(성동·광진·강동·동작) 안의 비(非)처치 토허구역",
            r[r.자치구.isin(ctrl_gu)], ctrl_n, "통제 자치구 거래"),
        "강남송파_통제동": block(
            "강남·송파의 비처치 법정동 안의 비(非)처치 토허구역",
            r[r.자치구.isin(["강남구", "송파구"])], len(gs), "강남·송파 비처치동 거래"),
    }
    io.save_result("contamination", res)
    print(f"\n저장: output/results/contamination.json")


if __name__ == "__main__":
    main()
