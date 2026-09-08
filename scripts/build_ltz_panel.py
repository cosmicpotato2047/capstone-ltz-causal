"""법정동 × 월 토허구역 지정 상태 패널을 만든다.

    python scripts/build_ltz_panel.py

입력  data/policy/notices/zones.csv        (extract_zones.py)
출력  data/policy/ltz_panel.csv            긴 형식: 자치구·법정동·연월·상태·계열
      output/results/ltz_panel.json        요약

상태는 셋이다.

    미지정   그 달에 토허구역이 아니었다
    지정     공고로 확인된 지정 구간 안이다
    불명     지정 구간 사이에 뚫린 구멍. 재지정 공고를 아직 못 읽었을 뿐일
             수도 있고 실제로 풀렸다가 다시 지정된 것일 수도 있다

**구멍을 지정으로 메우지 않는다.** 메우면 '읽지 못한 것'이 '지정된 것'으로
둔갑한다. 결정기록 0009 가 한 번 틀렸던 자리가 정확히 그 지점이다.

한계: 공고는 구역 단위이고 이 패널은 법정동 단위다. 한 동의 일부만 지정된
경우도 '지정'으로 표시된다. 2025-02-13 부분 해제처럼 단지 단위로 갈린
사건은 이 패널로 표현할 수 없다 (lib/policy.KEPT_2025 를 따로 쓴다).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import io, policy  # noqa: E402

io.setup_stdout()

START, END = "2006-01-01", "2026-08-01"


def main() -> None:
    z = pd.read_csv(io.POLICY / "notices" / "zones.csv", dtype={"지번": str})
    z = z[z.효력일.notna() & (z.효력일 != "") & z.종료일.notna() & (z.종료일 != "")].copy()
    z["s"] = pd.to_datetime(z.효력일).dt.to_period("M").dt.to_timestamp()
    z["e"] = pd.to_datetime(z.종료일)

    months = pd.date_range(START, END, freq="MS")
    recs = []
    for (gu, dong), g in z.groupby(["자치구", "법정동"]):
        on = np.zeros(len(months), bool)
        ser = {}
        for _, r in g.iterrows():
            m = (months >= r.s) & (months <= r.e)
            on |= m
            for i in np.flatnonzero(m):
                ser.setdefault(i, set()).add(r.계열)

        # 구멍 = 켜진 적 있는 구간의 사이에 낀 꺼짐
        idx = np.flatnonzero(on)
        state = np.where(on, "지정", "미지정").astype(object)
        if len(idx):
            inner = np.zeros(len(months), bool)
            inner[idx[0]:idx[-1] + 1] = True
            state[inner & ~on] = "불명"

        for i, mth in enumerate(months):
            if state[i] == "미지정":
                continue
            recs.append({"자치구": gu, "법정동": dong, "연월": mth.strftime("%Y-%m"),
                         "상태": state[i], "계열": "|".join(sorted(ser.get(i, ())))})

    p = pd.DataFrame(recs)
    out = io.POLICY / "ltz_panel.csv"
    p.to_csv(out, index=False, encoding="utf-8-sig")

    n_dong = p.groupby(["자치구", "법정동"]).ngroups
    print(f"패널 {len(p):,}행 (지정·불명인 달만 기록) · 법정동 {n_dong}개")
    print(f"저장 {out.relative_to(io.ROOT)}\n")
    print(p.상태.value_counts().to_string())

    print("\n자치구별 — 한 번이라도 지정된 법정동 수")
    g = (p[p.상태 == "지정"].groupby("자치구").법정동.nunique()
         .sort_values(ascending=False))
    for gu, n in g.items():
        mark = " ←분석 대상" if gu in policy.SGG_NAME.values() else ""
        print(f"  {gu:<8}{n:>3}개{mark}")

    io.save_result("ltz_panel", {
        "기간": [START[:7], END[:7]],
        "법정동": int(n_dong),
        "지정_월수": int((p.상태 == "지정").sum()),
        "불명_월수": int((p.상태 == "불명").sum()),
        "자치구별_지정동수": {k: int(v) for k, v in g.items()},
    })


if __name__ == "__main__":
    main()
