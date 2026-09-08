"""법정동별 토허구역 지정 구간(spell)을 일 단위로 만든다.

    python scripts/build_ltz_spells.py

입력  data/policy/notices/zones.csv            공고 × 구역 (원자료)
      data/policy/notices/policy_timeline.csv  공고별 면적기준
출력  data/policy/ltz_spells.csv               법정동 × 연속 구간 (일 단위)
      data/policy/ltz_events.csv               상태가 바뀐 시점만
      data/policy/ltz_panel.csv                법정동 × 월 (히트맵·병합용)

**구간이 원본이고 패널은 파생이다.** 지정 효력일은 2020-06-23 처럼 날짜다.
월 단위로 먼저 뭉개면 6월 1~22일의 미지정 기간이 사라지고, 사건 시점
정렬이 3주씩 어긋난다. 그래서 일 단위 구간을 먼저 만들고 패널을 거기서
뽑는다.

연속된 재지정은 한 구간으로 합친다. 분석에서 의미 있는 단위는 '2020-06-23
부터 2025-06-22 까지 끊김 없이 처치'이지 1년짜리 공고 5건이 아니다.
합칠 때 근거 공고 목록과 면적기준 변화는 남긴다.

상태는 둘이다.
    지정   공고로 확인된 구간
    불명   지정 구간 사이에 뚫린 구멍. 재지정 공고를 아직 못 읽었을 수도
           있어 지정으로 메우지 않는다 (결정기록 0009).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import io, policy  # noqa: E402

io.setup_stdout()

N = io.POLICY / "notices"
# 재지정 사이 이 정도 틈은 같은 구간으로 본다. 공고가 만료 당일에
# 이어지면 틈이 0~1일이다.
JOIN_DAYS = 1


def main() -> None:
    z = pd.read_csv(N / "zones.csv", dtype={"지번": str})
    z = z[z.효력일.notna() & (z.효력일 != "") & z.종료일.notna() & (z.종료일 != "")].copy()
    z["s"] = pd.to_datetime(z.효력일)
    z["e"] = pd.to_datetime(z.종료일)

    tl = pd.read_csv(N / "policy_timeline.csv")
    tl["stem"] = tl.파일.str.rsplit(".", n=1).str[0]
    rule = dict(zip(tl.stem, tl["주거기준_㎡초과"]))

    spells = []
    for (gu, dong), g in z.groupby(["자치구", "법정동"]):
        g = g.sort_values("s")
        cur = None
        for _, r in g.iterrows():
            if cur and r.s <= cur["종료일"] + pd.Timedelta(days=JOIN_DAYS):
                cur["종료일"] = max(cur["종료일"], r.e)
                cur["_src"].append(r.파일)
                cur["_ser"].update(str(r.계열).split("|"))
                cur["_area"].append(r.면적_km2)
                cur["_jib"].append(bool(str(r.지번).strip() not in ("", "nan")))
                cur["_rule"].append(rule.get(r.파일))
            else:
                if cur:
                    spells.append(cur)
                cur = {"자치구": gu, "법정동": dong, "시작일": r.s, "종료일": r.e,
                       "_src": [r.파일], "_ser": set(str(r.계열).split("|")), "_area": [r.면적_km2],
                       "_jib": [bool(str(r.지번).strip() not in ("", "nan"))],
                       "_rule": [rule.get(r.파일)]}
        if cur:
            spells.append(cur)

    rows = []
    for k, g in pd.DataFrame(spells).groupby(["자치구", "법정동"]):
        g = g.sort_values("시작일")
        prev_end = None
        for _, r in g.iterrows():
            if prev_end is not None and r.시작일 > prev_end + pd.Timedelta(days=1):
                rows.append({"자치구": k[0], "법정동": k[1],
                             "시작일": prev_end + pd.Timedelta(days=1),
                             "종료일": r.시작일 - pd.Timedelta(days=1),
                             "상태": "불명", "계열": "", "지정방식": "",
                             "면적기준_㎡": "", "구역면적_㎢": "",
                             "공고건수": 0, "근거공고": ""})
            jb = r._jib
            # 면적기준은 시간 순으로 적는다. 2022-06 에 18㎡ -> 6㎡ 로
            # 강화됐는데 값으로 정렬하면 거꾸로 읽힌다.
            rules = []
            for x in r._rule:
                if pd.notna(x) and (not rules or rules[-1] != int(x)):
                    rules.append(int(x))
            rows.append({
                "자치구": k[0], "법정동": k[1],
                "시작일": r.시작일, "종료일": r.종료일, "상태": "지정",
                "계열": "|".join(sorted(x for x in r._ser if x)),
                "지정방식": "동전체" if not any(jb) else ("일부" if all(jb) else "혼합"),
                "면적기준_㎡": "→".join(str(x) for x in rules),
                "구역면적_㎢": max([x for x in r._area if pd.notna(x)], default=""),
                "공고건수": len(r._src),
                "근거공고": "|".join(sorted({Path(x).stem for x in r._src}))[:120],
            })
            prev_end = r.종료일

    S = pd.DataFrame(rows).sort_values(["자치구", "법정동", "시작일"])
    for c in ("시작일", "종료일"):
        S[c] = pd.to_datetime(S[c])          # 안에서는 Timestamp 로 다룬다
    # 파일에는 날짜만 적는다. 시각이 붙으면 사람이 읽기 나쁘다.
    S.assign(**{c: S[c].dt.date for c in ("시작일", "종료일")})      .to_csv(io.POLICY / "ltz_spells.csv", index=False, encoding="utf-8-sig")

    # 사건 = 구간의 시작·끝
    ev = []
    for _, r in S.iterrows():
        ev.append({"자치구": r.자치구, "법정동": r.법정동,
                   "일자": pd.Timestamp(r.시작일),
                   "전이": "지정 시작" if r.상태 == "지정" else "기록 끊김",
                   "계열": r.계열, "근거공고": r.근거공고})
        ev.append({"자치구": r.자치구, "법정동": r.법정동,
                   "일자": pd.Timestamp(r.종료일) + pd.Timedelta(days=1),
                   "전이": "지정 종료" if r.상태 == "지정" else "기록 재개",
                   "계열": r.계열, "근거공고": r.근거공고})
    E = pd.DataFrame(ev)
    E["일자"] = pd.to_datetime(E.일자)
    E = E.sort_values(["일자", "자치구", "법정동"])
    E.assign(일자=E.일자.dt.date).to_csv(
        io.POLICY / "ltz_events.csv", index=False, encoding="utf-8-sig")

    # 패널은 구간에서 뽑는다. 그 달의 절반 넘게 지정이면 지정으로 본다.
    months = pd.period_range("2006-01", "2026-08", freq="M")
    pan = []
    for _, r in S.iterrows():
        s, e = pd.Timestamp(r.시작일), pd.Timestamp(r.종료일)
        for m in pd.period_range(s.to_period("M"), e.to_period("M"), freq="M"):
            if m not in months:
                continue
            lo, hi = max(s, m.start_time), min(e, m.end_time)
            frac = (hi - lo).days / (m.end_time - m.start_time).days
            pan.append({"자치구": r.자치구, "법정동": r.법정동,
                        "연월": str(m), "상태": r.상태, "계열": r.계열,
                        "지정일수비율": round(frac, 3)})
    P = (pd.DataFrame(pan)
         .sort_values(["자치구", "법정동", "연월", "상태"])
         .drop_duplicates(subset=["자치구", "법정동", "연월"], keep="first"))
    P.to_csv(io.POLICY / "ltz_panel.csv", index=False, encoding="utf-8-sig")

    print(f"구간  {len(S):,}행  ({int((S.상태=='지정').sum())} 지정 / "
          f"{int((S.상태=='불명').sum())} 불명)   ltz_spells.csv")
    print(f"사건  {len(E):,}행                      ltz_events.csv")
    print(f"패널  {len(P):,}행                      ltz_panel.csv")
    print(f"법정동 {S.groupby(['자치구','법정동']).ngroups}개\n")
    print("우리 처치 4개 동의 구간")
    t = S[(S.자치구.isin(["강남구", "송파구"])) & S.법정동.isin(policy.TREATED_2020)]
    t = t.assign(시작일=t.시작일.dt.date, 종료일=t.종료일.dt.date)
    print(t[["자치구", "법정동", "시작일", "종료일", "상태", "지정방식",
             "면적기준_㎡", "공고건수", "계열"]].to_string(index=False))


if __name__ == "__main__":
    main()
