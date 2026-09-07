"""공고 본문에서 구역 목록(자치구·법정동·지번·면적)을 뽑는다.

    python scripts/extract_zones.py

입력  data/policy/notices/notice_text/*.txt   (build_policy_timeline.py 가 생성)
출력  data/policy/notices/zones.csv

공고의 구역 표는 자치구명이 한 번 나오고 그 아래로 여러 동이 따라오는 구조다.
따라서 줄을 훑으며 '현재 자치구'를 유지한 채 동·지번·면적을 수집한다.

용도: 통제군 오염 점검. 신속통합기획·공공재개발·모아타운 구역이 우리
통제 자치구·법정동에 걸리는지 확인한다.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import io, policy  # noqa: E402

io.setup_stdout()

TD = io.POLICY / "notices" / "notice_text"
GU_LIST = ("종로구 중구 용산구 성동구 광진구 동대문구 중랑구 성북구 강북구 도봉구 "
           "노원구 은평구 서대문구 마포구 양천구 강서구 구로구 금천구 영등포구 "
           "동작구 관악구 서초구 강남구 송파구 강동구").split()
GU_RE = re.compile(r"^\s*(" + "|".join(g[0] + r"\s*" + g[1:] for g in GU_LIST) + r")\s*$")
# 동 + 지번 + (일대) + (구역명) — 면적은 다음 줄에 오는 경우가 많다
ROW_RE = re.compile(r"^\s*([가-힣]{2,5}동\d?가?)\s*"
                    r"([\d]+(?:-\d+)?)?\s*(?:번지)?\s*(일대)?\s*(?:\(([^)]{1,30})\))?\s*$")
AREA_RE = re.compile(r"^\s*([\d]+\.[\d]{2,4})\s*$")
SERIES_RE = [
    ("신속통합기획", ["신속통합기획", "신통기획"]),
    ("공공재개발", ["공공재개발"]),
    ("모아타운", ["모아타운"]),
    ("국제교류복합지구", ["국제교류복합지구"]),
    ("주요재건축단지", ["압구정", "여의도", "목동 택지", "성수전략"]),
]


def series_of(t: str) -> str:
    return "|".join(n for n, kws in SERIES_RE if any(k in t for k in kws)) or "미분류"


def main() -> None:
    rows = []
    for f in sorted(TD.glob("*.txt")):
        raw = f.read_text(encoding="utf-8")
        ser = series_of(raw)
        lines = [re.sub(r"[ \t]+", " ", x) for x in raw.splitlines()]
        cur_gu, pending = "", None
        for ln in lines:
            g = GU_RE.match(ln)
            if g:
                cur_gu = g.group(1).replace(" ", "")
                pending = None
                continue
            if pending is not None:
                a = AREA_RE.match(ln)
                if a:
                    pending["면적_km2"] = float(a.group(1))
                rows.append(pending)
                pending = None
                if a:
                    continue
            m = ROW_RE.match(ln)
            if m and cur_gu:
                dong, jibun, ilda, tag = m.groups()
                pending = {"파일": f.stem, "계열": ser, "자치구": cur_gu, "법정동": dong,
                           "지번": jibun or "", "일대": bool(ilda), "구역명": tag or "",
                           "면적_km2": None}
        if pending:
            rows.append(pending)

    if not rows:
        sys.exit("구역을 찾지 못했습니다.")
    df = pd.DataFrame(rows).drop_duplicates(
        subset=["파일", "자치구", "법정동", "지번"]).reset_index(drop=True)
    out = io.POLICY / "notices" / "zones.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"구역 {len(df):,}건 · 파일 {df.파일.nunique()}개 · 저장 {out.name}\n")

    print("=" * 84)
    print("계열별 구역 수")
    print("=" * 84)
    for s, n in df.계열.value_counts().items():
        print(f"  {s:<40}{n:>6,}")

    # 통제군 오염 점검
    ctrl_gu = [policy.SGG_NAME[c] for c in policy.CONTROL_SGG]
    trt_gu = [policy.SGG_NAME[c] for c in policy.TREATMENT_SGG]
    print("\n" + "=" * 84)
    print("통제군 오염 점검 — 정비사업 구역이 우리 자치구에 있는가")
    print("=" * 84)
    reno = df[df.계열.str.contains("신속통합기획|공공재개발|모아타운", na=False)]
    for label, gus in (("통제 자치구", ctrl_gu), ("처치 자치구", trt_gu)):
        sub = reno[reno.자치구.isin(gus)]
        print(f"\n[{label}]  {len(sub)}건")
        for gu, g in sub.groupby("자치구"):
            dongs = g.법정동.value_counts()
            print(f"  {gu}: {len(g)}건 — " +
                  ", ".join(f"{d}({n})" for d, n in dongs.head(8).items()))

    print("\n" + "=" * 84)
    print("주 분석 통제동 — 강남·송파의 비처치 법정동에 걸린 구역")
    print("=" * 84)
    hit = reno[(reno.자치구.isin(["강남구", "송파구"])) &
               (~reno.법정동.isin(policy.TREATED_2020))]
    if hit.empty:
        print("  없음 — 주 분석의 통제동은 정비사업 지정에 걸리지 않는다")
    else:
        for _, r in hit.iterrows():
            print(f"  {r.자치구} {r.법정동} {r.지번} {r.구역명} "
                  f"{r.면적_km2 or ''} [{r.계열}] ({r.파일})")


if __name__ == "__main__":
    main()
