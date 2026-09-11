"""토허구역 지정 이력을 서울 지도 위에서 시점별로 본다.

    python scripts/plot_ltz_map.py

입력  data/geo/seoul_neighborhoods_simple.json   법정동 467개 경계
      data/geo/seoul_municipalities_simple.json  자치구 25개 경계
      data/policy/ltz_spells.csv                 법정동별 지정 구간
출력  output/ltz_map.html                        자체 완결 HTML 한 장

경계는 서버에서 SVG 경로 문자열로 미리 바꾼다. 상태는 구간 표만 넘겨
브라우저가 그 달의 색을 계산한다. 달마다 467개 칸을 미리 펴서 보내면
파일이 몇 배로 커진다.

바깥 라이브러리를 쓰지 않는다. 서울은 경도 폭이 좁아 위도 보정만 한
선형 투영으로 충분하다.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import geo, io, policy  # noqa: E402

io.setup_stdout()

LO, HI = "2020-01", "2026-08"
PREC = 2          # SVG 좌표 소수 자릿수. 폭 1000 기준 0.01 = 약 5m

# 계열별 색. 우리 처치 계열을 가장 눈에 띄게 둔다.
SERIES_COLOR = [
    ("서울전역아파트", "#2c3e50"),
    ("국제교류복합지구", "#c0392b"),
    ("주요재건축단지", "#e67e22"),
    ("용산정비창", "#8e44ad"),
    ("자연녹지", "#16a085"),
    ("신규주택공급후보지", "#2980b9"),
    ("외국인", "#7f8c8d"),
]
OTHER, UNKNOWN, NONE_C = "#34495e", "#bdc3c7", "#f4f4f4"

TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "ltz_map.html"


def rings(geom):
    if geom["type"] == "Polygon":
        return geom["coordinates"]
    return [r for poly in geom["coordinates"] for r in poly]


def to_path(geom, proj) -> str:
    out = []
    for ring in rings(geom):
        pts = [proj(x, y) for x, y in ring]
        if len(pts) < 3:
            continue
        out.append("M" + "L".join(f"{x},{y}" for x, y in pts) + "Z")
    return "".join(out)


def build_payload() -> dict:
    nb = json.loads(geo.NEIGHBORHOODS.read_text(encoding="utf-8"))
    mu = json.loads(geo.MUNICIPALITIES.read_text(encoding="utf-8"))

    xs, ys = [], []
    for f in nb["features"]:
        for r in rings(f["geometry"]):
            for x, y in r:
                xs.append(x)
                ys.append(y)
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    k = math.cos(math.radians((y0 + y1) / 2))      # 위도 보정
    w = 1000.0
    h = w * (y1 - y0) / ((x1 - x0) * k)

    def proj(x, y):
        return (round((x - x0) / (x1 - x0) * w, PREC),
                round((y1 - y) / (y1 - y0) * h, PREC))

    gu_of = geo.municipalities()
    dong = []
    for f in nb["features"]:
        p = f["properties"]
        cd = str(p["EMD_CD"])
        dong.append({"cd": cd, "nm": p["EMD_KOR_NM"], "gu": gu_of.get(cd[:5], ""),
                     "d": to_path(f["geometry"], proj)})
    gu_paths = [{"nm": f["properties"]["SIG_KOR_NM"],
                 "d": to_path(f["geometry"], proj)} for f in mu["features"]]

    # 구간 — 이름이 아니라 법정동코드로 잇는다
    s = pd.read_csv(io.POLICY / "ltz_spells.csv")
    s["cd"] = [geo.code_of(g, n) for g, n in zip(s.자치구, s.법정동)]
    s = s[s.cd.notna()]
    months = pd.period_range(LO, HI, freq="M")
    mi = {str(m): i for i, m in enumerate(months)}

    spells = []
    for r in s.itertuples():
        a = pd.Timestamp(r.시작일).to_period("M")
        b = pd.Timestamp(r.종료일).to_period("M")
        if b < months[0] or a > months[-1]:
            continue
        i0 = mi.get(str(a), 0)
        i1 = mi.get(str(b), len(months) - 1)
        spells.append({"cd": r.cd, "a": i0, "b": i1,
                       "s": 1 if r.상태 == "지정" else 0,
                       "c": str(r.계열) if pd.notna(r.계열) else "",
                       "w": str(r.지정방식) if pd.notna(r.지정방식) else "",
                       "n": str(r.근거공고)[:80] if pd.notna(r.근거공고) else ""})

    return {
        "months": [str(m) for m in months],
        "dong": dong, "gu": gu_paths, "spells": spells,
        "W": round(w), "H": round(h),
        "series": SERIES_COLOR, "other": OTHER, "unknown": UNKNOWN, "none": NONE_C,
        "treated": sorted(policy.TREATED_2020),
        "endIdx": mi.get(str(policy.IDENTIFICATION_END.to_period("M")),
                         len(months) - 1),
        "events": [{"ym": a, "label": b} for a, b in [
            ("2020-06", "2020.6 최초 지정"),
            ("2021-04", "2021.4 재건축단지"),
            ("2022-06", "2022.6 기준 18→6㎡"),
            ("2025-02", "2025.2 부분 해제"),
            ("2025-03", "2025.3 강남3구·용산"),
            ("2025-10", "2025.10 서울 전역")]],
    }


def main() -> None:
    payload = build_payload()
    if not TEMPLATE_PATH.exists():
        sys.exit(f"{TEMPLATE_PATH} 가 없습니다.")
    html = TEMPLATE_PATH.read_text(encoding="utf-8").replace(
        "__DATA__", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    out = io.ROOT / "output" / "ltz_map.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")

    print(f"저장 {out.relative_to(io.ROOT)}  ({out.stat().st_size/1e6:.2f}MB)")
    print(f"  법정동 {len(payload['dong'])}개 · 자치구 {len(payload['gu'])}개 "
          f"· 구간 {len(payload['spells']):,}개")
    print(f"  시점 {len(payload['months'])}개월 ({LO} ~ {HI}) "
          f"· 식별 구간 끝 {payload['months'][payload['endIdx']]}")


if __name__ == "__main__":
    main()
