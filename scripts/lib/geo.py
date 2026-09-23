"""서울 행정경계와 법정동 기준표.

    from lib import geo
    geo.DONGS["강남구"]        # 강남구의 법정동 이름 집합
    geo.code_of("강남구", "대치동")   # '11680106'
    geo.normalize("신월7동")    # '신월동'  (행정동 -> 법정동)

출처: https://github.com/southkorea/seoul-maps  (juso/2015)
      도로명주소 사업의 법정구역 경계. 법정동 467개, 자치구 25개.
받은 날 2026-09-11. 원본 파일은 data/geo/ 에 그대로 둔다.

**왜 기준표가 필요한가.** 공고는 법정동과 행정동을 섞어 쓴다. '신월7동'은
행정동이고 법정동은 '신월동'이다. 거래 자료는 법정동을 쓰므로 맞춰야 한다.
또 공고 판독이 만들어낸 없는 동('인동', '미안동' 같은 조각)을 걸러내려면
실재하는 법정동 목록이 있어야 한다.

**이름이 아니라 코드로 잇는다.** 실거래 자료에는 자치구와 법정동명이
어긋난 기록이 소수 있다(140만건 중 28건). 법정동코드 10자리의 앞 8자리가
경계 자료의 EMD_CD 와 같으므로 그것으로 잇는다 ([[0008-유지단지-지번매칭]]
과 같은 원리다).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

GEO = Path(__file__).resolve().parents[2] / "data" / "geo"
NEIGHBORHOODS = GEO / "seoul_neighborhoods_simple.json"
MUNICIPALITIES = GEO / "seoul_municipalities_simple.json"


@lru_cache(maxsize=1)
def _load() -> tuple[dict, dict, dict]:
    gu_of_code = {}
    for f in json.loads(MUNICIPALITIES.read_text(encoding="utf-8"))["features"]:
        p = f["properties"]
        gu_of_code[str(p["SIG_CD"])] = p["SIG_KOR_NM"]

    dongs = defaultdict(set)          # 자치구 -> {법정동}
    code = {}                         # (자치구, 법정동) -> EMD_CD
    for f in json.loads(NEIGHBORHOODS.read_text(encoding="utf-8"))["features"]:
        p = f["properties"]
        cd, nm = str(p["EMD_CD"]), p["EMD_KOR_NM"]
        gu = gu_of_code.get(cd[:5], "")
        if gu:
            dongs[gu].add(nm)
            code[(gu, nm)] = cd
    return gu_of_code, dict(dongs), code


GU = property(lambda self: _load()[0])


def municipalities() -> dict[str, str]:
    """자치구코드 -> 이름."""
    return _load()[0]


DONGS: dict[str, set[str]] = {}
ALL_DONGS: set[str] = set()


def _init() -> None:
    global DONGS, ALL_DONGS
    DONGS = _load()[1]
    ALL_DONGS = {d for s in DONGS.values() for d in s}


_init()


def code_of(gu: str, dong: str) -> str | None:
    return _load()[2].get((gu, dong))


def exists(gu: str, dong: str) -> bool:
    return dong in DONGS.get(gu, ())


# 행정동 -> 법정동 되돌리기.
#   금호4가동 -> 금호동4가   (가 붙은 동)
#   신월7동   -> 신월동      (번호만 붙은 동)
#   상계3동   -> 상계동
_GA = re.compile(r"^([가-힣]{1,4})(\d)가동$")
_NUM = re.compile(r"^([가-힣]{1,4})\d+동$")


def normalize(dong: str, gu: str = "") -> str:
    """공고 표기를 법정동 표기로 되돌린다. 되돌릴 수 없으면 그대로 준다.

    자치구를 주면 그 구에 실재하는 법정동인지까지 확인한다."""
    pool = DONGS.get(gu) if gu else ALL_DONGS
    if pool and dong in pool:
        return dong
    m = _GA.match(dong)
    if m:
        c = f"{m.group(1)}동{m.group(2)}가"
        if not pool or c in pool:
            return c
    m = _NUM.match(dong)
    if m:
        c = f"{m.group(1)}동"
        if not pool or c in pool:
            return c
    return dong


def polygons():
    """법정동 경계를 DataFrame 으로. 열: cd8, 법정동, 자치구, geom.

    shapely 가 있어야 한다. 파급효과(백로그 12)와 공간 회귀불연속(16)에서 쓴다.
    """
    import pandas as pd
    from shapely.geometry import shape
    feats = json.loads(NEIGHBORHOODS.read_text(encoding="utf-8"))["features"]
    df = pd.DataFrame([{"cd8": str(f["properties"]["EMD_CD"]),
                        "법정동": f["properties"]["EMD_KOR_NM"],
                        "geom": shape(f["geometry"])} for f in feats])
    gu = municipalities()
    df["자치구"] = df.cd8.str[:5].map(gu)
    return df


def touching(dongs: set[str], gus: set[str]) -> set[str]:
    """주어진 법정동 덩어리에 **맞닿은** 법정동 이름 집합 (덩어리 자신은 뺀다).

    `dongs` 는 법정동 이름, `gus` 는 그 동이 속한 자치구 이름이다. 이름만으로는
    다른 구의 같은 이름 동에 걸리므로 자치구를 함께 받는다.

        geo.touching({"대치동","삼성동","청담동","잠실동"}, {"강남구","송파구"})

    경계가 단순화된 자료라 꼭짓점이 미세하게 어긋날 수 있다. 그래서 '닿았다'를
    교차(intersects)로 본다. 실제로 강남·송파 처치 4개 동에 14개 동이 닿는다.
    """
    from shapely.ops import unary_union
    g = polygons()
    core = g[(g.자치구.isin(gus)) & (g.법정동.isin(dongs))]
    if core.empty:
        raise ValueError("맞닿은 동을 찾을 덩어리가 비어 있다")
    u = unary_union(core.geom.tolist())
    hit = {n for n, gm in zip(g.법정동, g.geom) if gm.intersects(u)}
    return hit - set(core.법정동)
