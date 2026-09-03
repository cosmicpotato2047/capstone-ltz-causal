"""처치 정의 — 토허구역 지정 이력과 처치 배정.

처치 정의를 바꿔야 할 때 고칠 곳은 이 파일 하나다.
논문의 '처치를 어떻게 정의했는가' 절이 이 파일에 대응한다.
근거는 data/policy/ltz_events.csv 및 data/policy/notices/ 의 공고 원문.
"""
from __future__ import annotations

import pandas as pd

# --- 사건 날짜 -------------------------------------------------------------
# 발표일과 효력발생일은 다른 사건이다. 시장은 발표에 먼저 반응하고,
# 법적 처치는 효력일에 시작된다.
EVENTS = {
    "E2020": {  # 서울시 공고 2020-1832
        "announce": pd.Timestamp("2020-06-18"),
        "effect": pd.Timestamp("2020-06-23"),
        "label": "2020.6 국제교류복합지구 인근 지정",
    },
    "E2021": {  # 2021.4 주요 재건축단지
        "announce": pd.Timestamp("2021-04-21"),
        "effect": pd.Timestamp("2021-04-27"),
        "label": "2021.4 압구정·여의도·목동·성수 지정",
    },
    "E2025FEB": {  # 서울시 공고 2025-497
        "announce": pd.Timestamp("2025-02-12"),
        "effect": pd.Timestamp("2025-02-13"),
        "label": "2025.2 부분 해제 (305곳 중 291곳)",
    },
    "E2025MAR": {  # 서울시 공고 2025-953
        "announce": pd.Timestamp("2025-03-19"),
        "effect": pd.Timestamp("2025-03-24"),
        "label": "2025.3 강남3구·용산 전체 아파트 확대지정",
    },
    "E2025OCT": {  # 10.15 대책
        "announce": pd.Timestamp("2025-10-15"),
        "effect": pd.Timestamp("2025-10-20"),
        "label": "2025.10 서울 25개구 전역 확대",
    },
}

# 인과추정 유효 구간의 끝. 이후 통제군 소멸 + 규제지역과 교란 (결정기록 0002)
IDENTIFICATION_END = pd.Timestamp("2025-10-19")

# 신고지연으로 과소집계되는 최근 월을 잘라낸다 (결정기록 0004)
SAMPLE_END = pd.Timestamp("2026-05-31")

# --- 처치 지역 -------------------------------------------------------------
TREATED_2020 = {"삼성동", "청담동", "대치동", "잠실동"}

TREATED_2021 = {"압구정동", "여의도동", "목동", "성수동1가"}

TREATED_2025MAR_SGG = {"강남구", "서초구", "송파구", "용산구"}

# --- 2025.2 해제 시 지정이 유지된 재건축 단지 -------------------------------
# 자료마다 수가 다르다: 서울시 설명자료 14곳 / 공고 2025-1831 표기 13행 /
# '우성1·2·3차'를 분리하면 15개. 원문 토지조서 확보 전까지 두 정의를 병행한다.
# 아래는 국토부 실거래 자료의 aptNm 표기 기준.
KEPT_2025_NARROW = {  # 공고 표기와 1:1 대응이 명확한 것만
    "은마", "진흥아파트", "현대1", "개포우성1", "개포우성2",
    "주공아파트 5단지", "우성4차", "아시아선수촌아파트",
}
KEPT_2025_BROAD = KEPT_2025_NARROW | {  # 표기 대응이 모호한 후보까지 포함
    "선경1차(1동-7동)", "선경2차(8동-12동)", "선경3차",
    "한보미도맨션1", "한보미도맨션2",
    "쌍용대치2", "쌍용대치아파트1동,2동,3동,5동,6동",
    "대치우성아파트1동,2동,3동,5동,6동,7동", "우성아파트",
}

# --- 자치구 ---------------------------------------------------------------
SGG_NAME = {
    "11170": "용산구", "11200": "성동구", "11215": "광진구", "11590": "동작구",
    "11650": "서초구", "11680": "강남구", "11710": "송파구", "11740": "강동구",
}
TREATMENT_SGG = ["11680", "11650", "11710", "11170"]
CONTROL_SGG = ["11200", "11215", "11740", "11590"]
ALL_SGG = TREATMENT_SGG + CONTROL_SGG


def assign_2020(df: pd.DataFrame) -> pd.Series:
    """2020.6 지정의 처치 여부."""
    return df["umdNm"].isin(TREATED_2020)


def assign_kept_2025(df: pd.DataFrame, definition: str = "narrow") -> pd.Series:
    """2025.2 해제 시 지정이 유지된 단지 여부. definition: narrow | broad."""
    names = KEPT_2025_NARROW if definition == "narrow" else KEPT_2025_BROAD
    return df["aptNm"].isin(names)


def event_quarter(dates: pd.Series, event: str) -> pd.Series:
    """사건 효력일 기준 분기. 0 = 사건이 포함된 분기."""
    import numpy as np
    e = EVENTS[event]["effect"]
    m = (dates.dt.year - e.year) * 12 + (dates.dt.month - e.month)
    return np.floor(m / 3).astype(int)
