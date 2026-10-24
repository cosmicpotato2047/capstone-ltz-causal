"""결과 JSON 안의 값을 경로 문자열로 꺼낸다.

대시보드(`build_dashboard.py`)와 보고서 표 생성기(`build_tables.py`)가 같이 쓴다.
둘 다 **숫자를 손으로 적지 않고 `output/results/*.json` 에서 읽는다**는 원칙을
따르므로(결정기록 0007), 경로를 읽는 규칙도 한 곳에 둔다.

    dig(obj, "거래량[k=0].effect_pct")        리스트에서 k 가 0 인 **첫** 칸
    dig(obj, "나_2022_기준강화[기준_㎡=40.0]")  값에 점이 있어도 된다
    dig(obj, "결과[1].붕괴_Mbar_구간")         1번 칸

**번호보다 라벨을 쓴다.** 번호는 스크립트가 줄 순서를 바꾸면 조용히 다른 값을
가리키지만, 라벨은 그때 `None` 으로 드러난다.

점으로 쪼개면 `[기준_㎡=40.0]` 의 소수점까지 갈라지므로 대괄호를 하나의
토큰으로 읽는다.
"""
from __future__ import annotations

import re

TOKEN = re.compile(r"([^.\[\]]+)|\[([^\]]*)\]")


def dig(obj, path: str):
    """경로를 따라간다. 없으면 None."""
    cur = obj
    for name, sel in TOKEN.findall(path):
        if name:
            if not isinstance(cur, dict) or name not in cur:
                return None
            cur = cur[name]
            continue
        if not isinstance(cur, list):
            return None
        if "=" in sel:
            key, _, want = sel.partition("=")
            cur = next((r for r in cur if isinstance(r, dict)
                        and str(r.get(key)) == want), None)
            if cur is None:
                return None
        else:
            i = int(sel)
            if not (-len(cur) <= i < len(cur)):
                return None
            cur = cur[i]
    return cur


def rows(obj, path: str, where: dict | None = None) -> list[dict]:
    """리스트를 꺼내고, `where` 가 있으면 그 값들이 모두 맞는 칸만 남긴다.

    `dig` 의 대괄호 거르개는 **첫 칸 하나**만 주므로, 표의 여러 줄을 뽑을 때는
    이쪽을 쓴다. 예: 문턱 민감도 표에서 `표본 == "기본"` 인 다섯 줄.
    """
    got = dig(obj, path)
    if not isinstance(got, list):
        return []
    if not where:
        return [r for r in got if isinstance(r, dict)]
    return [r for r in got if isinstance(r, dict)
            and all(str(r.get(k)) == str(v) for k, v in where.items())]
