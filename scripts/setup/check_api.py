"""인증키 동작 확인용 스모크 테스트.

    python scripts/check_api.py

.env 의 DATA_GO_KR_KEY 를 읽어 강남구(11680) 2024년 7월 거래를 1건만 조회한다.
성공하면 그 달의 총 거래 건수(totalCount)를 출력한다.
"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import io  # noqa: E402
io.setup_stdout()


ENDPOINT = "https://apis.data.go.kr/1613000/RTMSDataSvcAptTrade/getRTMSDataSvcAptTrade"
LAWD_CD = "11680"   # 서울 강남구
DEAL_YMD = "202407"


load_key = io.load_key   # lib/io.py 로 통합


def main() -> None:
    key = load_key()
    params = {
        "serviceKey": key,          # requests 가 알아서 URL 인코딩한다 (그래서 Decoding 키)
        "LAWD_CD": LAWD_CD,
        "DEAL_YMD": DEAL_YMD,
        "pageNo": 1,
        "numOfRows": 1,
    }
    res = requests.get(ENDPOINT, params=params, timeout=20)
    res.encoding = "utf-8"

    if res.status_code != 200:
        sys.exit(f"HTTP {res.status_code}\n{res.text[:500]}")

    try:
        root = ET.fromstring(res.text)
    except ET.ParseError:
        sys.exit(f"XML 파싱 실패. 응답 원문:\n{res.text[:800]}")

    code = root.findtext(".//resultCode") or root.findtext(".//returnReasonCode") or "?"
    msg = root.findtext(".//resultMsg") or root.findtext(".//returnAuthMsg") or "?"

    if code not in ("00", "000"):
        print(f"[실패] resultCode={code}  resultMsg={msg}")
        if "SERVICE_KEY" in (msg or "").upper():
            print("  → 키 문제입니다. (1) Decoding 키가 맞는지 (2) 포털에서 활용신청이 '승인' 상태인지")
            print("     (3) 승인 직후라면 반영에 최대 1시간 걸릴 수 있습니다.")
        sys.exit(1)

    total = root.findtext(".//totalCount")
    print(f"[성공] resultCode={code} ({msg})")
    print(f"강남구 {DEAL_YMD} 아파트 매매 신고 건수: {total} 건")

    item = root.find(".//item")
    if item is not None:
        print("\n샘플 1건:")
        for tag in ("umdNm", "aptNm", "excluUseAr", "floor", "dealAmount", "dealDay", "buildYear"):
            print(f"  {tag:12s} = {(item.findtext(tag) or '').strip()}")


if __name__ == "__main__":
    main()
