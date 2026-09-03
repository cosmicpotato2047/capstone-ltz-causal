"""신청 승인된 엔드포인트가 어느 쪽인지 확인한다. 키 값은 출력하지 않는다."""
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import io  # noqa: E402
io.setup_stdout()


CANDIDATES = [
    ("아파트 매매 실거래가 자료",
     "https://apis.data.go.kr/1613000/RTMSDataSvcAptTrade/getRTMSDataSvcAptTrade"),
    ("아파트 매매 실거래 상세 자료",
     "https://apis.data.go.kr/1613000/RTMSDataSvcAptTradeDev/getRTMSDataSvcAptTradeDev"),
]
BASE = {"LAWD_CD": "11680", "DEAL_YMD": "202407", "pageNo": 1, "numOfRows": 1}


load_key = io.load_key   # lib/io.py 로 통합


def main() -> None:
    key = load_key()
    winner = None

    for name, url in CANDIDATES:
        print("=" * 60)
        print(f"{name}\n{url}")
        try:
            r = requests.get(url, params={**BASE, "serviceKey": key}, timeout=20)
        except Exception as e:
            print(f"  요청 실패: {e}")
            continue
        r.encoding = "utf-8"
        try:
            root = ET.fromstring(r.text)
        except ET.ParseError:
            print(f"  HTTP {r.status_code} / XML 아님: {r.text[:150]}")
            continue

        err = root.findtext(".//errMsg") or root.findtext(".//returnAuthMsg")
        code = root.findtext(".//resultCode")
        if err:
            print(f"  [실패] HTTP {r.status_code}  {err.strip()}")
            continue
        if code in ("00", "000"):
            total = root.findtext(".//totalCount")
            print(f"  [성공] 강남구 202407 신고 건수 = {total} 건")
            winner = (name, url)
            item = root.find(".//item")
            if item is not None:
                tags = [c.tag for c in item]
                print(f"  제공 필드 {len(tags)}개: {', '.join(tags)}")
        else:
            print(f"  [실패] resultCode={code} {root.findtext('.//resultMsg')}")

    print("=" * 60)
    if winner:
        print(f"사용할 엔드포인트: {winner[0]}\n  {winner[1]}")
    else:
        print("둘 다 실패. 마이페이지 > 오픈API > 개발계정 에서 승인 상태를 확인하세요.")


if __name__ == "__main__":
    main()
