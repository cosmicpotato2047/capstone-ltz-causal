"""신청 승인된 엔드포인트가 어느 쪽인지 확인한다. 키 값은 출력하지 않는다."""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

# Windows 콘솔 인코딩(cp949/cp1252)에서 한글 출력이 깨지지 않도록
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CANDIDATES = [
    ("아파트 매매 실거래가 자료",
     "https://apis.data.go.kr/1613000/RTMSDataSvcAptTrade/getRTMSDataSvcAptTrade"),
    ("아파트 매매 실거래 상세 자료",
     "https://apis.data.go.kr/1613000/RTMSDataSvcAptTradeDev/getRTMSDataSvcAptTradeDev"),
]
BASE = {"LAWD_CD": "11680", "DEAL_YMD": "202407", "pageNo": 1, "numOfRows": 1}


def load_key() -> str:
    env = Path(__file__).resolve().parent.parent / ".env"
    if not env.exists():
        sys.exit(".env 파일이 없습니다.")
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("DATA_GO_KR_KEY="):
            k = line.split("=", 1)[1].strip().strip("'\"")
            if k:
                return k
    sys.exit(".env 의 DATA_GO_KR_KEY 가 비어 있습니다.")


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
