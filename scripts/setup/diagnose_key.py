"""인증키 403 원인 진단. 키 값 자체는 절대 출력하지 않는다."""
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import io  # noqa: E402
io.setup_stdout()


ENDPOINT = "https://apis.data.go.kr/1613000/RTMSDataSvcAptTrade/getRTMSDataSvcAptTrade"
BASE = {"LAWD_CD": "11680", "DEAL_YMD": "202407", "pageNo": 1, "numOfRows": 1}


raw_key = io.load_key   # lib/io.py 로 통합


def inspect(k: str) -> str:
    print("=" * 50)
    print("1) 키 형태 점검  (값은 출력하지 않음)")
    print("=" * 50)
    stripped = k.strip()
    print(f"  길이(원본/공백제거)   : {len(k)} / {len(stripped)}")
    if k != stripped:
        print("  [!] 앞뒤 공백 또는 줄바꿈이 있습니다 -> 코드가 strip 하므로 치명적이진 않음")
    if stripped[:1] in ("'", '"') or stripped[-1:] in ("'", '"'):
        print("  [!!] 따옴표가 포함돼 있습니다. .env 에서 따옴표를 제거하세요.")
    if " " in stripped:
        print("  [!!] 키 중간에 공백이 있습니다. 복사가 잘렸을 수 있습니다.")
    print(f"  '%' 포함             : {'%' in stripped}   (True 면 Encoding 키)")
    print(f"  '+' '/' '=' 포함     : {any(c in stripped for c in '+/=')}   (True 면 Decoding 키 특징)")
    print(f"  끝 2글자가 '=='      : {stripped.endswith('==')}")
    if len(stripped) < 40:
        print("  [!!] 너무 짧습니다. 일반 인증키는 보통 80자 이상입니다. 복사 누락 의심.")
    return stripped


def call(label: str, url: str, params=None) -> bool:
    try:
        r = requests.get(url, params=params, timeout=20)
    except Exception as e:
        print(f"  {label:22s} -> 요청 실패: {e}")
        return False
    body = r.text
    msg = ""
    try:
        root = ET.fromstring(body)
        msg = (root.findtext(".//errMsg") or root.findtext(".//resultMsg")
               or root.findtext(".//returnAuthMsg") or "")
    except ET.ParseError:
        msg = body[:80].replace("\n", " ")
    ok = r.status_code == 200 and "ERROR" not in msg.upper()
    print(f"  {label:22s} -> HTTP {r.status_code}  {msg.strip()[:60]}")
    return ok


def main() -> None:
    k = inspect(raw_key())

    print()
    print("=" * 50)
    print("2) 전달 방식 4가지 전부 시도")
    print("=" * 50)
    results = {}
    # A. 원본 키를 params 로 (Decoding 키의 정석)
    results["A"] = call("A. params(원본)", ENDPOINT, {**BASE, "serviceKey": k})
    # B. 원본 키를 URL 에 직접 (이미 인코딩된 Encoding 키의 정석)
    qs = urllib.parse.urlencode(BASE)
    results["B"] = call("B. URL직접(원본)", f"{ENDPOINT}?serviceKey={k}&{qs}")
    # C. 원본을 한 번 디코딩해서 params 로 (Encoding 키를 넣었을 때의 구제책)
    dec = urllib.parse.unquote(k)
    if dec != k:
        results["C"] = call("C. params(디코딩)", ENDPOINT, {**BASE, "serviceKey": dec})
    # D. 원본을 인코딩해서 URL 직접 (Decoding 키를 넣었을 때의 대안 경로)
    enc = urllib.parse.quote(k, safe="")
    if enc != k:
        results["D"] = call("D. URL직접(인코딩)", f"{ENDPOINT}?serviceKey={enc}&{qs}")

    print()
    print("=" * 50)
    print("3) 판정")
    print("=" * 50)
    win = [name for name, ok in results.items() if ok]
    if win:
        print(f"  성공한 방식: {', '.join(win)}")
        print("  -> 키는 정상입니다. 해당 방식으로 수집 코드를 맞추면 됩니다.")
    else:
        print("  4가지 전부 실패했습니다. 전달 방식 문제가 아니라 키/신청 상태 문제입니다.")
        print()
        print("  확인 순서:")
        print("   (1) data.go.kr > 마이페이지 > 오픈API > 개발계정 에서")
        print("       '아파트 매매 실거래가 자료' 신청 상태가 [승인] 인지 확인")
        print("       (목록에 이 API 자체가 없으면 활용신청을 아직 안 하신 겁니다)")
        print("   (2) 승인 직후라면 반영에 최대 1시간 걸립니다. 시간을 두고 재시도")
        print("   (3) 상세페이지의 '일반 인증키(Decoding)' 를 다시 복사")
        print("       (마이페이지 상단 인증키가 아니라 해당 API 상세페이지의 키)")


if __name__ == "__main__":
    main()
