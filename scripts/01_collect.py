"""[01] 국토부 아파트 매매 실거래 상세자료 수집.

(자치구, 계약년월) 단위로 조회해 data/raw/<LAWD_CD>/<YYYYMM>.json 에 캐시한다.
이미 받은 달은 건너뛰므로 중단 후 재실행해도 이어서 받는다.

    python scripts/01_collect.py                     # 기본 8개 자치구, 전 기간
    python scripts/01_collect.py --districts 11680 --start 202401 --end 202412
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import io, policy  # noqa: E402

io.setup_stdout()

ENDPOINT = "https://apis.data.go.kr/1613000/RTMSDataSvcAptTradeDev/getRTMSDataSvcAptTradeDev"
NUM_OF_ROWS = 1000
SLEEP = 0.12          # 명세상 30 tps. 넉넉히 억제
MAX_RETRY = 4


def months(start: str, end: str) -> list[str]:
    out, y, m = [], int(start[:4]), int(start[4:])
    ey, em = int(end[:4]), int(end[4:])
    while (y, m) <= (ey, em):
        out.append(f"{y}{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def fetch_page(key: str, lawd: str, ymd: str, page: int) -> tuple[list[dict], int]:
    params = {"serviceKey": key, "LAWD_CD": lawd, "DEAL_YMD": ymd,
              "pageNo": page, "numOfRows": NUM_OF_ROWS}
    last = None
    for attempt in range(MAX_RETRY):
        try:
            r = requests.get(ENDPOINT, params=params, timeout=30)
            r.encoding = "utf-8"
            root = ET.fromstring(r.text)
        except Exception as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
            continue

        err = root.findtext(".//errMsg") or root.findtext(".//returnAuthMsg")
        if err:
            raise RuntimeError(f"{lawd}/{ymd} API 오류: {err.strip()}")

        code = root.findtext(".//resultCode")
        if code not in ("00", "000"):
            msg = root.findtext(".//resultMsg") or ""
            if "NODATA" in msg.upper().replace(" ", "") or code == "03":
                return [], 0
            last = RuntimeError(f"resultCode={code} {msg}")
            time.sleep(1.5 * (attempt + 1))
            continue

        rows = [{c.tag: (c.text or "").strip() for c in item}
                for item in root.iter("item")]
        return rows, int(root.findtext(".//totalCount") or 0)

    raise RuntimeError(f"{lawd}/{ymd} p{page} 재시도 소진: {last}")


def fetch_month(key: str, lawd: str, ymd: str) -> list[dict]:
    rows, total = fetch_page(key, lawd, ymd, 1)
    page = 1
    while len(rows) < total:
        page += 1
        time.sleep(SLEEP)
        more, _ = fetch_page(key, lawd, ymd, page)
        if not more:
            break
        rows.extend(more)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--districts", default=",".join(policy.ALL_SGG),
                    help="쉼표 구분 법정동코드 5자리")
    ap.add_argument("--start", default="200601")
    ap.add_argument("--end", default="202608")
    args = ap.parse_args()

    key = io.load_key()
    codes = [c.strip() for c in args.districts.split(",") if c.strip()]
    ms = months(args.start, args.end)
    print(f"자치구 {len(codes)}개 x {len(ms)}개월 = {len(codes) * len(ms)} 요청 예정")

    for lawd in codes:
        outdir = io.RAW / lawd
        outdir.mkdir(parents=True, exist_ok=True)
        got = skipped = 0
        for ymd in ms:
            f = outdir / f"{ymd}.json"
            if f.exists():
                skipped += 1
                continue
            rows = fetch_month(key, lawd, ymd)
            f.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            got += len(rows)
            time.sleep(SLEEP)
        name = policy.SGG_NAME.get(lawd, lawd)
        print(f"  {lawd} {name}: 신규 {got:,}건 / 캐시된 달 {skipped}개")

    print("완료")


if __name__ == "__main__":
    main()
