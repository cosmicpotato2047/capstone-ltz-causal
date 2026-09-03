"""경로 상수, 인증키 로딩, 데이터 입출력."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
POLICY = ROOT / "data" / "policy"
FIGURES = ROOT / "output" / "figures"
RESULTS = ROOT / "output" / "results"

TRADES = PROCESSED / "trades.parquet"


def setup_stdout() -> None:
    """Windows 콘솔(cp949/cp1252)에서 한글 출력이 깨지지 않도록."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_key() -> str:
    """.env 에서 공공데이터포털 Decoding 키를 읽는다."""
    env = ROOT / ".env"
    if not env.exists():
        sys.exit(".env 파일이 없습니다. .env.example 을 복사해 키를 채우세요.")
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("DATA_GO_KR_KEY="):
            key = line.split("=", 1)[1].strip().strip("'\"")
            if key:
                if "%" in key:
                    print("[경고] 키에 '%' 가 있습니다. Encoding 키가 아니라 "
                          "Decoding 키를 넣으세요.\n")
                return key
    sys.exit(".env 의 DATA_GO_KR_KEY 가 비어 있습니다.")


def load_trades():
    """정제된 거래 자료를 읽는다."""
    import pandas as pd
    if not TRADES.exists():
        sys.exit(f"{TRADES} 가 없습니다. scripts/02_build.py 를 먼저 실행하세요.")
    return pd.read_parquet(TRADES)


def save_result(name: str, payload: dict) -> Path:
    """추정 결과를 JSON 으로 저장한다. 문서에 숫자를 주입할 때 이 파일을 읽는다."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / f"{name}.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                 encoding="utf-8")
    return p
