# 서울 토지거래허가구역의 인과효과 추정

컴퓨터공학 졸업작품 (2026-2학기). 서울시 토지거래허가구역(토허제) 지정·해제가
아파트 **거래량과 가격**에 미친 인과효과를 실거래 미시자료로 추정한다.

- [주제 제안서](docs/proposal.md)
- [선행연구 정리](docs/prior_work.md)
- [주차별 결과 보고서](docs/reports/)

## 재현

```bash
pip install -r requirements.txt
cp .env.example .env          # 공공데이터포털 Decoding 키 입력
python run_all.py --collect   # 수집부터 전부 (수십 분)
python run_all.py             # 수집 생략, 정제부터
python run_all.py --only 04   # 특정 단계만
python run_all.py --extra     # 선행연구 재현 + 문서 수치 검증 포함
```

데이터 출처는 국토교통부 「아파트 매매 실거래 **상세** 자료」(공공데이터포털).
동일 기관의 「아파트 매매 실거래가 자료」와는 별개 서비스이므로 활용신청이
개별적으로 필요하다. 엔드포인트는 `RTMSDataSvcAptTradeDev`.

**실거래 자료는 취소와 지연신고로 소급 갱신된다.** 수집 시점이 다르면 결과도
달라지므로 스냅샷 시점을 명시한다. 현재 스냅샷: **2026-09-03 수집**.

## 구조

```
run_all.py            전체 파이프라인 진입점
requirements.txt      패키지 버전 (Python 3.13.7 에서 검증)

scripts/
  lib/                공용 모듈 — import 전용
    io.py             경로 상수, 인증키, 데이터 입출력, 결과 저장
    policy.py         처치 정의 (지정 이력, 처치동, 유지단지) ← 정의 변경은 여기 한 곳
    econ.py           고정효과 제거, event study, 이중차분
  01_collect.py       수집   (자치구, 계약년월) 단위 JSON 캐시. 중단 후 재개 가능
  02_build.py         정제   취소거래 제외, 단가 산출, 검증 → parquet
  04_event_study.py   분석   2020.6 지정 event study
  setup/              환경 점검 (일회성)
  explore/            탐색 스크립트
  replicate/          선행연구 재현

data/
  policy/
    notices/          공고 원문 PDF + manifest.csv (링크 소멸 대비)
    ltz_events.csv    코드화된 토허구역 지정 이력
  raw/                원자료 스냅샷 (git 제외, 303MB)
  processed/          정제 결과 (git 제외)

output/
  figures/            그림
  results/            추정 결과 JSON — 문서의 수치는 여기서 가져온다

docs/
  proposal.md         주제 제안서
  prior_work.md       선행연구
  journal/            작업 일지 (작업한 날만, 일자별 파일)
  decisions/          설계 결정 기록 (번호별) → 논문 방법론 절의 원재료
  reports/            주차별 제출 보고서
  papers/             참고 논문 (git 제외, 저작권)
```

### 설계 원칙

- **처치 정의는 `scripts/lib/policy.py` 한 곳에서만** 관리한다. 검증 결과로
  정의가 바뀌어도 한 줄만 고치면 전체에 반영된다.
- **모든 추정 결과를 `output/results/*.json` 으로 저장**한다. 문서의 수치를
  손으로 옮기지 않기 위해서다.
- **정제 단계에 검증을 넣는다**(`02_build.py`의 `validate`). 표본 수·결측·
  날짜 범위가 기대를 벗어나면 파이프라인이 중단된다.

## 일정

2026-09-03 착수 ~ 2026-11-20 마감. 매주 목요일 결과 보고.
제출 시점마다 `week-NN` 태그를 남긴다.

## 라이선스

코드 MIT / 문서 CC BY 4.0. 자세한 내용은 [LICENSE](LICENSE) 참조.

## 어디를 보면 되나

| 알고 싶은 것 | 볼 곳 |
|---|---|
| **지금 뭘 하고 있고 다음은 뭔가** | [docs/backlog.md](docs/backlog.md) — 맨 위 "현재 상태" |
| 이번 주에 뭘 했나 | [docs/reports/](docs/reports/) — 주차별 보고서 |
| 왜 그렇게 설계했나 | [docs/decisions/](docs/decisions/) — 번호별 결정 기록 |
| 용어가 무슨 뜻인가 | [docs/glossary.md](docs/glossary.md) |
| 그날그날 무슨 일이 있었나 | [docs/journal/](docs/journal/) |
| 숫자의 원본 | `output/results/*.json` — 문서와 어긋나면 이쪽이 맞다 |
