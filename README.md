# 서울 토지거래허가구역의 인과효과 추정

컴퓨터공학 졸업작품 (2026-2학기). 서울시 토지거래허가구역(토허제) 지정·해제가
아파트 거래량과 가격에 미친 인과효과를 실거래 미시자료로 추정한다.

## 구조

```
scripts/     수집 -> 정제 -> 분석 파이프라인
data/policy/ 토허구역·규제지역 지정 이력 (서울시 공고 원문 검증)
data/raw/    원자료 스냅샷 (git 제외, 303MB. collect.py 로 재생성)
docs/
  proposal.md      주제 제안서
  prior_work.md    선행연구 정리
  journal/         작업 일지 (일자별)
  decisions/       설계 결정 기록 (번호별)
  reports/         주차별 결과 보고서
output/      분석 결과 (그림, 계수표)
```

## 실행

```bash
cp .env.example .env          # 공공데이터포털 Decoding 키 입력
python scripts/collect.py --districts 11680,11650,11710,11170,11200,11215,11740,11590 \
                          --start 200601 --end 202608
python scripts/build_dataset.py
python scripts/event_study.py
```

데이터 출처: 국토교통부 아파트 매매 실거래 상세 자료 (공공데이터포털).
실거래 자료는 취소·지연신고로 소급 갱신되므로 **수집 시점**을 명시한다.
현재 스냅샷: **2026-09-03 수집**.

## 일정

2026-09-03 착수 ~ 2026-11-20 마감. 매주 목요일 결과 보고.
