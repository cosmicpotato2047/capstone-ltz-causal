"""표지 정보와 참고문헌.

참고문헌은 학과 안내서가 준 예시 형식을 그대로 따른다.

    [7] 박재학, 이재봉, 최영환. (2010). "제목", 학술지명, 제25권, 제6호, pp. 1~8.

번호는 여기서 매기지 않는다 — 본문에 `[@키]` 로 적으면 **처음 나온 순서대로**
`build_report_docx.py` 가 [1][2][3] 을 붙이고, 인용되지 않은 항목은 콘솔에
알린다. 안내서가 'Not Found 는 감점' 이라고 한 URL 은 넣기 전에 실제로 열어
확인한 것만 적는다.
"""

META = {
    "제목": "서울 토지거래허가구역 인과효과 추정",
    "부제": "실거래 47만 건과 정책 공고 원문을 결합한 미시 패널 구축",
    "학과": "컴퓨터공학과",
    "학번": "(학번)",
    "성명": "이찬희",
    "지도교수": "김성곤",
    "제출일": "2026. 11. 20.",
}

REFS = {
    # ── 국내 선행연구 (이 연구가 직접 재현한 세 편) ──────────────────────
    "jeon2025":
        "전상현, 김성용, 박종선. (2025). “토지거래허가구역 지정이 아파트 "
        "가격에 미치는 영향에 관한 연구”, 융합사회와 공공정책, 제19권, "
        "제1호, pp. 169~189.",
    "lee2025":
        "이한웅, 이춘원. (2025). “토지거래허가구역 지정이 매매가와 전세가에 "
        "미치는 영향: 잠실동 아파트 사례분석”, 대한부동산학회지, 제43권, "
        "제2호, pp. 161~180.",
    "yang2025":
        "양지영, 정준호. (2025). “토지거래허가구역 지정의 풍선효과 분석: "
        "잠실·삼성·대치·청담동 사례를 중심으로”, 한국콘텐츠학회논문지, "
        "제25권, 제5호, p. 447.",

    # ── 이중차분·사건연구 방법론 ─────────────────────────────────────────
    "rambachan2023honest":
        "Rambachan, A., & Roth, J. (2023). “A More Credible Approach to "
        "Parallel Trends”, Review of Economic Studies, Vol. 90, No. 5, "
        "pp. 2555~2591.",
    "dechaisemartin2020twoway":
        "de Chaisemartin, C., & D’Haultfœuille, X. (2020). "
        "“Two-Way Fixed Effects Estimators with Heterogeneous Treatment "
        "Effects”, American Economic Review, Vol. 110, No. 9, "
        "pp. 2964~2996.",
    "goodmanbacon2021":
        "Goodman-Bacon, A. (2021). “Difference-in-Differences with "
        "Variation in Treatment Timing”, Journal of Econometrics, "
        "Vol. 225, No. 2, pp. 254~277.",
    "callaway2021":
        "Callaway, B., & Sant’Anna, P. H. C. (2021). "
        "“Difference-in-Differences with Multiple Time Periods”, "
        "Journal of Econometrics, Vol. 225, No. 2, pp. 200~230.",
    "sunabraham2021":
        "Sun, L., & Abraham, S. (2021). “Estimating Dynamic Treatment "
        "Effects in Event Studies with Heterogeneous Treatment Effects”, "
        "Journal of Econometrics, Vol. 225, No. 2, pp. 175~199.",
    "arkhangelsky2021":
        "Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & "
        "Wager, S. (2021). “Synthetic Difference-in-Differences”, "
        "American Economic Review, Vol. 111, No. 12, pp. 4088~4118.",

    # ── 합성대조군 ───────────────────────────────────────────────────────
    "abadie2010synthetic":
        "Abadie, A., Diamond, A., & Hainmueller, J. (2010). “Synthetic "
        "Control Methods for Comparative Case Studies: Estimating the Effect "
        "of California’s Tobacco Control Program”, Journal of the "
        "American Statistical Association, Vol. 105, No. 490, pp. 493~505.",

    # ── 적은 클러스터 추론 ───────────────────────────────────────────────
    "mackinnon2017wild":
        "MacKinnon, J. G., & Webb, M. D. (2017). “Wild Bootstrap "
        "Inference for Wildly Different Cluster Sizes”, Journal of "
        "Applied Econometrics, Vol. 32, No. 2, pp. 233~254.",
    "webb2014reworking":
        "Webb, M. D. (2014). “Reworking Wild Bootstrap Based Inference "
        "for Clustered Errors”, Working Paper No. 1315, Queen’s "
        "University Department of Economics, Kingston, Canada.",
    "fisher1935design":
        "Fisher, R. A. (1935). “The Design of Experiments”, "
        "Oliver and Boyd, Edinburgh.",

    # ── 사양 곡선 ────────────────────────────────────────────────────────
    "simonsohn2020specification":
        "Simonsohn, U., Simmons, J. P., & Nelson, L. D. (2020). "
        "“Specification Curve Analysis”, Nature Human Behaviour, "
        "Vol. 4, No. 11, pp. 1208~1214.",

    # ── 가격지수 ─────────────────────────────────────────────────────────
    "bailey1963regression":
        "Bailey, M. J., Muth, R. F., & Nourse, H. O. (1963). “A "
        "Regression Method for Real Estate Price Index Construction”, "
        "Journal of the American Statistical Association, Vol. 58, No. 304, "
        "pp. 933~942.",
    "caseshiller1987":
        "Case, K. E., & Shiller, R. J. (1987). “Prices of Single-Family "
        "Homes Since 1970: New Indexes for Four Cities”, New England "
        "Economic Review, pp. 45~56.",

    # ── 쏠림(bunching) ───────────────────────────────────────────────────
    "bestkleven2018":
        "Best, M. C., & Kleven, H. J. (2018). “Housing Market Responses "
        "to Transaction Taxes: Evidence from Notches and Stimulus in the "
        "U.K.”, Review of Economic Studies, Vol. 85, No. 1, pp. 157~193.",

    # ── 선행 작업 (필자 본인). 이 캡스톤의 출발점 ────────────────────────
    "prior2025":
        "이찬희. (2025). “Seoul_Apt_Market — 서울 25개 자치구 아파트 "
        "가격지수와 사이클 분석”, GitHub 저장소. "
        "https://github.com/cosmicpotato2047/Seoul_Apt_Market",

    # ── 자료 출처와 공개 주소 (열어서 확인한 것만 적는다) ────────────────
    "repo":
        "이찬희. (2026). “서울 토지거래허가구역 인과효과 추정 — 코드·"
        "자료·결정기록”, GitHub 저장소. "
        "https://github.com/cosmicpotato2047/capstone-ltz-causal",
    "site":
        "이찬희. (2026). “연구 결과 공개 페이지 — 주차 보고서·발표 "
        "자료·결과 대시보드”, GitHub Pages. "
        "https://cosmicpotato2047.github.io/capstone-ltz-causal/",
    "molit_api":
        "국토교통부. “아파트 매매 실거래가 상세 자료"
        "(RTMSDataSvcAptTradeDev)”, 공공데이터포털 오픈 API. "
        "https://www.data.go.kr/",
}
