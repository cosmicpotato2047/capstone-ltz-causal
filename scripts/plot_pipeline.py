r"""[V6] 파이프라인 도식 — 원자료에서 보고서까지 무엇이 무엇을 먹는가.

    python scripts/plot_pipeline.py

입력  run_all.py 의 STEPS (단계 수를 여기서 읽는다)
      output/results/, output/figures/, paper/tables/ (산출물 수를 센다)
출력  output/figures/pipeline.png
      paper/generated/counts.tex  (본문이 쓰는 \pipe... 매크로)

최종 보고서 3.5절(재현 장치)에 들어간다. 그림의 단계 수를 **손으로 적지
않는다** — 주차 보고서가 한 편 늘면 문서 단계도 하나 늘기 때문에, 손으로
적으면 바로 유령 숫자가 된다(3.7절이 그 사건이다). 그래서 run_all.py 를
import 해서 세고, 산출물은 실제 파일 수를 센다.

같은 이유로 **본문의 단계 수도 손으로 적지 않는다**. 주차 보고서가 한 편
늘면 48이 49가 되므로, 여기서 매크로 파일을 하나 써 두고 본문은 그것을 쓴다.

도식이 말하려는 것은 셋이다.

  1. 입구가 셋(실거래 API · 공고 PDF · 법정동 경계)인데 **출구가 하나**다 —
     output/results/*.json 가 현재 값의 유일한 출처다 ([[0007-문서유형별-수치기록]]).
  2. 공고는 '참고 자료'가 아니라 **자료의 한 축**이다. PDF 에서 지정 구간까지
     내려오는 가운데 줄기가 없으면 처치 변수 자체가 없다.
  3. 점검 넷은 분석 **뒤**가 아니라 매 실행에 걸린다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.patches import FancyBboxPatch        # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from lib import io                                   # noqa: E402
import run_all                                       # noqa: E402

io.setup_stdout()
for f in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
    if f in {x.name for x in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

FP = ROOT / "output" / "figures" / "pipeline.png"
TEX = ROOT / "paper" / "generated" / "counts.tex"

# 매크로 이름 -> counts() 의 열쇠. 본문은 \pipeSteps 처럼 쓴다.
MACROS = {
    "pipeSteps": "수집제외", "pipeStepsAll": "전체",
    "pipeData": "자료", "pipePolicy": "정책", "pipeAnalysis": "분석",
    "pipeReplicate": "재현", "pipeViz": "시각화", "pipeDocs": "문서",
    "pipeResults": "결과json", "pipeFigures": "그림", "pipeTables": "표",
    "pipeDecisions": "결정기록",
}

# 색. 입력은 회색, 우리가 지은 것은 파랑, 유일한 출처는 진하게, 점검은 주황.
C_IN, C_MID, C_HUB, C_OUT, C_CHK = "#eaeaea", "#dbe9f6", "#2f5e8f", "#eef3f8", "#fbe7d4"
E_IN, E_MID, E_HUB, E_OUT, E_CHK = "#9a9a9a", "#7aa6cc", "#2f5e8f", "#9db8d0", "#d9924a"

GROUPS = ("수집", "자료", "정책", "분석", "재현", "시각화", "문서")


def counts() -> dict[str, int]:
    """단계 수와 산출물 수를 센다. 어느 것도 손으로 적지 않는다."""
    steps = run_all.STEPS + run_all.report_steps()
    g: dict[str, int] = {}
    for _, grp, *_ in steps:
        g[grp] = g.get(grp, 0) + 1
    g["전체"] = sum(g.values())
    g["수집제외"] = g["전체"] - g.get("수집", 0)
    g["결과json"] = len(list((ROOT / "output" / "results").glob("*.json")))
    # 자기 자신을 세는 문제. 이 그림이 있기 전에 세면 하나 적고 있고 나서 세면
    # 하나 많다 — 매 실행마다 숫자가 흔들린다. 합집합으로 두면 둘 다 같다.
    g["그림"] = len({*(ROOT / "output" / "figures").glob("*.png"), FP})
    g["표"] = len(list((ROOT / "paper" / "tables").glob("*.tex")))
    g["결정기록"] = len(list((ROOT / "docs" / "decisions").glob("[0-9]*.md")))
    return g


def write_macros(n: dict[str, int]) -> None:
    """본문이 쓸 단계 수 매크로. 손으로 적으면 바로 유령 숫자가 된다."""
    lines = [
        "% 이 파일은 scripts/plot_pipeline.py 가 만든다. 손으로 고치지 않는다.",
        "% 단계 수는 run_all.py 의 STEPS 에서, 산출물 수는 실제 파일에서 센다.",
        "",
    ]
    for macro, key in MACROS.items():
        lines.append(r"\newcommand{\%s}{%d}" % (macro, n[key]))
    TEX.parent.mkdir(parents=True, exist_ok=True)
    TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def box(ax, x, y, w, h, title, body, fc, ec, tc="#1a1a1a", ts=10.5, bs=8.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                linewidth=1.3, facecolor=fc, edgecolor=ec, zorder=2))
    cx = x + w / 2
    ax.text(cx, y + h - (h * 0.30 if body else h * 0.5), title, ha="center",
            va="center", fontsize=ts, fontweight="bold", color=tc, zorder=3)
    if body:
        ax.text(cx, y + h * 0.30, body, ha="center", va="center",
                fontsize=bs, color=tc, zorder=3, linespacing=1.5)


def arrow(ax, x0, y0, x1, y1, label="", side="right", dashed=False, rad=0.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=15, linewidth=1.5,
                                color="#5a5a5a", shrinkA=1, shrinkB=1,
                                linestyle="--" if dashed else "-",
                                connectionstyle="arc3,rad=" + str(rad)), zorder=1)
    if label:
        dx = 1.2 if side == "right" else -1.2
        ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2, label, fontsize=8.2,
                color="#444444", ha="left" if side == "right" else "right",
                va="center", zorder=3, linespacing=1.4)


def main() -> None:
    n = counts()
    # tight_layout 에 맡기지 않는다 — 좌표를 그림 전체에 직접 건다. 제목도
    # suptitle 이 아니라 데이터 좌표로 찍어야 겹치지 않는다.
    fig = plt.figure(figsize=(10.2, 11.2))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    L, W = 7.0, 91.0            # 가로 전체 띠의 왼쪽 끝과 너비
    CX = L + W / 2              # 가운데 열
    X1, X2, X3, CW = 7.0, 38.5, 70.0, 28.0
    C1X, C2X, C3X = X1 + CW / 2, X2 + CW / 2, X3 + CW / 2

    # ── 제목 ────────────────────────────────────────────────────────────────
    ax.text(3, 97.6, "원자료에서 보고서까지 — " + str(n["수집제외"])
            + "단계 파이프라인 하나로 돈다",
            fontsize=14, fontweight="bold", color="#1a1a1a", ha="left", va="top")
    ax.text(3, 94.6, "python run_all.py   (수집 포함 " + str(n["전체"])
            + "단계 · 수집 제외 " + str(n["수집제외"]) + "단계)",
            fontsize=9.4, color="#666666", ha="left", va="top")

    # ── 1단: 입력 셋 ────────────────────────────────────────────────────────
    box(ax, X1, 83, CW, 9, "실거래 상세 자료",
        "국토교통부 공개 API\nRTMSDataSvcAptTradeDev\n활용신청이 따로 필요", C_IN, E_IN, bs=8.4)
    box(ax, X2, 83, CW, 9, "정책 공고 원문",
        "서울시보 · 전자관보 PDF\n원본 2.9 GB — 저장소 밖\n발췌본만 저장소에", C_IN, E_IN, bs=8.4)
    box(ax, X3, 83, CW, 9, "법정동 경계",
        "행정구역 폴리곤\n맞닿은 동 판정에 쓴다", C_IN, E_IN, bs=8.4)

    # ── 2단: 정제·추출 ──────────────────────────────────────────────────────
    arrow(ax, C1X, 82.5, C1X, 75.5, "01  수집", "right")
    arrow(ax, C2X, 82.5, C2X, 75.5, "P1  글꼴별 오프셋 교정", "right")
    arrow(ax, C3X, 82.5, C3X, 75.5)

    box(ax, X1, 63.5, CW, 11, "정제 표 두 장",
        "원자료 482,231건\n취소 6,293건(1.30%) 제외\n신고지연으로 2026-05 절단\n분석 8개 구 · 서울 25개 구",
        C_MID, E_MID, bs=8.2)
    box(ax, X2, 63.5, CW, 11, "법정동 지정 구간",
        "본문 텍스트 → 구역 표 1,168건\n→ 법정동별 [시작일, 종료일)\n지정 · 재지정 · 해제를\n시간 순으로 이어 붙인다",
        C_MID, E_MID, bs=8.2)
    box(ax, X3, 63.5, CW, 11, "공간 보조표",
        "맞닿은 동 11개\n파급 분해의 비교 단위", C_MID, E_MID, bs=8.2)

    # ── 3단: 결합 ───────────────────────────────────────────────────────────
    arrow(ax, C1X, 63, CX - 14, 54.0, "02 · 02s", "left", rad=-0.12)
    arrow(ax, C2X, 63, CX, 54.0, "P4 · P5", "right")
    arrow(ax, C3X, 63, CX + 14, 54.0, "", rad=0.12)

    box(ax, L, 43, W, 10.5, "미시 패널 — 거래 한 건이 한 행",
        "aptSeq (단지 식별자) · excluUseAr (전용면적) · floor (층) · buildYear · 계약일\n"
        "+ 계약일 × 법정동으로 그 시점의 규제 상태를 붙인다 (lib/policy)\n"
        "단지 고정효과를 쓸 수 있게 되는 지점 — 결과 크기를 두 배 가른 설계 선택 (0003)",
        C_MID, E_MID, ts=11.5, bs=8.6)

    # ── 4단: 분석 ───────────────────────────────────────────────────────────
    arrow(ax, CX, 42.5, CX, 36.0)
    box(ax, L, 24, W, 11.5, "분석 " + str(n["분석"]) + "단계",
        "사건연구 (효력일 정렬 · 거래 없는 달은 0)        적은 클러스터 추론 (WCR · 무작위화)\n"
        "이질성 (가격대 · 전용면적 · 허가 면적 기준)        파급 분해 (같은 동 · 맞닿은 동 · 기준점)\n"
        "자체 가격지수 둘 (반복매매 · 헤도닉)        강건성 일곱 (위약 · 합성대조군 · 사양곡선 …)\n"
        "+ 선행연구 재현 " + str(n["재현"]) + "편",
        C_MID, E_MID, ts=11.5, bs=8.4)

    # ── 5단: 유일한 출처 ────────────────────────────────────────────────────
    arrow(ax, CX, 23.5, CX, 18.0)
    box(ax, L + 14, 11.5, W - 28, 6.5, "output/results/*.json",
        "현재 값의 유일한 출처 — 파일 " + str(n["결과json"]) + "개 (0007)",
        C_HUB, E_HUB, tc="#ffffff", ts=12.5, bs=9.2)

    # ── 6단: 소비 ───────────────────────────────────────────────────────────
    arrow(ax, CX - 16, 11.0, C1X, 7.0, "", rad=0.1)
    arrow(ax, CX, 11.0, C2X, 7.0)
    arrow(ax, CX + 16, 11.0, C3X, 7.0, "", rad=-0.1)

    box(ax, X1, 1.0, CW, 6, "그림 " + str(n["그림"]) + "장",
        "output/figures/", C_OUT, E_OUT, ts=10.5)
    box(ax, X2, 1.0, CW, 6, "보고서 표 " + str(n["표"]) + "개",
        "paper/tables/*.tex", C_OUT, E_OUT, ts=10.5)
    box(ax, X3, 1.0, CW, 6, "결과 대시보드",
        "output/dashboard.html", C_OUT, E_OUT, ts=10.5)

    # ── 점검 레일: 왼쪽에서 전 구간에 걸린다 ────────────────────────────────
    ax.add_patch(FancyBboxPatch((1.2, 1.0), 3.4, 91.0,
                                boxstyle="round,pad=0.3,rounding_size=1.0",
                                linewidth=1.2, linestyle="--",
                                facecolor=C_CHK, edgecolor=E_CHK, zorder=0))
    ax.text(2.9, 46.5, "자동 점검 넷 — 매 실행마다 전 구간에 걸린다        "
                       "C1 폐기된 값 · C2 고아 산출물 · C3 의존 어긋남 · T1 표 생성",
            fontsize=8.8, color="#8a5520", ha="center", va="center",
            rotation=90, zorder=3)
    for y in (87.5, 69.0, 48.0, 29.5, 14.5, 4.0):
        arrow(ax, 4.8, y, 6.4, y, "", dashed=True)

    FP.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FP, dpi=150, facecolor="white")
    plt.close(fig)

    write_macros(n)

    print("[V6] " + str(FP.relative_to(ROOT)) + " · " + str(TEX.relative_to(ROOT)))
    print("  단계  전체 " + str(n["전체"]) + " · 수집 제외 " + str(n["수집제외"]))
    print("        " + " · ".join(g + " " + str(n.get(g, 0)) for g in GROUPS))
    print("  산출  결과 JSON " + str(n["결과json"]) + " · 그림 " + str(n["그림"])
          + " · 표 " + str(n["표"]))


if __name__ == "__main__":
    main()
