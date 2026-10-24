"""토허구역 지정 이력을 법정동 × 월 히트맵으로 그린다.

    python scripts/plot_ltz_heatmap.py              # 서울 전역 + 분석 8개구
    python scripts/plot_ltz_heatmap.py --scope gu   # 분석 8개구만

입력  data/policy/ltz_panel.csv     (build_ltz_panel.py)
출력  output/figures/ltz_heatmap_seoul.png
      output/figures/ltz_heatmap_analysis.png

이 그림은 장식이 아니라 **처치군을 고르는 도구**다. 한 줄이 한 법정동,
가로가 시간이다. 색이 끊기지 않고 이어지면 재지정으로 계속 처치받은
것이고, 중간에 끊기면 해제된 것이다.

'불명'(회색 빗금)은 지정 구간 사이에 뚫린 구멍이다. 재지정 공고를 아직
못 읽었을 뿐일 수도 있어 지정으로 메우지 않는다 (결정기록 0009).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch          # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import io, policy  # noqa: E402

io.setup_stdout()

for f in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
    if f in {x.name for x in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

# 0 미지정 / 1 불명 / 2 지정
COLORS = ["#f2f2f2", "#c9c9c9", "#1f4e79"]
# 우리 처치 계열은 따로 칠해 눈에 띄게 한다
COLOR_MAIN = "#c0392b"

EVENTS = [("2020-06", "2020.6\n최초 지정"), ("2021-04", "2021.4\n재건축단지"),
          ("2022-06", "2022.6\n기준 18→6㎡"), ("2025-02", "2025.2\n부분 해제"),
          ("2025-03", "2025.3\n강남3구·용산"), ("2025-10", "2025.10\n서울 전역")]


def draw(pan: pd.DataFrame, dongs: pd.DataFrame, title: str, out: Path,
         lo: str, hi: str, n_read: int, label_size: float = 7.0,
         extra_note: str = "", width: float = 15.0, row_h: float = 0.16) -> None:
    months = pd.period_range(lo, hi, freq="M")
    mi = {m: i for i, m in enumerate(months)}
    M = np.zeros((len(dongs), len(months)), int)
    main = np.zeros_like(M, bool)

    idx = {(r.자치구, r.법정동): i for i, r in dongs.reset_index().iterrows()}
    for r in pan.itertuples():
        k, m = (r.자치구, r.법정동), pd.Period(r.연월, freq="M")
        if k in idx and m in mi:
            M[idx[k], mi[m]] = 2 if r.상태 == "지정" else 1
            if r.상태 == "지정" and "국제교류복합지구" in str(r.계열):
                main[idx[k], mi[m]] = True

    h = max(4.0, row_h * len(dongs) + 2.2)
    fig, ax = plt.subplots(figsize=(width, h))
    ax.imshow(M, aspect="auto", interpolation="nearest",
              cmap=ListedColormap(COLORS), vmin=0, vmax=2)
    ov = np.zeros((*M.shape, 4))
    ov[main] = matplotlib.colors.to_rgba(COLOR_MAIN)
    ax.imshow(ov, aspect="auto", interpolation="nearest")

    ax.set_yticks(range(len(dongs)))
    ax.set_yticklabels([f"{r.자치구[:-1]} {r.법정동}" for r in dongs.itertuples()],
                       fontsize=label_size)
    tick = [i for i, m in enumerate(months) if m.month == 1]
    ax.set_xticks(tick)
    ax.set_xticklabels([str(months[i].year) for i in tick], fontsize=9)
    ax.set_xlabel("계약 연월")

    for j, (ym, label) in enumerate(EVENTS):
        p = pd.Period(ym, freq="M")
        if p in mi:
            ax.axvline(mi[p], color="#333", lw=0.9, ls="--", alpha=0.7)
            # 2025년 사건들이 몰려 있어 이름표를 두 줄로 엇갈려 놓는다.
            # 간격은 **포인트**로 준다 — 행 수로 주면 행이 적은 그림에서
            # 이름표가 제목 위로 올라가 겹친다.
            ax.annotate(label, xy=(mi[p], -0.5), xycoords="data",
                        xytext=(0, 6 + 20 * (j % 2)), textcoords="offset points",
                        fontsize=7, ha="center", va="bottom",
                        color="#333", annotation_clip=False)

    ax.set_title(title, fontsize=13, pad=62, loc="left")
    ax.text(0, len(dongs) + 2.5,
            f"공고 원문에서 구역 목록을 읽어낸 {n_read}건 기준. "
            "읽지 못한 공고의 지정은 '미지정'으로 보이므로 빈 칸이 곧 미지정은 아니다."
            + (f"\n{extra_note}" if extra_note else ""),
            fontsize=7.5, color="#666", va="top")

    ax.legend(handles=[Patch(color=COLOR_MAIN, label="지정 (국제교류복합지구)"),
                       Patch(color=COLORS[2], label="지정 (그 밖의 계열)"),
                       Patch(color=COLORS[1], label="불명 — 공고를 아직 못 읽은 구간"),
                       Patch(color=COLORS[0], label="미지정")],
              loc="upper left", bbox_to_anchor=(1.005, 1), fontsize=8,
              frameon=False)
    ax.set_ylim(len(dongs) - 0.5, -0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.relative_to(io.ROOT)}  ({len(dongs)}개 동 × {len(months)}개월)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", default="2019-01")
    ap.add_argument("--hi", default="2026-08")
    a = ap.parse_args()

    pan = pd.read_csv(io.POLICY / "ltz_panel.csv")
    z = pd.read_csv(io.POLICY / "notices" / "zones.csv")
    n_read = z.파일.nunique()
    io.FIGURES.mkdir(parents=True, exist_ok=True)

    # 정렬: 우리 처치 계열 먼저, 그다음 자치구·법정동
    d = (pan.groupby(["자치구", "법정동"])
         .agg(주계열=("계열", lambda s: s.str.contains("국제교류복합지구").any()),
              첫달=("연월", "min")).reset_index())
    d = d.sort_values(["주계열", "자치구", "첫달"], ascending=[False, True, True])

    print("히트맵 생성")
    draw(pan, d, "서울 토지거래허가구역 지정 이력 — 법정동별 (공고 원문 기준)",
         io.FIGURES / "ltz_heatmap_seoul.png", a.lo, a.hi, n_read)

    ours = d[d.자치구.isin(policy.SGG_NAME.values())]
    draw(pan, ours, "토지거래허가구역 지정 이력 — 분석 대상 8개 자치구",
         io.FIGURES / "ltz_heatmap_analysis.png", a.lo, a.hi, n_read)

    # 보고서용. 8개 구 전체(115행)를 한 쪽에 넣으면 동 이름이 안 읽히므로
    # **추정이 실제로 쓰는 동**만 남긴다 — 처치군(A·B그룹)과 강남·송파 통제동.
    # 좁게 그리는 것이 중요하다. 가로 15인치짜리를 본문 폭(16cm)에 맞추면
    # 2.4배로 줄어들어 글자가 뭉개진다.
    # 처치군은 **서울 전역에서** 뽑는다. B그룹의 여의도동·목동은 영등포구·
    # 양천구라 분석 8개 자치구 밖이다 — ours 에서 고르면 두 동이 빠진다.
    keep = policy.TREATED_2020 | policy.TREATED_2021
    pick = d[d.법정동.isin(keep) | d.자치구.isin(["강남구", "송파구"])].copy()
    pick["처치"] = pick.법정동.isin(keep)
    pick = pick.sort_values(["처치", "자치구", "첫달"],
                            ascending=[False, True, True])
    draw(pan, pick, "토지거래허가구역 지정 이력 — 추정에 쓰는 법정동",
         io.FIGURES / "ltz_heatmap_report.png", a.lo, a.hi, n_read,
         label_size=8.5, width=9.0, row_h=0.21,
         extra_note=f"위 {int(pick.처치.sum())}줄이 처치군(2020.6 A그룹 "
                    f"{len(policy.TREATED_2020)} · 2021.4 B그룹 "
                    f"{len(policy.TREATED_2021)}), 아래 "
                    f"{len(pick) - int(pick.처치.sum())}줄이 강남·송파 통제 "
                    f"후보다. 서울 법정동 {len(d)}개 중 {len(pick)}개를 그렸다.")


if __name__ == "__main__":
    main()
