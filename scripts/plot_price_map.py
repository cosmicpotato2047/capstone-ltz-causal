"""서울 25개 자치구 아파트값 지도를 그린다. 배경화면용.

    python scripts/plot_price_map.py
    python scripts/plot_price_map.py --w 3840 --h 2160 --months 12 --theme light

입력  data/geo/seoul_municipalities_simple.json
      data/processed/trades_seoul.parquet
출력  output/figures/seoul_price_map.png

값은 **전용 84㎡ 중위 실거래가**다. 평균은 비싼 거래 몇 건에 끌려 올라가고
(고가 자치구에서 중위가보다 2~3억 높다), 전체 중위가는 자치구마다 주력
평형이 달라 비교가 흔들린다. 84㎡ 로 고정하면 둘 다 피하면서 '국민평형'
이라 읽기도 쉽다. 72~90㎡ 를 84㎡ 로 본다.

기간 기본값은 최근 6개월이다. 신고지연 때문에 표본을 2026-05-31 에서
자르므로(결정기록 0004) '오늘 기준'은 만들 수 없고, 3개월은 종로구가
얇으며, 12개월은 2025-10-20 서울 전역 지정 전후가 섞인다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize   # noqa: E402
from matplotlib.patches import Polygon                # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import geo, io  # noqa: E402

io.setup_stdout()

for f in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
    if f in {x.name for x in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

AREA_LO, AREA_HI = 72.0, 90.0        # 전용 84㎡ 로 보는 범위

# 도심은 자치구가 작고 몰려 있어 무게중심에 그대로 찍으면 글자가 겹친다.
# 도 단위로 조금씩 밀어 준다. (경도, 위도)
NUDGE = {
    "중구": (0.004, 0.008), "종로구": (-0.004, 0.012), "서대문구": (-0.010, 0.000),
    "성동구": (-0.002, 0.004), "광진구": (0.018, -0.002), "동대문구": (0.004, 0.006),
    "용산구": (-0.002, -0.006), "마포구": (-0.008, -0.004), "성북구": (0.002, 0.004),
    "영등포구": (-0.004, 0.002), "동작구": (0.002, -0.002),
}

THEMES = {
    "dark": dict(bg="#11151c", fg="#f2f4f7", mut="#8b97a8", edge="#11151c",
                 ramp=["#1b3a5c", "#2a6f97", "#4e9f3d", "#d4a017", "#e07a2f",
                       "#c1272d"]),
    "light": dict(bg="#fbfbfa", fg="#1a1a1a", mut="#777", edge="#fbfbfa",
                  ramp=["#e8eef5", "#9dc3e0", "#f2d88f", "#e8945a", "#c1272d"]),
}


def rings(geom):
    if geom["type"] == "Polygon":
        return geom["coordinates"]
    return [r for poly in geom["coordinates"] for r in poly]


def centroid(ring) -> tuple[float, float]:
    """다각형 무게중심. 경계가 오목해도 대체로 안쪽에 떨어진다."""
    p = np.asarray(ring, float)
    x, y = p[:, 0], p[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y2 - x2 * y
    a = cr.sum() / 2.0
    if abs(a) < 1e-12:
        return float(x.mean()), float(y.mean())
    return float(((x + x2) * cr).sum() / (6 * a)), float(((y + y2) * cr).sum() / (6 * a))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=int, default=2560)
    ap.add_argument("--h", type=int, default=1440)
    ap.add_argument("--months", type=int, default=6, help="최근 몇 개월")
    ap.add_argument("--theme", choices=list(THEMES), default="dark")
    ap.add_argument("--plain", action="store_true",
                    help="왼쪽 글 기둥 없이 지도만. 바탕화면 아이콘이 가릴 때")
    a = ap.parse_args()
    T = THEMES[a.theme]

    df = pd.read_parquet(io.TRADES_SEOUL)
    end = df.deal_date.max()
    lo = (end.to_period("M") - (a.months - 1)).to_timestamp()
    s = df[(df.deal_date >= lo) & df.area_m2.between(AREA_LO, AREA_HI)]
    if s.empty:
        sys.exit("해당 기간에 84㎡ 거래가 없습니다.")
    g = s.groupby("sgg_nm").amount_manwon.agg(["median", "size"])
    g["eok"] = g["median"] / 10000

    mu = json.loads(geo.MUNICIPALITIES.read_text(encoding="utf-8"))
    feats = []
    for f in mu["features"]:
        nm = f["properties"]["SIG_KOR_NM"]
        rs = rings(f["geometry"])
        big = max(rs, key=len)
        feats.append({"nm": nm, "rings": rs, "c": centroid(big),
                      "eok": float(g.eok.get(nm, np.nan)),
                      "n": int(g["size"].get(nm, 0))})

    cmap = LinearSegmentedColormap.from_list("seoul", T["ramp"], N=256)
    vals = np.array([f["eok"] for f in feats])
    norm = Normalize(vmin=np.nanmin(vals), vmax=np.nanmax(vals))

    dpi = 160
    fig = plt.figure(figsize=(a.w / dpi, a.h / dpi), dpi=dpi)
    fig.patch.set_facecolor(T["bg"])
    ax = fig.add_axes([0.02, 0.03, 0.96, 0.94] if a.plain
                      else [0.29, 0.03, 0.69, 0.94])
    ax.set_facecolor(T["bg"])
    ax.set_axis_off()

    for f in feats:
        col = cmap(norm(f["eok"])) if np.isfinite(f["eok"]) else T["mut"]
        for r in f["rings"]:
            ax.add_patch(Polygon(r, closed=True, facecolor=col,
                                 edgecolor=T["edge"], linewidth=1.4, zorder=2))

    xs = [p[0] for f in feats for r in f["rings"] for p in r]
    ys = [p[1] for f in feats for r in f["rings"] for p in r]
    ax.set_xlim(min(xs), max(xs))
    ax.set_ylim(min(ys), max(ys))
    ax.set_aspect(1 / np.cos(np.radians((min(ys) + max(ys)) / 2)))

    base = a.w / 2560
    for f in feats:
        x, y = f["c"]
        if not np.isfinite(f["eok"]):
            continue
        # 밝은 칸에는 어두운 글씨를, 어두운 칸에는 밝은 글씨를
        r_, g_, b_, _ = cmap(norm(f["eok"]))
        lum = 0.299 * r_ + 0.587 * g_ + 0.114 * b_
        tc = "#111" if lum > 0.6 else "#fff"
        dx, dy = NUDGE.get(f["nm"], (0.0, 0.0))
        x, y = x + dx, y + dy
        ax.text(x, y + 0.0035, f["nm"], ha="center", va="bottom", color=tc,
                fontsize=14 * base, fontweight="bold", zorder=3)
        ax.text(x, y - 0.0035, f"{f['eok']:.1f}억", ha="center", va="top", color=tc,
                fontsize=18 * base, fontweight="bold", zorder=3)

    L = 0.045
    if not a.plain:
        fig.text(L, 0.955, "서울 아파트값 지도", color=T["fg"],
                 fontsize=40 * base, fontweight="bold", va="top")
        fig.text(L, 0.885, "전용 84㎡ 중위 실거래가", color=T["fg"],
                 fontsize=19 * base, va="top")
        fig.text(L, 0.848, f"{lo.year}년 {lo.month}월 ~ {end.year}년 {end.month}월"
                 f"  ·  {int(g['size'].sum()):,}건",
                 color=T["mut"], fontsize=15 * base, va="top")

        # 비싼 곳과 싼 곳을 나란히
        top = g.sort_values("eok", ascending=False)
        y = 0.735
        for label, rows in (("높은 곳", top.head(5)), ("낮은 곳", top.tail(5)[::-1])):
            fig.text(L, y, label, color=T["mut"], fontsize=13 * base,
                     fontweight="bold", va="top")
            y -= 0.040
            for nm, v in rows.eok.items():
                fig.text(L, y, nm, color=T["fg"], fontsize=15 * base, va="top")
                fig.text(L + 0.165, y, f"{v:.1f}억", color=T["fg"], fontsize=15 * base,
                         fontweight="bold", ha="right", va="top")
                y -= 0.036
            y -= 0.030

        cax = fig.add_axes([L, 0.075, 0.165, 0.014])
        cax.imshow(np.linspace(0, 1, 256).reshape(1, -1), aspect="auto", cmap=cmap)
        cax.set_xticks([]); cax.set_yticks([])
        for sp in cax.spines.values():
            sp.set_visible(False)
        fig.text(L, 0.062, f"{np.nanmin(vals):.0f}억", color=T["mut"],
                 fontsize=12 * base, va="top")
        fig.text(L + 0.165, 0.062, f"{np.nanmax(vals):.0f}억", color=T["mut"],
                 fontsize=12 * base, ha="right", va="top")
        fig.text(L, 0.032, "국토교통부 아파트 매매 실거래 상세자료",
                 color=T["mut"], fontsize=11 * base, va="top")

    name = "seoul_price_map_plain" if a.plain else "seoul_price_map"
    out = io.FIGURES / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor=T["bg"])
    plt.close(fig)

    print(f"저장 {out.relative_to(io.ROOT)}  {a.w}x{a.h}  "
          f"({out.stat().st_size/1e6:.2f}MB)")
    print(f"  기간 {lo.date()} ~ {end.date()} · 84㎡ 거래 {int(g['size'].sum()):,}건")
    top = g.sort_values("eok", ascending=False)
    print("  " + " · ".join(f"{i} {v:.1f}억" for i, v in top.eok.head(3).items())
          + "  …  "
          + " · ".join(f"{i} {v:.1f}억" for i, v in top.eok.tail(2).items()))


if __name__ == "__main__":
    main()
