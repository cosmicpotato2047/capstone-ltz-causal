"""[14] 최종 보고서의 표를 결과 JSON 에서 생성한다.

    python scripts/build_tables.py          -> paper/tables/*.tex
    python scripts/build_tables.py --list   -> 표 목록과 \\input 줄만 출력

**왜 손으로 적지 않는가.** 결정기록 0007 이 "현재 값의 출처는
`output/results/*.json` 하나" 라고 정했고, 그 뒤 0004 에서 여섯 달 된 값이
문서에 남아 있던 것을 실제로 잡았다. 보고서는 표가 스무 개 넘게 들어가는
문서라 **같은 사고가 가장 크게 날 자리**다.

그래서 표는 전부 여기서 만든다. 본문에서는 `\\input{tables/…}` 로 불러 쓴다.
분석을 다시 돌리면 표도 따라 바뀐다 — 대시보드와 같은 원칙이다.

**적는 것은 숫자가 아니라 '어디서 꺼낼지' 와 '어떻게 보일지' 다.** 가리키는
키가 없으면 그 칸에 `--` 가 들어가고 콘솔에 알린다. 조용히 사라지지 않는다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import io, jpath  # noqa: E402

io.setup_stdout()

OUT = io.ROOT / "paper" / "tables"
MISS = "--"

# 서식: s 문자열 · i 정수(천단위) · f1/f2 소수 · p1 퍼센트(부호) · p0 퍼센트
#       pp %p(부호) · g 유효숫자 3
FMT = {
    "s":  lambda v: tex(str(v)),
    "i":  lambda v: f"{int(v):,}",
    "f1": lambda v: f"{float(v):.1f}",
    "f2": lambda v: f"{float(v):.2f}",
    "f3": lambda v: f"{float(v):.3f}",
    "g":  lambda v: f"{float(v):.3g}",
    "p1": lambda v: f"{float(v):+.1f}",
    "p0": lambda v: f"{float(v):.0f}",
    "pp": lambda v: f"{float(v):+.1f}",
    # p 값은 정확히 0 이 될 수 없다. 반올림해서 0 이 된 것을 0.000 으로 쓰면
    # 틀린 표기다. 관행대로 상한으로 적는다.
    "pv": lambda v: "$<0.001$" if float(v) < 0.001 else f"{float(v):.3f}",
}


def tex(s: str) -> str:
    """LaTeX 에서 뜻을 가지는 글자를 막는다."""
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


# ─── 표 정의 ────────────────────────────────────────────────────────────────
# file  : output/results/<file>.json
# path  : 그 안의 리스트 경로
# where : 그 리스트에서 남길 줄의 조건
# cols  : (JSON 열 이름, 표 머리, 서식)
# align : 열 정렬 (기본은 첫 열 l, 나머지 r)
TABLES = [
    dict(name="rq4_axes", file="05b_spec_curve", path="축별_영향.거래량",
         label="tab:rq4-axes", where=None,
         caption="설계 선택이 추정치를 가르는 폭 (거래량, 지정 직후 3개월). "
                 "사건 시점 정렬이 통제군 정의의 17배다.",
         note="축별로 그 축의 값들이 내는 중위 추정치의 최대 차이. "
              "달력 월 $-40.5\\%$ 대 효력일 $-80.1\\%$.",
         cols=[("축", "설계 선택", "s"), ("폭_pct포인트", "추정치 폭 (\\%p)", "f1")]),

    dict(name="pretrend_ref", file="06d_pretrend", path="A_기준분기",
         label="tab:pretrend-ref", where=None,
         caption="기준 분기를 옮겨 본 거래량 효과. $k=-2$ 만 크게 튄다.",
         note="$k=-2$ 는 12·16 대책 직후라 눌려 있는 분기다. 나머지는 "
              "$-73 \\sim -80\\%$ 로 좁게 모인다.",
         cols=[("기준", "기준 분기", "s"), ("k0", "$k{=}0$", "f1"),
               ("k1", "$k{=}1$", "f1"), ("k4", "$k{=}4$", "f1"),
               ("사전유의", "사전 유의 칸", "i")]),

    dict(name="inference", file="06_inference", path=None,
         label="tab:inference", where=None,
         caption="적은 클러스터 추론. 관행 $p$ 를 버리고 둘을 함께 적는다.",
         note="법정동 27개 중 처치가 4개다. WCR 은 Webb 6점 9,999벌, "
              "무작위화는 4개를 고르는 17,550가지 전수.",
         rowsfn="inference_rows",
         cols=[("사양", "사양", "s"), ("효과", "효과 (\\%)", "f1"),
               ("관행", "관행 $p$", "pv"), ("wcr", "WCR $p$", "pv"),
               ("ri", "무작위화 순위", "s")]),

    dict(name="threshold", file="06f_threshold", path="결과",
         label="tab:threshold", where={"표본": "기본"},
         caption="고가·저가 경계를 옮기면. 15억에 계단이 없다.",
         note="경계를 어디에 두든 고가 효과는 $-44 \\sim -54\\%$ 로 평평하고, "
              "저가 효과는 경계를 따라 매끈하게 움직인다.",
         cols=[("문턱_억", "경계 (억)", "f0"), ("저가효과_pct", "저가 (\\%)", "f1"),
               ("고가효과_pct", "고가 (\\%)", "f1"),
               ("차이_계수", "차이 (로그)", "f3"),
               ("WCR_p", "WCR $p$", "pv"), ("RI_p_등꼬리", "무작위화 $p$", "pv")]),

    dict(name="area_2022", file="06g_area_exemption", path="나_2022_기준강화",
         label="tab:area-2022", where={"표본": "기본"},
         caption="2022-06-23 허가 면적 기준 강화(대지 18 $\\to$ 6\\,m$^2$). "
                 "소형만 줄었다.",
         note="그날 처치동 소형만 새로 허가 대상이 된다. 대출 가설은 이 날짜에 "
              "아무것도 예측하지 않는다.",
         cols=[("기준_㎡", "소형 경계 (m$^2$)", "f0"),
               ("처치_소형_거래", "처치 소형 거래", "i"),
               ("소형효과_pct", "소형 (\\%)", "f1"),
               ("대형효과_pct", "대형 (\\%)", "f1"),
               ("WCR_p", "WCR $p$", "pv"), ("RI_p_등꼬리", "무작위화 $p$", "pv")]),

    dict(name="area_2021", file="06g_area_exemption", path="다_2021_위약",
         label="tab:area-2021", where={"표본": "기본"},
         caption="위약 --- 2021-06-23 재지정. 면적 기준을 그대로 두었다.",
         note="같은 계절, 같은 행정 절차인데 소형에 바뀐 것이 없다. 0 이어야 한다.",
         cols=[("기준_㎡", "소형 경계 (m$^2$)", "f0"),
               ("소형효과_pct", "소형 (\\%)", "f1"),
               ("대형효과_pct", "대형 (\\%)", "f1"),
               ("WCR_p", "WCR $p$", "pv"), ("RI_p_등꼬리", "무작위화 $p$", "pv")]),

    dict(name="size_ddd", file="09_release_2025", path="크기_삼중차분",
         label="tab:size-ddd", where=None,
         caption="면적 기준이 바뀌자 효과의 모양이 바뀌었다. "
                 "2022년에 적어 둔 예측이 2025년 자료에서 확인됐다.",
         note="2020년 지정은 대지 18\\,m$^2$ 초과만 허가 대상이라 소형이 면제됐고, "
              "2025년 재지정은 6\\,m$^2$ 라 면제가 거의 없다.",
         cols=[("사건", "사건", "s"), ("허가대상", "허가 대상", "s"),
               ("대형_pct", "대형 (\\%)", "f1"), ("소형_pct", "소형 (\\%)", "f1"),
               ("차이_계수", "차이 (로그)", "f3"), ("차이_p", "$p$", "pv")]),

    dict(name="spill_rings", file="12_spillover", path="층2_고리별",
         label="tab:spill-rings", where=None,
         caption="2020-06 지정의 거리 고리별 효과. 서울 나머지 19개 구 대비.",
         note="어느 고리도 거래량이 양($+$)이 아니다. 거래가 옆 동으로 옮겨 간 "
              "흔적이 없다.",
         cols=[("고리", "고리", "s"), ("거래량_pct", "거래량 (\\%)", "f1"),
               ("거래량_p", "$p$", "pv"), ("가격_pct", "가격 (\\%)", "f1"),
               ("가격_p", "$p$", "pv")]),

    dict(name="spill_bench", file="12_spillover", path="사다리_맞닿은동",
         label="tab:spill-bench", where=None,
         caption="같은 자료, 같은 창, 같은 사양. 기준점만 바꿨다 (맞닿은 동).",
         note="$+115.5\\%$ 는 옆 동이 늘어서가 아니라 처치동이 $-57.8\\%$ 로 "
              "줄어서다. 선행연구가 보고한 `인근 급등' 이 이것이다.",
         cols=[("단계", "기준점", "s"), ("거래량_pct", "거래량 (\\%)", "f1"),
               ("거래량_p", "$p$", "pv"), ("가격_pct", "가격 (\\%)", "f1"),
               ("가격_p", "$p$", "pv")]),

    dict(name="index_ladder", file="08_hedonic", path="지수_사다리",
         label="tab:index-ladder", where=None,
         caption="구성 통제의 사다리. 아래로 갈수록 더 많은 것을 고정한다.",
         note="약점이 정반대인 두 방법(헤도닉과 반복매매)이 $-5.9\\%$ 와 "
              "$-6.0\\%$ 에서 만난다. 집계 지수의 $-10 \\sim -12\\%$ 는 "
              "절반쯤이 구성 변화였다.",
         cols=[("지수", "가격지수", "s"), ("사전비", "지정 전 비", "f3"),
               ("사후비", "지정 후 비", "f3"), ("변화_pct", "변화 (\\%)", "f1")]),

    dict(name="placebo_date", file="13_placebo", path="A_가짜사건일",
         label="tab:placebo-date", where=None,
         caption="위약 --- 지정이 없던 날을 사건일로 두고 같은 추정을 돌렸다.",
         note="진짜($-33.5\\%$)가 가짜 최대($-21.5\\%$)의 1.5배다. 다만 2016·2017 "
              "창에는 8·2 와 9·13 대책이 들어 있어 완전한 위약이 아니다.",
         cols=[("사건일", "가짜 사건일", "s"), ("pct", "효과 (\\%)", "f1"),
               ("p", "$p$", "pv")]),

    dict(name="contamination", file="13b_contamination", path="결과",
         label="tab:contamination",
         where={"사양": "사건연구 k=0"},
         caption="통제군 오염 동을 빼도 지정 직후 효과는 그대로다.",
         note="가장 이른 오염이 사건월 9 다. 지정 직후 3개월에는 오염이 없다. "
              "제외 목록은 결과를 보기 전(2026-09-07)에 정했다.",
         cols=[("표본", "표본", "s"), ("법정동", "법정동", "i"),
               ("효과_pct", "효과 (\\%)", "f1"),
               ("WCR_p_양측", "WCR $p$", "pv"),
               ("RI_순위", "무작위화 순위", "i")]),

    dict(name="bgroup", file="21_b_group", path="대칭",
         label="tab:bgroup", where=None,
         caption="두 사건을 같은 자로. 같은 통제군 정의, 같은 창, 같은 추정량.",
         note="창 전체($\\pm$12개월) 이중차분이다. 늘 인용하는 $-79.8\\%$ 는 "
              "지정 직후 3개월이라 재는 것이 다르다. 두 사건의 차이를 쌓아서 "
              "직접 추정하면 $+8.7\\%$, 무작위화 $p=0.715$ 다.",
         cols=[("이름", "사양", "s"), ("처치동", "처치 동", "i"),
               ("효과_pct", "효과 (\\%)", "f1"),
               ("WCR_양측", "WCR $p$", "pv"), ("무작위_p", "무작위화 $p$", "pv")]),

    dict(name="ladder_b", file="21_b_group", path="사다리",
         label="tab:ladder-b", where=None,
         caption="2021.4 B그룹 --- 기준점을 바꿔도 $-41.7 \\sim -44.9\\%$ 다.",
         note="목동이 속한 양천구에 쓸 수 있는 비지정 동이 0개라 "
              "`같은 구 안' 통제군을 주 사양으로 쓸 수 없었다.",
         cols=[("이름", "기준점", "s"), ("동", "동", "i"),
               ("효과_pct", "효과 (\\%)", "f1"),
               ("WCR_양측", "WCR $p$", "pv"),
               ("무작위_p", "무작위화 $p$", "pv")]),

    dict(name="cancel", file="18_cancellations", path="이중차분.2025-03-24 재지정",
         label="tab:cancel", where=None,
         caption="취소거래가 규제 직후에 몰리는가 (2025-03-24 재지정).",
         note="창을 바꾸면 부호가 바뀐다. 처치군 특유의 집중이 없다. "
              "취소를 다시 넣어도 주 결과는 $0.3\\%$p 움직인다.",
         cols=[("창_개월", "창 (개월)", "i"), ("처치_전", "처치 전", "f2"),
               ("처치_후", "처치 후", "f2"), ("통제_전", "통제 전", "f2"),
               ("통제_후", "통제 후", "f2"), ("DID_pp", "DID (\\%p)", "pp")]),
]

FMT["f0"] = lambda v: f"{float(v):.0f}"


def inference_rows(d: dict) -> list[dict]:
    """06_inference 는 사양별로 두 블록(WCR·무작위화)에 흩어져 있어 합친다."""
    wb, ri = d.get("와일드부트스트랩", {}), d.get("무작위화추론", {})
    out = []
    for k in wb:
        w, r = wb[k], ri.get(k, {})
        out.append({
            "사양": k,
            "효과": w.get("효과_pct"),
            "관행": w.get("관행_p_t(G-1)"),
            "wcr": w.get("부트_p_양측"),
            "ri": (f"{r['순위_작은쪽부터']:,} / {r['경우의수']:,}"
                   if "순위_작은쪽부터" in r else None),
        })
    return out


def render(t: dict, data: dict) -> tuple[str, list[str]]:
    cols = t["cols"]
    if t.get("rowsfn"):
        rows = globals()[t["rowsfn"]](data)
    else:
        rows = jpath.rows(data, t["path"], t.get("where"))
    missing = []
    body = []
    for r in rows:
        cells = []
        for key, _hdr, f in cols:
            v = r.get(key)
            if v is None:
                cells.append(MISS)
                missing.append(f"{t['name']}: {key}")
            else:
                try:
                    cells.append(FMT[f](v))
                except (TypeError, ValueError):
                    cells.append(tex(str(v)))
                    missing.append(f"{t['name']}: {key} (서식 {f} 불가)")
        body.append("  " + " & ".join(cells) + r" \\")
    if not rows:
        missing.append(f"{t['name']}: 행이 하나도 없다 ({t['path']})")

    align = t.get("align") or ("l" + "r" * (len(cols) - 1))
    head = " & ".join(f"\\textbf{{{h}}}" for _k, h, _f in cols)
    note = (f"\n  \\begin{{minipage}}{{\\linewidth}}\\footnotesize\n"
            f"  \\textit{{주.}} {t['note']}\n  \\end{{minipage}}\n"
            if t.get("note") else "")
    tex_s = (
        "% 자동 생성 — scripts/build_tables.py. 손으로 고치지 않는다.\n"
        f"% 출처: output/results/{t['file']}.json"
        + (f" : {t['path']}" if t.get("path") else "") + "\n"
        "\\begin{table}[htbp]\n  \\centering\n"
        f"  \\caption{{{t['caption']}}}\n"
        f"  \\label{{{t['label']}}}\n"
        f"  \\begin{{tabular}}{{{align}}}\n"
        "    \\toprule\n"
        f"    {head} \\\\\n"
        "    \\midrule\n"
        + "\n".join("  " + b for b in body) + "\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        + note +
        "\\end{table}\n")
    return tex_s, missing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="목록과 \\input 줄만")
    a = ap.parse_args()

    if a.list:
        for t in TABLES:
            print(f"  \\input{{tables/{t['name']}}}   % {t['label']} — {t['caption'][:40]}")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict] = {}
    allmiss, written = [], 0
    for t in TABLES:
        if t["file"] not in cache:
            p = io.RESULTS / f"{t['file']}.json"
            cache[t["file"]] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        s, miss = render(t, cache[t["file"]])
        (OUT / f"{t['name']}.tex").write_text(s, encoding="utf-8")
        written += 1
        allmiss += miss

    print(f"표 {written}개 생성 -> {OUT.relative_to(io.ROOT)}")
    if allmiss:
        print(f"\n  빈 칸 {len(allmiss)}개 — 경로를 확인하라")
        for m in dict.fromkeys(allmiss):
            print(f"    [없음] {m}")
    else:
        print("  빈 칸 없음.")


if __name__ == "__main__":
    main()
