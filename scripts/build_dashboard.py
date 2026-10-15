"""[20] 결과 대시보드 — 결과 ↔ 근거 ↔ 재현 명령을 한 장에 잇는다.

    python scripts/build_dashboard.py      -> output/dashboard.html

**왜 필요한가.** 결과가 결정기록 31편·주차 보고서 6편·JSON 25개에 흩어져 있다.
"이 숫자 어디서 나왔나"를 묻는 사람(심사위원, 그리고 석 달 뒤의 나)이 매번
저장소를 뒤져야 한다.

이 페이지는 **숫자마다 세 가지를 붙인다** — 어느 결정기록인지, 어느 스크립트가
만들었는지, 어떻게 다시 돌리는지. 숫자의 **계보**를 보이는 것이 목적이다.

**숫자를 손으로 적지 않는다.** `output/results/*.json` 에서 읽어 쓴다. 적는 것은
**어디서 꺼낼지(경로)와 어떻게 읽을지(설명)** 뿐이다. 그래서 분석을 다시 돌리면
이 페이지도 따라 바뀐다 — [[0007-문서유형별-수치기록]] 이 "현재 값의 출처는
하나"라고 한 원칙을 화면에도 적용한 것이다.

없는 키를 가리키면 **그 자리에 '없음' 이 뜨고 끝까지 돈다.** 숫자가 조용히
사라지는 것보다 눈에 띄는 쪽이 낫다.
"""
from __future__ import annotations

import html
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import io  # noqa: E402

io.setup_stdout()

MISSING = "—"


# (제목, JSON 파일, 키 경로, 서식, 읽는 법, 결정기록, 스크립트)
# 키 경로는 점으로 잇는다. 리스트는 [0] 처럼 쓴다.
ROWS = [
    ("얼마나 줄었나", [
        ("지정 직후 3개월", "04_event_study", "거래량[k=0].effect_pct", "pct",
         "2020-06-23 지정. 처치 4개 동 대 강남·송파 통제동, 기준 분기 k=−1",
         "0013·0014", "04_event_study.py"),
        ("무작위화 순위", "06_inference", "무작위화추론.사건연구 k=0.순위_작은쪽부터", "rank17550",
         "27개 동 중 4개를 고르는 모든 경우를 돌려 진짜 배정이 몇 등인지 센다",
         "0019", "06_inference.py"),
        ("합성대조군 — 직후 3개월", "17_scm", "직후3개월_pct", "pct",
         "서울 법정동 151개를 가중평균해 '가짜 대치동' 을 만든 뒤 비교",
         "0017", "17_scm.py"),
        ("사양 곡선 — 음수 비율", "05b_spec_curve", "거래량.음수_비율", "pct0",
         "처치군·통제군·창·날짜·빈칸·척도를 전부 조합한 192가지",
         "0028", "05b_spec_curve.py"),
    ]),
    ("언제까지 갔나", [
        ("24개월 평균", "13_placebo", "실제.pct", "pct",
         "사후 24개월을 하나로 묶은 이중차분. 직후 효과와 다른 값이다",
         "0016", "13_placebo.py"),
        ("붕괴값 — 24개월", "11_honest_did", "결과[대상=거래량 — 24개월 평균].붕괴_Mbar_구간", "num",
         "사전 추세가 사후에도 이어졌다고 가정할 때 결론이 무너지는 지점",
         "0026", "11_honest_did.py"),
    ]),
    ("누구에게 달랐나", [
        ("2022 면적 기준 강화 — 소형", "06g_area_exemption", "나_2022_기준강화[기준_㎡=40.0].소형효과_pct", "pct",
         "허가 대상이 대지 18㎡ → 6㎡ 로 좁아진 날. 소형만 새로 규제에 걸린다",
         "0022", "06g_area_exemption.py"),
        ("2021 위약 — 소형", "06g_area_exemption", "다_2021_위약[기준_㎡=40.0].소형효과_pct", "pct",
         "같은 절차였지만 면적 기준을 그대로 뒀다. 0 이어야 한다",
         "0022", "06g_area_exemption.py"),
        ("2025 재지정 — 소형", "09_release_2025", "크기_삼중차분[사건=2025-03-24 재지정].소형_pct", "pct",
         "기준이 6㎡ 라 소형 면제가 없다. 3주차에 미리 적어 둔 예측이 맞았다",
         "0023", "09_release_2025.py"),
    ]),
    ("어디로 번졌나", [
        ("같은 동 안 — 규제 유지 단지", "12_spillover", "층1[집단=유지군].거래량_pct", "pct",
         "2025-02 해제 때 규제가 그대로 남은 16개 단지. 먼 통제군 대비",
         "0024", "12_spillover.py"),
        ("맞닿은 동 — 처치동 기준", "12_spillover", "사다리_맞닿은동[기준점=R0 처치동].거래량_pct", "pct",
         "선행연구가 말한 '인근 급등'. 옆 동이 늘어서가 아니라 처치동이 줄어서다",
         "0024", "12_spillover.py"),
        ("맞닿은 동 — 서울 19개 구 기준", "12_spillover", "사다리_맞닿은동[2].거래량_pct", "pct",
         "같은 동, 같은 기간. 기준점만 한 칸 옮기면 부호가 뒤집힌다",
         "0024", "12_spillover.py"),
    ]),
    ("가격은 잡혔나", [
        ("반복매매지수 12개월", "08_hedonic", "지수_사다리[지수=반복매매].변화_pct", "pct",
         "같은 집이 두 번 팔린 쌍만 쓴다. 구성 변화가 원천적으로 안 들어온다",
         "0018·0025", "08_hedonic.py"),
        ("헤도닉 + 단지 12개월", "08_hedonic", "지수_사다리[지수=헤도닉+단지].변화_pct", "pct",
         "모든 거래를 쓰되 면적·층·단지를 통제한다. 약점이 반복매매와 정반대다",
         "0025", "08_hedonic.py"),
        ("붕괴값 — 가격", "11_honest_did", "결과[2].붕괴_Mbar_구간", "num",
         "거래량 직후는 1.62 다. 가격은 그 7분의 1 에서 무너진다",
         "0026", "11_honest_did.py"),
        ("가격 합성대조군 p", "08_hedonic", "가격_SCM.경험적p", "num",
         "가짜 대치동 중 진짜가 몇 등인지. 거래량은 1위, 가격은 중간",
         "0025", "08_hedonic.py"),
    ]),
    ("믿어도 되는가", [
        ("B그룹 − A그룹 차이", "21_b_group", "stacked.효과_pct", "pct",
         "2021.4 압구정·여의도·목동·성수. 두 패널을 쌓아 차이를 직접 추정",
         "0031", "21_b_group.py"),
        ("그 차이의 무작위화 p", "21_b_group", "stacked.무작위_순위", "rank5000",
         "0 과 구분되지 않는다 = B 가 A 와 다르다고 할 수 없다",
         "0031", "21_b_group.py"),
        ("사양 곡선 — 가격 음수 비율", "05b_spec_curve", "가격.음수_비율", "pct0",
         "거래량은 192개 전부 0 아래인데 가격은 다섯 중 하나가 0 위로 간다",
         "0028", "05b_spec_curve.py"),
        ("취소거래를 넣으면", "18_cancellations", "사건연구.k0_차이pp", "pp",
         "자료를 만들 때 뺀 6,293건을 다시 넣어도 주 결과가 이만큼만 움직인다",
         "0030", "18_cancellations.py"),
    ]),
    ("자료", [
        ("분석 표본 (8개 자치구)", "02_build", "표본", "int",
         "2006-01 ~ 2026-05. 취소거래와 신고지연 구간을 뺀 뒤",
         "0004", "02_build.py"),
        ("서울 25개 자치구", "02_build_seoul", "표본", "int",
         "합성대조군 기증자 풀과 2021 B그룹 분석에 쓴다",
         "0017·0031", "02_build.py --scope seoul"),
        ("단지 수", "02_build", "단지", "int",
         "aptSeq 기준. 단지 고정효과의 단위다",
         "0003", "02_build.py"),
        ("반복매매 쌍", "02_build", "반복매매쌍", "int",
         "(단지, 전용면적, 층) 조합의 재거래. 보유 180일 미만 제외",
         "0018", "02_build.py"),
    ]),
]


TOKEN = re.compile(r"([^.\[\]]+)|\[([^\]]*)\]")


def dig(obj, path: str):
    """경로를 따라간다. 없으면 None.

        거래량[k=0].effect_pct          리스트에서 k 가 0 인 **첫** 칸
        나_2022_기준강화[기준_㎡=40.0]    값에 점이 있어도 된다
        결과[1].붕괴_Mbar_구간           1번 칸

    **라벨로 찾는 쪽을 쓴다.** 번호는 스크립트가 줄 순서를 바꾸면 조용히 다른
    값을 가리키지만, 라벨은 그때 '없음' 으로 드러난다.

    점으로 쪼개면 `[기준_㎡=40.0]` 의 소수점까지 갈라진다. 그래서 대괄호를
    하나의 토큰으로 읽는다.
    """
    cur = obj
    for name, sel in TOKEN.findall(path):
        if name:
            if not isinstance(cur, dict) or name not in cur:
                return None
            cur = cur[name]
            continue
        if not isinstance(cur, list):
            return None
        if "=" in sel:
            key, _, want = sel.partition("=")
            cur = next((r for r in cur if isinstance(r, dict)
                        and str(r.get(key)) == want), None)
            if cur is None:
                return None
        else:
            i = int(sel)
            if not (-len(cur) <= i < len(cur)):
                return None
            cur = cur[i]
    return cur


def fmt(v, kind: str) -> tuple[str, str]:
    """값 -> (보일 문자열, 색 클래스)."""
    if v is None:
        return MISSING, "na"
    try:
        if kind == "pct":
            return f"{float(v):+.1f}%", "down" if float(v) < 0 else "up"
        if kind == "pct0":
            return f"{float(v):.0f}%", "flat"
        if kind == "pp":
            return f"{float(v):+.1f}%p", "flat"
        if kind == "num":
            return f"{float(v):.3g}", "flat"
        if kind == "int":
            return f"{int(v):,}", "flat"
        if kind.startswith("rank"):
            n = int(kind[4:])
            return f"{int(v):,} / {n:,}", "down"
    except (TypeError, ValueError):
        return str(v)[:24], "na"
    return str(v), "flat"


def head() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=io.ROOT,
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        sha = ""
    return sha


def main() -> None:
    cache: dict[str, dict] = {}
    rows_html, missing = [], []
    for section, items in ROWS:
        rows_html.append(f'<tr class="sec"><td colspan="4">{html.escape(section)}</td></tr>')
        for title, fname, path, kind, how, dec, script in items:
            if fname not in cache:
                p = io.RESULTS / f"{fname}.json"
                cache[fname] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
            v = dig(cache[fname], path)
            if v is None:
                missing.append(f"{title}  <-  {fname}.json : {path}")
            txt, cls = fmt(v, kind)
            decs = " · ".join(
                f'<a href="../docs/decisions/" title="결정기록 {d}">{d}</a>'
                for d in dec.split("·"))
            rows_html.append(
                f'<tr><td class="t">{html.escape(title)}</td>'
                f'<td class="v {cls}">{txt}</td>'
                f'<td class="h">{html.escape(how)}</td>'
                f'<td class="s">{decs}<br><code>python scripts/{html.escape(script)}</code></td></tr>')

    n_total = sum(len(i) for _, i in ROWS)
    doc = TEMPLATE.format(
        rows="\n".join(rows_html), sha=head(),
        n=n_total, nmiss=len(missing),
        warn=("" if not missing else
              '<p class="warn">가리키는 키가 없는 칸 {}개. 아래 콘솔 출력을 보라.</p>'
              .format(len(missing))))
    out = io.ROOT / "output" / "dashboard.html"
    out.write_text(doc, encoding="utf-8")

    print(f"지표 {n_total}개 · 없는 키 {len(missing)}개")
    for m in missing:
        print(f"  [없음] {m}")
    print(f"저장: {out}")


TEMPLATE = """<!DOCTYPE html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>결과 대시보드 · 서울 토허구역 인과추정</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Nanum+Myeongjo:wght@700;800&family=IBM+Plex+Sans+KR:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{color-scheme:light;--ground:#f4f6f7;--surface:#fcfcfb;--surface-2:#eceef0;
 --ink:#12161a;--ink-2:#454b52;--ink-3:#7d848c;--rule:#c9ced4;--rule-soft:#e2e6e9;
 --accent:#2a78d6;--warn:#e34948}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{color-scheme:dark;
 --ground:#141719;--surface:#1a1d20;--surface-2:#232729;--ink:#f2f4f5;--ink-2:#b9c0c7;
 --ink-3:#8b939b;--rule:#343a3f;--rule-soft:#282d31;--accent:#3987e5;--warn:#e66767}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--ground);color:var(--ink);
 font-family:"IBM Plex Sans KR","Malgun Gothic",system-ui,sans-serif;font-size:15px;line-height:1.6}}
.wrap{{max-width:1080px;margin:0 auto;padding:48px 24px 90px}}
h1{{font-family:"Nanum Myeongjo","Batang",serif;font-size:27px;font-weight:800;margin:0 0 8px}}
.sub{{color:var(--ink-2);margin:0 0 4px;font-size:15px}}
.meta{{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--ink-3);margin:0 0 30px}}
.warn{{border-left:3px solid var(--warn);background:var(--surface);padding:10px 14px;
 font-size:14px;margin:0 0 20px}}
table{{width:100%;border-collapse:collapse;background:var(--surface);
 border:1px solid var(--rule-soft)}}
td{{padding:11px 14px;border-bottom:1px solid var(--rule-soft);vertical-align:top}}
tr.sec td{{background:var(--surface-2);font-family:"Nanum Myeongjo","Batang",serif;
 font-weight:700;font-size:15.5px;padding:10px 14px;border-top:1px solid var(--rule)}}
td.t{{width:24%;font-weight:500}}
td.v{{width:15%;font-family:"IBM Plex Mono",monospace;font-size:17px;font-weight:500;
 white-space:nowrap}}
td.v.down{{color:var(--accent)}} td.v.up{{color:var(--warn)}}
td.v.na{{color:var(--ink-3)}} td.v.flat{{color:var(--ink)}}
td.h{{width:38%;color:var(--ink-2);font-size:13.5px;line-height:1.55}}
td.s{{width:23%;font-size:12px;color:var(--ink-3)}}
td.s a{{color:var(--ink-3)}}
td.s code{{font-family:"IBM Plex Mono",monospace;font-size:11.5px;color:var(--ink-3);
 display:inline-block;margin-top:3px;word-break:break-all}}
footer{{margin-top:34px;padding-top:18px;border-top:1px solid var(--rule-soft);
 color:var(--ink-3);font-size:13px}}
footer code{{font-family:"IBM Plex Mono",monospace;font-size:12px}}
a{{color:var(--accent)}}
</style>
<div class="wrap">
<h1>결과 대시보드</h1>
<p class="sub">숫자마다 <strong>어느 결정기록에서 나왔는지</strong>와
<strong>어떻게 다시 돌리는지</strong>를 붙인다.</p>
<p class="meta">지표 {n}개 · 스냅샷 2026-09-03 · 커밋 {sha} ·
 이 표는 <code>output/results/*.json</code> 에서 생성된다 (손으로 적지 않는다)</p>
{warn}
<table>
{rows}
</table>
<footer>
전체 파이프라인 <code>python run_all.py</code> (42단계) ·
이 페이지 <code>python scripts/build_dashboard.py</code><br>
숫자가 문서와 어긋나면 <code>output/results/*.json</code> 쪽이 맞다 (결정기록 0007).
</footer>
</div>
"""


if __name__ == "__main__":
    main()
