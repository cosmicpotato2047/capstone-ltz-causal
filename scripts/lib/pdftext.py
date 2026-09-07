"""서울시보 PDF의 깨진 글꼴을 복원해 본문 텍스트를 얻는다.

시보 지면의 표는 자소가 아니라 **글리프 번호**로 저장된 서브셋 글꼴을 쓴다.
PyMuPDF 가 뽑아낸 문자는 실제 글자에서 일정한 값만큼 밀려 있고,
그 폭은 글꼴마다 다르다. 공고 2022-827호의 표를 예로 들면

    '⹚ⲯ⹚⪇'  ->  '지정지역'   (한글 오프셋 3242)
    '5355ᗞ'   ->  '2022년'     (ASCII  오프셋 3)

밀린 폭이 글꼴 안에서는 일정하므로, 글꼴마다 오프셋 하나씩만 찾으면 된다.
찾는 방법은 사전 대조다 — 후보 오프셋마다 되돌려보고 공고에 반드시 나오는
낱말(지정·구역·번지 …)이 가장 많이 살아나는 값을 고른다.

밀린 방향은 글꼴마다 다르다. 어떤 글꼴은 '#'->' ' 처럼 뒤로(+3), 어떤 글꼴은
'0'->'1' 처럼 앞으로(-1) 밀려 있어서 양쪽을 다 본다.

한글과 숫자가 서로 다른 글꼴에 담긴 공고도 있다. 그래서 한글 오프셋과
ASCII 오프셋을 따로 찾는다. 숫자만 든 글꼴은 낱말 대조가 통하지 않으므로
'글자 구성이 그럴듯한 정도'(쉼표가 '+' 로 깨지지 않았는가)를 함께 본다.

    from lib.pdftext import text_of
    t = text_of(path)          # 정상 지면은 그대로, 깨진 지면은 복원해서

한계: 글리프 번호가 일정하게 밀린 것이 아니라 뒤섞인 글꼴(주로 제목 글꼴)은
복원되지 않는다. 본문은 살아나므로 지정기간·근거공고·구역표는 얻을 수 있다.
그림으로만 된 지면(텍스트 층이 없는 PDF)도 당연히 복원 대상이 아니다.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import fitz

HAN0, HAN1 = 0xAC00, 0xD7A3
# 깨진 글리프 번호가 떨어지는 구간. 되돌린 결과가 한글일 때만 바꾸므로
# 넓게 잡아도 佭·侍 같은 진짜 한자는 건드리지 않는다.
LO, HI = 0x0100, 0xA000
# 공고에 반드시 나오는 낱말. 오프셋이 맞으면 이들이 한꺼번에 살아난다.
ANCHORS = ("지정", "구역", "번지", "일대", "면적", "동", "구", "토지", "거래",
           "허가", "서울", "지역", "기간", "위치", "비고", "시도", "자치")
# 오프셋 후보를 만들 때 기준으로 삼는 글자 — 지 정 구 동 역 허 가 토 서
TARGETS = (0xC9C0, 0xC815, 0xAD6C, 0xB3D9, 0xC5ED, 0xD5C8, 0xAC00, 0xD1A0, 0xC11C)

# 숫자가 제자리에 있으면 살아나는 표현들 — 연도, 지번, 세자리 끊은 면적
NUM = (re.compile(r"20[12]\d\s*[년\-.]"), re.compile(r"\d+(?:-\d+)?\s*번지"),
       re.compile(r"\d{1,3},\d{3}"), re.compile(r"\d+\s*㎡"))
# 제자리에 있는 표에 나올 법한 글자. 한 칸 밀리면 쉼표가 '+' 로 바뀐다.
PLAUSIBLE = set("0123456789,.-()~%")

# 시보 PDF 에는 제어문자와 짝 없는 서로게이트가 섞여 있다. 파일로 쓸 때
# 터지므로 여기서 한 번 걷어낸다.
CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def _clean(t: str) -> str:
    t = CTRL.sub("", t).encode("utf-8", "ignore").decode("utf-8")
    return re.sub(r"[ \t]+", " ", t)


def _shift(s: str, han: int, asc: int) -> str:
    out = []
    for ch in s:
        c = ord(ch)
        if han and LO <= c <= HI:
            v = c - han + HAN0
            out.append(chr(v) if HAN0 <= v <= HAN1 else ch)
        elif asc and 0x21 <= c <= 0x7E:
            v = c - asc
            out.append(chr(v) if 0x20 <= v <= 0x7E else ch)
        else:
            out.append(ch)
    return "".join(out)


def _score(s: str) -> int:
    return sum(s.count(a) for a in ANCHORS)


def _best_offset(sample: str) -> int:
    """되돌렸을 때 닻 낱말이 가장 많이 살아나는 한글 오프셋. 0이면 멀쩡한 글꼴."""
    if _score(sample) >= 8:
        return 0
    cands: dict[int, int] = defaultdict(int)
    for ch in sample:
        c = ord(ch)
        if LO <= c <= HI:                  # 이 글자가 한글이려면 필요한 오프셋
            for tgt in TARGETS:
                cands[c - (tgt - HAN0)] += 1     # 음수도 후보다
    best, best_s = 0, _score(sample)
    for d, _ in sorted(cands.items(), key=lambda kv: -kv[1])[:600]:
        if d == 0:
            continue
        s = _score(_shift(sample, d, 0))
        if s > best_s + 2 and s >= 4:      # 우연한 한두 개로는 안 바꾼다
            best, best_s = d, s
    return best


def _ascii_score(t: str) -> float:
    """숫자가 제자리인 정도. 낱말 일치와 글자 구성을 함께 본다.

    숫자만 든 글꼴은 낱말 일치가 거의 없어 그것만으로는 방향을 못 고른다.
    반면 한 칸 밀리면 쉼표가 '+' 로, 숫자가 기호로 바뀌어 구성이 무너진다."""
    vis = [c for c in t if 0x21 <= ord(c) <= 0x7E]
    ratio = sum(c in PLAUSIBLE for c in vis) / len(vis) if vis else 0.0
    return 3 * sum(len(r.findall(t)) for r in NUM) + 12 * ratio


def _ascii_offset(sample: str) -> int:
    """한글을 먼저 되돌린 표본을 넣어야 한다. '번지'가 살아야 지번이 보인다."""
    if len(set(re.findall(r"20[12]\d", sample))) >= 2:    # 연도가 이미 살아있다
        return 0
    best, best_s = 0, _ascii_score(sample) + 0.5          # 제자리에 이점을 준다
    for d in (1, -1, 2, -2, 3, -3, 4, -4, 5, -5):
        sc = _ascii_score(_shift(sample, 0, d))
        if sc > best_s:
            best, best_s = d, sc
    return best


def _offsets(sample: str) -> tuple[int, int]:
    """(한글 오프셋, ASCII 오프셋). 한글을 먼저 풀어야 지번이 보인다."""
    han = _best_offset(sample)
    return han, _ascii_offset(_shift(sample, han, 0))


def _spans(page) -> list[tuple[str, str]]:
    """(글꼴이름, 문자열) 목록. 줄바꿈은 ('\n', '\n') 으로 끼워 넣는다."""
    out = []
    for blk in page.get_text("dict")["blocks"]:
        for line in blk.get("lines", []):
            for sp in line["spans"]:
                out.append((sp["font"], sp["text"]))
            out.append(("\n", "\n"))
    return out


def _by_font(path: str | Path) -> tuple[list[tuple[str, str]], dict[str, str]]:
    spans = [s for pg in fitz.open(path) for s in _spans(pg)]
    bucket: dict[str, list[str]] = defaultdict(list)
    for font, txt in spans:
        if font != "\n":
            bucket[font].append(txt)
    return spans, {f: " ".join(c)[:6000] for f, c in bucket.items()}


def text_of(path: str | Path) -> str:
    """글꼴별로 오프셋을 찾아 되돌린 전체 본문."""
    spans, samples = _by_font(path)
    fix = {f: _offsets(s) for f, s in samples.items()}
    out = []
    for font, txt in spans:
        if font == "\n":
            out.append("\n")
            continue
        han, asc = fix[font]
        out.append(_shift(txt, han, asc) if (han or asc) else txt)
    return _clean("".join(out))


def report(path: str | Path) -> dict[str, tuple[int, int]]:
    """어떤 글꼴에 어떤 오프셋을 적용했는지 (점검용)."""
    return {f: _offsets(s) for f, s in _by_font(path)[1].items()}
