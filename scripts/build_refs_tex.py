r"""[T2] 제출본 참고문헌을 학과 형식으로 만든다.

    python scripts/build_refs_tex.py

입력  paper/refs_kr.py           학과 안내서 형식으로 적어 둔 서지
출력  paper/generated/refs_v2.tex   thebibliography 블록

**왜 BibTeX 을 안 쓰나.** 학과 안내서가 요구하는 형식이 표준 .bst 어느
것과도 다르다.

    [7] 박재학, 이재봉, 최영환. (2010). "제목", 학술지명, 제25권, 제6호, pp. 1~8.

.bst 를 새로 짜는 것보다 서지 문자열을 한 번 적어 두고 그대로 찍는 쪽이
정확하다. 서지는 분석 결과가 아니라 고정값이므로 손으로 적어도 낡지 않는다.

번호는 **본문에 처음 나온 순서**로 매긴다. 주제별로 묶어 매겼더니 서론의 첫
인용이 [18] 로 나와 오식처럼 보였다. 읽는 쪽에서 [1] [2] [3] 은 '나온 차례'
라고 읽히므로 그쪽에 맞춘다(컴퓨터공학 분야의 관행이기도 하다).

나온 차례를 알려면 **문서의 \input 순서**를 알아야 한다. 파일 이름 순으로 훑으면
v2_*.tex 가 뒤로 밀려 차례가 어긋나므로, main_ko_v2.tex 에서 읽어 온다.

**main_ko.tex(연구용)은 그대로 BibTeX 을 쓴다.** 이 파일은 main_ko_v2.tex
(제출본)에서만 읽는다.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import io  # noqa: E402

io.setup_stdout()

sys.path.insert(0, str(io.ROOT / "paper"))
import refs_kr  # noqa: E402

OUT = io.ROOT / "paper" / "generated" / "refs_v2.tex"
SECTIONS = io.ROOT / "paper" / "sections"

# LaTeX 에서 뜻을 가지는 글자. 서지 문자열은 평문이라 막아 준다.
ESC = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "$": r"\$"}
URL = re.compile(r"https?://\S+")


def plain(s: str) -> str:
    """URL 바깥의 평문을 LaTeX 으로 옮긴다."""
    for a, b in ESC.items():
        s = s.replace(a, b)
    # 쪽 범위의 '~'. LaTeX 에서 ~ 는 **줄바꿈 금지 공백**이라 그냥 두면
    # 'pp. 169 189' 로 찍힌다. 붙임표 둘은 en dash 가 되며 APA 의 쪽 범위
    # 표기이기도 하다 (안내서가 'APA Style 을 기본으로' 라고 했다).
    return s.replace("~", "--")


def esc(s: str) -> str:
    """URL 은 \\url{} 로 감싸고 **그 안은 건드리지 않는다.**

    먼저 밑줄을 막고 나중에 감싸면 주소 안에 `\\_` 가 들어가 **링크가 깨진다**.
    안내서가 'Not Found 는 감점' 이라고 한 그 자리다.
    """
    out, last = [], 0
    for m in URL.finditer(s):
        out.append(plain(s[last:m.start()]))
        out.append("\\url{" + m.group(0) + "}")
        last = m.end()
    out.append(plain(s[last:]))
    return "".join(out)


MAIN = io.ROOT / "paper" / "main_ko_v2.tex"


def cited_keys() -> list[str]:
    """본문이 부르는 열쇠를 **처음 나온 차례대로** 모은다."""
    rx = re.compile(r"\\cite[pt]?\{([^}]*)\}")
    order = re.findall(r"\\input\{sections/([A-Za-z0-9_가-힣]+)\}",
                       MAIN.read_text(encoding="utf-8"))
    seen: list[str] = []
    for stem in order:
        p = SECTIONS / f"{stem}.tex"
        if not p.exists():
            print(f"  [없음] main_ko_v2.tex 가 읽는 파일이 없다 — {stem}.tex")
            continue
        for m in rx.finditer(p.read_text(encoding="utf-8")):
            for k in m.group(1).split(","):
                k = k.strip()
                if k and k not in seen:
                    seen.append(k)
    # \input 에 걸리지 않은 절이 있으면 알린다 (main_ko 전용 파일 등)
    linked = {SECTIONS / f"{s}.tex" for s in order}
    for p in sorted(SECTIONS.glob("*.tex")):
        if p not in linked and rx.search(p.read_text(encoding="utf-8")):
            print(f"  [참고] main_ko_v2 가 읽지 않는 절에 인용이 있다 — {p.name}")
    return seen


def main() -> None:
    used = cited_keys()
    refs = refs_kr.REFS

    missing = [k for k in used if k not in refs]
    unused = [k for k in refs if k not in used]
    for k in missing:
        print(f"  [없음]    본문이 부르는데 refs_kr.py 에 없다 — {k}")
    for k in unused:
        print(f"  [안 쓰임] {k}")
    if missing:
        sys.exit("참고문헌이 비어 있는 채로 내보내지 않는다.")

    # 인용된 것만, 처음 나온 차례대로 싣는다.
    keys = used
    width = str(len(keys))
    lines = [
        "% 자동 생성 — scripts/build_refs_tex.py. 손으로 고치지 않는다.",
        "% 출처: paper/refs_kr.py (학과 안내서 형식)",
        r"\begin{thebibliography}{" + width + "}",
        r"\setlength{\itemsep}{6pt}",
    ]
    for k in keys:
        lines.append(r"\bibitem{" + k + "}")
        lines.append("  " + esc(refs[k]))
    lines.append(r"\end{thebibliography}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  {OUT.relative_to(io.ROOT)}  —  {len(keys)}편 "
          f"(본문 인용 {len(used)}개)")


if __name__ == "__main__":
    main()
