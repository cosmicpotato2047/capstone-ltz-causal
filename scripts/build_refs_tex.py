"""[T2] 제출본 참고문헌을 학과 형식으로 만든다.

    python scripts/build_refs_tex.py

입력  paper/refs_kr.py           학과 안내서 형식으로 적어 둔 서지
출력  paper/generated/refs_v2.tex   thebibliography 블록

**왜 BibTeX 을 안 쓰나.** 학과 안내서가 요구하는 형식이 표준 .bst 어느
것과도 다르다.

    [7] 박재학, 이재봉, 최영환. (2010). "제목", 학술지명, 제25권, 제6호, pp. 1~8.

.bst 를 새로 짜는 것보다 서지 문자열을 한 번 적어 두고 그대로 찍는 쪽이
정확하다. 서지는 분석 결과가 아니라 고정값이므로 손으로 적어도 낡지 않는다.

번호는 refs_kr.py 에 적힌 순서로 매겨진다 — 안내서의 예시도 학위논문·단행본·
특허·학회논문·저널 순으로 묶여 있으므로 주제별 묶음이 오히려 가깝다.

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


def esc(s: str) -> str:
    for a, b in ESC.items():
        s = s.replace(a, b)
    # URL 은 줄바꿈이 되게 \url{} 로 감싼다
    return re.sub(r"(https?://\S+)", lambda m: r"\url{" + m.group(1) + "}", s)


def cited_keys() -> set[str]:
    """본문이 실제로 부르는 열쇠. v2 전용 절까지 함께 훑는다."""
    rx = re.compile(r"\\cite[pt]?\{([^}]*)\}")
    keys: set[str] = set()
    for p in sorted(SECTIONS.glob("*.tex")):
        for m in rx.finditer(p.read_text(encoding="utf-8")):
            keys |= {k.strip() for k in m.group(1).split(",")}
    return keys


def main() -> None:
    used = cited_keys()
    refs = refs_kr.REFS

    missing = sorted(used - set(refs))
    unused = [k for k in refs if k not in used]
    for k in missing:
        print(f"  [없음]    본문이 부르는데 refs_kr.py 에 없다 — {k}")
    for k in unused:
        print(f"  [안 쓰임] {k}")
    if missing:
        sys.exit("참고문헌이 비어 있는 채로 내보내지 않는다.")

    # 인용된 것만 싣는다. 번호는 refs_kr.py 의 순서를 따른다.
    keys = [k for k in refs if k in used]
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
