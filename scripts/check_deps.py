"""스크립트가 실제로 import 하는 외부 패키지와 requirements.txt 를 맞춰 본다.

    python scripts/check_deps.py
    python scripts/check_deps.py --strict     # 어긋나면 1 로 끝낸다

**왜 필요한가.** 2026-10-15 클린 환경 재현(백로그 15)에서 `shapely` 와
`markdown` 이 `requirements.txt` 에 빠져 있던 것을 찾았다. 작업 환경에는 다른
경로로 깔려 있어 몇 주 동안 드러나지 않았고, **새 가상환경에서는 12 단계에서
멈춘다.** 받는 사람만 겪고 만드는 사람은 못 보는 종류의 고장이다.

`check_stale.py` 가 '아무도 안 만드는 산출물' 을 찾듯, 이것은 **'아무도 안 적은
의존' 을 찾는다.** 함수 안에서 늦게 import 하는 것(`lib/geo.py` 의 shapely)도
잡으려고 AST 로 전부 훑는다.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import io  # noqa: E402

io.setup_stdout()

# import 이름과 배포 이름이 다른 것
ALIAS = {"fitz": "PyMuPDF"}

# 직접 import 하지 않지만 있어야 하는 것 (왜 필요한지 한 줄로 남긴다)
INDIRECT = {
    "pyarrow": "pandas 가 parquet 을 읽고 쓸 때 쓴다. 직접 import 하지 않는다",
}


def local_modules(root: Path) -> set[str]:
    """저장소 안의 모듈 — 외부 의존이 아니다.

    손으로 적어 두면 저장소에 모듈이 하나 늘 때마다 오탐이 난다. 실제로
    `plot_pipeline.py` 가 단계 수를 세려고 `run_all` 을 import 하자마자
    'requirements.txt 에 없다'고 걸렸다. 그래서 파일을 찾아서 판단한다.
    """
    found = set()
    for d in (root, root / "scripts"):
        if not d.is_dir():
            continue
        found |= {p.stem for p in d.glob("*.py")}
        found |= {p.name for p in d.iterdir()
                  if p.is_dir() and (p / "__init__.py").exists()}
    return found


def imported(root: Path) -> dict[str, set[str]]:
    """스크립트마다 최상위 모듈 이름을 모은다 (함수 안의 import 도 포함)."""
    out: dict[str, set[str]] = {}
    local = local_modules(root.parent)
    for p in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError as e:
            print(f"  [건너뜀] {p.name} — 구문 오류 {e.lineno}행")
            continue
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                mods.add(n.module.split(".")[0])
        for m in mods:
            if m in sys.stdlib_module_names or m in local or m.startswith("_"):
                continue
            out.setdefault(ALIAS.get(m, m), set()).add(
                p.relative_to(root).as_posix())
    return out


def required(path: Path) -> set[str]:
    names = set()
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.split("#")[0].strip()
        if ln:
            names.add(re.split(r"[=<>~!\[]", ln)[0].strip())
    return names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true",
                    help="어긋나면 1 로 끝낸다 (기본은 알리기만)")
    a = ap.parse_args()

    used = imported(io.ROOT / "scripts")
    req = required(io.ROOT / "requirements.txt")
    low = {r.lower(): r for r in req}

    missing = {m: f for m, f in used.items() if m.lower() not in low}
    unused = {r for r in req if r.lower() not in {u.lower() for u in used}
              and r.lower() not in {k.lower() for k in INDIRECT}}

    print("=" * 78)
    print(f"의존 점검 — 스크립트가 쓰는 외부 패키지 {len(used)}개 · "
          f"requirements.txt {len(req)}줄")
    print("=" * 78)

    if missing:
        print("\n  [오류] requirements.txt 에 없는데 import 한다")
        for m, files in sorted(missing.items()):
            head = ", ".join(sorted(files)[:3])
            more = f" 외 {len(files) - 3}개" if len(files) > 3 else ""
            print(f"    {m:<16} <- {head}{more}")
    else:
        print("\n  빠진 것 없음. 쓰는 것이 전부 적혀 있다.")

    if unused:
        print("\n  [확인] 적혀 있는데 아무도 import 하지 않는다")
        for r in sorted(unused):
            print(f"    {r}")
        print("    -> 빼거나, 간접 의존이면 INDIRECT 에 이유와 함께 적는다.")

    if INDIRECT:
        print("\n  간접 의존 (직접 import 하지 않지만 필요)")
        for k, why in INDIRECT.items():
            mark = "O" if k.lower() in low else "**빠짐**"
            print(f"    {k:<16} {mark}  {why}")

    bad = len(missing) + len(unused) + sum(
        1 for k in INDIRECT if k.lower() not in low)
    print()
    if bad == 0:
        print("통과 — requirements.txt 가 실제 import 와 맞는다.")
    elif a.strict:
        sys.exit(1)


if __name__ == "__main__":
    main()
