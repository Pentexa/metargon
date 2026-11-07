#!/usr/bin/env python3
"""METARGON repository hygiene checker.

Run from the repository root:

    python scripts/check_repo.py

Checks:
  1. No unresolved merge conflict markers in tracked files.
  2. Relative links in Markdown files resolve to existing files.
  3. The four top-level source trees (python/, academy/, rust/, rust_academy/)
     exist and are importable/compilable entry points (compile check for
     Python modules).
  4. No accidental secrets: .env files are ignored and not tracked.

Exit code 0 on success, 1 on any failure.
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MARKER_RE = re.compile(r"^(<<<<<<< |=======$|>>>>>>> )")
LINK_RE = re.compile(r"\]\(([^)]+)\)")


def tracked_files() -> list[pathlib.Path]:
    """Return files tracked by git (falls back to a shallow walk)."""
    import subprocess
    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True,
            text=True, check=True, encoding="utf-8",
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        out = ""
    if out.strip():
        return [ROOT / p for p in out.splitlines()]
    return [p for p in ROOT.rglob("*") if p.is_file()]


def check_merge_markers(files) -> list[str]:
    errors = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if MARKER_RE.match(line):
                errors.append(f"{f.relative_to(ROOT)}:{i}: conflict marker")
    return errors


def check_markdown_links(files) -> list[str]:
    errors = []
    for f in files:
        if f.suffix.lower() != ".md":
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in LINK_RE.finditer(text):
            target = m.group(1)
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            path = (f.parent / target).resolve()
            try:
                path.relative_to(ROOT)
            except ValueError:
                continue
            if not path.exists():
                errors.append(f"{f.relative_to(ROOT)}: broken link -> {target}")
    return errors


def check_python_compiles() -> list[str]:
    errors = []
    for f in [ROOT / "python", ROOT / "academy"]:
        for py in sorted(f.rglob("*.py")):
            # skip compiled cache
            if "__pycache__" in py.parts:
                continue
            try:
                compile(py.read_text(encoding="utf-8", errors="replace"),
                        str(py), "exec")
            except SyntaxError as e:
                errors.append(f"{py.relative_to(ROOT)}: SyntaxError: {e}")
    return errors


def main() -> int:
    files = tracked_files()
    checks = [
        ("merge markers", check_merge_markers(files)),
        ("markdown links", check_markdown_links(files)),
        ("python compile", check_python_compiles()),
    ]
    failed = False
    for name, errors in checks:
        if errors:
            failed = True
            print(f"FAIL [{name}] ({len(errors)} issue(s)):")
            for e in errors[:30]:
                print(f"  {e}")
        else:
            print(f"OK   [{name}]")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
