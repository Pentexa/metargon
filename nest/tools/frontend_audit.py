"""NEST frontend validation (CI-friendly).

1. Extracts the inline <script> from nest/static/index.html.
2. Runs `node --check` on it when Node is available (catches syntax
   errors like the one that broke the original UI).
3. Falls back to a brace/backtick balance check when Node is absent.
4. Optionally runs the Node DOM smoke test (nest/tools/frontend_smoke.node.js)
   to exercise initial render, conversation send, memory panel, and
   control-wiring code paths.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
HTML = ROOT / "nest" / "static" / "index.html"
SMOKE = Path(__file__).resolve().parent / "frontend_smoke.node.js"


def extract_scripts(text: str) -> list[str]:
    return re.findall(r"<script>(.*?)</script>", text, re.S)


def balance_check(js: str) -> list[str]:
    problems = []
    for ch, close in (("`", "`"), ("{", "}"), ("(", ")"), ("[", "]")):
        if js.count(ch) != js.count(close):
            problems.append(f"unbalanced {ch!r}/{close!r}: "
                            f"{js.count(ch)} vs {js.count(close)}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="also run the Node DOM smoke test")
    args = ap.parse_args()

    text = HTML.read_text(encoding="utf-8")
    scripts = extract_scripts(text)
    if not scripts:
        print("FAIL: no inline <script> found")
        return 1
    js = "\n\n".join(scripts)
    problems = balance_check(js)

    node_ok = False
    try:
        r = subprocess.run(["node", "--version"], capture_output=True)
        node_ok = r.returncode == 0
    except FileNotFoundError:
        node_ok = False

    if node_ok:
        r = subprocess.run(["node", "--check"], input=js.encode(),
                           capture_output=True)
        if r.returncode != 0:
            print("FAIL: node --check errored:")
            print(r.stderr.decode()[:2000])
            return 1
        print("node --check: OK")
    else:
        print("node unavailable; balance checks only")

    if problems:
        print("FAIL:", "\n".join(problems))
        return 1
    print(f"balance checks: OK (inline scripts: {len(scripts)})")

    if args.smoke and node_ok:
        r = subprocess.run(["node", str(SMOKE)], capture_output=True)
        print(r.stdout.decode())
        if r.returncode != 0:
            print("FAIL: DOM smoke test errored:")
            print(r.stderr.decode()[:2000])
            return 1
        print("DOM smoke test: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
