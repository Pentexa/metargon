#!/usr/bin/env python3
"""Repository hygiene regression tests.

These guard the v0.9/quality audit fixes: unresolved merge conflict
markers previously broke the python/ test suite at import time
(benchmark_v09.py and test_benchmark_v09.py contained `<<<<<<< HEAD`
blocks that are invalid Python).
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
MARKER_RE = re.compile(r"^(<<<<<<< |=======$|>>>>>>> )")


def _tracked_text_files():
    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True,
            text=True, check=True, encoding="utf-8",
        ).stdout
        paths = [ROOT / p for p in out.splitlines()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        paths = list(ROOT.rglob("*"))
    return [p for p in paths if p.is_file()]


class TestNoMergeMarkers(unittest.TestCase):
    def test_no_unresolved_conflict_markers(self):
        offenders = []
        for f in _tracked_text_files():
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if MARKER_RE.match(line):
                    offenders.append(f"{f.relative_to(ROOT)}:{i}")
        self.assertEqual(offenders, [], "unresolved merge markers found")


if __name__ == "__main__":
    unittest.main()
