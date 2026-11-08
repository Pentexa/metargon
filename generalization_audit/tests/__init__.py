"""Tests for the generalization-audit package.

Bootstraps the repository root and python/ onto sys.path so the audit
tests can import the shared research modules (lab_env, policies) without
requiring a specific working directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (_ROOT, _ROOT / "python"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
