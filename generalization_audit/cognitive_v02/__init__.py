"""METARGON Cognitive Evaluation v0.2 — independent intelligence audit.

Independently evaluates whether METARGON's cognitive improvements
represent genuine learning, memory transfer and adaptation rather than
increased search effort or benchmark-specific behavior.

Fully self-contained under ``generalization_audit/cognitive_v02/``:
imports only the frozen Academy API (agent, memory, environment, splits);
never modifies learning algorithms.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (_ROOT, _ROOT / "python", _ROOT / "academy"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
