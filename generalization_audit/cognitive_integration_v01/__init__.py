"""Cognitive Integration Evaluation v0.1.

Independently evaluates whether memory (epoch_reset) and Cognitive
Exploration v0.1 together improve the Academy student more than either
alone, and whether any joint benefit is due to retained causal knowledge
rather than planning/search effort.

Reuses the unmodified Cognitive Evaluation v0.2 framework
(``generalization_audit/cognitive_v02``) and only:
- registers additional control arms (muted/shuffled variants of the
  combined configurations) via the public candidate interface;
- computes interaction and attribution deltas on top of the shared
  per-seed results.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (_ROOT, _ROOT / "python", _ROOT / "academy"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
