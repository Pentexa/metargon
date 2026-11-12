"""METARGON Cognitive Intelligence Benchmark v0.1.

Independent cognitive evaluation of the actual Academy student agent
(``MetargonAcademyAgent``). Measures six dimensions:

  1. causal discovery
  2. learning efficiency
  3. knowledge retention
  4. generalization
  5. adaptability
  6. planning efficiency

This package is fully self-contained under ``generalization_audit/
cognitive_v01/``: it imports the unmodified Academy student / environment /
baseline policies and never modifies them.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (_ROOT, _ROOT / "python", _ROOT / "academy"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
