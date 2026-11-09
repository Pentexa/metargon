"""METARGON experiment verification package.

Public API:

- ``load_manifest(path)`` / ``validate_manifest(dict)``
- ``VerifierContext`` + ``run_checks``
- ``VerificationReport``

The CLI lives in ``generalization_audit/tools/verify_experiment.py`` and
is invoked as ``python -m generalization_audit.tools.verify_experiment``.
"""
from .manifest import (
    ManifestError,
    load_manifest,
    sha256_file,
    validate_manifest,
)
from .checks import (
    CheckResult,
    INCONCLUSIVE,
    PASS,
    FAIL,
    VerifierContext,
    run_checks,
)
from .report import VerificationReport

__all__ = [
    "ManifestError", "load_manifest", "validate_manifest", "sha256_file",
    "CheckResult", "PASS", "FAIL", "INCONCLUSIVE",
    "VerifierContext", "run_checks", "VerificationReport",
]
