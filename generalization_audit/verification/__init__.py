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
    canonical_sha256,
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
from .artifact_gate import (
    FROZEN_TARGETS,
    GateReport,
    check_frozen_target_artifacts,
)

__all__ = [
    "ManifestError", "load_manifest", "validate_manifest", "sha256_file",
    "canonical_sha256",
    "CheckResult", "PASS", "FAIL", "INCONCLUSIVE",
    "VerifierContext", "run_checks", "VerificationReport",
    "FROZEN_TARGETS", "GateReport", "check_frozen_target_artifacts",
]
