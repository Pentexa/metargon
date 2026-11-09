"""Integration: committed example manifests must verify with zero FAILs.

These manifests describe real, committed experiments (Memory v0.5 and the
v0.12.1 L-final run). The verifier must not report FAIL on historical
evidence — it either verifies (PASS) or says INCONCLUSIVE when metadata is
absent. Failing here means the infrastructure mis-validates real evidence.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (_ROOT, _ROOT / "python", _ROOT / "generalization_audit"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from generalization_audit.verification import (  # noqa: E402
    FAIL,
    VerifierContext,
    load_manifest,
    run_checks,
    validate_manifest,
)

_MANIFESTS = [
    "generalization_audit/manifests/memory_v05.manifest.json",
    "generalization_audit/manifests/v121_final_l.manifest.json",
]


class TestExampleManifests(unittest.TestCase):

    def test_manifests_are_schema_valid(self):
        for rel in _MANIFESTS:
            path = _ROOT / rel
            self.assertTrue(path.exists(), rel)
            m = load_manifest(path)
            self.assertEqual(validate_manifest(m), [], rel)

    def test_manifests_verify_with_zero_fails(self):
        for rel in _MANIFESTS:
            path = _ROOT / rel
            m = load_manifest(path)
            results = run_checks(VerifierContext(m, path))
            fails = [r for r in results if r.status == FAIL]
            self.assertEqual(
                fails, [],
                f"{rel}: unexpected FAILs: {[(r.name, r.message) for r in fails]}")

    def test_final_l_manifest_captures_temporal_claim(self):
        path = _ROOT / _MANIFESTS[1]
        m = load_manifest(path)
        metrics = m["experiment"]["evaluation"]["metrics"]
        self.assertIn("exact_discovery_D1", metrics)
        self.assertIn("temporal_probes", metrics)
        self.assertEqual(m["experiment"]["seeds"], [11800, 11801, 11802, 11803])
        self.assertEqual(m["experiment"]["episodes_per_seed"], 40)

    def test_memory_v05_manifest_declares_historical_convention(self):
        path = _ROOT / _MANIFESTS[0]
        m = load_manifest(path)
        self.assertEqual(
            m["experiment"]["evaluation"]["comparison_convention"],
            "baseline_minus_candidate")


if __name__ == "__main__":
    unittest.main()
