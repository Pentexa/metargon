"""Artifact-gate regression tests (Quality v0.5 hotfix).

The frozen evaluation criteria (v02) are immutable, but the *permanent
artifact prohibition* (previous `test_future_targets_have_no_committed_
artifacts_yet`) is replaced by a provenance-aware publication gate:

- absent frozen target -> gate open (PASS)
- raw evidence without a covering manifest -> FAIL
- covering manifest with FAIL verification -> FAIL
- orphan artifact files undeclared by any manifest -> FAIL
- covering manifest verifies with zero FAIL -> PASS (evidence published)

All scenarios are simulated in temporary roots so the real repository and
its committed artifacts are never touched.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (_ROOT, _ROOT / "python", _ROOT / "generalization_audit"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from generalization_audit.verification import FAIL, PASS  # noqa: E402
from generalization_audit.verification.artifact_gate import (  # noqa: E402
    FROZEN_TARGETS,
    check_frozen_target_artifacts,
)
from generalization_audit.tools.verify_experiment import stamp_manifest  # noqa: E402

_TARGET = FROZEN_TARGETS[0][0]  # "results/v13"


def _write_synthetic_experiment(base: Path, target: str) -> Path:
    """Create a minimal (raw+summary) artifact under the target with a
    stamped manifest covering it. Returns the manifest path."""
    tgt = base / target
    tgt.mkdir(parents=True, exist_ok=True)
    artifact = tgt / "final.json"
    content = {
        "metadata": {"seeds": 2, "episodes_per_seed": 10},
        "raw": {"stable": [
            {"alice": {"success_rate": 0.8, "cpu_s": 1.0}},
            {"alice": {"success_rate": 0.9, "cpu_s": 1.0}},
        ]},
        "summary": {"stable": {
            "alice": {"success_rate": 0.85},
        }},
    }
    artifact.write_text(json.dumps(content), encoding="utf-8")

    manifests_dir = base / "generalization_audit" / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(artifact, manifests_dir).replace(os.sep, "/")
    manifest = {
        "schema_version": "experiment-manifest-v01",
        "protocol_version": "eval-protocol-v01",
        "experiment": {
            "name": "syn-v13", "format": "benchmark_v12",
            "family": "f", "conditions": ["stable"],
            "seeds": [0, 1], "episodes_per_seed": 10,
            "agent_version": "a", "harness": "h", "commit_sha": "abcdef0",
            "policies": ["alice"],
            "resources": {"per_policy_cpu_s": True,
                          "planner_max_expand": 700,
                          "memory_limit_bytes": 10 ** 9},
            "evaluation": {
                "comparison_convention": "candidate_minus_baseline",
                "bootstrap": {"method": "paired", "seed": 7,
                              "rounds": 200, "alpha": 0.05},
            },
        },
        "artifacts": [
            {"path": rel, "role": "raw",
             "contains_per_seed": True},
        ],
    }
    stamped = stamp_manifest(manifest, manifests_dir)
    mpath = manifests_dir / "syn_v13.manifest.json"
    mpath.write_text(json.dumps(stamped), encoding="utf-8")
    return mpath


class TestArtifactGateSimulated(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _gate(self):
        return check_frozen_target_artifacts(
            self.base, self.base / "generalization_audit" / "manifests")

    def test_absent_target_gate_open(self):
        reports = self._gate()
        self.assertEqual(len(reports), len(FROZEN_TARGETS))
        for r in reports:
            self.assertEqual(r.overall, PASS)
            self.assertTrue(all(c.status == PASS for c in r.checks))

    def test_evidence_without_manifest_fails(self):
        tgt = self.base / _TARGET
        tgt.mkdir(parents=True)
        (tgt / "final.json").write_text('{"raw": {}}', encoding="utf-8")
        reports = self._gate()
        self.assertTrue(any(r.overall == FAIL for r in reports))
        gate_r = next(r for r in reports if r.target == _TARGET)
        self.assertTrue(any("provenance" in c.name and c.status == FAIL
                            for c in gate_r.checks))

    def test_valid_provenance_passes(self):
        _write_synthetic_experiment(self.base, _TARGET)
        reports = self._gate()
        gate_r = next(r for r in reports if r.target == _TARGET)
        self.assertEqual(gate_r.overall, PASS)
        self.assertTrue(any("manifest" in c.name and c.status == PASS
                            for c in gate_r.checks))

    def test_tampered_artifact_fails_verification(self):
        mpath = _write_synthetic_experiment(self.base, _TARGET)
        # corrupt the raw artifact after stamping -> hash mismatch
        tgt = self.base / _TARGET
        art = tgt / "final.json"
        data = json.loads(art.read_text(encoding="utf-8"))
        data["summary"]["stable"]["alice"]["success_rate"] = 0.42
        art.write_text(json.dumps(data), encoding="utf-8")
        reports = self._gate()
        gate_r = next(r for r in reports if r.target == _TARGET)
        self.assertEqual(gate_r.overall, FAIL)
        self.assertTrue(any("manifest" in c.name and c.status == FAIL
                            for c in gate_r.checks))

    def test_orphan_artifact_not_declared_fails(self):
        _write_synthetic_experiment(self.base, _TARGET)
        tgt = self.base / _TARGET
        (tgt / "unlisted.json").write_text('{"extra": true}', encoding="utf-8")
        reports = self._gate()
        gate_r = next(r for r in reports if r.target == _TARGET)
        self.assertEqual(gate_r.overall, FAIL)
        self.assertTrue(any("provenance" in c.name and c.status == FAIL
                            for c in gate_r.checks))


class TestArtifactGateRealRepo(unittest.TestCase):
    """Integration against the real repository (read-only)."""

    def test_current_main_state_gate_open(self):
        # As of this branch, v0.13 / Memory v0.6 have no committed raw
        # evidence; the gate must be open (all PASS, zero FAIL).
        reports = check_frozen_target_artifacts(_ROOT)
        self.assertEqual(len(reports), len(FROZEN_TARGETS))
        for r in reports:
            self.assertEqual(r.overall, PASS,
                             f"{r.target}: {r.render_text()}")
            self.assertTrue(all(c.status == PASS for c in r.checks))


if __name__ == "__main__":
    unittest.main()
