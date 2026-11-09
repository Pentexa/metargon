"""Manifest schema + stamp validation tests."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (_ROOT, _ROOT / "python", _ROOT / "generalization_audit"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from generalization_audit.verification import load_manifest, validate_manifest  # noqa: E402
from generalization_audit.tools.verify_experiment import stamp_manifest  # noqa: E402


def _minimal() -> dict:
    return {
        "schema_version": "experiment-manifest-v01",
        "protocol_version": "eval-protocol-v01",
        "experiment": {
            "name": "t", "family": "f", "conditions": ["stable"],
            "seeds": [1], "episodes_per_seed": 10,
            "agent_version": "a", "harness": "h", "commit_sha": "abcdef0",
            "evaluation": {
                "comparison_convention": "candidate_minus_baseline",
                "bootstrap": {"method": "paired", "seed": 1, "rounds": 100,
                              "alpha": 0.05},
            },
        },
        "artifacts": [{
            "path": "raw.json", "role": "raw",
            "sha256": "0" * 64,
        }],
    }


class TestManifestSchema(unittest.TestCase):

    def test_minimal_valid(self):
        self.assertEqual(validate_manifest(_minimal()), [])

    def test_missing_required_experiment_field(self):
        m = _minimal()
        del m["experiment"]["family"]
        errs = validate_manifest(m)
        self.assertTrue(any("family" in e for e in errs))

    def test_missing_required_root_field(self):
        m = _minimal()
        del m["artifacts"]
        errs = validate_manifest(m)
        self.assertTrue(any("artifacts" in e for e in errs))

    def test_bad_sha_pattern(self):
        m = _minimal()
        m["artifacts"][0]["sha256"] = "xyz"
        errs = validate_manifest(m)
        self.assertTrue(any("sha256" in e for e in errs))

    def test_bad_role_enum(self):
        m = _minimal()
        m["artifacts"][0]["role"] = "rawx"
        errs = validate_manifest(m)
        self.assertTrue(any("enum" in e for e in errs))

    def test_bad_convention_enum(self):
        m = _minimal()
        m["experiment"]["evaluation"]["comparison_convention"] = "sideways"
        errs = validate_manifest(m)
        self.assertTrue(any("comparison_convention" in e for e in errs))

    def test_load_manifest_rejects_invalid(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.json"
            p.write_text(json.dumps(_minimal()).replace('"family"', '"fam"'),
                         encoding="utf-8")
            with self.assertRaises(Exception):
                load_manifest(p)


class TestStampManifest(unittest.TestCase):

    def test_stamp_fills_hashes_and_validates(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            raw = base / "raw.json"
            raw.write_text('{"a": 1}', encoding="utf-8")
            cfg = _minimal()
            cfg["artifacts"] = [{"path": "raw.json", "role": "raw"}]
            stamped = stamp_manifest(cfg, base)
            self.assertEqual(len(stamped["artifacts"][0]["sha256"]), 64)
            self.assertEqual(stamped["artifacts"][0]["size_bytes"], raw.stat().st_size)
            self.assertEqual(validate_manifest(stamped), [])

    def test_stamp_missing_artifact_raises(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            cfg = _minimal()
            cfg["artifacts"] = [{"path": "nope.json", "role": "raw"}]
            with self.assertRaises(Exception):
                stamp_manifest(cfg, base)


if __name__ == "__main__":
    unittest.main()
