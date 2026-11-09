"""CLI integration tests for the experiment verifier.

Exercises `main()` with real argv vectors: PASS/FAIL/INCONCLUSIVE exit
codes, `--against` contrast, `--replay-command` with a bounded generator,
and `--stamp`.
"""
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

from generalization_audit.tools.verify_experiment import (  # noqa: E402
    main,
    stamp_manifest,
)
from generalization_audit.verification import sha256_file  # noqa: E402
from generalization_audit.verification.checks import harness_paired_bootstrap  # noqa: E402


def _write_exp(base: Path, alice=(0.8, 0.9), bob=(0.5, 0.6),
               delta=None, raw_rows_extra=None) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (a, b) in enumerate(zip(alice, bob)):
        rows.append({
            "alice": {"success_rate": a, "cpu_s": 1.0,
                      "exact_discovery_D1": a},
            "bob": {"success_rate": b, "cpu_s": 1.0,
                    "exact_discovery_D1": b},
        })
    if raw_rows_extra:
        rows += raw_rows_extra
    if delta is None:
        delta = sum(alice) / len(alice) - sum(bob) / len(bob)
    diffs = [a - b for a, b in zip(alice, bob)]
    lo, hi = harness_paired_bootstrap(diffs, seed=7, rounds=200)
    content = {
        "raw": {"stable": rows},
        "summary": {"stable": {
            "alice": {"success_rate": sum(alice) / len(alice)},
            "bob": {"success_rate": sum(bob) / len(bob)},
            "comparison:alice_minus_bob": {
                "success_delta": round(delta, 10),
                "paired_bootstrap_95pct": [lo, hi]},
        }},
    }
    rel = "experiment.json"
    (base / rel).write_text(json.dumps(content), encoding="utf-8")
    manifest = {
        "schema_version": "experiment-manifest-v01",
        "protocol_version": "eval-protocol-v01",
        "experiment": {
            "name": "syn", "format": "benchmark_v12",
            "family": "f", "conditions": ["stable"],
            "seeds": [0, 1], "episodes_per_seed": 10,
            "agent_version": "a", "harness": "h", "commit_sha": "abcdef0",
            "policies": ["alice", "bob"],
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
            {"path": rel, "role": "raw", "contains_per_seed": True},
        ],
    }
    stamped = stamp_manifest(manifest, base)
    mpath = base / "experiment.manifest.json"
    mpath.write_text(json.dumps(stamped), encoding="utf-8")
    return mpath


class TestVerifyCLI(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_pass_exit_zero(self):
        mpath = _write_exp(self.base)
        rc = main(["--manifest", str(mpath)])
        self.assertEqual(rc, 0)

    def test_fail_exit_one(self):
        mpath = _write_exp(self.base, delta=-0.3)  # reversed under candidate conv
        rc = main(["--manifest", str(mpath)])
        self.assertEqual(rc, 1)

    def test_inconclusive_exit_two_when_artifact_missing(self):
        mpath = _write_exp(self.base)
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        manifest["artifacts"].append({
            "path": "missing.json", "role": "raw",
            "sha256": "0" * 64})
        mpath.write_text(json.dumps(manifest), encoding="utf-8")
        rc = main(["--manifest", str(mpath)])
        self.assertEqual(rc, 1)  # missing artifact is FAIL (manifest claims it)

    def test_json_report_flag(self):
        mpath = _write_exp(self.base)
        rc = main(["--manifest", str(mpath), "--json"])
        self.assertEqual(rc, 0)

    def test_stamp_output(self):
        raw = self.base / "r.json"
        raw.write_text('{"x": 1}', encoding="utf-8")
        cfg = {
            "schema_version": "experiment-manifest-v01",
            "protocol_version": "eval-protocol-v01",
            "experiment": {
                "name": "t", "family": "f", "conditions": ["stable"],
                "seeds": [1], "episodes_per_seed": 2,
                "agent_version": "a", "harness": "h", "commit_sha": "abcdef0",
                "evaluation": {
                    "comparison_convention": "candidate_minus_baseline",
                    "bootstrap": {"method": "paired", "seed": 1,
                                  "rounds": 100, "alpha": 0.05},
                },
            },
            "artifacts": [{"path": "r.json", "role": "raw"}],
        }
        cfg_path = self.base / "cfg.json"
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        out = self.base / "out.manifest.json"
        rc = main(["--stamp", str(cfg_path), "-o", str(out)])
        self.assertEqual(rc, 0)
        stamped = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(stamped["artifacts"][0]["sha256"],
                         sha256_file(raw))


class TestContrastCLI(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_exact_contrast(self):
        a = _write_exp(self.base / "a")
        b = _write_exp(self.base / "b")
        rc = main(["--manifest", str(a), "--against", str(b)])
        self.assertEqual(rc, 0)  # EXACT => pass

    def test_statistical_contrast(self):
        a = _write_exp(self.base / "a")
        b = _write_exp(self.base / "b", alice=(0.81, 0.89), bob=(0.51, 0.59))
        # hashes differ, means within tolerance 0.02? diff 0.01 -> statistical
        rc = main(["--manifest", str(a), "--against", str(b), "--tolerance", "0.02"])
        self.assertEqual(rc, 0)

    def test_non_reproducible_contrast(self):
        a = _write_exp(self.base / "a")
        b = _write_exp(self.base / "b", alice=(0.5, 0.5), bob=(0.5, 0.5))
        rc = main(["--manifest", str(a), "--against", str(b)])
        self.assertEqual(rc, 1)  # NON-REPRODUCIBLE => fail

    def test_replay_command_contrast(self):
        ref = _write_exp(self.base)
        # bounded generator reproducing the same experiment.json content in the
        # replay workdir (its own temp dir) -> EXACT reproducibility
        ref_file = self.base / "experiment.json"
        script = (
            "python -c \""
            "import pathlib; "
            f"d=pathlib.Path(r'{ref_file}').read_text(); "
            "pathlib.Path('experiment.json').write_text(d)\""
        )
        rc = main(["--manifest", str(ref), "--replay-command", script])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
