"""Unit tests for the verifier checks on synthetic experiments."""
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

from generalization_audit.verification import (  # noqa: E402
    FAIL,
    INCONCLUSIVE,
    PASS,
    VerifierContext,
    load_manifest,
    run_checks,
    sha256_file,
)
from generalization_audit.verification.checks import harness_paired_bootstrap  # noqa: E402
from generalization_audit.tools.verify_experiment import stamp_manifest  # noqa: E402


def _write_experiment(base: Path, raw: dict, summary: dict,
                      summary_only: bool = False) -> Path:
    """Write a combined artifact (raw+summary, like the real v121 files)
    and a stamped manifest."""
    rel = "experiment.json"
    content = dict(summary)
    if not summary_only:
        content["raw"] = raw["raw"]
    else:
        content.pop("raw", None)
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


def _raw_rows(alice_succ, bob_succ):
    rows = []
    for i, (a, b) in enumerate(zip(alice_succ, bob_succ)):
        rows.append({
            "alice": {"success_rate": a, "cpu_s": 1.0,
                      "exact_discovery_D1": a},
            "bob": {"success_rate": b, "cpu_s": 1.0,
                    "exact_discovery_D1": b},
        })
    return rows


def _benchmark_summary(raw_rows, delta=None, convention="candidate_minus_baseline"):
    alice = [r["alice"]["success_rate"] for r in raw_rows]
    bob = [r["bob"]["success_rate"] for r in raw_rows]
    if delta is None:
        delta = sum(alice) / len(alice) - sum(bob) / len(bob)
    sign = 1 if convention == "candidate_minus_baseline" else -1
    diffs = [sign * (a - b) for a, b in zip(alice, bob)]
    lo, hi = harness_paired_bootstrap(diffs, seed=7, rounds=200)
    return {
        "raw": {"stable": raw_rows},
        "summary": {
            "stable": {
                "alice": {"success_rate": sum(alice) / len(alice),
                          "exact_discovery_D1": sum(r["alice"]["exact_discovery_D1"] for r in raw_rows) / len(raw_rows)},
                "bob": {"success_rate": sum(bob) / len(bob),
                        "exact_discovery_D1": sum(r["bob"]["exact_discovery_D1"] for r in raw_rows) / len(raw_rows)},
                "comparison:alice_minus_bob": {
                    "success_delta": round(delta, 10),
                    "paired_bootstrap_95pct": [lo, hi]},
            }
        },
        "metadata": {"seeds": 2, "episodes_per_seed": 10},
    }


class TestBenchmarkFormatChecks(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _ctx(self, mpath):
        return VerifierContext(load_manifest(mpath), mpath)

    def test_clean_experiment_passes(self):
        rows = _raw_rows([0.8, 0.9], [0.5, 0.6])
        mpath = _write_experiment(self.base, {"raw": {"stable": rows}},
                                  _benchmark_summary(rows))
        results = run_checks(self._ctx(mpath))
        fails = [r for r in results if r.status == FAIL]
        self.assertEqual(fails, [], [r.message for r in fails])

    def test_tampered_summary_mean_fails(self):
        rows = _raw_rows([0.8, 0.9], [0.5, 0.6])
        summary = _benchmark_summary(rows)
        summary["summary"]["stable"]["alice"]["success_rate"] = 0.42
        mpath = _write_experiment(self.base, {"raw": {"stable": rows}}, summary)
        results = run_checks(self._ctx(mpath))
        self.assertTrue(any(r.status == FAIL and "raw" in r.name
                            for r in results))

    def test_reversed_delta_with_candidate_convention_fails(self):
        rows = _raw_rows([0.8, 0.9], [0.5, 0.6])
        summary = _benchmark_summary(rows, delta=-0.3)  # reversed sign
        mpath = _write_experiment(self.base, {"raw": {"stable": rows}}, summary)
        results = run_checks(self._ctx(mpath))
        self.assertTrue(any(r.status == FAIL and "direction" in r.name
                            for r in results))

    def test_declared_reversed_convention_passes(self):
        rows = _raw_rows([0.8, 0.9], [0.5, 0.6])
        summary = _benchmark_summary(rows, delta=-0.3)
        mpath = _write_experiment(self.base, {"raw": {"stable": rows}}, summary)
        manifest = load_manifest(mpath)
        manifest["experiment"]["evaluation"]["comparison_convention"] = \
            "baseline_minus_candidate"
        mpath.write_text(json.dumps(manifest), encoding="utf-8")
        results = run_checks(self._ctx(mpath))
        self.assertTrue(any(r.status == PASS and "direction" in r.name
                            for r in results))

    def test_missing_raw_is_inconclusive(self):
        rows = _raw_rows([0.8, 0.9], [0.5, 0.6])
        mpath = _write_experiment(self.base, {"raw": {"stable": rows}},
                                  _benchmark_summary(rows), summary_only=True)
        results = run_checks(self._ctx(mpath))
        self.assertTrue(any(r.status == INCONCLUSIVE for r in results))

    def test_duplicate_raw_row_fails(self):
        rows = _raw_rows([0.8, 0.9], [0.5, 0.6])
        rows.append(dict(rows[0]))
        mpath = _write_experiment(self.base, {"raw": {"stable": rows}},
                                  _benchmark_summary(rows))
        results = run_checks(self._ctx(mpath))
        self.assertTrue(any(r.status == FAIL and "unique" in r.name
                            for r in results))

    def test_discovery_fraction_not_integer_fails(self):
        rows = _raw_rows([0.8, 0.9], [0.5, 0.6])
        rows[0]["alice"]["exact_discovery_D1"] = 0.123
        mpath = _write_experiment(self.base, {"raw": {"stable": rows}},
                                  _benchmark_summary(rows))
        results = run_checks(self._ctx(mpath))
        self.assertTrue(any(r.status == FAIL and "fraction" in r.name
                            for r in results))

    def test_artifact_hash_mismatch_fails(self):
        rows = _raw_rows([0.8, 0.9], [0.5, 0.6])
        mpath = _write_experiment(self.base, {"raw": {"stable": rows}},
                                  _benchmark_summary(rows))
        manifest = load_manifest(mpath)
        manifest["artifacts"][0]["sha256"] = "f" * 64
        mpath.write_text(json.dumps(manifest), encoding="utf-8")
        results = run_checks(self._ctx(mpath))
        self.assertTrue(any(r.status == FAIL and "hash" in r.name
                            for r in results))


class TestAcademyMemoryFormat(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _memory_summary(self, flip=False):
        # two variants, 4 configs each; constant +0.25 / -0.25 delta, so
        # any paired bootstrap CI collapses to the constant delta
        orig = [0.5, 0.5, 0.5, 0.5]
        cand = [0.75, 0.75, 0.75, 0.75]
        delta = 0.25 if not flip else -0.25
        return {
            "conditions_detail": {
                "original": {"classic_test_per_config": orig},
                "fresh": {"classic_test_per_config": cand},
            },
            "comparisons": {
                "fresh_vs_original": {
                    "classic_test": {"mean": delta, "ci": [delta, delta]},
                },
            },
        }

    def _write(self, summary, convention="candidate_minus_baseline"):
        base = self.base
        (base / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        manifest = {
            "schema_version": "experiment-manifest-v01",
            "protocol_version": "eval-protocol-v01",
            "experiment": {
                "name": "mem", "format": "academy_memory",
                "family": "academy", "conditions": ["classic"],
                "seeds": [1, 2], "episodes_per_seed": 4,
                "agent_version": "a", "harness": "h",
                "commit_sha": "abcdef0",
                "policies": ["original", "fresh"],
                "resources": {"per_policy_cpu_s": False,
                              "planner_max_expand": 1500,
                              "memory_limit_bytes": 10 ** 9},
                "evaluation": {
                    "comparison_convention": convention,
                    "bootstrap": {"method": "paired", "seed": 99,
                                  "rounds": 500, "alpha": 0.05},
                },
            },
            "artifacts": [{"path": "summary.json", "role": "summary"}],
        }
        stamped = stamp_manifest(manifest, base)
        mpath = base / "m.manifest.json"
        mpath.write_text(json.dumps(stamped), encoding="utf-8")
        return mpath

    def test_candidate_convention_pass(self):
        mpath = self._write(self._memory_summary())
        results = run_checks(VerifierContext(load_manifest(mpath), mpath))
        fails = [r for r in results if r.status == FAIL]
        self.assertEqual(fails, [], [r.message for r in fails])

    def test_flipped_signs_fail_under_candidate_convention(self):
        mpath = self._write(self._memory_summary(flip=True))
        results = run_checks(VerifierContext(load_manifest(mpath), mpath))
        self.assertTrue(any(r.status == FAIL and "direction" in r.name
                            for r in results))

    def test_flipped_signs_pass_under_baseline_convention(self):
        mpath = self._write(self._memory_summary(flip=True),
                            convention="baseline_minus_candidate")
        results = run_checks(VerifierContext(load_manifest(mpath), mpath))
        fails = [r for r in results if r.status == FAIL]
        self.assertEqual(fails, [], [r.message for r in fails])


if __name__ == "__main__":
    unittest.main()
