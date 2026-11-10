"""Transfer-evaluation protocol checks regression tests (Quality v0.7).

Exercises PASS / FAIL / INCONCLUSIVE semantics of every transfer check on
synthetic manifests + artifacts. Also validates the frozen protocol file
itself (schema/registration invariants).
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

from generalization_audit.verification import (  # noqa: E402
    FAIL,
    INCONCLUSIVE,
    PASS,
    VerifierContext,
    load_manifest,
    validate_manifest,
)
from generalization_audit.verification.transfer_checks import (  # noqa: E402
    REQUIRED_ROLES,
    TransferProtocol,
    check_transfer_attribution_consistency,
    check_transfer_control_roles,
    check_transfer_frozen_immutability,
    check_transfer_memory_ablation_paired,
    check_transfer_protocol_registered,
    check_transfer_unseen_separation,
    run_transfer_checks,
)
from generalization_audit.tools.verify_experiment import stamp_manifest  # noqa: E402


def _base_manifest() -> dict:
    return {
        "schema_version": "experiment-manifest-v01",
        "protocol_version": "transfer-eval-protocol-v01",
        "experiment": {
            "name": "tr", "format": "academy_memory",
            "family": "academy", "conditions": ["classic"],
            "seeds": [1, 2], "episodes_per_seed": 4,
            "agent_version": "a", "harness": "h", "commit_sha": "abcdef0",
            "policies": ["fresh", "original", "wiped", "shuffled",
                         "planonly"],
            "action_budget_max_steps": 80,
            "resources": {"per_policy_cpu_s": True,
                          "planner_max_expand": 700,
                          "memory_limit_bytes": 10 ** 9},
            "evaluation": {
                "comparison_convention": "candidate_minus_baseline",
                "bootstrap": {"method": "paired", "seed": 7,
                              "rounds": 200, "alpha": 0.05},
                "transfer": {
                    "roles": {
                        "fresh": "fresh",
                        "trained_with_memory": "original",
                        "trained_memory_wiped": "wiped",
                        "shuffled_memory": "shuffled",
                        "planning_only": "planonly",
                    },
                    "eval_mode": "frozen",
                    "unseen_configs_declared": True,
                    "split_source": "frozen-l.json",
                    "frozen_immutability_asserted": True,
                },
            },
        },
        "artifacts": [
            {"path": "summary.json", "role": "summary"},
        ],
    }


def _write(base: Path, manifest: dict, summary: dict) -> Path:
    (base / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    stamped = stamp_manifest(manifest, base)
    mpath = base / "tr.manifest.json"
    mpath.write_text(json.dumps(stamped), encoding="utf-8")
    return mpath


def _summary_with_vectors(lengths: dict[str, int]) -> dict:
    return {
        "conditions_detail": {
            name: {"classic_frozen_per_config": [0.5] * n}
            for name, n in lengths.items()
        },
        "comparisons": {},
    }


class TestTransferProtocolFile(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.protocol = TransferProtocol.load(_ROOT)

    def test_protocol_registered_fields(self):
        d = self.protocol.data
        self.assertEqual(d["status"], "frozen")
        self.assertTrue(d["created"].startswith("2026-11-"))
        self.assertEqual(d["audit"], "evaluation-v07")

    def test_required_control_arms_defined(self):
        arms = self.protocol.data["mandatory_control_arms"]
        for role in REQUIRED_ROLES:
            self.assertIn(role, arms)

    def test_quantities_defined(self):
        q = self.protocol.data["quantities_to_distinguish"]
        for key in ("retained_memory_benefit", "within_episode_learning",
                    "goal_directed_exploration", "repeated_task_memorization",
                    "true_transfer_unseen_structures", "negative_transfer"):
            self.assertIn(key, q)

    def test_claim_levels_and_seeds(self):
        levels = self.protocol.data["claim_levels"]
        self.assertIn("level_1_preliminary", levels)
        self.assertIn("level_2_confirmatory", levels)
        seeds = self.protocol.data["seed_registration"]["frozen_seeds"]
        self.assertGreaterEqual(len(seeds), 8)
        self.assertEqual(len(seeds), len(set(seeds)))


class TestTransferChecksSemantics(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.protocol = TransferProtocol.load(_ROOT)

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, manifest, summary):
        mpath = _write(self.base, manifest, summary)
        m = load_manifest(mpath)
        return run_transfer_checks(VerifierContext(m, mpath), self.protocol)

    def test_full_declaration_passes(self):
        manifest = _base_manifest()
        summary = _summary_with_vectors(
            {"fresh": 16, "original": 16, "wiped": 16,
             "shuffled": 16, "planonly": 16})
        results = self._run(manifest, summary)
        fails = [r for r in results if r.status == FAIL]
        self.assertEqual(fails, [], [(r.name, r.message) for r in fails])
        self.assertTrue(any(r.name == "transfer.roles.resolve" and r.status == PASS
                            for r in results))

    def test_missing_roles_inconclusive(self):
        manifest = _base_manifest()
        del manifest["experiment"]["evaluation"]["transfer"]["roles"]
        summary = _summary_with_vectors({})
        results = self._run(manifest, summary)
        r = next(r for r in results if r.name == "transfer.roles.declared")
        self.assertEqual(r.status, INCONCLUSIVE)

    def test_partial_roles_fail(self):
        manifest = _base_manifest()
        roles = manifest["experiment"]["evaluation"]["transfer"]["roles"]
        del roles["shuffled_memory"]
        summary = _summary_with_vectors({})
        results = self._run(manifest, summary)
        r = next(r for r in results if r.name == "transfer.roles.declared")
        self.assertEqual(r.status, FAIL)

    def test_unresolved_role_fail(self):
        manifest = _base_manifest()
        manifest["experiment"]["evaluation"]["transfer"]["roles"]["planning_only"] = "nope"
        summary = _summary_with_vectors(
            {"fresh": 16, "original": 16, "wiped": 16,
             "shuffled": 16, "planonly": 16})
        results = self._run(manifest, summary)
        r = next(r for r in results if r.name == "transfer.roles.resolve")
        self.assertEqual(r.status, FAIL)

    def test_unseen_undeclared_inconclusive(self):
        manifest = _base_manifest()
        manifest["experiment"]["evaluation"]["transfer"]["unseen_configs_declared"] = False
        summary = _summary_with_vectors({})
        results = self._run(manifest, summary)
        r = next(r for r in results if r.name == "transfer.unseen.separation")
        self.assertEqual(r.status, INCONCLUSIVE)

    def test_frozen_without_attestation_inconclusive(self):
        manifest = _base_manifest()
        manifest["experiment"]["evaluation"]["transfer"]["frozen_immutability_asserted"] = False
        summary = _summary_with_vectors({})
        results = self._run(manifest, summary)
        r = next(r for r in results if r.name == "transfer.frozen.immutability")
        self.assertEqual(r.status, INCONCLUSIVE)

    def test_ablation_vector_length_mismatch_fail(self):
        manifest = _base_manifest()
        summary = _summary_with_vectors(
            {"fresh": 16, "original": 16, "wiped": 12,
             "shuffled": 16, "planonly": 16})
        results = self._run(manifest, summary)
        r = next(r for r in results if r.name == "transfer.ablation.paired")
        self.assertEqual(r.status, FAIL)

    def test_attribution_consistent_pass_inconsistent_fail(self):
        manifest = _base_manifest()
        good = {
            "conditions_detail": {},
            "comparisons": {},
            "runs": [{
                "variant": "original",
                "per": {"frozen_trace_mean": {
                    "source_hypothesis": 1.0, "source_bank": 2.0,
                    "source_atom_single": 1.0, "source_atom_pair": 0.0,
                    "source_novel_single": 2.0, "source_conjunctive_probe": 0.0,
                    "source_other": 0.0, "plans_used": 6.0}},
            }],
        }
        results = self._run(manifest, good)
        r = next(r for r in results if r.name == "transfer.attribution.consistency")
        self.assertEqual(r.status, PASS)

        bad = json.loads(json.dumps(good))
        bad["runs"][0]["per"]["frozen_trace_mean"]["plans_used"] = 10.0
        results = self._run(manifest, bad)
        r = next(r for r in results if r.name == "transfer.attribution.consistency")
        self.assertEqual(r.status, FAIL)

    def test_budget_fairness_recording(self):
        manifest = _base_manifest()
        summary = _summary_with_vectors({})
        results = self._run(manifest, summary)
        self.assertTrue(any(r.name == "transfer.budgets.planner" and r.status == PASS
                            for r in results))
        self.assertTrue(any(r.name == "transfer.budgets.steps" and r.status == PASS
                            for r in results))
        # no raw rows -> cpu measurement inconclusive
        r = next(r for r in results if r.name == "transfer.budgets.cpu")
        self.assertEqual(r.status, INCONCLUSIVE)

    def test_schema_accepts_transfer_section(self):
        manifest = _base_manifest()
        mpath = _write(self.base, manifest, _summary_with_vectors({}))
        self.assertEqual(validate_manifest(load_manifest(mpath)), [])

    def test_protocol_mismatch_fail(self):
        manifest = _base_manifest()
        manifest["protocol_version"] = "some-other-protocol"
        summary = _summary_with_vectors({})
        results = self._run(manifest, summary)
        r = next(r for r in results if r.name == "transfer.protocol.registered")
        self.assertEqual(r.status, FAIL)


if __name__ == "__main__":
    unittest.main()
