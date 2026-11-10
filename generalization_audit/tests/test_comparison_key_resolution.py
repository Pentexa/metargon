"""Regression tests: semantic comparison-key resolution (audit v0.6).

Covers the ``v12_temporal_minus_notp_exact_D1`` case and the bare
``temporal_minus_notp_exact_D1`` case where several policy pairs are
data-indistinguishable (identical raw vectors) and the resolver must
resolve deterministically without manual intervention.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (_ROOT, _ROOT / "python", _ROOT / "generalization_audit"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from generalization_audit.verification.checks import (  # noqa: E402
    _mean,
    resolve_comparison_with_delta,
)

_POLICIES = ["metargon12", "metargon13", "metargon13-gen",
             "metargon13-tw", "metargon13-tpex",
             "metargon13-notp", "metargon12-notp",
             "oracle", "random", "qlearning", "episodic"]

# v0.12-style single-slot rows (from the committed v0.12 study)
_V12_D1 = [0.0, 0.0, 0.675, 0.0]
# v0.13 rolling rows (from the committed final_l raw)
_V13_D1 = [0.775, 0.0, 0.8, 0.0]
_NOTP = [0.0, 0.0, 0.0, 0.0]


def _rows():
    return [
        {"metargon12": {"exact_discovery_D1": v, "temporal_probes": 60},
         "metargon13": {"exact_discovery_D1": v13, "temporal_probes": 62},
         "metargon13-gen": {"exact_discovery_D1": v, "temporal_probes": 60},
         "metargon13-tw": {"exact_discovery_D1": v13, "temporal_probes": 62},
         "metargon13-tpex": {"exact_discovery_D1": v13, "temporal_probes": 62},
         "metargon13-notp": {"exact_discovery_D1": n, "temporal_probes": 0},
         "metargon12-notp": {"exact_discovery_D1": n, "temporal_probes": 0},
         "oracle": {"exact_discovery_D1": 1.0, "temporal_probes": 0},
         "random": {"exact_discovery_D1": 0.0, "temporal_probes": 0},
         "qlearning": {"exact_discovery_D1": 0.0, "temporal_probes": 0},
         "episodic": {"exact_discovery_D1": 0.0, "temporal_probes": 0}}
        for v, v13, n in zip(_V12_D1, _V13_D1, _NOTP)
    ]


class TestSemanticComparisonKeyResolution(unittest.TestCase):

    def setUp(self):
        self.rows = _rows()

    def test_v12_temporal_key_resolves_to_metargon12_pair(self):
        delta = _mean(_V12_D1) - _mean(_NOTP)  # 0.16875
        status, res = resolve_comparison_with_delta(
            self.rows, "comparison:v12_temporal_minus_notp_exact_D1",
            _POLICIES, delta)
        self.assertEqual(status, "ok")
        self.assertEqual(res[0], "metargon12")
        self.assertEqual(res[1], "metargon12-notp")
        self.assertEqual(res[2], "exact_discovery_D1")

    def test_temporal_key_resolves_when_family_ambiguous(self):
        delta = _mean(_V13_D1) - _mean(_NOTP)  # 0.39375
        status, res = resolve_comparison_with_delta(
            self.rows, "comparison:temporal_minus_notp_exact_D1",
            _POLICIES, delta)
        self.assertEqual(status, "ok")
        self.assertEqual(res[0], "metargon13")
        self.assertEqual(res[1], "metargon13-notp")

    def test_duplicate_roster_entries_do_not_cause_ambiguity(self):
        dup_policies = _POLICIES + ["metargon12-notp"]
        delta = _mean(_V12_D1) - _mean(_NOTP)
        status, res = resolve_comparison_with_delta(
            self.rows, "comparison:v12_temporal_minus_notp_exact_D1",
            dup_policies, delta)
        self.assertEqual(status, "ok")
        self.assertEqual(res[1], "metargon12-notp")

    def test_reversed_delta_reports_none(self):
        # A delta that no pair reproduces must not silently resolve.
        status, res = resolve_comparison_with_delta(
            self.rows, "comparison:v12_temporal_minus_notp_exact_D1",
            _POLICIES, 0.9)
        self.assertEqual(status, "none")

    def test_plain_policy_key_resolves_directly(self):
        alice = [0.8, 0.9]
        bob = [0.5, 0.6]
        rows = [
            {"alice": {"success_rate": a}, "bob": {"success_rate": b}}
            for a, b in zip(alice, bob)
        ]
        delta = _mean(alice) - _mean(bob)
        status, res = resolve_comparison_with_delta(
            rows, "comparison:alice_minus_bob", ["alice", "bob"], delta,
            metric_hint="success_rate")
        self.assertEqual(status, "ok")
        self.assertEqual(res, ("alice", "bob", "success_rate"))


if __name__ == "__main__":
    unittest.main()
