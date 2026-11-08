"""Metric unit tests for the audit (discovery, forgetting, adaptation)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (_ROOT, _ROOT / "python"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from generalization_audit.metrics.discovery import (
    discovery_from_rows, summarize_discovery, forgetting_between,
    adaptation_summary, verified_antecedents_by_door,
)


class _H:
    def __init__(self, door, ante, status):
        self.door = door
        self.antecedent = ante
        self.status = status


class _Spec:
    def __init__(self, cond):
        self.config = cond


class _Cfg:
    def __init__(self, d1, d2):
        self._d1 = frozenset(d1)
        self._d2 = frozenset(d2)

    def cond_for(self, door):
        return self._d1 if door == "D1" else self._d2


class TestDiscovery(unittest.TestCase):
    def test_exact_discovery(self):
        true1 = {("sw", "S1", True)}
        spec = _Spec(_Cfg(true1, {("load", "P2", "B")}))
        rows = [(spec, [_H("D1", true1, "verified")])]
        stats = discovery_from_rows(rows)
        s = summarize_discovery(stats)
        self.assertEqual(s["D1"]["exact_rate"], 1.0)
        self.assertEqual(s["D1"]["covered_rate"], 1.0)
        self.assertEqual(s["D1"]["false_verified_per_episode"], 0.0)

    def test_false_verified_counted(self):
        true1 = {("sw", "S1", True)}
        spec = _Spec(_Cfg(true1, {("load", "P2", "B")}))
        rows = [(spec, [
            _H("D1", {("sw", "S1", False)}, "verified"),
            _H("D1", true1, "verified"),
        ])]
        s = summarize_discovery(discovery_from_rows(rows))
        self.assertEqual(s["D1"]["exact_rate"], 1.0)
        self.assertEqual(s["D1"]["false_verified_per_episode"], 1.0)


class TestForgetting(unittest.TestCase):
    def test_jaccard_complement(self):
        pre = [_H("D1", {("sw", "S1", True)}, "verified")]
        post = [_H("D1", {("sw", "S1", True)}, "verified"),
                _H("D1", {("load", "P1", "A")}, "verified")]
        self.assertEqual(forgetting_between(pre, post, "D1"), 0.0)
        self.assertEqual(forgetting_between(post, pre, "D1"), 0.5)


class TestAdaptation(unittest.TestCase):
    def test_recovery_first_episode(self):
        summary = adaptation_summary([False, False], [False, True, True], 2)
        self.assertEqual(summary["first_recovery_episode"], 3)
        self.assertEqual(summary["pre_success_rate"], 0.0)
        self.assertAlmostEqual(summary["post_success_rate"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
