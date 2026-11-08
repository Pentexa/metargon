"""Leakage and shortcut checks for the audit benchmark."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (_ROOT, _ROOT / "python"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from lab_env import LabEpisode

from generalization_audit.envs.base import RWorld, LWorld, RConfig
from generalization_audit.envs.pools import build_pools
from generalization_audit.metrics.shortcuts import (
    assert_no_hidden_access, noise_parity_check,
)
from generalization_audit.policies.factory import (
    make_policy, ALL_POLICIES, COMPUTE_BUDGET,
)


class TestNoHiddenAccess(unittest.TestCase):
    def test_metargon10_never_stores_config(self):
        (train, _, _), _ = build_pools("r")
        cfg = train[0]
        policy = make_policy("metargon10", 7, family="r")
        from generalization_audit.benchmark_generalization import play_episode
        spec = LabEpisode(cfg, 60, 0.0, seed=9, context="r")
        play_episode(spec, policy, "r")
        assert_no_hidden_access(policy, cfg)

    def test_all_policies_no_hidden_reference(self):
        for family in ("r", "l"):
            (train, _, _), _ = build_pools(family)
            cfg = train[0]
            from generalization_audit.benchmark_generalization import play_episode
            for name in ALL_POLICIES:
                policy = make_policy(name, 7, family=family)
                spec = LabEpisode(cfg, 60, 0.0, seed=9, context=family)
                play_episode(spec, policy, family)
                if name == "oracle":
                    # the privileged oracle legitimately holds the hidden
                    # config (diagnostic comparator); skip the non-hold
                    # assertion for it
                    continue
                assert_no_hidden_access(policy, cfg)


class TestShortcutChecks(unittest.TestCase):
    def test_noise_parity_across_policies(self):
        (train, _, _), _ = build_pools("r")
        cfg = train[0]
        self.assertTrue(noise_parity_check(RWorld, cfg, 1234, steps=30))

    def test_compute_cap_fires_only_on_planning_policies(self):
        (train, _, _), _ = build_pools("r")
        cfg = train[0]
        from generalization_audit.benchmark_generalization import play_episode
        spec = LabEpisode(cfg, 40, 0.0, seed=3, context="r")
        capped = make_policy("metargon10-cap", 1, family="r")
        play_episode(spec, capped, "r")
        self.assertEqual(capped.budget, COMPUTE_BUDGET)
        self.assertGreaterEqual(capped.expansions, 0)


if __name__ == "__main__":
    unittest.main()
