"""Stream determinism and conflict-pair structure tests."""
from __future__ import annotations

import unittest

from academy.envs.gridworld import is_solvable

from generalization_audit.cognitive_v02 import config as C
from generalization_audit.cognitive_v02.streams import (
    conflict_pair,
    config_signature,
    eval_stream,
    phase_train_stream,
    retention_eval_configs,
    retention_eval_stream,
    rule_sig,
    single_config_stream,
    stream_signature,
    train_stream,
)


class TestStreams(unittest.TestCase):

    def test_train_stream_deterministic(self):
        s1 = train_stream(0, 5)
        s2 = train_stream(0, 5)
        self.assertEqual(stream_signature(s1), stream_signature(s2))
        self.assertEqual(len(s1), 5)

    def test_eval_streams_use_distinct_pools(self):
        val = eval_stream("val", 0, 4)
        test = eval_stream("test", 0, 4)
        ood = eval_stream("ood", 0, 4)
        for sp in val:
            self.assertTrue(sp.config.topology in ("TP1", "TP2"))
        self.assertEqual(len(val), 4)
        self.assertEqual(len(test), 4)
        self.assertEqual(len(ood), 4)

    def test_noisy_eval_uses_noise(self):
        noisy = eval_stream("test", 0, 3, noise_q=0.08)
        self.assertTrue(all(sp.noise_q == 0.08 for sp in noisy))

    def test_conflict_pair_properties(self):
        a, b = conflict_pair(0)
        self.assertEqual(a.topology, b.topology)
        self.assertEqual(a.d2_rule, b.d2_rule)
        self.assertEqual(a.d1_delay, b.d1_delay)
        self.assertEqual(a.d2_delay, b.d2_delay)
        self.assertNotEqual(a.d1_rule, b.d1_rule)
        # B is the flipped-polarity variant of A
        atom_a = next(iter(a.d1_rule))
        self.assertIn((atom_a[0], atom_a[1], not atom_a[2]), b.d1_rule)
        self.assertTrue(is_solvable(b))

    def test_conflict_pair_deterministic(self):
        a1, b1 = conflict_pair(3)
        a2, b2 = conflict_pair(3)
        self.assertEqual(config_signature(a1), config_signature(a2))
        self.assertEqual(config_signature(b1), config_signature(b2))

    def test_retention_eval_configs_share_d1_rule(self):
        a, _ = conflict_pair(0)
        cfgs = retention_eval_configs(a)
        self.assertGreaterEqual(len(cfgs), 1)
        sig = rule_sig(a.d1_rule)
        for c in cfgs:
            self.assertEqual(c.topology, a.topology)
            self.assertEqual(rule_sig(c.d1_rule), sig)

    def test_retention_stream_length_and_signature(self):
        a, _ = conflict_pair(0)
        st = retention_eval_stream(a, 1, 5)
        self.assertEqual(len(st), 5)
        self.assertEqual(stream_signature(st), stream_signature(st))

    def test_phase_stream_single_config(self):
        a, _ = conflict_pair(0)
        st = phase_train_stream(a, 0, 3, "phaseA1")
        self.assertEqual(len(st), 3)
        self.assertTrue(all(config_signature(s.config) ==
                            config_signature(a) for s in st))
        self.assertTrue(all(s.pool_tag == "phaseA1" for s in st))


if __name__ == "__main__":
    unittest.main()
