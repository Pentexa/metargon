"""Protocol stream determinism and structure tests."""
from __future__ import annotations

import unittest

from generalization_audit.cognitive_v01 import config
from generalization_audit.cognitive_v01.protocols import (
    config_signature,
    eval_stream,
    revisit_occurrences,
    revisit_stream,
    stream_signature,
    switch_pair,
    switch_split_point,
    switch_stream,
    train_stream,
)


class TestProtocolStreams(unittest.TestCase):

    def test_train_stream_deterministic_and_correct_length(self):
        s1 = train_stream(0, 6)
        s2 = train_stream(0, 6)
        self.assertEqual(stream_signature(s1), stream_signature(s2))
        self.assertEqual(len(s1), 6)
        self.assertTrue(all(sp.pool_tag == "train" for sp in s1))
        self.assertTrue(all(sp.max_steps == config.MAX_STEPS for sp in s1))

    def test_eval_streams_are_disjoint_by_construction(self):
        val = eval_stream("val", 0, 4)
        test = eval_stream("test", 0, 4)
        ood = eval_stream("ood", 0, 4)
        self.assertEqual(len(val), 4)
        self.assertEqual(len(test), 4)
        self.assertEqual(len(ood), 4)
        self.assertTrue(all(sp.pool_tag == "val" for sp in val))
        self.assertTrue(all(sp.pool_tag == "test" for sp in test))
        self.assertTrue(all(sp.pool_tag == "ood" for sp in ood))

    def test_revisit_stream_cycles_and_repeats(self):
        stream = revisit_stream(0, n_configs=3)
        self.assertEqual(len(stream), 7)
        occ = revisit_occurrences(stream)
        self.assertEqual(len(occ), 3)          # A, B, C
        for sig, idxs in occ.items():
            self.assertGreaterEqual(len(idxs), 2)  # each config repeats
        # A (the first config) appears at 0, 3, 6 -> 3 occurrences
        first_sig = config_signature(stream[0].config)
        self.assertEqual(len(occ[first_sig]), 3)
        # first and last episode are the same config (retention check)
        self.assertEqual(config_signature(stream[0].config),
                         config_signature(stream[-1].config))

    def test_switch_pair_properties(self):
        a, b = switch_pair(0)
        self.assertEqual(a.topology, b.topology)
        self.assertEqual(a.d2_rule, b.d2_rule)
        self.assertNotEqual(a.d1_rule, b.d1_rule)
        self.assertEqual(a.d1_delay, b.d1_delay)
        self.assertEqual(a.d2_delay, b.d2_delay)

    def test_switch_stream_structure(self):
        stream = switch_stream(0, half=4)
        self.assertEqual(len(stream), 8)
        sp = switch_split_point(stream)
        self.assertEqual(sp, 4)
        self.assertEqual(config_signature(stream[0].config),
                         config_signature(stream[sp - 1].config))
        self.assertEqual(config_signature(stream[sp].config),
                         config_signature(stream[-1].config))
        self.assertNotEqual(config_signature(stream[0].config),
                            config_signature(stream[sp].config))

    def test_stream_signature_stable_across_calls(self):
        s1 = switch_stream(1, half=2)
        s2 = switch_stream(1, half=2)
        self.assertEqual(stream_signature(s1), stream_signature(s2))


if __name__ == "__main__":
    unittest.main()
