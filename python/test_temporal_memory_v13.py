"""v0.13 temporal-memory tests: rolling completion generation and
trigger-proximity support weighting."""
import unittest

from temporal_memory import TemporalMemory, VERIFIED, PREDICTED, HYPOTHESIS


class TemporalMemoryV13Tests(unittest.TestCase):
    def test_defaults_preserve_v12_behavior(self):
        t = TemporalMemory(seed=1)
        self.assertEqual(t.completion_mode, "active_first")
        self.assertEqual(t.support_weighting, "flat")
        self.assertEqual(t._first_buf["D1"], [])

    def test_rolling_generates_true_pair_across_spurious_batch(self):
        """v0.12's single-slot generation loses the true pair (load,P1,A)
        -> (sw,S1,False) because the simultaneous (loadany,P1) consumes the
        slot. Rolling generation must create it."""
        t = TemporalMemory(seed=1, completion_mode="rolling")
        # step 5: loading P1 turns on (load,P1,A) AND (loadany,P1) at once
        t.update("D1", 5, frozenset({
            ("load", "P1", "A"), ("loadany", "P1")}))
        # step 9: S1 flips off — the true second for D1
        t.update("D1", 9, frozenset({("sw", "S1", False)}))
        pairs = {(h.first, h.second) for h in t.hypotheses}
        self.assertIn((frozenset({("load", "P1", "A")}),
                       frozenset({("sw", "S1", False)})), pairs)

    def test_rolling_generates_same_step_batch_pairs(self):
        t = TemporalMemory(seed=2, completion_mode="rolling")
        t.update("D1", 3, frozenset({
            ("load", "P1", "A"), ("loadany", "P1")}))
        pairs = {(h.first, h.second) for h in t.hypotheses}
        self.assertIn((frozenset({("load", "P1", "A")}),
                       frozenset({("loadany", "P1")})), pairs)

    def test_rolling_never_pairs_atom_with_itself(self):
        t = TemporalMemory(seed=3, completion_mode="rolling")
        t.update("D1", 4, frozenset({("sw", "S1", True)}))
        t.update("D1", 6, frozenset({("sw", "S1", True)}))
        for h in t.hypotheses:
            self.assertNotEqual(h.first, h.second)

    def test_rolling_buffer_pruned_by_window(self):
        t = TemporalMemory(seed=4, completion_mode="rolling", window=10)
        t.update("D1", 1, frozenset({("sw", "S1", True)}))
        # old first should not pair after window expires
        t.update("D1", 12, frozenset({("sw", "S2", True)}))
        for h in t.hypotheses:
            self.assertNotEqual(h.first, frozenset({("sw", "S1", True)}))

    def test_proximity_weighting_credits_close_trigger(self):
        t = TemporalMemory(seed=5, completion_mode="rolling",
                           support_weighting="proximity")
        t.update("D1", 10, frozenset({("sw", "S1", True)}))
        t.update("D1", 12, frozenset({("sw", "S1", False)}))
        h = next(h for h in t.hypotheses
                 if h.second == frozenset({("sw", "S1", False)}))
        # completion at step 12; open at step 12 => gap 0 => weight 1.0
        t.observe_open("D1", True, 12)
        self.assertEqual(h.sup, 1.0)
        self.assertEqual(h.status, PREDICTED)

    def test_proximity_weighting_halves_distant_credit(self):
        t = TemporalMemory(seed=6, completion_mode="rolling",
                           support_weighting="proximity")
        t.update("D1", 10, frozenset({("sw", "S1", True)}))
        t.update("D1", 12, frozenset({("sw", "S1", False)}))
        h = next(h for h in t.hypotheses
                 if h.second == frozenset({("sw", "S1", False)}))
        # completion at step 12; open at step 15 => gap 3 => weight 0.5
        t.observe_open("D1", True, 15)
        self.assertAlmostEqual(h.sup, 0.5)

    def test_flat_weighting_is_legacy(self):
        t = TemporalMemory(seed=7, completion_mode="rolling",
                           support_weighting="flat")
        t.update("D1", 10, frozenset({("sw", "S1", True)}))
        t.update("D1", 12, frozenset({("sw", "S1", False)}))
        h = next(h for h in t.hypotheses
                 if h.second == frozenset({("sw", "S1", False)}))
        t.observe_open("D1", True, 15)
        self.assertEqual(h.sup, 1.0)

    def test_episode_end_clears_rolling_buffer(self):
        t = TemporalMemory(seed=8, completion_mode="rolling")
        t.update("D1", 2, frozenset({("sw", "S1", True)}))
        self.assertTrue(t._first_buf["D1"])
        t.episode_end()
        self.assertEqual(t._first_buf["D1"], [])
        self.assertEqual(t._pending["D1"], [])

    def test_correct_rule_verifies_after_repeated_support(self):
        t = TemporalMemory(seed=9, completion_mode="rolling",
                           support_weighting="proximity")
        first = frozenset({("load", "P1", "A")})
        second = frozenset({("sw", "S1", False)})
        for i in range(4):
            t.update("D1", 10 + i * 20, first)
            t.update("D1", 12 + i * 20, second)
            t.observe_open("D1", True, 12 + i * 20)
        h = next(h for h in t.hypotheses
                 if (h.first, h.second) == (first, second))
        self.assertEqual(h.status, VERIFIED)

    def test_contradiction_still_works(self):
        t = TemporalMemory(seed=10, completion_mode="rolling")
        t.update("D1", 10, frozenset({("sw", "S1", True)}))
        t.update("D1", 12, frozenset({("sw", "S1", False)}))
        h = next(h for h in t.hypotheses
                 if h.second == frozenset({("sw", "S1", False)}))
        # door stays closed well past the credit window => contradiction
        for step in range(17, 40):
            t.observe_nonevent("D1", step)
        self.assertGreater(h.con, 0)
        self.assertNotEqual(h.status, VERIFIED)


if __name__ == "__main__":
    unittest.main()
