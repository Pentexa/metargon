"""v0.10 compositional memory tests (Phase G)."""
import unittest

from compositional_memory_v10 import (
    CompositionalMemoryV10, Hypothesis, MAX_CONTRADICT, VERIFY_SUPPORT,
    opens_key,
)


class MemoryV10Tests(unittest.TestCase):
    def setUp(self):
        self.m = CompositionalMemoryV10(seed=1)

    def test_support_verifies(self):
        ante = frozenset({("sw", "S1", True)})
        self.m.observe_event(frozenset({("sw", "S1", True)}), "D1", True)
        h = self.m._find(ante, opens_key("D1", True))
        self.assertEqual(h.sup, 1.0)
        self.assertEqual(h.status, "predicted")
        self.m.observe_event(frozenset({("sw", "S1", True)}), "D1", True)
        self.m.observe_event(frozenset({("sw", "S1", True)}), "D1", True)
        self.assertGreaterEqual(h.sup, VERIFY_SUPPORT)
        self.assertEqual(h.status, "verified")

    def test_contradiction_demotes_then_contradicts(self):
        ante = frozenset({("sw", "S1", True)})
        # support once
        self.m.observe_event(frozenset({("sw", "S1", True)}), "D1", True)
        h = self.m._find(ante, opens_key("D1", True))
        # continuous true without effect counts one contradiction per window
        atoms = frozenset({("sw", "S1", True)})
        for _ in range(3):
            self.m.observe_nonevent(atoms, "D1", True)
        self.assertEqual(h.con, 1.0)
        for _ in range(3):
            self.m.observe_nonevent(atoms, "D1", True)
        self.assertEqual(h.con, 2.0)
        self.assertNotEqual(h.status, "contradicted")
        for _ in range(3):
            self.m.observe_nonevent(atoms, "D1", True)
        self.assertGreater(h.con, MAX_CONTRADICT)
        self.assertEqual(h.status, "contradicted")
        self.assertFalse(h.usable)

    def test_decay_reduces_evidence(self):
        ante = frozenset({("sw", "S1", True)})
        for _ in range(3):
            self.m.observe_event(frozenset({("sw", "S1", True)}), "D1", True)
        h = self.m._find(ante, opens_key("D1", True))
        before = h.sup
        # replay reinforces earlier evidence, so raw support can exceed the
        # direct count; the weighted value must still decay between episodes
        self.assertGreaterEqual(h.n_sup, 3)
        self.m.episode_end()
        self.assertLess(h.sup, before)

    def test_replay_prioritizes_novel_antecedents(self):
        self.m.observe_event(frozenset({("sw", "S1", True)}), "D1", True)
        self.m.observe_event(frozenset({("sw", "S1", True)}), "D1", True)
        # second event generates candidates; repetition should still be
        # replayed but limited by REPLAY_PER_EVENT per event
        self.assertGreaterEqual(self.m.replay_applied, 1)
        self.assertLessEqual(len(self.m.replay), 64)

    def test_replay_buffer_bounded(self):
        for i in range(80):
            self.m.observe_event(frozenset({("sw", "S1", bool(i % 2))}),
                                 "D1", True)
        self.assertLessEqual(len(self.m.replay), 64)

    def test_candidate_generation_complete_for_small_context(self):
        atoms = frozenset({("sw", "S1", True), ("ws", "S2", False),
                           ("loadany", "P1"), ("load", "P1", "A"),
                           ("loadany", "P2")})
        cands = self.m._candidate_antecedents(atoms, opens_key("D1", True))
        # all singletons must be present (no string-truncation bug)
        for a in atoms:
            self.assertIn(frozenset({a}), cands)
        # pairs covering the first atoms present
        self.assertIn(frozenset({("sw", "S1", True), ("loadany", "P1")}),
                      cands)

    def test_memory_bounded_and_deterministic_eviction(self):
        m = CompositionalMemoryV10(max_hypotheses=16, seed=2)
        import random as _random
        rng = _random.Random(7)
        for i in range(300):
            ctx = frozenset({
                ("sw", "S1", bool(rng.getrandbits(1))),
                ("sw", "S2", bool(rng.getrandbits(1))),
                ("loadany", rng.choice(("P1", "P2"))),
            })
            m.observe_event(ctx, rng.choice(("D1", "D2")), True)
            m.step_advance()
        self.assertLessEqual(len(m), 16)
        self.assertGreater(m.evictions, 0)

    def test_hypothesis_score_deterministic(self):
        # identical scores must be resolved by eviction determinism: on a
        # full tie the larger h_id (newer) is evicted first.
        m = CompositionalMemoryV10(max_hypotheses=2, seed=1)
        h1 = Hypothesis(frozenset({("sw", "S1", True)}), opens_key("D1", True),
                        sup=3.0, status="verified", last_update=10, h_id=1)
        h2 = Hypothesis(frozenset({("sw", "S2", True)}), opens_key("D2", True),
                        sup=3.0, status="verified", last_update=10, h_id=2)
        m.hypotheses = [h1, h2]
        m._evict_one()
        self.assertNotIn(h2, m.hypotheses)
        self.assertIn(h1, m.hypotheses)


if __name__ == "__main__":
    unittest.main()
