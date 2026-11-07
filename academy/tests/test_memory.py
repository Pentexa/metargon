"""Tests for the compositional hypothesis memory: evidence flow, bounds,
maintain/verification semantics, and checkpoint roundtrip."""

import unittest

from academy.agents.memory import (
    CompositionalMemory, VERIFIED, CONTRADICTED, PREDICTED, HYPOTHESIS,
    consequence_key,
)

A = frozenset({("sw", "S1", True)})
B = frozenset({("load", "P1", "A")})
AB = frozenset({("sw", "S1", True), ("load", "P1", "A")})


class TestMemory(unittest.TestCase):
    def test_event_generates_and_supports(self):
        m = CompositionalMemory()
        m.observe_event(A, "D1", True, A)
        hyps = m.query("D1", opens=True)
        self.assertTrue(hyps)
        self.assertEqual(hyps[0].sup, 1)

    def test_after_state_context_supports_persistent_cause(self):
        # the true cause only holds *after* the door opens
        m = CompositionalMemory()
        before = frozenset({("sw", "S1", False)})
        after = frozenset({("sw", "S1", True)})
        m.observe_event(before, "D1", True, after)
        q = m.query("D1", opens=True)
        self.assertTrue(any(h.antecedent == after for h in q))

    def test_maintain_verifies_persistent_cause(self):
        m = CompositionalMemory()
        m.observe_event(frozenset(), "D1", True, A)
        m.observe_maintain(A, "D1")
        m.observe_maintain(A, "D1")
        m.observe_maintain(A, "D1")
        q = m.query("D1", opens=True)
        self.assertEqual(q[0].status, VERIFIED)

    def test_nonevent_contradiction(self):
        m = CompositionalMemory()
        m.observe_event(A, "D1", True, A)
        # three contradiction windows (3 consecutive true steps each)
        for _ in range(9):
            m.observe_nonevent(A, "D1", True)
        self.assertEqual(m.query("D1", opens=True), [])

    def test_bounded_eviction(self):
        m = CompositionalMemory(max_hypotheses=5)
        for i in range(20):
            ant = frozenset({
                ("sw", "S1", bool(i & 1)),
                ("sw", "S2", bool(i & 2)),
                ("load", "P1", "A") if i % 3 == 0 else ("empty", "P1"),
            })
            m._add(ant, consequence_key("D1", True))
        self.assertLessEqual(len(m), 5)
        self.assertGreater(m.evictions, 0)

    def test_roundtrip(self):
        m = CompositionalMemory()
        m.observe_event(A, "D1", True, A)
        m.observe_maintain(A, "D1")
        m.observe_maintain(A, "D1")
        s = m.save_state()
        m2 = CompositionalMemory()
        m2.load_state(s)
        self.assertEqual([h.as_dict() for h in m.hypotheses],
                         [h.as_dict() for h in m2.hypotheses])
        self.assertEqual(m.step, m2.step)
        self.assertEqual(m.generated, m2.generated)

    def test_consequence_key(self):
        self.assertEqual(consequence_key("D1", True), "door:D1:opens")
        self.assertEqual(consequence_key("D2", False), "door:D2:closes")


if __name__ == "__main__":
    unittest.main()
