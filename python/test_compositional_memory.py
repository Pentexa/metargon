"""Phase C tests: compositional causal memory correctness."""
import unittest

from compositional_memory import (
    CONTRADICTED, HYPOTHESIS, MAX_CONTRADICT, MAX_HYPOTHESES, OPENS,
    PREDICTED, VERIFY_SUPPORT, VERIFIED, CompositionalMemory, Hypothesis,
    antecedent_text, atoms_from_observation, opens_key,
)


def atoms_of(*items):
    return frozenset(items)


def fake_obs(**kw):
    class O:
        pass
    o = O()
    o.x = kw.get("x", 0); o.y = kw.get("y", 0)
    o.ax = kw.get("ax", 2); o.ay = kw.get("ay", 3)
    o.bx = kw.get("bx", 4); o.by = kw.get("by", 3)
    o.s1 = kw.get("s1", False); o.s2 = kw.get("s2", False)
    o.p1 = kw.get("p1", "none"); o.p2 = kw.get("p2", "none")
    o.d1 = kw.get("d1", False); o.d2 = kw.get("d2", False)
    o.finished = False
    return o


class LifecycleTests(unittest.TestCase):
    def test_atom_extraction_from_observation(self):
        a = atoms_from_observation(fake_obs(p1="A", s1=True))
        self.assertIn(("load", "P1", "A"), a)
        self.assertIn(("loadany", "P1"), a)
        self.assertIn(("sw", "S1", True), a)
        self.assertIn(("sw", "S2", False), a)
        self.assertNotIn(("load", "P2", "A"), a)

    def test_event_generates_hypothesis(self):
        m = CompositionalMemory()
        ctx = atoms_of(("load", "P1", "A"), ("sw", "S1", True))
        m.observe_event(ctx, "D1", True)
        self.assertEqual(len(m), 3)  # two singletons + one pair
        q = m.query("D1", True)
        self.assertTrue(q)
        # all candidates participated in one confirmed event -> PREDICTED,
        # but none is VERIFIED yet (no auto-verification)
        self.assertTrue(all(h.status == PREDICTED for h in q))
        self.assertFalse(any(h.status == VERIFIED for h in q))

    def test_hypothesis_does_not_auto_verify(self):
        m = CompositionalMemory()
        ctx = atoms_of(("sw", "S1", True), ("load", "P1", "A"))
        for _ in range(VERIFY_SUPPORT - 1):
            m.observe_event(ctx, "D1", True)
        self.assertFalse(any(h.status == VERIFIED for h in m.query("D1", True)))
        m.observe_event(ctx, "D1", True)
        self.assertTrue(any(h.status == VERIFIED for h in m.query("D1", True)))

    def test_contradiction_invalidates(self):
        m = CompositionalMemory()
        ctx = atoms_of(("sw", "S1", True))
        # build support
        for _ in range(VERIFY_SUPPORT):
            m.observe_event(ctx, "D2", True)
        q = m.query("D2", True)
        self.assertTrue(any(h.status == VERIFIED for h in q))
        # contradict: antecedent true continuously, door never opens; each
        # contradiction requires a full CONTRADICT_WINDOW of consecutive steps
        for _ in range((MAX_CONTRADICT + 1) * 3):
            m.observe_nonevent(ctx, "D2", True)
        q = m.query("D2", True)
        self.assertFalse(any(h.status == VERIFIED for h in q))
        self.assertFalse(any(h.usable for h in m.hypotheses
                             if h.consequence == opens_key("D2", True)))

    def test_nonevent_requires_continuous_window(self):
        m = CompositionalMemory()
        ctx = atoms_of(("sw", "S1", True))
        m.observe_nonevent(ctx, "D1", True)
        m.observe_nonevent(atoms_of(), "D1", True)   # antecedent drops
        m.observe_nonevent(ctx, "D1", True)
        m.observe_nonevent(ctx, "D1", True)
        # only two consecutive true steps -> no contradiction yet
        for h in m.hypotheses:
            self.assertEqual(h.con, 0)

    def test_antecedent_must_be_subset_of_context(self):
        m = CompositionalMemory()
        ctx = atoms_of(("sw", "S1", True), ("load", "P1", "A"))
        m.observe_event(ctx, "D1", True)
        # a candidate with an extra atom (not in context) should never have
        # matched, so it is not supported
        q = m.query("D1", True)
        self.assertTrue(all(h.sup <= 1 for h in q))

    def test_eviction_bounded_deterministic(self):
        m = CompositionalMemory(max_hypotheses=8)
        # generate > max hypotheses by observing many distinct contexts
        for i in range(20):
            ctx = atoms_of(("sw", "S1", bool(i % 2)), ("load", "P1", "A" if i % 2 else "B"))
            m.observe_event(ctx, "D1", True)
            m.step_advance()
        self.assertLessEqual(len(m), 8)
        # deterministic: same sequence -> same result
        m2 = CompositionalMemory(max_hypotheses=8)
        for i in range(20):
            ctx = atoms_of(("sw", "S1", bool(i % 2)), ("load", "P1", "A" if i % 2 else "B"))
            m2.observe_event(ctx, "D1", True)
            m2.step_advance()
        self.assertEqual([(h.antecedent, h.sup, h.con, h.status)
                          for h in m.hypotheses],
                         [(h.antecedent, h.sup, h.con, h.status)
                          for h in m2.hypotheses])

    def test_support_prefers_correct_antecedent(self):
        """The true rule's antecedent should accumulate more support than
        a singleton that only coincides occasionally."""
        m = CompositionalMemory()
        true_ante = atoms_of(("sw", "S1", True), ("load", "P1", "A"))
        for _ in range(6):
            m.observe_event(true_ante, "D1", True)
            m.step_advance()
        # singleton candidate ("sw","S1",True) appears in context but the
        # pair should have >= support of any singleton
        pair = m._find(true_ante, opens_key("D1", True))
        single = m._find(atoms_of(("sw", "S1", True)), opens_key("D1", True))
        self.assertIsNotNone(pair)
        self.assertIsNotNone(single)
        self.assertGreaterEqual(pair.sup, single.sup)

    def test_diagnostics_report(self):
        m = CompositionalMemory()
        m.observe_event(atoms_of(("sw", "S1", True)), "D1", True)
        d = m.diagnostics()
        self.assertEqual(d["size"], 1)
        self.assertEqual(d["max"], MAX_HYPOTHESES)
        self.assertIn("status_counts", d)


if __name__ == "__main__":
    unittest.main()
