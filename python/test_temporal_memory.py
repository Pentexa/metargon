"""v0.12 temporal memory tests (Mission B prototype, v2 API)."""
import unittest

from temporal_memory import TemporalMemory, VERIFIED, CONTRADICTED
from compositional_memory_v10 import opens_key


class TemporalMemoryTests(unittest.TestCase):
    def test_ordered_hypothesis_learned(self):
        tm = TemporalMemory(seed=1)
        first = frozenset({("sw", "S1", True)})
        second = frozenset({("loadany", "P1")})
        cons = opens_key("D1", True)
        # forward completion: first then second
        tm.update("D1", 10, first)
        tm.update("D1", 12, second)          # completion fires
        h = tm._find(first, second, cons)
        self.assertIsNotNone(h)
        self.assertGreaterEqual(h.sup, 0.0)  # not yet credited
        # door opens 2 steps later -> credit support
        tm.update("D1", 13, frozenset())
        tm.observe_open("D1", True, 14)
        self.assertGreaterEqual(h.sup, 1.0)
        for step in (20, 30, 40):
            tm.update("D1", step, first)
            tm.update("D1", step + 2, second)
            tm.observe_open("D1", True, step + 3)
        self.assertEqual(h.status, VERIFIED)
        self.assertEqual(tm.best_for_door("D1"), h)

    def test_reverse_order_contradicts(self):
        tm = TemporalMemory(seed=2)
        first = frozenset({("sw", "S1", True)})
        second = frozenset({("loadany", "P1")})
        # reverse completions: second arrives first, then first
        for step in (20, 28, 36, 44):
            tm.update("D1", step, second)
            tm.update("D1", step + 1, first)   # completion (second->first)
            tm.observe_nonevent("D1", step + 8)
        cons = opens_key("D1", True)
        rev = tm._find(second, first, cons)     # wrongly-ordered hypothesis
        self.assertIsNotNone(rev)
        self.assertEqual(rev.status, CONTRADICTED)
        # the forward hypothesis was never created from reversals
        self.assertIsNone(tm._find(first, second, cons))

    def test_bounded(self):
        import random as _random
        rng = _random.Random(0)
        tm = TemporalMemory(max_hypotheses=8, seed=3)
        pool = [
            frozenset({("sw", "S1", True)}),
            frozenset({("sw", "S1", False)}),
            frozenset({("sw", "S2", True)}),
            frozenset({("sw", "S2", False)}),
            frozenset({("loadany", "P1")}),
            frozenset({("loadany", "P2")}),
            frozenset({("load", "P1", "A")}),
            frozenset({("load", "P2", "B")}),
        ]
        for i in range(300):
            a = pool[rng.randrange(len(pool))]
            b = pool[rng.randrange(len(pool))]
            if a == b:
                b = pool[(pool.index(a) + 1) % len(pool)]
            tm.update("D1", i, a)
            tm.update("D1", i + 1, b)
            tm.observe_nonevent("D1", i + 8)
        self.assertLessEqual(len(tm), 8)
        self.assertGreater(tm.evictions, 0)

    def test_diagnostics(self):
        tm = TemporalMemory(seed=4)
        d = tm.diagnostics()
        self.assertIn("size", d)
        self.assertIn("status_counts", d)


if __name__ == "__main__":
    unittest.main()
