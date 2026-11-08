"""Tests for memory v0.4 context-aware evidence mechanisms.

Focus: separate general knowledge from task-specific evidence, prevent
destructive cross-task contradictions, and preserve v0.3 behavior when
all new flags are disabled.
"""

from __future__ import annotations

import unittest

from academy.agents.memory import (
    CONTRADICTED, HYPOTHESIS, PREDICTED, VERIFIED, CompositionalMemory,
    ctx_key,
)

# two distinct PUBLIC-observation context signatures (opaque tuples)
CTX_A = ((0, 0), (True, False), ("A", None), (1, 1), (2, 2), (False, False))
CTX_B = ((0, 0), (False, True), (None, "B"), (3, 3), (4, 4), (False, False))
KEY_A = ctx_key(CTX_A)
KEY_B = ctx_key(CTX_B)

ATOMS = frozenset({("sw", "S2", True)})


class TestContextScoped(unittest.TestCase):
    def test_default_global_behavior_preserved(self):
        m = CompositionalMemory()
        m.set_context(CTX_A)
        m.observe_event(ATOMS, "D1", True, None)
        # 9 nonevents => con=3 > 2 => contradicted (global cumulative)
        for _ in range(9):
            m.observe_nonevent(ATOMS, "D1", True)
        self.assertTrue(all(h.status == CONTRADICTED for h in m.hypotheses))

    def test_context_scoped_keeps_hypothesis_alive_across_contexts(self):
        m = CompositionalMemory(context_scoped=True)
        m.set_context(CTX_A)
        m.observe_event(ATOMS, "D1", True, None)
        for _ in range(4):
            m.observe_event(ATOMS, "D1", True, None)   # sup_A = 5
        # now switch context and contradict heavily
        m.set_context(CTX_B)
        for _ in range(12):
            m.observe_nonevent(ATOMS, "D1", True)      # con_B grows
        h = m.hypotheses[0]
        # in CTX_B the hypothesis is contradicted…
        self.assertEqual(h.status, CONTRADICTED)
        # …but evidence in CTX_A is preserved (reversible confidence)
        self.assertGreaterEqual(h.ctx_support(KEY_A), 5)
        self.assertGreaterEqual(h.ctx_contradiction(KEY_B), 1)
        # switch back to CTX_A: status recovers to verified
        m.set_context(CTX_A)
        m._recompute_status(h)
        self.assertEqual(h.status, VERIFIED)

    def test_context_scoped_generation_uses_active_context(self):
        m = CompositionalMemory(context_scoped=True)
        m.set_context(CTX_A)
        m.observe_event(ATOMS, "D1", True, None)
        h = m.hypotheses[0]
        self.assertEqual(h.ctx_support(KEY_A), 1)


class TestTaskChangeReset(unittest.TestCase):
    def test_reset_prevents_cross_task_nonevent_accumulation(self):
        m = CompositionalMemory(task_change_reset=True)
        m.set_context(CTX_A)
        m.observe_event(ATOMS, "D1", True, None)
        # two nonevents in A (window 3, not reached)
        m.observe_nonevent(ATOMS, "D1", True)
        m.observe_nonevent(ATOMS, "D1", True)
        m.set_context(CTX_B)          # task switch resets windows
        m.observe_nonevent(ATOMS, "D1", True)
        m.observe_nonevent(ATOMS, "D1", True)
        # with a reset the count never reaches 3 across tasks
        self.assertFalse(any(h.status == CONTRADICTED for h in m.hypotheses))

    def test_no_reset_accumulates_across_contexts(self):
        m = CompositionalMemory(task_change_reset=False)
        m.set_context(CTX_A)
        m.observe_event(ATOMS, "D1", True, None)
        for _ in range(5):
            m.observe_nonevent(ATOMS, "D1", True)
        m.set_context(CTX_B)
        for _ in range(5):
            m.observe_nonevent(ATOMS, "D1", True)
        # cross-task accumulation reaches contradiction threshold
        self.assertTrue(any(h.status == CONTRADICTED for h in m.hypotheses))


class TestContextGateTransfer(unittest.TestCase):
    def test_gate_hides_unproven_in_new_context(self):
        m = CompositionalMemory(context_scoped=True, context_gate_transfer=True)
        m.set_context(CTX_A)
        m.observe_event(ATOMS, "D1", True, None)
        # in A the hypothesis is queryable
        self.assertEqual(len(m.query("D1")), 1)
        # in B (no evidence yet) it is hidden
        m.set_context(CTX_B)
        self.assertEqual(len(m.query("D1")), 0)
        # evidence in B makes it available again
        m.observe_event(ATOMS, "D1", True, None)
        self.assertEqual(len(m.query("D1")), 1)


class TestCheckpointCompatibility(unittest.TestCase):
    def test_v03_state_loads_with_v04_defaults(self):
        m = CompositionalMemory()
        m.set_context(CTX_A)
        m.observe_event(ATOMS, "D1", True, None)
        old = m.save_state()
        for k in ("context_scoped", "task_change_reset",
                  "context_gate_transfer", "current_ctx"):
            old.pop(k, None)
        m2 = CompositionalMemory()
        m2.load_state(old)
        self.assertFalse(m2.context_scoped)
        self.assertIsNone(m2.current_ctx)
        self.assertEqual(len(m2.hypotheses), len(m.hypotheses))

    def test_v04_flags_roundtrip(self):
        m = CompositionalMemory(context_scoped=True, task_change_reset=True,
                                context_gate_transfer=True)
        m.set_context(CTX_A)
        m.observe_event(ATOMS, "D1", True, None)
        m2 = CompositionalMemory()
        m2.load_state(m.save_state())
        self.assertTrue(m2.context_scoped)
        self.assertTrue(m2.task_change_reset)
        self.assertTrue(m2.context_gate_transfer)
        self.assertEqual(m2.current_ctx, m.current_ctx)


if __name__ == "__main__":
    unittest.main()
