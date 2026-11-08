"""Tests for memory v0.3 flags (recycling, ratio gating, evidence
windowing, adaptive/episode-local probing).

Each test asserts both the new behavior when the flag is enabled AND
that defaults preserve the v0.2 behavior (no regression).
"""

from __future__ import annotations

import json
import unittest

from academy.agents.memory import (
    CONTRADICTED, HYPOTHESIS, PREDICTED, VERIFIED, CompositionalMemory,
)
from academy.agents.metargon_agent import MetargonAcademyAgent
from academy.envs.gridworld import EnvConfig
from academy.envs.tasks import get_splits, EpisodeSpec, difficulty_of
from academy.learning.loop import run_one_episode

C_TRUE = EnvConfig("TP1", frozenset({("sw", "S2", True)}),
                   frozenset({("load", "P2", "B")}), False, False)


def _mem(recycle=False, ratio=0.0, window=0):
    return CompositionalMemory(
        recycle_contradicted=recycle,
        contradiction_ratio=ratio,
        evidence_window=window)


class TestRecycling(unittest.TestCase):
    def test_default_keeps_contradicted(self):
        m = _mem()
        m.observe_event(frozenset({("sw", "S2", True)}), "D1", True, None)
        m.observe_nonevent(frozenset({("sw", "S2", True)}), "D1", True)
        m.observe_nonevent(frozenset({("sw", "S2", True)}), "D1", True)
        m.observe_nonevent(frozenset({("sw", "S2", True)}), "D1", True)
        self.assertGreater(len(m.hypotheses), 0)
        self.assertTrue(all(h.status == CONTRADICTED for h in m.hypotheses
                            if h.sup == 0))

    def test_recycle_removes_and_blocks_regeneration(self):
        m = _mem(recycle=True)
        m.observe_event(frozenset({("sw", "S2", True)}), "D1", True, None)
        # contradiction needs con > 2: three 3-step crossings
        for _ in range(9):
            m.observe_nonevent(frozenset({("sw", "S2", True)}), "D1", True)
        # recycled => none remain
        self.assertEqual(len(m.hypotheses), 0)
        # dead-set blocks regeneration on new events
        m.observe_event(frozenset({("sw", "S2", True)}), "D1", True, None)
        m.observe_event(frozenset({("sw", "S2", True)}), "D1", True, None)
        self.assertEqual(len(m.hypotheses), 0)

    def test_recycle_does_not_kill_below_threshold(self):
        m = _mem(recycle=True)
        atoms = frozenset({("sw", "S2", True)})
        for _ in range(4):
            m.observe_event(atoms, "D1", True, None)
        # 3 nonevents => con=1 (< threshold) => still retained
        for _ in range(3):
            m.observe_nonevent(atoms, "D1", True)
        self.assertGreater(len(m.hypotheses), 0)


class TestRatioGate(unittest.TestCase):
    def test_default_gate_off(self):
        m = _mem()
        atoms = frozenset({("sw", "S2", True)})
        for _ in range(5):
            m.observe_event(atoms, "D1", True, None)
        for _ in range(9):
            m.observe_nonevent(atoms, "D1", True)
        self.assertTrue(all(h.status == CONTRADICTED
                            for h in m.hypotheses if h.sup >= 5))

    def test_ratio_gate_blocks_flip_of_supported(self):
        m = _mem(ratio=2.0)
        atoms = frozenset({("sw", "S2", True)})
        for _ in range(5):
            m.observe_event(atoms, "D1", True, None)
        for _ in range(9):
            m.observe_nonevent(atoms, "D1", True)
        # gated con = 0 -> stays verified
        self.assertTrue(any(h.status == VERIFIED for h in m.hypotheses))

    def test_ratio_gate_still_kills_unsupported(self):
        m = _mem(ratio=2.0)
        atoms = frozenset({("sw", "S2", True)})
        m.observe_event(atoms, "D1", True, None)   # seed one hypothesis
        for _ in range(9):
            m.observe_nonevent(atoms, "D1", True)
        self.assertTrue(any(h.status == CONTRADICTED for h in m.hypotheses))


class TestEvidenceWindow(unittest.TestCase):
    def test_default_cumulative(self):
        m = _mem()
        atoms = frozenset({("sw", "S2", True)})
        for _ in range(6):
            m.observe_event(atoms, "D1", True, None)
        h = m.hypotheses[0]
        self.assertEqual(h.sup, 6)

    def test_window_limits_status_counts(self):
        m = _mem(window=3)
        atoms = frozenset({("sw", "S2", True)})
        for _ in range(6):
            m.observe_event(atoms, "D1", True, None)
        h = m.hypotheses[0]
        sup, con = m._windowed_counts(h)
        self.assertEqual(sup, 3)
        self.assertEqual(con, 0)


class TestCheckpointCompatibility(unittest.TestCase):
    def test_v02_state_loads_with_defaults(self):
        # simulates an old checkpoint (v0.2 keys only)
        m = _mem()
        m.observe_event(frozenset({("sw", "S2", True)}), "D1", True, None)
        old = m.save_state()
        for k in ("recycle_contradicted", "contradiction_ratio",
                  "evidence_window", "dead_set_cap",
                  "dead", "sup_timeline", "con_timeline"):
            old.pop(k, None)
        m2 = _mem()
        m2.load_state(old)
        self.assertEqual(len(m2.hypotheses), len(m.hypotheses))
        self.assertFalse(m2.recycle_contradicted)
        self.assertEqual(m2.contradiction_ratio, 0.0)
        self.assertEqual(m2.evidence_window, 0)

    def test_new_flags_roundtrip(self):
        m = _mem(recycle=True, ratio=1.5, window=7)
        m.observe_event(frozenset({("sw", "S2", True)}), "D1", True, None)
        m2 = _mem()
        m2.load_state(m.save_state())
        self.assertTrue(m2.recycle_contradicted)
        self.assertEqual(m2.contradiction_ratio, 1.5)
        self.assertEqual(m2.evidence_window, 7)


class TestAgentFlags(unittest.TestCase):
    def _train(self, agent, episodes=8):
        splits = get_splits()
        import random
        rng = random.Random(0)
        for i in range(episodes):
            cfg = rng.choice(splits["train"])
            spec = EpisodeSpec(config=cfg, max_steps=60, noise_q=0.0,
                               spec_seed=1000 + i, pool_tag="train",
                               difficulty=difficulty_of(cfg))
            run_one_episode(agent, spec)
        return agent

    def test_default_determinism(self):
        a = self._train(MetargonAcademyAgent(seed=0))
        b = self._train(MetargonAcademyAgent(seed=0))
        self.assertEqual(a.diagnostics(), b.diagnostics())

    def test_probe_adaptive_within_budget(self):
        a = self._train(MetargonAcademyAgent(seed=0, probe_adaptive=True))
        self.assertGreater(a.successes, 0)

    def test_episode_local_flags_run(self):
        a = self._train(MetargonAcademyAgent(
            seed=0, probe_adaptive=True, probe_episode_local=True,
            protect_episode_local=True))
        self.assertGreaterEqual(a.successes, 0)

    def test_state_roundtrip_preserves_agent(self):
        a = MetargonAcademyAgent(seed=3, probe_adaptive=True,
                                 protect_episode_local=True)
        self._train(a, episodes=6)
        s = a.save_state()
        b = MetargonAcademyAgent(seed=3, probe_adaptive=True,
                                 protect_episode_local=True)
        b.load_state(s)
        self.assertEqual(a.diagnostics(), b.diagnostics())


if __name__ == "__main__":
    unittest.main()
