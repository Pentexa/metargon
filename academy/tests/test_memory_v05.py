"""Tests for memory v0.5: frozen-policy evaluation safety and the
evidence-bank reuse mechanism."""

from __future__ import annotations

import unittest

from academy.agents.memory import VERIFIED, CompositionalMemory
from academy.agents.metargon_agent import MetargonAcademyAgent
from academy.envs.gridworld import EnvConfig
from academy.envs.tasks import EpisodeSpec, difficulty_of, get_splits
from academy.learning.loop import run_one_episode

ATOMS = frozenset({("sw", "S2", True)})


class TestFrozenPolicy(unittest.TestCase):
    def test_frozen_policy_blocks_learning(self):
        agent = MetargonAcademyAgent(seed=0)
        agent.frozen_policy = True
        before = agent.memory.generated
        # simulate a step that would normally teach the agent
        agent.learn(None, "N", None)
        self.assertEqual(agent.memory.generated, before)
        self.assertEqual(agent.steps, 0)

    def test_frozen_policy_survives_checkpoint_roundtrip(self):
        agent = MetargonAcademyAgent(seed=0)
        agent.frozen_policy = True
        st = agent.save_state()
        agent2 = MetargonAcademyAgent(seed=0)
        agent2.load_state(st)
        self.assertTrue(agent2.frozen_policy)

    def test_default_frozen_off(self):
        agent = MetargonAcademyAgent(seed=0)
        self.assertFalse(agent.frozen_policy)


class TestEvidenceBank(unittest.TestCase):
    def test_bank_captures_verified(self):
        m = CompositionalMemory(evidence_bank_cap=16)
        m.observe_event(ATOMS, "D1", True, None)
        for _ in range(4):
            m.observe_event(ATOMS, "D1", True, None)
        self.assertEqual(len(m.evidence_bank), 1)
        ante, cons = m.evidence_bank[0]
        self.assertEqual(ante, ATOMS)
        self.assertIn("D1", cons)

    def test_bank_cap_fifo(self):
        m = CompositionalMemory(evidence_bank_cap=2)
        for i in range(4):
            # different antecedents via different atoms
            atoms = frozenset({("sw", "S1", bool(i % 2))})
            m.observe_event(atoms, "D1", True, None)
            for _ in range(4):
                m.observe_event(atoms, "D1", True, None)
        self.assertLessEqual(len(m.evidence_bank), 2)

    def test_bank_deduplicates(self):
        m = CompositionalMemory(evidence_bank_cap=16)
        for _ in range(3):
            m.observe_event(ATOMS, "D1", True, None)
            for _ in range(4):
                m.observe_event(ATOMS, "D1", True, None)
        self.assertEqual(len(m.evidence_bank), 1)

    def test_bank_roundtrip(self):
        m = CompositionalMemory(evidence_bank_cap=8)
        m.observe_event(ATOMS, "D1", True, None)
        for _ in range(4):
            m.observe_event(ATOMS, "D1", True, None)
        m2 = CompositionalMemory(evidence_bank_cap=8)
        m2.load_state(m.save_state())
        self.assertEqual(len(m2.evidence_bank), 1)
        self.assertEqual(m2.evidence_bank[0][0], ATOMS)

    def test_bank_retrieval_marker(self):
        m = CompositionalMemory(evidence_bank_cap=8)
        m.observe_event(ATOMS, "D1", True, None)
        for _ in range(4):
            m.observe_event(ATOMS, "D1", True, None)
        m.mark_retrieval()
        self.assertEqual(m.bank_retrievals, 1)

    def test_bank_off_by_default(self):
        m = CompositionalMemory()
        self.assertEqual(m.evidence_bank_cap, 0)
        self.assertEqual(len(m.evidence_bank), 0)


class TestAgentBankIntegration(unittest.TestCase):
    def _train(self, agent, episodes=6):
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

    def test_bank_agent_trains_and_retrieves(self):
        a = self._train(MetargonAcademyAgent(
            seed=0, memory_evidence_bank_cap=16))
        self.assertGreaterEqual(len(a.memory.evidence_bank), 0)
        self.assertTrue(hasattr(a.memory, "bank_retrievals"))

    def test_bank_agent_state_roundtrip(self):
        a = self._train(MetargonAcademyAgent(
            seed=3, memory_evidence_bank_cap=16))
        st = a.save_state()
        b = MetargonAcademyAgent(seed=3, memory_evidence_bank_cap=16)
        b.load_state(st)
        self.assertEqual(len(b.memory.evidence_bank),
                         len(a.memory.evidence_bank))
        self.assertEqual(b.memory.bank_retrievals,
                         a.memory.bank_retrievals)

    def test_v03_checkpoint_compat(self):
        a = MetargonAcademyAgent(seed=0)
        self._train(a, episodes=4)
        st = a.save_state()
        for k in ("evidence_bank_cap", "evidence_bank", "bank_retrievals",
                  "_ctx_signature", "frozen_policy"):
            st.pop(k, None)
        b = MetargonAcademyAgent(seed=0)
        b.load_state(st)      # must not raise
        self.assertEqual(b.memory.evidence_bank_cap, 0)
        self.assertFalse(b.frozen_policy)


if __name__ == "__main__":
    unittest.main()
