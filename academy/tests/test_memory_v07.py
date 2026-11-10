"""Tests for memory v0.7: episode-scoped evidence, cross-epoch
contradiction protection, uncertainty-aware retrieval, and memory-to-
action attribution counters."""

from __future__ import annotations

import unittest

from academy.agents.memory import (
    CONTRADICTED, HYPOTHESIS, VERIFIED, CompositionalMemory, consequence_key,
)
from academy.agents.metargon_agent import MetargonAcademyAgent
from academy.envs.gridworld import EnvConfig, TOPOLOGIES
from academy.envs.tasks import EpisodeSpec, difficulty_of

R1 = frozenset({("sw", "S2", True)})
R2 = frozenset({("sw", "S1", False)})
D1_OPEN = consequence_key("D1", True)


def _spec():
    return EpisodeSpec(config=EnvConfig(topology="TP1", d1_rule=R1,
                                        d2_rule=R2,
                                        d1_delay=False, d2_delay=False),
                       max_steps=60, noise_q=0.0, spec_seed=1,
                       pool_tag="train",
                       difficulty=difficulty_of(EnvConfig(
                           topology="TP1", d1_rule=R1, d2_rule=R2,
                           d1_delay=False, d2_delay=False)))


class TestEpisodeScopedEvidence(unittest.TestCase):
    def test_epoch_contradiction_does_not_destroy_verified(self):
        m = CompositionalMemory(episode_scoped=True, verify_support=3,
                                max_contradict=2)
        h = m._add(R1, D1_OPEN)
        m.begin_episode(0)
        for _ in range(3):
            m._support(h)
        self.assertEqual(h.status, VERIFIED)
        self.assertEqual(h.verified_epochs, [0])
        # different mechanism in episode 1: same observable context, but
        # the door does not open -> counts as contradiction *in this epoch*
        m.begin_episode(1)
        for _ in range(9):
            m.observe_nonevent(R1, "D1", True)
        self.assertGreaterEqual(h.epoch_contradiction(1), 2)
        self.assertEqual(h.status, CONTRADICTED)   # demoted in epoch 1
        # the hypothesis is NOT destroyed: at the next task boundary the
        # status is recomputed against the fresh epoch (neutral), so it
        # becomes retrievable again as a cross-epoch transfer candidate.
        m.begin_episode(2)
        self.assertEqual(h.status, HYPOTHESIS)
        self.assertIn(h, m.query("D1", opens=True))
        # same rule reappears in episode 2: hypothesis can be re-verified
        for _ in range(3):
            m._support(h)
        self.assertEqual(h.status, VERIFIED)
        self.assertEqual(h.verified_epochs, [0, 2])

    def test_global_mode_survives_contradiction_once(self):
        # without episode scoping the same contradiction is global
        m = CompositionalMemory(verify_support=3, max_contradict=2)
        h = m._add(R1, D1_OPEN)
        for _ in range(3):
            m._support(h)
        self.assertEqual(h.status, VERIFIED)
        for _ in range(9):
            m.observe_nonevent(R1, "D1", True)
        self.assertEqual(h.status, CONTRADICTED)
        self.assertFalse(h.usable)

    def test_epoch_evidence_recorded_per_epoch(self):
        m = CompositionalMemory(episode_scoped=True)
        h = m._add(R1, D1_OPEN)
        m.begin_episode(5)
        m._support(h)
        m.begin_episode(9)
        m._support(h)
        self.assertEqual(h.epoch_support(5), 1)
        self.assertEqual(h.epoch_support(9), 1)
        self.assertEqual(h.epoch_support(7), 0)

    def test_checkpoint_roundtrip_epoch_fields(self):
        m = CompositionalMemory(episode_scoped=True)
        h = m._add(R1, D1_OPEN)
        m.begin_episode(0)
        for _ in range(3):
            m._support(h)
        m.begin_episode(1)
        m._contradict(h)
        m2 = CompositionalMemory(episode_scoped=True)
        m2.load_state(m.save_state())
        h2 = m2.hypotheses[0]
        self.assertEqual(h2.verified_epochs, h.verified_epochs)
        self.assertEqual(h2.epoch_support(0), h.epoch_support(0))
        self.assertEqual(h2.epoch_contradiction(1),
                         h.epoch_contradiction(1))
        self.assertEqual(m2._current_epoch, m._current_epoch)

    def test_off_by_default(self):
        m = CompositionalMemory()
        self.assertFalse(m.episode_scoped)
        self.assertIsNone(m._current_epoch)


class TestCrossEpochRetrieval(unittest.TestCase):
    def test_untested_cross_epoch_verified_retrievable(self):
        m = CompositionalMemory(episode_scoped=True, verify_support=3,
                                min_verified_epochs_for_transfer=2)
        h = m._add(R1, D1_OPEN)
        for ep in (0, 1):
            m.begin_episode(ep)
            for _ in range(3):
                m._support(h)
        self.assertEqual(h.verified_epoch_count, 2)
        m.begin_episode(3)          # new task, hypothesis untested
        q = m.query("D1", opens=True)
        self.assertIn(h, q)
        # cross-epoch candidate is retrievable but ranked below an
        # active-epoch verified hypothesis
        h2 = m._add(R2, D1_OPEN)
        m.begin_episode(3)
        for _ in range(3):
            m._support(h2)
        q = m.query("D1", opens=True)
        self.assertEqual(q[0], h2)          # active-epoch verified first
        self.assertIn(h, q)

    def test_retrieval_key_priority(self):
        m = CompositionalMemory(episode_scoped=True, verify_support=3,
                                min_verified_epochs_for_transfer=2)
        h_act = m._add(R1, D1_OPEN)
        h_cross = m._add(R2, D1_OPEN)
        m.begin_episode(0)
        for _ in range(3):
            m._support(h_act)
        m.begin_episode(1)
        for _ in range(3):
            m._support(h_cross)
        m.begin_episode(2)
        # active-epoch verified hypothesis ranks above an untested
        # cross-epoch candidate
        for _ in range(3):
            m._support(h_act)
        self.assertGreater(m._retrieval_key(h_act), m._retrieval_key(h_cross))


class TestAgentEpisodeHook(unittest.TestCase):
    def test_episode_start_declares_epoch(self):
        a = MetargonAcademyAgent(seed=0, memory_episode_scoped=True)
        self.assertEqual(a.memory._current_epoch, None)
        a.episode_start(_spec(), TOPOLOGIES["TP1"])
        self.assertEqual(a.memory._current_epoch, 0)
        a.finish_episode(True)
        a.episode_start(_spec(), TOPOLOGIES["TP1"])
        self.assertEqual(a.memory._current_epoch, 1)

    def test_agent_flags_roundtrip(self):
        a = MetargonAcademyAgent(
            seed=1, memory_episode_scoped=True,
            memory_min_verified_epochs_for_transfer=3)
        st = a.save_state()
        b = MetargonAcademyAgent(seed=1)
        b.load_state(st)
        self.assertTrue(b.memory_episode_scoped)
        self.assertTrue(b.memory.episode_scoped)
        self.assertEqual(b.memory_min_verified_epochs_for_transfer, 3)

    def test_attribution_trace_fields_present(self):
        a = MetargonAcademyAgent(seed=0, memory_episode_scoped=True)
        for k in ("hypothesis_retrieved", "cross_epoch_transfer_retrieved",
                  "memory_actions", "memory_correct", "memory_incorrect"):
            self.assertIn(k, a.trace)
            self.assertEqual(a.trace[k], 0)


class TestAttributionMetrics(unittest.TestCase):
    def test_attribution_metrics(self):
        from academy.tools.memory_experiment_v07 import attribution_metrics
        rows = [
            {"success_rate": 1.0, "trace": {
                "source_hypothesis": 1, "memory_actions": 1,
                "memory_correct": 1, "memory_incorrect": 0,
                "cross_epoch_transfer_retrieved": 2}},
            {"success_rate": 0.0, "trace": {
                "source_hypothesis": 1, "memory_actions": 1,
                "memory_correct": 0, "memory_incorrect": 1,
                "cross_epoch_transfer_retrieved": 0}},
            {"success_rate": 0.5, "trace": {
                "source_hypothesis": 0, "memory_actions": 0,
                "memory_correct": 0, "memory_incorrect": 0,
                "cross_epoch_transfer_retrieved": 0}},
        ]
        m = attribution_metrics(rows)
        self.assertEqual(m["memory_actions"], 2)
        self.assertEqual(m["memory_correct"], 1)
        self.assertEqual(m["memory_incorrect"], 1)
        self.assertEqual(m["cross_epoch_transfer_retrieved"], 2)
        self.assertEqual(m["rows_with_memory"], 2)
        self.assertEqual(m["rows_without_memory"], 1)
        self.assertEqual(m["success_with_memory"], 0.5)
        self.assertEqual(m["success_without_memory"], 0.5)
        self.assertEqual(m["memory_attributable_improvement"], 0.0)

    def test_attribution_all_without_memory(self):
        from academy.tools.memory_experiment_v07 import attribution_metrics
        rows = [{"success_rate": 0.0, "trace": {}},
                {"success_rate": 1.0, "trace": {}}]
        m = attribution_metrics(rows)
        self.assertIsNone(m["memory_attributable_improvement"])
        self.assertEqual(m["memory_actions"], 0)


if __name__ == "__main__":
    unittest.main()
