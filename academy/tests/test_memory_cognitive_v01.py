"""Tests for Cognitive Memory v0.1 (CM v0.1): the reliability-based
retrieval layer.

CM v0.1 separates *historical verification* from *current-context
applicability*: retrieval priority is driven by a Laplace-smoothed
support proportion over the active evidence basis (per-epoch when
``episode_scoped``), with history contributing only a bounded bonus.
Repeated prediction failures (completed contradiction windows) demote a
hypothesis without destroying its evidence, and fresh support re-promotes
it. All new state is checkpoint-compatible (defaults on load).
"""

from __future__ import annotations

import unittest

from academy.agents.memory import (
    CONTRADICTED, HYPOTHESIS, VERIFIED, CognitiveMemoryV01,
    CompositionalMemory, consequence_key,
)
from academy.agents.metargon_agent import MetargonAcademyAgent
from academy.envs.gridworld import EnvConfig, TOPOLOGIES
from academy.envs.tasks import EpisodeSpec, difficulty_of

R1 = frozenset({("sw", "S2", True)})
R2 = frozenset({("sw", "S1", False)})
D1_OPEN = consequence_key("D1", True)


def _cognitive(episode_scoped=True, **kw):
    kw.setdefault("cognitive_reliability", True)
    kw.setdefault("episode_scoped", episode_scoped)
    return CompositionalMemory(verify_support=3, max_contradict=2, **kw)


def _verify_in(m, h, epochs):
    for ep in epochs:
        m.begin_episode(ep)
        for _ in range(3):
            m._support(h)


class TestCognitiveReliability(unittest.TestCase):
    def test_off_by_default(self):
        m = CompositionalMemory()
        self.assertFalse(m.cognitive_reliability)
        # default sort path is confidence-based, untouched
        h = m._add(R1, D1_OPEN)
        for _ in range(3):
            m._support(h)
        q = m.query("D1", opens=True)
        self.assertEqual(q[0], h)

    def test_reliability_neutral_at_epoch_start_and_tracks_evidence(self):
        m = _cognitive()
        h = m._add(R1, D1_OPEN)
        m.begin_episode(0)
        self.assertAlmostEqual(m._cognitive_reliability(h), 0.5)
        m._support(h)
        self.assertAlmostEqual(m._cognitive_reliability(h), 2.0 / 3.0)
        for _ in range(2):
            m._support(h)
        self.assertAlmostEqual(m._cognitive_reliability(h), 4.0 / 5.0)
        for _ in range(3):
            m._contradict(h)
        self.assertAlmostEqual(m._cognitive_reliability(h), 4.0 / 8.0)

    def test_reliability_uses_active_epoch_counts(self):
        m = _cognitive()
        h = m._add(R1, D1_OPEN)
        _verify_in(m, h, (0, 1))
        m.begin_episode(2)          # new task: no local evidence yet
        self.assertAlmostEqual(m._cognitive_reliability(h), 0.5)
        m._contradict(h)            # one failure in the active epoch
        self.assertAlmostEqual(m._cognitive_reliability(h), 1.0 / 3.0)

    def test_repeated_failure_streak(self):
        m = _cognitive()
        h = m._add(R1, D1_OPEN)
        m.begin_episode(0)
        self.assertEqual(h.consecutive_failures, 0)
        m._contradict(h)
        m._contradict(h)
        self.assertEqual(h.consecutive_failures, 2)
        m._support(h)               # support resets the streak
        self.assertEqual(h.consecutive_failures, 0)
        m._contradict(h)
        m.begin_episode(1)          # task boundary resets current failures
        self.assertEqual(h.consecutive_failures, 0)


class TestCognitiveRetrievalOrder(unittest.TestCase):
    def test_historical_verification_does_not_dominate_new_task(self):
        m = _cognitive()
        h = m._add(R1, D1_OPEN)
        n = m._add(R2, D1_OPEN)
        _verify_in(m, h, (0, 1))   # h verified in two past epochs
        m.begin_episode(2)          # new task: h untested, n untouched
        q = m.query("D1", opens=True)
        # h gets a bounded history bonus but the gap to neutral is small
        self.assertIn(h, q)
        self.assertIn(n, q)
        self.assertGreater(m._cognitive_retrieval_key(h),
                           m._cognitive_retrieval_key(n))
        # ... and an active-epoch verified hypothesis dominates both
        h2 = m._add(frozenset({("sw", "S2", False)}), D1_OPEN)
        for _ in range(3):
            m._support(h2)
        q2 = m.query("D1", opens=True)
        self.assertEqual(q2[0], h2)

    def test_single_failure_tolerated_above_neutral(self):
        # a transient failure (e.g. a delay artifact) must not demote a
        # verified hypothesis below neutral exploration
        m = _cognitive()
        h = m._add(R1, D1_OPEN)
        n = m._add(R2, D1_OPEN)
        _verify_in(m, h, (0, 1))
        m.begin_episode(2)
        self.assertEqual(m.query("D1", opens=True)[0], h)
        for _ in range(3):
            m.observe_nonevent(R1, "D1", True)
        self.assertEqual(h.consecutive_failures, 1)
        q = m.query("D1", opens=True)
        self.assertEqual(q[0], h)          # still ranked above neutral

    def test_repeated_failures_demote_stale_below_neutral(self):
        m = _cognitive()
        h = m._add(R1, D1_OPEN)
        n = m._add(R2, D1_OPEN)
        _verify_in(m, h, (0, 1))
        m.begin_episode(2)
        self.assertEqual(m.query("D1", opens=True)[0], h)
        # two completed failure windows: h is demoted below neutral n
        for _ in range(6):
            m.observe_nonevent(R1, "D1", True)
        self.assertEqual(h.consecutive_failures, 2)
        q = m.query("D1", opens=True)
        self.assertEqual(q[0], n)
        # but h is not destroyed: evidence and history are intact
        self.assertIn(h, m.hypotheses)
        self.assertEqual(h.verified_epoch_count, 2)
        self.assertEqual(h.con, 2)
        self.assertEqual(h.sup, 6)

    def test_fresh_support_revives_previously_failing_hypothesis(self):
        m = _cognitive()
        h = m._add(R1, D1_OPEN)
        n = m._add(R2, D1_OPEN)
        _verify_in(m, h, (0, 1))
        m.begin_episode(2)
        self.assertEqual(m.query("D1", opens=True)[0], h)
        for _ in range(6):
            m.observe_nonevent(R1, "D1", True)
        # h has failed twice: neutral n now outranks it (not destroyed)
        self.assertEqual(m.query("D1", opens=True)[0], n)
        self.assertIn(h, m.hypotheses)
        # the rule reappears in the next task: h recovers and dominates
        m.begin_episode(3)
        self.assertEqual(h.consecutive_failures, 0)
        self.assertEqual(h.status, HYPOTHESIS)
        for _ in range(3):
            m._support(h)
        self.assertEqual(h.status, VERIFIED)
        self.assertEqual(m.query("D1", opens=True)[0], h)

    def test_query_order_deterministic(self):
        m = _cognitive()
        h = m._add(R1, D1_OPEN)
        n = m._add(R2, D1_OPEN)
        m.begin_episode(0)
        q1 = m.query("D1", opens=True)
        q2 = m.query("D1", opens=True)
        self.assertEqual([x.h_id for x in q1],
                         [x.h_id for x in q2])

    def test_effective_confidence_returns_reliability(self):
        m = _cognitive()
        h = m._add(R1, D1_OPEN)
        m.begin_episode(0)
        self.assertAlmostEqual(m.effective_confidence(h), 0.5)
        m._support(h)
        self.assertAlmostEqual(m.effective_confidence(h), 2.0 / 3.0)


class TestCognitiveCheckpointCompat(unittest.TestCase):
    def test_legacy_state_without_cognitive_fields_loads(self):
        m = _cognitive()
        h = m._add(R1, D1_OPEN)
        _verify_in(m, h, (0, 1))
        m.begin_episode(1)
        m._contradict(h)
        state = m.save_state()
        # strip every CM v0.1 field -> a v0.7-era checkpoint
        for hd in state["hypotheses"]:
            hd.pop("consecutive_failures", None)
        for k in ("cognitive_reliability", "cognitive_history_cap",
                  "cognitive_failure_penalty"):
            state.pop(k, None)
        m2 = CompositionalMemory(episode_scoped=True)
        m2.load_state(state)
        self.assertFalse(m2.cognitive_reliability)
        self.assertEqual(m2.cognitive_history_cap, 5)
        self.assertEqual(m2.cognitive_failure_penalty, 8.0)
        h2 = m2.hypotheses[0]
        self.assertEqual(h2.consecutive_failures, 0)
        self.assertEqual(h2.verified_epochs, h.verified_epochs)
        self.assertEqual(h2.epoch_support(0), h.epoch_support(0))
        self.assertEqual(h2.epoch_contradiction(1),
                         h.epoch_contradiction(1))

    def test_roundtrip_preserves_cognitive_state(self):
        m = _cognitive()
        h = m._add(R1, D1_OPEN)
        m.begin_episode(0)
        m._support(h)
        m._contradict(h)
        m2 = CompositionalMemory()
        m2.load_state(m.save_state())
        self.assertTrue(m2.cognitive_reliability)
        self.assertTrue(m2.episode_scoped)
        self.assertEqual(m2.cognitive_history_cap, m.cognitive_history_cap)
        self.assertEqual(m2.hypotheses[0].consecutive_failures, 1)
        self.assertEqual(m2._current_epoch, m._current_epoch)

    def test_saved_state_is_a_superset_of_legacy_format(self):
        m = _cognitive()
        h = m._add(R1, D1_OPEN)
        m.begin_episode(0)
        m._support(h)
        state = m.save_state()
        # every field the v0.7 loader required is still present
        for key in ("hypotheses", "next_id", "step", "evictions",
                    "generated", "continuous_true", "dead",
                    "sup_timeline", "con_timeline", "evidence_bank",
                    "atom_stats", "atom_ctxs", "pair_stats",
                    "door_event_count", "atom_event_support",
                    "atom_nonevent_support", "door_nonevent_count",
                    "current_epoch", "current_ctx", "max_hypotheses",
                    "verify_support", "max_contradict",
                    "predict_threshold", "contradict_window",
                    "candidates_per_event", "pair_base",
                    "recycle_contradicted", "contradiction_ratio",
                    "evidence_window", "dead_set_cap",
                    "context_scoped", "task_change_reset",
                    "context_gate_transfer", "evidence_bank_cap",
                    "atom_transfer", "bank_refresh_on_contradiction",
                    "episode_scoped",
                    "min_verified_epochs_for_transfer"):
            self.assertIn(key, state, key)
        hd = state["hypotheses"][0]
        for key in ("antecedent", "consequence", "sup", "con", "status",
                    "created_step", "last_update", "h_id",
                    "sup_by_ctx", "con_by_ctx", "sup_by_epoch",
                    "con_by_epoch", "verified_epochs"):
            self.assertIn(key, hd, key)


class TestCognitiveMemoryAgentIntegration(unittest.TestCase):
    def _spec(self):
        return EpisodeSpec(
            config=EnvConfig(topology="TP1", d1_rule=R1, d2_rule=R2,
                             d1_delay=False, d2_delay=False),
            max_steps=60, noise_q=0.0, spec_seed=1, pool_tag="train",
            difficulty=difficulty_of(EnvConfig(
                topology="TP1", d1_rule=R1, d2_rule=R2,
                d1_delay=False, d2_delay=False)))

    def test_factory_constructs_cognitive_memory(self):
        a = MetargonAcademyAgent(
            seed=0, memory_factory=CognitiveMemoryV01,
            memory_episode_scoped=True, probe_reset_per_episode=True)
        self.assertIsInstance(a.memory, CognitiveMemoryV01)
        self.assertTrue(a.memory.cognitive_reliability)
        self.assertTrue(a.memory.episode_scoped)
        a.episode_start(self._spec(), TOPOLOGIES["TP1"])
        self.assertIsNotNone(a.memory._current_epoch)

    def test_agent_checkpoint_roundtrip_cognitive(self):
        a = MetargonAcademyAgent(
            seed=1, memory_factory=CognitiveMemoryV01,
            memory_episode_scoped=True, probe_reset_per_episode=True)
        a.episode_start(self._spec(), TOPOLOGIES["TP1"])
        st = a.save_state()
        b = MetargonAcademyAgent(
            seed=1, memory_factory=CognitiveMemoryV01,
            memory_episode_scoped=True, probe_reset_per_episode=True)
        b.load_state(st)
        self.assertTrue(b.memory.cognitive_reliability)
        self.assertTrue(b.memory.episode_scoped)
        self.assertEqual(b.memory._current_epoch,
                         a.memory._current_epoch)

    def test_cognitive_diagnostics_fields_present(self):
        a = MetargonAcademyAgent(seed=0, memory_factory=CognitiveMemoryV01,
                                 memory_episode_scoped=True)
        d = a.memory.diagnostics()
        for k in ("cognitive_reliability", "cognitive_reliability_mean",
                  "cognitive_high_reliability", "cognitive_under_attack",
                  "cognitive_streak_max"):
            self.assertIn(k, d)


if __name__ == "__main__":
    unittest.main()
