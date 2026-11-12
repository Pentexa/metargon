"""Tests for Cognitive Memory v0.2 (CM v0.2): protect exploration.

v0.2 does NOT re-rank hypotheses (PR #50's negative finding). It adds a
minimal, context-sensitive eligibility filter + cap on hypothesis *reuse*
so that exploration slots are preserved:

- a hypothesis is reusable only after it has ``protection_min_active_support``
  support events in the active evidence basis (epoch > context > cumulative);
- hypotheses under repeated-failure attack (> ``protection_max_failures``
  completed failure windows) are excluded from reuse (evidence kept);
- at most ``protection_max_reuse_per_door`` reusable hypotheses are returned
  per door for probing (after the baseline sort), protecting exploration.

All flags default off; legacy checkpoints load unchanged.
"""

from __future__ import annotations

import unittest

from academy.agents.memory import (
    VERIFIED, CognitiveMemoryV02, CompositionalMemory, consequence_key,
)
from academy.agents.metargon_agent import MetargonAcademyAgent
from academy.envs.gridworld import EnvConfig, TOPOLOGIES
from academy.envs.tasks import EpisodeSpec, difficulty_of

R1 = frozenset({("sw", "S2", True)})
R2 = frozenset({("sw", "S1", False)})
D1_OPEN = consequence_key("D1", True)


def _v02(**kw):
    kw.setdefault("protect_exploration", True)
    kw.setdefault("episode_scoped", True)
    return CompositionalMemory(verify_support=3, max_contradict=2, **kw)


class TestExplorationProtection(unittest.TestCase):
    def test_off_by_default(self):
        m = CompositionalMemory()
        self.assertFalse(m.protect_exploration)
        h = m._add(R1, D1_OPEN)
        m.begin_episode(0)
        # default path returns usable hypotheses regardless of local support
        self.assertEqual(m.query("D1", opens=True), [h])

    def test_no_local_evidence_blocks_reuse(self):
        m = _v02()
        h = m._add(R1, D1_OPEN)      # verified in past epochs
        m.begin_episode(0)
        m._support(h)
        m.begin_episode(1)           # new task, no local evidence yet
        self.assertEqual(m.query("D1", opens=True), [])
        m._support(h)                # one active-epoch support -> eligible
        self.assertEqual(m.query("D1", opens=True), [h])

    def test_cap_limits_reuse_per_door(self):
        m = _v02()
        h = m._add(R1, D1_OPEN)
        n = m._add(R2, D1_OPEN)
        m.begin_episode(0)
        m._support(h)
        m._support(n)
        q = m.query("D1", opens=True)
        self.assertEqual(len(q), 1)          # cap default 1
        # baseline order preserved: the single returned one is the top by
        # the epoch-reset retrieval key, not by insertion
        self.assertIn(q[0], (h, n))
        # protection path (opens=None) is untouched by filter/cap
        qa = m.query("D1")
        self.assertEqual({x.h_id for x in qa}, {h.h_id, n.h_id})

    def test_repeated_failures_exclude_from_reuse_only(self):
        m = _v02()
        h = m._add(R1, D1_OPEN)
        n = m._add(R2, D1_OPEN)
        m.begin_episode(0)
        m._support(h)
        m._support(n)
        for _ in range(2):
            m._contradict(n)         # 2 completed failure windows
        self.assertEqual(n.consecutive_failures, 2)
        q = m.query("D1", opens=True)
        self.assertEqual(q, [h])             # n excluded
        # evidence kept: n survives with intact counts
        self.assertIn(n, m.hypotheses)
        self.assertEqual(n.sup, 1)
        self.assertEqual(n.con, 2)
        # task boundary resets streaks -> n reusable again
        m.begin_episode(1)
        self.assertEqual(n.consecutive_failures, 0)
        m._support(n)
        self.assertIn(n, m.query("D1", opens=True))

    def test_support_resets_streak(self):
        m = _v02()
        h = m._add(R1, D1_OPEN)
        m.begin_episode(0)
        m._contradict(h)
        self.assertEqual(h.consecutive_failures, 1)
        m._support(h)
        self.assertEqual(h.consecutive_failures, 0)

    def test_eligibility_uses_active_epoch_not_history(self):
        m = _v02()
        h = m._add(R1, D1_OPEN)
        m.begin_episode(0)
        for _ in range(3):
            m._support(h)
        self.assertEqual(h.status, VERIFIED)
        m.begin_episode(1)           # new task: verified history, no local
        self.assertEqual(m.query("D1", opens=True), [])

    def test_ablated_gate_only_no_cap(self):
        m = _v02(protection_max_reuse_per_door=10)
        h = m._add(R1, D1_OPEN)
        n = m._add(R2, D1_OPEN)
        m.begin_episode(0)
        m._support(h)
        m._support(n)
        q = m.query("D1", opens=True)
        self.assertEqual({x.h_id for x in q}, {h.h_id, n.h_id})

    def test_trusted_transfer_tier(self):
        # hypotheses verified across >= K epochs are eligible as transfer
        # candidates even without active-epoch support (still capped)
        m = _v02(protection_allow_trusted_transfer=True,
                 min_verified_epochs_for_transfer=2)
        h = m._add(R1, D1_OPEN)
        for ep in (0, 1):
            m.begin_episode(ep)
            for _ in range(3):
                m._support(h)
        self.assertEqual(h.verified_epoch_count, 2)
        m.begin_episode(3)           # new task: no local evidence
        self.assertEqual(m.query("D1", opens=True), [h])   # trusted tier
        # ... but a failing streak excludes it
        for _ in range(2):
            m._contradict(h)
        self.assertEqual(m.query("D1", opens=True), [])
        m.begin_episode(4)           # boundary resets streak
        self.assertEqual(m.query("D1", opens=True), [h])
        # without the tier flag, no reuse (gate-only behavior)
        m2 = _v02()
        h2 = m2._add(R1, D1_OPEN)
        for ep in (0, 1):
            m2.begin_episode(ep)
            for _ in range(3):
                m2._support(h2)
        m2.begin_episode(3)
        self.assertEqual(m2.query("D1", opens=True), [])


class TestV02CheckpointCompat(unittest.TestCase):
    def test_legacy_state_without_v02_fields_loads(self):
        m = _v02()
        h = m._add(R1, D1_OPEN)
        m.begin_episode(0)
        m._support(h)
        m._contradict(h)
        state = m.save_state()
        for hd in state["hypotheses"]:
            hd.pop("consecutive_failures", None)
        for k in ("protect_exploration", "protection_min_active_support",
                  "protection_max_failures", "protection_max_reuse_per_door",
                  "protection_allow_trusted_transfer"):
            state.pop(k, None)
        m2 = CompositionalMemory(episode_scoped=True)
        m2.load_state(state)
        self.assertFalse(m2.protect_exploration)
        self.assertEqual(m2.protection_min_active_support, 1)
        self.assertEqual(m2.protection_max_failures, 1)
        self.assertEqual(m2.protection_max_reuse_per_door, 1)
        self.assertFalse(m2.protection_allow_trusted_transfer)
        self.assertEqual(m2.hypotheses[0].consecutive_failures, 0)
        self.assertEqual(m2.hypotheses[0].epoch_support(0),
                         h.epoch_support(0))

    def test_roundtrip_preserves_v02_state(self):
        m = _v02()
        h = m._add(R1, D1_OPEN)
        m.begin_episode(0)
        m._support(h)
        m._contradict(h)
        m2 = CompositionalMemory()
        m2.load_state(m.save_state())
        self.assertTrue(m2.protect_exploration)
        self.assertTrue(m2.episode_scoped)
        self.assertEqual(m2.protection_max_reuse_per_door, 1)
        self.assertEqual(m2.hypotheses[0].consecutive_failures, 1)

    def test_saved_state_is_superset_of_legacy(self):
        m = _v02()
        h = m._add(R1, D1_OPEN)
        m.begin_episode(0)
        m._support(h)
        state = m.save_state()
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


class TestV02AgentIntegration(unittest.TestCase):
    def _spec(self):
        return EpisodeSpec(
            config=EnvConfig(topology="TP1", d1_rule=R1, d2_rule=R2,
                             d1_delay=False, d2_delay=False),
            max_steps=60, noise_q=0.0, spec_seed=1, pool_tag="train",
            difficulty=difficulty_of(EnvConfig(
                topology="TP1", d1_rule=R1, d2_rule=R2,
                d1_delay=False, d2_delay=False)))

    def test_factory_constructs_v02_memory(self):
        a = MetargonAcademyAgent(
            seed=0, memory_factory=CognitiveMemoryV02,
            memory_episode_scoped=True, probe_reset_per_episode=True)
        self.assertIsInstance(a.memory, CognitiveMemoryV02)
        self.assertTrue(a.memory.protect_exploration)
        self.assertTrue(a.memory.episode_scoped)
        a.episode_start(self._spec(), TOPOLOGIES["TP1"])
        self.assertIsNotNone(a.memory._current_epoch)

    def test_agent_checkpoint_roundtrip_v02(self):
        a = MetargonAcademyAgent(
            seed=1, memory_factory=CognitiveMemoryV02,
            memory_episode_scoped=True, probe_reset_per_episode=True)
        a.episode_start(self._spec(), TOPOLOGIES["TP1"])
        st = a.save_state()
        b = MetargonAcademyAgent(
            seed=1, memory_factory=CognitiveMemoryV02,
            memory_episode_scoped=True, probe_reset_per_episode=True)
        b.load_state(st)
        self.assertTrue(b.memory.protect_exploration)
        self.assertEqual(b.memory._current_epoch, a.memory._current_epoch)

    def test_diagnostics_fields_present(self):
        a = MetargonAcademyAgent(seed=0, memory_factory=CognitiveMemoryV02,
                                 memory_episode_scoped=True)
        d = a.memory.diagnostics()
        for k in ("protect_exploration", "reuse_eligible",
                  "reuse_filtered", "reuse_filtered_streak"):
            self.assertIn(k, d)


if __name__ == "__main__":
    unittest.main()
