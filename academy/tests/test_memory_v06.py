"""Tests for memory v0.6: cross-task atom transfer, context-conditioned
confidence, adaptive bank forgetting, per-episode probe bookkeeping,
frozen-evaluation integrity, and the corrected comparison-sign
convention."""

from __future__ import annotations

import unittest

from academy.agents.memory import (
    CONTRADICTED, VERIFIED, CompositionalMemory, consequence_key,
)
from academy.agents.metargon_agent import MetargonAcademyAgent
from academy.envs.gridworld import EnvConfig
from academy.envs.tasks import EpisodeSpec, difficulty_of, get_splits
from academy.learning.loop import run_one_episode
from academy.eval.metrics import paired_difference
from academy.tools.memory_experiment_v06 import (
    check_frozen_integrity, compare_delta, probe_relevance,
    rule_atoms, rule_clauses, summarize_probe_relevance,
)

ATOMS = frozenset({("sw", "S2", True)})
ATOMS2 = frozenset({("sw", "S1", False), ("empty", "P1")})


class TestAtomTransfer(unittest.TestCase):
    def test_verified_hypothesis_populates_profile(self):
        m = CompositionalMemory(atom_transfer=True)
        h = m._add(ATOMS, consequence_key("D1", True))
        for _ in range(4):
            m._support(h)
        stats = m._atom_stats.get("D1", {})
        atom = next(iter(ATOMS))
        self.assertIn(atom, stats)
        sup, con, n_ctxs, _ = stats[atom]
        self.assertGreaterEqual(sup, 1)
        self.assertEqual(con, 0)
        self.assertEqual(n_ctxs, 0)
        self.assertEqual(m.transfer_atoms("D1", 3), [atom])

    def test_contradicted_hypothesis_demotes(self):
        m = CompositionalMemory(atom_transfer=True)
        h = m._add(ATOMS, consequence_key("D1", True))
        for _ in range(4):
            m._support(h)
        for _ in range(4):
            m._contradict(h)
        stats = m._atom_stats.get("D1", {})
        atom = next(iter(ATOMS))
        self.assertIn(atom, stats)
        sup, con, _, _ = stats[atom]
        self.assertGreaterEqual(con, 1)
        self.assertEqual(m.transfer_atoms("D1", 3), [])

    def test_profile_bounded(self):
        m = CompositionalMemory(atom_transfer=True)
        m._ATOM_STATS_CAP = 4
        for i in range(8):
            h = m._add(frozenset({("sw", "S1", bool(i % 2))}),
                       consequence_key("D1", True))
            for _ in range(3):
                m._support(h)
        self.assertLessEqual(len(m._atom_stats.get("D1", {})), 4)

    def test_atom_stats_roundtrip(self):
        m = CompositionalMemory(atom_transfer=True)
        h = m._add(ATOMS, consequence_key("D1", True))
        for _ in range(4):
            m._support(h)
        m2 = CompositionalMemory(atom_transfer=True)
        m2.load_state(m.save_state())
        atom = next(iter(ATOMS))
        self.assertEqual(m2.transfer_atoms("D1", 3), [atom])

    def test_pair_stats_from_verified_conjunction(self):
        m = CompositionalMemory(atom_transfer=True)
        pair = frozenset({("sw", "S1", True), ("empty", "P1")})
        h = m._add(pair, consequence_key("D1", True))
        for _ in range(4):
            m._support(h)
        self.assertIn(pair, m._pair_stats.get("D1", {}))
        pe = m._pair_stats["D1"][pair]
        self.assertGreaterEqual(pe[0], 1)
        self.assertEqual(pe[1], 0)
        self.assertEqual(m.transfer_pairs("D1", k=4), [pair])

    def test_pair_roundtrip(self):
        m = CompositionalMemory(atom_transfer=True)
        pair = frozenset({("sw", "S1", True), ("empty", "P1")})
        h = m._add(pair, consequence_key("D1", True))
        for _ in range(4):
            m._support(h)
        m2 = CompositionalMemory(atom_transfer=True)
        m2.load_state(m.save_state())
        self.assertEqual(m2.transfer_pairs("D1", k=4), [pair])

    def test_event_level_atom_evidence(self):
        m = CompositionalMemory(atom_transfer=True)
        before = frozenset({("sw", "S1", True), ("empty", "P1")})
        m.observe_event(before, "D1", True, None)
        stats = m._atom_stats.get("D1", {})
        for atom in before:
            self.assertIn(atom, stats)
            self.assertGreaterEqual(stats[atom][0], 1)
        top = m.transfer_atoms("D1", k=3)
        self.assertEqual(set(top), set(before))

    def test_event_atom_conditional_probability_filter(self):
        m = CompositionalMemory(atom_transfer=True)
        # D1 opens 10 times; atom A present at all 10 opens, atom B at 2.
        a = ("sw", "S1", True)
        b = ("sw", "S2", True)
        for i in range(10):
            before = frozenset({a}) | (frozenset({b}) if i < 2 else frozenset())
            m.observe_event(before, "D1", True, None)
        # nonevents: default-state atom C present at all 20 closed steps,
        # atom A only at 2 (so A has lift, C does not).
        c = ("empty", "P1")
        for i in range(20):
            atoms = frozenset({c}) | (frozenset({a}) if i < 2 else frozenset())
            m.observe_nonevent(atoms, "D1", True)
        top = m.transfer_atoms("D1", k=6)
        self.assertIn(a, top)
        self.assertNotIn(b, top)
        self.assertNotIn(c, top)

    def test_event_atom_evidence_off_by_default(self):
        m = CompositionalMemory()      # atom_transfer off
        before = frozenset({("sw", "S1", True)})
        m.observe_event(before, "D1", True, None)
        self.assertEqual(m._atom_stats, {})

    def test_pair_self_contradictory_excluded(self):
        m = CompositionalMemory(atom_transfer=True)
        bad = frozenset({("sw", "S1", True), ("sw", "S1", False)})
        h = m._add(bad, consequence_key("D1", True))
        for _ in range(4):
            m._support(h)
        self.assertEqual(m.transfer_pairs("D1", k=4), [])


class TestBankRefresh(unittest.TestCase):
    def test_bank_entry_removed_on_contradiction(self):
        m = CompositionalMemory(evidence_bank_cap=16,
                                bank_refresh_on_contradiction=True)
        h = m._add(ATOMS, consequence_key("D1", True))
        for _ in range(4):
            m._support(h)
        self.assertEqual(len(m.evidence_bank), 1)
        for _ in range(4):
            m._contradict(h)
        self.assertEqual(m.evidence_bank, [])
        self.assertEqual(h.status, CONTRADICTED)

    def test_bank_kept_without_refresh_flag(self):
        m = CompositionalMemory(evidence_bank_cap=16)
        h = m._add(ATOMS, consequence_key("D1", True))
        for _ in range(4):
            m._support(h)
        for _ in range(4):
            m._contradict(h)
        self.assertEqual(len(m.evidence_bank), 1)


class TestEffectiveConfidence(unittest.TestCase):
    def test_global_fallback(self):
        m = CompositionalMemory()
        h = m._add(ATOMS, consequence_key("D1", True))
        for _ in range(4):
            m._support(h)
        self.assertAlmostEqual(m.effective_confidence(h), 5.0 / 6.0)

    def test_new_context_zero_local_evidence(self):
        m = CompositionalMemory(context_scoped=True)
        h = m._add(ATOMS, consequence_key("D1", True))
        m.set_context((0, 0, "none"))
        for _ in range(4):
            m._support(h)
        self.assertAlmostEqual(m.effective_confidence(h), 5.0 / 6.0)
        m.set_context((5, 5, "A"))       # new task context
        self.assertEqual(m.effective_confidence(h), 0.0)


class TestProbeReset(unittest.TestCase):
    def test_episode_reset_clears_tried(self):
        a = MetargonAcademyAgent(seed=0, probe_reset_per_episode=True)
        from academy.envs.gridworld import TOPOLOGIES
        geom = TOPOLOGIES["TP1"]
        a.tried = {"D1": {frozenset({("sw", "S1", True)})},
                    "D2": set()}
        a.pending_probe["D1"] = (frozenset({("sw", "S1", True)}),
                                  a.memory.step + 5, "hypothesis")
        spec = EpisodeSpec(config=EnvConfig(topology="TP1",
                                            d1_rule=ATOMS,
                                            d2_rule=ATOMS2,
                                            d1_delay=False,
                                            d2_delay=False),
                           max_steps=60, noise_q=0.0,
                           spec_seed=1, pool_tag="train",
                           difficulty=difficulty_of(EnvConfig(
                               topology="TP1", d1_rule=ATOMS,
                               d2_rule=ATOMS2,
                               d1_delay=False, d2_delay=False)))
        a.episode_start(spec, geom)
        self.assertEqual(a.tried["D1"], set())
        self.assertEqual(a.pending_probe, {})

    def test_no_reset_by_default(self):
        a = MetargonAcademyAgent(seed=0)
        from academy.envs.gridworld import TOPOLOGIES
        geom = TOPOLOGIES["TP1"]
        a.tried = {"D1": {frozenset({("sw", "S1", True)})},
                    "D2": set()}
        spec = EpisodeSpec(config=EnvConfig(topology="TP1",
                                            d1_rule=ATOMS,
                                            d2_rule=ATOMS2,
                                            d1_delay=False,
                                            d2_delay=False),
                           max_steps=60, noise_q=0.0,
                           spec_seed=1, pool_tag="train",
                           difficulty=difficulty_of(EnvConfig(
                               topology="TP1", d1_rule=ATOMS,
                               d2_rule=ATOMS2,
                               d1_delay=False, d2_delay=False)))
        a.episode_start(spec, geom)
        self.assertEqual(a.tried["D1"],
                         {frozenset({("sw", "S1", True)})})


class TestCheckpointFlags(unittest.TestCase):
    def test_agent_flags_roundtrip(self):
        a = MetargonAcademyAgent(
            seed=1, memory_atom_transfer=True,
            memory_bank_refresh_on_contradiction=True,
            probe_reset_per_episode=True,
            probe_min_effective_confidence=0.5,
            probe_adaptive=True, probe_bank_reuse=True)
        st = a.save_state()
        b = MetargonAcademyAgent(seed=1)
        b.load_state(st)
        self.assertTrue(b.memory_atom_transfer)
        self.assertTrue(b.memory_bank_refresh_on_contradiction)
        self.assertTrue(b.probe_reset_per_episode)
        self.assertEqual(b.probe_min_effective_confidence, 0.5)
        self.assertTrue(b.probe_adaptive)
        self.assertTrue(b.probe_bank_reuse)

    def test_backward_compat_missing_flag_keys(self):
        a = MetargonAcademyAgent(seed=0)
        st = a.save_state()
        for k in ("probe_adaptive", "probe_episode_local",
                  "protect_episode_local", "memory_atom_transfer",
                  "memory_bank_refresh_on_contradiction",
                  "probe_reset_per_episode",
                  "probe_min_effective_confidence",
                  "probe_bank_reuse"):
            st.pop(k, None)
        st["memory"].pop("atom_transfer", None)
        st["memory"].pop("bank_refresh_on_contradiction", None)
        st["memory"].pop("atom_stats", None)
        st["memory"].pop("atom_ctxs", None)
        b = MetargonAcademyAgent(seed=0)
        b.load_state(st)      # must not raise
        self.assertFalse(b.probe_reset_per_episode)
        self.assertFalse(b.memory.atom_transfer)


class TestFrozenIntegrity(unittest.TestCase):
    def test_frozen_integrity_ok_when_clean(self):
        a = MetargonAcademyAgent(seed=0)
        a.frozen_policy = True
        check_frozen_integrity(a, a.memory.generated, a.memory.step,
                               a.steps, "test")   # must not raise

    def test_frozen_integrity_detects_leak(self):
        a = MetargonAcademyAgent(seed=0)
        with self.assertRaises(AssertionError):
            check_frozen_integrity(a, a.memory.generated - 1,
                                   a.memory.step, a.steps, "test")


class TestComparisonSign(unittest.TestCase):
    def test_paired_sign_convention(self):
        a = [0.6, 0.7, 0.8]
        b = [0.4, 0.5, 0.6]
        d_ab = compare_delta(a, b, seed=7)
        d_ba = compare_delta(b, a, seed=7)
        self.assertGreater(d_ab["mean"], 0.0)
        self.assertLess(d_ba["mean"], 0.0)
        self.assertAlmostEqual(d_ab["mean"], -d_ba["mean"], places=6)
        # matches the documented paired_difference order
        self.assertAlmostEqual(d_ab["mean"],
                               paired_difference(a, b, seed=7)["mean"])


class TestRelevanceHelpers(unittest.TestCase):
    def test_rule_atoms_and_clauses(self):
        conj = frozenset({("sw", "S1", True), ("empty", "P1")})
        disj = (frozenset({("sw", "S2", True)}),
                frozenset({("empty", "P2")}))
        self.assertEqual(rule_atoms(conj), conj)
        self.assertEqual(rule_clauses(conj), [conj])
        self.assertEqual(len(rule_clauses(disj)), 2)
        self.assertEqual(rule_atoms(disj),
                         frozenset({("sw", "S2", True),
                                    ("empty", "P2")}))

    def test_probe_relevance_modes(self):
        rule = frozenset({("sw", "S1", True), ("empty", "P1")})
        self.assertEqual(probe_relevance(rule, rule)["exact"], 1)
        partial = frozenset({("sw", "S1", True)})
        self.assertEqual(probe_relevance(partial, rule)["partial"], 1)
        irr = frozenset({("sw", "S2", True)})
        self.assertEqual(probe_relevance(irr, rule)["irrelevant"], 1)

    def test_summarize_probe_relevance(self):
        config = EnvConfig(topology="TP1", d1_rule=ATOMS,
                           d2_rule=ATOMS2, d1_delay=False,
                           d2_delay=False)
        planned = [["D1", sorted(ATOMS), "hypothesis"],
                   ["D1", [["sw", "S2", True]], "atom_single"],
                   ["D1", [["sw", "S1", True]], "novel_single"]]
        s = summarize_probe_relevance(planned, config)
        self.assertEqual(s["n"], 3)
        self.assertEqual(s["exact"], 2)      # both match the true D1 rule
        self.assertEqual(s["partial"], 0)
        self.assertEqual(s["irrelevant"], 1)
        self.assertEqual(s["by_source"]["hypothesis"]["exact"], 1)


class TestAgentAtomTransferIntegration(unittest.TestCase):
    def test_agent_trains_and_transfers(self):
        splits = get_splits()
        import random
        rng = random.Random(0)
        a = MetargonAcademyAgent(
            seed=0, memory_atom_transfer=True,
            probe_reset_per_episode=True,
            probe_min_effective_confidence=0.5)
        for i in range(6):
            cfg = rng.choice(splits["train"])
            spec = EpisodeSpec(config=cfg, max_steps=60, noise_q=0.0,
                               spec_seed=1000 + i, pool_tag="train",
                               difficulty=difficulty_of(cfg))
            run_one_episode(a, spec)
        # profile exists and is checkpointed
        self.assertGreater(a.memory.atom_profile_size(), 0)
        st = a.save_state()
        b = MetargonAcademyAgent(seed=0, memory_atom_transfer=True)
        b.load_state(st)
        self.assertEqual(b.memory.atom_profile_size(),
                         a.memory.atom_profile_size())


if __name__ == "__main__":
    unittest.main()
