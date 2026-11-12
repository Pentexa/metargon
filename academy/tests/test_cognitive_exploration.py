"""Tests for cognitive exploration v0.1 (knowledge-gap probing, hold-
based probe attribution, stagnation recovery, decomposition, and the
probe-attempt instrumentation)."""

from __future__ import annotations

import unittest

from academy.agents.metargon_agent import MetargonAcademyAgent
from academy.agents.planner import maximal_feasible_subset
from academy.envs.gridworld import EnvConfig, TOPOLOGIES, make_ws

R1 = frozenset({("sw", "S2", True)})
R2 = frozenset({("sw", "S1", False)})
TP1 = TOPOLOGIES["TP1"]


def _agent(**kw):
    kw.setdefault("seed", 0)
    return MetargonAcademyAgent(**kw)


class TestBaselineUnaffected(unittest.TestCase):
    def test_all_flags_off_by_default(self):
        a = _agent()
        self.assertFalse(a.cognitive_exploration)
        self.assertFalse(a.cog_knowledge_gaps)
        self.assertFalse(a.cog_info_gain)
        self.assertFalse(a.cog_recovery)
        self.assertFalse(a.cog_decompose)
        self.assertFalse(a.cog_probe_hold)

    def test_master_switch_turns_subs_on(self):
        a = _agent(cognitive_exploration=True)
        self.assertTrue(a.cog_knowledge_gaps)
        self.assertTrue(a.cog_info_gain)
        self.assertTrue(a.cog_recovery)
        self.assertTrue(a.cog_decompose)
        self.assertTrue(a.cog_probe_hold)

    def test_explicit_subflag_overrides_master(self):
        a = _agent(cognitive_exploration=True, cog_recovery=False)
        self.assertFalse(a.cog_recovery)
        self.assertTrue(a.cog_knowledge_gaps)


class TestPlannerDecomposition(unittest.TestCase):
    def test_full_subset_keeps_all_atoms(self):
        belief = make_ws(TP1)
        # S2 and P1 are both in the start-side component (doors closed)
        target = frozenset({("sw", "S2", True), ("loadany", "P1")})
        sub = maximal_feasible_subset(belief, target, TP1)
        self.assertEqual(sub, target)

    def test_infeasible_atom_dropped(self):
        # S1 sits on the far side of a closed door from the start region:
        # it must be dropped, while a start-side atom stays
        belief = make_ws(TP1)          # doors closed
        target = frozenset({("sw", "S2", True), ("sw", "S1", True),
                            ("loadany", "P1")})
        sub = maximal_feasible_subset(belief, target, TP1)
        self.assertIn(("sw", "S2", True), sub)
        self.assertIn(("loadany", "P1"), sub)
        self.assertNotIn(("sw", "S1", True), sub)

    def test_empty_when_nothing_feasible(self):
        belief = make_ws(TP1)
        belief = belief._replace(pos=(5, 0))     # wall cell: no component
        sub = maximal_feasible_subset(
            belief, frozenset({("sw", "S1", True)}), TP1)
        self.assertEqual(sub, frozenset())


class TestKnowledgeGapModel(unittest.TestCase):
    def test_uncovered_starts_full(self):
        a = _agent(cognitive_exploration=True)
        unc = a._uncovered_atoms("D1")
        self.assertGreater(len(unc), 0)
        # no evidence yet -> every vocab atom is uncovered
        self.assertEqual(len(unc), len(a._local_vocab()))

    def test_evidence_covers_atoms(self):
        a = _agent(cognitive_exploration=True)
        a._note_local_evidence("D1", {("sw", "S2", True)}, opened=True)
        a._note_local_evidence("D1", {("sw", "S2", True)}, opened=True)
        a._note_local_evidence("D1", {("sw", "S2", True)}, opened=True)
        unc = a._uncovered_atoms("D1")
        self.assertNotIn(("sw", "S2", True), unc)

    def test_candidate_score_rewards_novelty(self):
        a = _agent(cognitive_exploration=True)
        # mark all but one atom covered
        for atom in a._local_vocab():
            if atom != ("sw", "S2", True):
                a._note_local_evidence("D1", {atom}, opened=True)
                a._note_local_evidence("D1", {atom}, opened=True)
        novel = a._candidate_score("D1", frozenset({("sw", "S2", True)}),
                                   1, "novel_single", 3)
        known = a._candidate_score("D1", frozenset({("sw", "S1", False)}),
                                   1, "novel_single", 3)
        self.assertGreater(novel, known)


class TestStagnationRecovery(unittest.TestCase):
    def test_recovery_plan_avoids_recent_states(self):
        a = _agent(cognitive_exploration=True)
        a.geom = TP1
        belief = make_ws(TP1)
        a.belief = belief
        a._recent_states.append(("stateA",))
        a._recent_states.append(("stateB",))
        plan = a._recovery_plan()
        self.assertIsNotNone(plan)
        self.assertTrue(plan)

    def test_stall_detected_on_invalid_streak(self):
        a = _agent(cognitive_exploration=True)
        a._invalid_streak = 3
        a._steps_since_state_change = 0
        self.assertTrue(a._detect_stall())

    def test_no_stall_while_holding(self):
        a = _agent(cognitive_exploration=True)
        a._invalid_streak = 3
        a._active_hold = {"door": "D1", "src": "hypothesis",
                          "held_steps": 0}
        self.assertFalse(a._detect_stall())


class TestHoldMechanism(unittest.TestCase):
    def _belief_with(self, atoms):
        from academy.envs.gridworld import make_ws
        b = make_ws(TP1)
        # S1 true
        if ("sw", "S1", True) in atoms:
            b = b._replace(switch_state={"S1": True, "S2": False})
        if ("sw", "S2", True) in atoms:
            b = b._replace(switch_state={"S1": False, "S2": True})
        return b

    def _agent_ready(self, **kw):
        a = _agent(cognitive_exploration=True, **kw)
        a.geom = TP1
        return a

    def test_hold_starts_when_antecedent_holds(self):
        a = self._agent_ready()
        a.belief = self._belief_with({("sw", "S1", True)})
        ok = a._maybe_start_hold("D1", frozenset({("sw", "S1", True)}),
                                 "hypothesis")
        self.assertTrue(ok)
        self.assertIsNotNone(a._active_hold)

    def test_hold_starts_only_for_model_backed(self):
        a = self._agent_ready()
        a.belief = self._belief_with({("sw", "S1", True)})
        self.assertFalse(a._maybe_start_hold(
            "D1", frozenset({("sw", "S1", True)}), "novel_single"))

    def test_hold_resolves_open(self):
        a = self._agent_ready()
        a.belief = self._belief_with({("sw", "S1", True)})
        a._maybe_start_hold("D1", frozenset({("sw", "S1", True)}),
                            "hypothesis")
        # door opens
        a.belief = a.belief._replace(
            door_open={"D1": True, "D2": False})
        plan, door = a._resolve_hold()
        self.assertIsNone(plan)
        self.assertEqual(door, "D1")
        self.assertIsNone(a._active_hold)

    def test_hold_preserves_condition_and_then_fails(self):
        a = self._agent_ready(hold_budget=1)
        a.belief = self._belief_with({("sw", "S1", True)})
        a._maybe_start_hold("D1", frozenset({("sw", "S1", True)}),
                            "hypothesis")
        # holds, no open -> preserving action returned once
        plan, door = a._resolve_hold()
        self.assertIsNotNone(plan)
        self.assertEqual(door, "D1")
        # budget exhausted, still closed -> genuine failure marked
        plan, door = a._resolve_hold()
        self.assertIsNone(plan)
        self.assertIn(frozenset({("sw", "S1", True)}),
                      a.tried.get("D1", set()))
        self.assertEqual(a.trace["holds_resolved_fail"], 1)

    def test_hold_interrupted_keeps_candidate(self):
        a = self._agent_ready()
        a.belief = self._belief_with({("sw", "S1", True)})
        a._maybe_start_hold("D1", frozenset({("sw", "S1", True)}),
                            "hypothesis")
        # condition breaks: S1 flips off
        a.belief = a.belief._replace(
            switch_state={"S1": False, "S2": False})
        plan, door = a._resolve_hold()
        self.assertIsNone(plan)
        self.assertIsNone(a._active_hold)
        self.assertNotIn(frozenset({("sw", "S1", True)}),
                         a.tried.get("D1", set()))


class TestProbeAttemptInstrumentation(unittest.TestCase):
    def test_new_and_repeat_attempts(self):
        a = _agent(cognitive_exploration=True)
        a._collect_probe_log = True
        ante = frozenset({("sw", "S1", True)})
        a._log_probe_attempt("D1", ante, "novel_single")
        a._log_probe_attempt("D1", ante, "novel_single")
        self.assertEqual(a.trace["probes_new"], 1)
        self.assertEqual(a.trace["probes_repeat"], 0)
        a._mark_probe_completed("D1", ante, "novel_single")
        a._log_probe_attempt("D2", frozenset({("sw", "S2", True)}),
                              "novel_single")
        a._log_probe_attempt("D1", ante, "novel_single")
        self.assertEqual(a.trace["probes_new"], 2)
        self.assertEqual(a.trace["probes_repeat"], 1)


class TestCheckpointFlags(unittest.TestCase):
    def test_save_load_preserves_cognitive_flags(self):
        a = _agent(cognitive_exploration=True, cog_recovery=False)
        a._note_local_evidence("D1", {("sw", "S1", True)}, opened=True)
        s = a.save_state()
        b = _agent()
        b.load_state(s)
        self.assertTrue(b.cognitive_exploration)
        self.assertFalse(b.cog_recovery)
        self.assertEqual(b._local_atom_evidence,
                         a._local_atom_evidence)


if __name__ == "__main__":
    unittest.main()
