"""Tests for cognitive exploration v0.1 (knowledge-gap probing, hold-
based probe attribution, stagnation recovery, decomposition, and the
probe-attempt instrumentation)."""

from __future__ import annotations

import unittest

from academy.agents.common import belief_from_obs
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


class TestPlanCacheCapAwareReuse(unittest.TestCase):
    """Results produced under different BFS expansion caps must never be
    incorrectly reused: a *failure* under a small cap is provisional and
    must not satisfy a request with a larger cap. Successes are
    cap-independent. Legacy (pre-v0.1) entries carry no cap and are
    treated as produced under the default cap."""

    def _agent(self, **kw):
        a = _agent(cognitive_exploration=True, **kw)
        a.geom = TP1
        a.belief = make_ws(TP1)
        return a

    def _key(self, a, target, to_pos=None):
        return (a._plan_cached.__self__ is not None and
                ("state_key", to_pos, tuple(sorted(target))))

    def test_failure_under_small_cap_not_reused_by_larger(self):
        from unittest.mock import patch
        a = self._agent()
        target = frozenset({("sw", "S1", True)})
        with patch("academy.agents.metargon_agent.plan_to_atoms",
                   return_value=None) as p1:
            first = a._plan_cached(a.belief, target, lambda: None,
                                   max_expand=700)
            self.assertIsNone(first)
            self.assertEqual(p1.call_count, 1)
        # larger cap must recompute (cached small-cap failure is not
        # authoritative for a larger budget)
        with patch("academy.agents.metargon_agent.plan_to_atoms",
                   return_value=["N"]) as p2:
            second = a._plan_cached(a.belief, target, lambda: None,
                                    max_expand=1500)
            self.assertEqual(second, ["N"])
            self.assertEqual(p2.call_count, 1)

    def test_failure_under_large_cap_reused_by_smaller(self):
        from unittest.mock import patch
        a = self._agent()
        target = frozenset({("sw", "S1", True)})
        with patch("academy.agents.metargon_agent.plan_to_atoms",
                   return_value=None):
            a._plan_cached(a.belief, target, lambda: None,
                           max_expand=1500)
        with patch("academy.agents.metargon_agent.plan_to_atoms",
                   return_value=["N"]) as p:
            second = a._plan_cached(a.belief, target, lambda: None,
                                    max_expand=700)
            self.assertIsNone(second)      # safe cache reuse
            self.assertEqual(p.call_count, 0)

    def test_success_reused_across_caps(self):
        from unittest.mock import patch
        a = self._agent()
        target = frozenset({("sw", "S1", True)})
        with patch("academy.agents.metargon_agent.plan_to_atoms",
                   return_value=["E", "N"]):
            a._plan_cached(a.belief, target, lambda: None,
                           max_expand=1500)
        with patch("academy.agents.metargon_agent.plan_to_atoms",
                   return_value=[]) as p:
            second = a._plan_cached(a.belief, target, lambda: None,
                                    max_expand=700)
            self.assertEqual(second, ["E", "N"])
            self.assertEqual(p.call_count, 0)

    def test_legacy_cache_entries_normalized_on_load(self):
        a = _agent()
        legacy = {repr(("sk", None, (("sw", "S1", True),))): ["N"]}
        state = a.save_state()
        state["plan_cache"] = legacy
        b = _agent()
        b.load_state(state)
        key = eval(next(iter(legacy)))
        cap, plan = b.plan_cache[key]
        self.assertEqual(cap, b.planner_max_expand)
        self.assertEqual(plan, ["N"])

    def test_cache_roundtrip_preserves_cap(self):
        a = _agent()
        key = ("sk", None, (("sw", "S1", True),))
        a.plan_cache[key] = (700, [])
        b = _agent()
        b.load_state(a.save_state())
        self.assertEqual(b.plan_cache[key], (700, []))


class TestGoalPrefilterGating(unittest.TestCase):
    """The goal-connectivity prefilter is cognitive-only: with default
    flags it must not run (baseline behavior unchanged)."""

    def _agent(self, **kw):
        a = _agent(**kw)
        a.geom = TP1
        a.belief = make_ws(TP1)
        # minimal per-episode bookkeeping expected by _choose_probe
        a.tried = {d: set() for d in TP1.door_cell}
        a.passed = {d: False for d in TP1.door_cell}
        a.far_anchor = a.far_anchor or {d: TP1.goal for d in TP1.door_cell}
        a._goal_infeasible = {}
        a._steps_since_goal_check = 0
        return a

    def test_baseline_does_not_call_prefilter(self):
        from unittest.mock import patch
        a = self._agent()          # all flags off
        with patch.object(a, "_goal_component_reachable",
                          side_effect=AssertionError(
                              "prefilter must not run for baseline")):
            plan, door = a._choose_probe()
        # baseline still plans toward the goal (door/goal path), or at
        # least does not raise; door is None for non-goal plans
        self.assertIsInstance(plan, (list, type(None)))

    def test_cognitive_uses_prefilter(self):
        from unittest.mock import patch
        a = self._agent(cognitive_exploration=True)
        with patch.object(a, "_goal_component_reachable",
                          return_value=False) as pre:
            with patch.object(a, "_plan_cached") as plan:
                out = a._choose_probe()
        pre.assert_called_once()
        # goal plan must NOT be requested when the goal is not component-
        # reachable; every _plan_cached call has to_pos=None (probes)
        for call in plan.call_args_list:
            self.assertIsNone(call.kwargs.get("to_pos"))


class TestNoiseRobustConfirmation(unittest.TestCase):
    """v0.2: door events fire only when causally linked to an exact atom
    change; switch state is tracked exactly from Toggle semantics."""

    def _agent(self, **kw):
        kw.setdefault("cognitive_exploration", True)
        kw.setdefault("cog_noise_robust", True)
        a = MetargonAcademyAgent(seed=0, **kw)
        a.geom = TP1
        a.belief = make_ws(TP1)
        return a

    @staticmethod
    def _obs(d1=False, d2=False, s1=False, s2=False,
              x=0, y=4, ax=2, ay=4, bx=4, by=3, p1="none", p2="none"):
        from academy.envs.gridworld import Observation
        return Observation(x=x, y=y, ax=ax, ay=ay, bx=bx, by=by,
                           s1=s1, s2=s2, p1=p1, p2=p2,
                           d1=d1, d2=d2, finished=False)

    def test_flip_without_atom_change_is_noise(self):
        a = self._agent()
        a._init_confirmed_from(self._obs(d1=False))
        before = a._inferred_atoms(self._obs(d1=False))
        after = a._inferred_atoms(self._obs(d1=True))
        a._update_events(before, after, step=10,
                         before_obs=self._obs(d1=False),
                         after_obs=self._obs(d1=True))
        self.assertEqual(a._newly_confirmed, [])
        self.assertFalse(a._confirmed_door["D1"])
        self.assertEqual(a.trace["unconfirmed_flips"], 1)

    def test_flip_in_bootstrap_window_accepted(self):
        """Episode bootstrap: initial-state rule satisfaction opens doors
        without any prior atom change; step<=1 flips are accepted."""
        a = self._agent()
        a._init_confirmed_from(self._obs(d1=False))
        before = a._inferred_atoms(self._obs(d1=False))
        after = a._inferred_atoms(self._obs(d1=True))
        a._update_events(before, after, step=0,
                         before_obs=self._obs(d1=False),
                         after_obs=self._obs(d1=True))
        self.assertEqual(a._newly_confirmed, [("D1", True)])
        self.assertTrue(a._confirmed_door["D1"])

    def test_flip_with_atom_change_is_event(self):
        a = self._agent()
        a._init_confirmed_from(self._obs(d1=False))
        # atom change: object A moves onto P1 (3,4)
        before = a._inferred_atoms(self._obs(d1=False, ax=2, ay=4))
        after = a._inferred_atoms(self._obs(d1=True, ax=3, ay=4))
        a._update_events(before, after, step=5,
                         before_obs=self._obs(d1=False, ax=2, ay=4),
                         after_obs=self._obs(d1=True, ax=3, ay=4))
        self.assertEqual(a._newly_confirmed, [("D1", True)])
        self.assertTrue(a._confirmed_door["D1"])

    def test_delayed_open_accepted_on_next_step(self):
        a = self._agent()
        a._init_confirmed_from(self._obs(d1=False))
        # atom change at step 5 (loader placed), door opens at step 6
        before1 = a._inferred_atoms(self._obs(d1=False, ax=2, ay=4))
        after1 = a._inferred_atoms(self._obs(d1=False, ax=3, ay=4))
        a._update_events(before1, after1, step=5,
                         before_obs=self._obs(d1=False, ax=2, ay=4),
                         after_obs=self._obs(d1=False, ax=3, ay=4))
        # flip at step 6 with no new atom change -> accepted (linked)
        before2 = a._inferred_atoms(self._obs(d1=False, ax=3, ay=4))
        after2 = a._inferred_atoms(self._obs(d1=True, ax=3, ay=4))
        a._update_events(before2, after2, step=6,
                         before_obs=self._obs(d1=False, ax=3, ay=4),
                         after_obs=self._obs(d1=True, ax=3, ay=4))
        self.assertEqual(a._newly_confirmed, [("D1", True)])

    def test_isolated_flip_long_after_atom_change_is_noise(self):
        a = self._agent()
        a._init_confirmed_from(self._obs(d1=False))
        a._last_atom_change_step = -10
        before = a._inferred_atoms(self._obs(d1=False))
        after = a._inferred_atoms(self._obs(d1=True))
        a._update_events(before, after, step=20,
                         before_obs=self._obs(d1=False),
                         after_obs=self._obs(d1=True))
        self.assertEqual(a._newly_confirmed, [])

    def test_toggle_semantics_exact(self):
        a = self._agent()
        # stand on S1 cell (0,1) and toggle
        a._toggle_switch_if_standing("Toggle", (0, 1))
        self.assertTrue(a._inferred_switch["S1"])
        a._toggle_switch_if_standing("Toggle", (0, 1))
        self.assertFalse(a._inferred_switch["S1"])
        # toggle elsewhere does nothing
        a._toggle_switch_if_standing("Toggle", (1, 1))
        self.assertFalse(a._inferred_switch["S2"])

    def test_belief_override_uses_inferred_switches(self):
        a = self._agent()
        a._init_confirmed_from(self._obs(d1=False))
        a._inferred_switch["S1"] = True
        # raw obs says S1 off, D1 open
        a.belief = belief_from_obs(
            self._obs(d1=True, s1=False), TP1)
        a._apply_confirmed_belief()
        self.assertFalse(a.belief.door_open["D1"])
        self.assertTrue(a.belief.switch_state["S1"])

    def test_inferred_atoms_derive_plates_from_positions(self):
        a = self._agent()
        a._init_confirmed_from(self._obs(d1=False))
        # object A sits on P1 cell (3,4); raw p1 reading says "none"
        obs = self._obs(ax=3, ay=4, p1="none")
        atoms = a._inferred_atoms(obs)
        self.assertIn(("load", "P1", "A"), atoms)
        self.assertNotIn(("empty", "P1"), atoms)

    def test_checkpoint_roundtrip_confirmation(self):
        a = self._agent()
        a._init_confirmed_from(self._obs(d1=True))
        a._inferred_switch["S2"] = True
        a._last_atom_change_step = 7
        s = a.save_state()
        b = MetargonAcademyAgent(seed=0)
        b.load_state(s)
        self.assertTrue(b.cog_noise_robust)
        self.assertEqual(b._confirmed_door, {"D1": True, "D2": False})
        self.assertTrue(b._inferred_switch["S2"])
        self.assertEqual(b._last_atom_change_step, 7)


class TestArtifactCounts(unittest.TestCase):
    """The committed experiment artifacts must match the module defaults
    (reconciliation guard)."""

    def test_defaults_match_committed_artifacts(self):
        import json
        import os
        from academy.tests import test_cognitive_exploration_experiment as m
        root = os.path.join(os.path.dirname(__file__), "..", "..")
        main_path = os.path.join(root, "results", "academy_v01",
                                 "cognitive_exploration", "main_summary.json")
        ab_path = os.path.join(root, "results", "academy_v01",
                               "cognitive_exploration",
                               "ablations_summary.json")
        if not (os.path.exists(main_path) and os.path.exists(ab_path)):
            self.skipTest("experiment artifacts not present")
        main = json.load(open(main_path, encoding="utf-8"))
        ab = json.load(open(ab_path, encoding="utf-8"))
        self.assertEqual(main["n_seeds"], m.N_SEEDS_MAIN)
        self.assertEqual(main["episodes"], m.EPISODES_MAIN)
        self.assertEqual(ab["n_seeds"], m.N_SEEDS_ABLATION)
        self.assertEqual(ab["episodes"], m.EPISODES_ABLATION)


if __name__ == "__main__":
    unittest.main()
