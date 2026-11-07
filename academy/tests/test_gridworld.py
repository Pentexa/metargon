"""Tests for the MechanismGrid environment: public mechanics, hidden door
dynamics (conjunctive, disjunctive, delay), and strict observation
separation (no hidden-state leakage)."""

import unittest

from academy.envs.gridworld import (
    ACTIONS, TOPOLOGIES, EnvConfig, MechanismGrid, Observation,
    atoms_from_state, is_solvable, make_ws, public_next,
    rule_holds, transition_with_hidden,
)
from academy.envs.tasks import _dict_to_config, _config_to_dict


D1_TP1 = frozenset({("sw", "S2", True)})
D2_TP1 = frozenset({("sw", "S1", True)})
DISJ_D2 = (frozenset({("sw", "S1", True)}),
            frozenset({("load", "P1", "A")}))


class TestPublicMechanics(unittest.TestCase):
    def setUp(self):
        self.geom = TOPOLOGIES["TP1"]
        self.st = make_ws(self.geom)

    def test_start_positions(self):
        self.assertEqual(self.st.pos, (0, 4))
        self.assertEqual(self.geom.goal, (6, 0))

    def test_move_invalid_into_wall(self):
        # (0,2) is a wall in TP1; from (0,3) N is invalid
        st = make_ws(self.geom)._replace(pos=(0, 3))
        self.assertIsNone(public_next(st, "N", self.geom))

    def test_toggle_only_on_switch(self):
        st = make_ws(self.geom)._replace(pos=(4, 4))  # S2 cell
        nxt = public_next(st, "Toggle", self.geom)
        self.assertIsNotNone(nxt)
        self.assertTrue(nxt.switch_state["S2"])
        st2 = make_ws(self.geom)._replace(pos=(0, 0))
        self.assertIsNone(public_next(st2, "Toggle", self.geom))

    def test_push_moves_object_and_agent(self):
        # TP1: A at (2,4); push E from (1,4)
        st = make_ws(self.geom)._replace(pos=(1, 4))
        nxt = public_next(st, "PushE", self.geom)
        self.assertIsNotNone(nxt)
        self.assertEqual(nxt.pos, (2, 4))
        self.assertEqual(nxt.objs["A"], (3, 4))

    def test_push_blocked_by_wall(self):
        # B at (4,3) TP1; push N from (4,4): target (4,2) wall
        st = make_ws(self.geom)._replace(pos=(4, 4))
        self.assertIsNone(public_next(st, "PushN", self.geom))


class TestHiddenDynamics(unittest.TestCase):
    def setUp(self):
        self.geom = TOPOLOGIES["TP1"]
        self.cfg = EnvConfig("TP1", D1_TP1, D2_TP1, False, False)

    def test_door_opens_when_rule_holds(self):
        st = make_ws(self.geom)
        st = st._replace(pos=(4, 4))
        st = st._replace(switch_state={"S1": False, "S2": True})
        nxt, events = transition_with_hidden(st, "W", self.cfg, self.geom)
        # moving W from (4,4) to (3,4) keeps S2 on -> D1 opens
        self.assertIsNotNone(nxt)
        self.assertTrue(nxt.door_open["D1"])
        self.assertIn(("D1", "open"), events)

    def test_door_stays_closed_without_rule(self):
        st = make_ws(self.geom)
        nxt, events = transition_with_hidden(st, "N", self.cfg, self.geom)
        self.assertIsNotNone(nxt)
        self.assertFalse(nxt.door_open["D1"])
        self.assertEqual(events, [])

    def test_delay_arms_then_opens(self):
        cfg = EnvConfig("TP1", D1_TP1, D2_TP1, True, False)
        st = make_ws(self.geom)
        st = st._replace(pos=(4, 4), switch_state={"S1": False, "S2": True})
        nxt, ev = transition_with_hidden(st, "W", cfg, self.geom)
        self.assertFalse(nxt.door_open["D1"])
        self.assertTrue(nxt.armed["D1"])
        self.assertEqual(ev, [])
        nxt2, ev2 = transition_with_hidden(nxt, "N", cfg, self.geom)
        self.assertTrue(nxt2.door_open["D1"])
        self.assertIn(("D1", "open"), ev2)

    def test_open_door_closes_when_rule_breaks(self):
        st = make_ws(self.geom)
        st = st._replace(pos=(4, 4), switch_state={"S1": False, "S2": True})
        nxt, _ = transition_with_hidden(st, "W", self.cfg, self.geom)
        self.assertTrue(nxt.door_open["D1"])
        # walk back onto S2, then toggle it off -> rule breaks -> closes
        nxt1, _ = transition_with_hidden(nxt, "E", self.cfg, self.geom)
        self.assertEqual(nxt1.pos, (4, 4))
        nxt2, ev2 = transition_with_hidden(nxt1, "Toggle", self.cfg, self.geom)
        self.assertIn(("D1", "close"), ev2)
        self.assertFalse(nxt2.door_open["D1"])

    def test_disjunctive_rule(self):
        cfg = EnvConfig("TP1", D1_TP1, DISJ_D2, False, False)
        # D2 opens if S1 on OR P1 loaded with A
        st = make_ws(self.geom)
        st = st._replace(pos=(4, 4), switch_state={"S1": True, "S2": False})
        self.assertTrue(rule_holds(cfg.d2_rule, st.plate_load, st.switch_state))

    def test_solvability_basic(self):
        cfg = EnvConfig("TP1", D1_TP1, D2_TP1, False, False)
        self.assertTrue(is_solvable(cfg))


class TestNoLeakage(unittest.TestCase):
    def test_observation_has_no_hidden_fields(self):
        field_names = set(Observation._fields)
        self.assertEqual(field_names,
                         {"x", "y", "ax", "ay", "bx", "by", "s1", "s2",
                          "p1", "p2", "d1", "d2", "finished"})

    def test_config_serialization_roundtrip(self):
        for c in (EnvConfig("TP1", D1_TP1, D2_TP1, True, False),
                  EnvConfig("TP1", D1_TP1, DISJ_D2, False, True)):
            d = _config_to_dict(c)
            c2 = _dict_to_config(d)
            self.assertEqual(c.key, c2.key)

    def test_env_surface_never_exposes_rule(self):
        env = MechanismGrid(EnvConfig("TP1", D1_TP1, D2_TP1, False, False))
        env.reset(0)
        obs = env.observe()
        self.assertNotIn("rule", dir(obs))
        self.assertNotIn("config", dir(obs))


if __name__ == "__main__":
    unittest.main()
