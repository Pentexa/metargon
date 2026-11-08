"""Family semantics and interface-compatibility tests for the audit."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (_ROOT, _ROOT / "python"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from lab_env import Observation, make_state, GOAL

from generalization_audit.envs.base import (
    RWorld, LWorld, RConfig, LConfig, HOLD_K,
)
from generalization_audit.envs.pools import build_pools, config_from_json
from generalization_audit.policies.factory import (
    ALL_POLICIES, make_policy,
)


FAMILIES = ("r", "l")


class TestFamilyRSemantics(unittest.TestCase):
    # safe back-and-forth movement between (0,1)-(2,1) that never enters a
    # door cell and leaves switch state unchanged
    HOLD_MOVES = ("E", "E", "W", "W")

    def test_hold_threshold_opens_then_closes(self):
        from generalization_audit.envs.pools import _r_transition
        cfg = RConfig(d1_cond=frozenset({("sw", "S1", True)}),
                      d2_cond=frozenset({("load", "P2", "B")}))
        # start with S1=True (agent stands on S1 after toggling)
        st = make_state()
        st = st._replace(pos=(0, 1))
        st.switch_state["S1"] = True
        hold = {"D1": 0, "D2": 0}
        # HOLD_K-1 steps of non-toggling movement: door stays closed
        for i in range(HOLD_K - 1):
            st, events, hold = _r_transition(
                st, self.HOLD_MOVES[i % len(self.HOLD_MOVES)], cfg,
                HOLD_K, hold)
            self.assertIsNotNone(st)
            self.assertFalse(st.door_open["D1"])
        # the K-th step opens it
        st, events, hold = _r_transition(
            st, self.HOLD_MOVES[(HOLD_K - 1) % len(self.HOLD_MOVES)],
            cfg, HOLD_K, hold)
        self.assertIsNotNone(st)
        self.assertTrue(st.door_open["D1"])

    def test_condition_loss_closes_door(self):
        from generalization_audit.envs.pools import _r_transition
        cfg = RConfig(d1_cond=frozenset({("sw", "S1", True)}),
                      d2_cond=frozenset({("load", "P2", "B")}))
        st = make_state()
        st = st._replace(pos=(0, 1))
        st.switch_state["S1"] = True
        hold = {"D1": 0, "D2": 0}
        for i in range(HOLD_K):
            st, events, hold = _r_transition(
                st, self.HOLD_MOVES[i % len(self.HOLD_MOVES)], cfg,
                HOLD_K, hold)
        self.assertIsNotNone(st)
        self.assertTrue(st.door_open["D1"])
        # toggle S1 off (stand on S1 and toggle) -> condition lost
        st = st._replace(pos=(0, 1))
        st.switch_state["S1"] = False
        st, events, hold = _r_transition(
            st, "E", cfg, HOLD_K, hold)
        self.assertIsNotNone(st)
        self.assertFalse(st.door_open["D1"])


class TestFamilyLSemantics(unittest.TestCase):
    def test_correct_order_opens_reverse_does_not(self):
        from generalization_audit.envs.pools import _l_transition
        cfg = LConfig(
            d1_first=frozenset({("sw", "S1", True)}),
            d1_second=frozenset({("load", "P1", "A")}),
            d2_first=frozenset({("sw", "S2", True)}),
            d2_second=frozenset({("load", "P2", "B")}),
        )
        # reverse order: load P1 (A) first, then S1 on
        st = make_state()
        st.plate_load["P1"] = "A"
        st = st._replace(pos=(0, 1))
        phase = {"D1": 0, "D2": 0}
        ps = {"D1": None, "D2": None}
        st, ev, phase, ps = _l_transition(st, "Toggle", cfg, 30,
                                           phase, ps, 5)
        self.assertIsNotNone(st)
        self.assertFalse(st.door_open["D1"], "reverse order must not open")
        # even with a further step the door must stay closed
        st, ev, phase, ps = _l_transition(st, "E", cfg, 30,
                                           phase, ps, 6)
        self.assertIsNotNone(st)
        self.assertFalse(st.door_open["D1"], "reverse order must not open")
        # correct order: S1 on first, then load appears
        st2 = make_state()
        st2 = st2._replace(pos=(0, 1))
        st2.switch_state["S1"] = True
        phase = {"D1": 0, "D2": 0}
        ps = {"D1": None, "D2": None}
        st2, ev, phase, ps = _l_transition(st2, "E", cfg, 30,
                                            phase, ps, 6)
        self.assertIsNotNone(st2)
        self.assertFalse(st2.door_open["D1"])
        st2.plate_load["P1"] = "A"
        st2, ev, phase, ps = _l_transition(st2, "E", cfg, 30,
                                            phase, ps, 7)
        self.assertIsNotNone(st2)
        self.assertTrue(st2.door_open["D1"], "correct order must open D1")

    def test_window_expiry_resets_sequence(self):
        from generalization_audit.envs.pools import _l_transition
        cfg = LConfig(
            d1_first=frozenset({("sw", "S1", True)}),
            d1_second=frozenset({("load", "P1", "A")}),
            d2_first=frozenset({("sw", "S2", True)}),
            d2_second=frozenset({("load", "P2", "B")}),
        )
        st = make_state()
        st.switch_state["S1"] = True
        phase = {"D1": 1, "D2": 0}
        ps = {"D1": 5, "D2": None}
        st, ev, phase, ps = _l_transition(st, "E", cfg, 30, phase, ps, 36)
        self.assertEqual(phase["D1"], 0, "window expiry must reset phase")


class TestInterfaceCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pools = {}
        for family in FAMILIES:
            (train, val, test), n = build_pools(family)
            cls.pools[family] = (train, val, test)
            cls._n[family] = n

    _n = {}

    def test_all_policies_run_one_episode_each_family(self):
        for family in FAMILIES:
            cfg = self.pools[family][0][0]
            for name in ALL_POLICIES:
                policy = make_policy(name, 3, family=family)
                from generalization_audit.benchmark_generalization import (
                    play_episode)
                from lab_env import LabEpisode
                spec = LabEpisode(cfg, 80, 0.0, seed=1, context=family)
                r = play_episode(spec, policy, family)
                self.assertIn(r["success"], (True, False))
                self.assertGreaterEqual(r["steps"], 0)
                self.assertGreater(r["steps"], 0)

    def test_determinism_same_seed(self):
        for family in FAMILIES:
            cfg = self.pools[family][0][0]
            from lab_env import LabEpisode
            from generalization_audit.benchmark_generalization import (
                play_episode)
            spec = LabEpisode(cfg, 80, 0.0, seed=11, context=family)
            r1 = play_episode(spec, make_policy("metargon10", 5, family=family),
                               family)
            r2 = play_episode(spec, make_policy("metargon10", 5, family=family),
                               family)
            self.assertEqual((r1["success"], r1["steps"], r1["invalid"]),
                             (r2["success"], r2["steps"], r2["invalid"]))


if __name__ == "__main__":
    unittest.main()
