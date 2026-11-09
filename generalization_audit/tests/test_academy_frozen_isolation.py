"""Evaluation v0.4 audit — frozen-policy learning isolation (behavioral).

Independent behavioral check of the Academy agent's `frozen_policy`
flag: when set, `learn()` and `learn_from_experience()` must be no-ops,
and the flag must survive a state save/load round-trip. This is the
runtime counterpart of the v0.5 frozen-regime protocol claim.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (_ROOT, _ROOT / "python"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from academy.agents.metargon_agent import MetargonAcademyAgent  # noqa: E402


def _obs(d1=False, d2=False, plate=None, switch=False):
    plate = plate or "A"
    return SimpleNamespace(
        plate_load={"P1": plate, "P2": None},
        switch_state={"S1": bool(switch), "S2": False},
        x=0, y=0, ax=0, ay=0, bx=0, by=0,
        s1=bool(switch), s2=False, p1=plate, p2=None,
        d1=d1, d2=d2)


def _result(after):
    return SimpleNamespace(observation=after)


class TestFrozenPolicyIsolation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.agent = MetargonAcademyAgent(seed=7)
        cls.agent.geom = SimpleNamespace(door_cell=("D1", "D2"))

    def test_frozen_learn_is_noop(self):
        agent = MetargonAcademyAgent(seed=11)
        agent.geom = SimpleNamespace(door_cell=("D1", "D2"))
        agent.frozen_policy = True
        before_steps = agent.steps
        before_hyp = len(agent.memory.hypotheses)
        agent.learn(_obs(False, False), "N", _result(_obs(True, False)))
        agent.learn_from_experience(SimpleNamespace(
            obs_before=_obs(False, False), obs_after=_obs(True, False),
            events=[("D1", "open")]))
        self.assertEqual(agent.steps, before_steps, "steps must not advance")
        self.assertEqual(len(agent.memory.hypotheses), before_hyp,
                         "no hypotheses may be created while frozen")

    def test_unfrozen_learn_advances_steps(self):
        agent = MetargonAcademyAgent(seed=13)
        agent.geom = SimpleNamespace(door_cell=("D1", "D2"))
        before = agent.steps
        agent.learn(_obs(False, False), "N", _result(_obs(True, False)))
        self.assertGreater(agent.steps, before, "learn() must advance steps")

    def test_frozen_flag_roundtrips_through_state(self):
        agent = MetargonAcademyAgent(seed=17)
        agent.frozen_policy = True
        state = agent.save_state()
        self.assertTrue(state["frozen_policy"])
        fresh = MetargonAcademyAgent(seed=17)
        fresh.load_state(state)
        self.assertTrue(fresh.frozen_policy)

    def test_frozen_policy_blocks_replay_path(self):
        agent = MetargonAcademyAgent(seed=19, replay_enabled=True)
        agent.geom = SimpleNamespace(door_cell=("D1", "D2"))
        agent.frozen_policy = True
        # seed a hypothesis so the replay path has something to adjudicate
        agent.learn(_obs(False, False), "N", _result(_obs(True, False)))
        # (above is a no-op while frozen: no hypothesis exists)
        self.assertEqual(len(agent.memory.hypotheses), 0)
        # now unfreeze and learn an event, then freeze again and attempt replay
        agent.frozen_policy = False
        agent.learn(_obs(False, False), "N", _result(_obs(True, False)))
        self.assertGreater(len(agent.memory.hypotheses), 0)
        agent.frozen_policy = True
        before = len(agent.memory.hypotheses)
        agent.learn_from_experience(SimpleNamespace(
            obs_before=_obs(False, False), obs_after=_obs(True, False),
            events=[("D1", "open")]))
        self.assertEqual(len(agent.memory.hypotheses), before,
                         "replay must be blocked while frozen")


if __name__ == "__main__":
    unittest.main()
