#!/usr/bin/env python3
"""Policy factories for the independent generalization audit.

Roster (all evaluated on identical, pre-generated worlds):

* metargon10  — the unmodified v0.10 student agent.
* random      — uniform random action (model-free lower bound).
* episodic    — lightweight episodic-memory replay controller (model-free).
* qlearning   — tabular epsilon-greedy Q-learning (model-free).
* bayes       — family-F Bayesian particle-filter planner (declared family;
                misspecified on families R and L by design).
* oracle      — privileged perfect-knowledge planner for the audited family
                (diagnostic only; excluded from rankings).
* *-cap      — compute-capped variants of metargon10 / bayes / qlearning
                that enforce an identical per-decision expansion budget
                (see ComputeCappedPolicy), for the matched-compute arm.

No policy is modified; the oracle is implemented here as a standalone
privileged comparator for the new families.
"""
from __future__ import annotations

import random
from collections import deque

from lab_env import ACTIONS, GOAL, SWITCHES, PLATES, make_state, public_next
from lab_env import LabConfig

from ..envs.base import RWorld, LWorld, RConfig, LConfig, eval_atoms
from ..envs.pools import _l_transition as _l_step_full

COMPUTE_BUDGET = 120  # per-decision expansion budget (matched-compute arm)


def _key(st) -> tuple:
    return (st.pos, st.objs["A"], st.objs["B"],
            tuple(st.switch_state[s] for s in SWITCHES),
            tuple(st.plate_load[p] for p in PLATES),
            tuple(st.door_open[d] for d in ("D1", "D2")))


# --------------------------------------------------------------------------
# Privileged family oracles (perfect knowledge, diagnostic only)
# --------------------------------------------------------------------------
def _r_plan(state, config: RConfig, goal=GOAL, hold_k=3, max_expand=4000):
    """Ground-truth BFS under family-R dynamics (perfect knowledge)."""
    if state.pos == goal:
        return []
    q = deque([(state, [], {"D1": 0, "D2": 0})])
    seen = set()
    seen.add((_key(state), 0, 0))
    while q and len(seen) < max_expand:
        st, path, hold = q.popleft()
        children = []
        for a in ACTIONS:
            nxt2, _ev = _r_step(st, a, config, hold_k, dict(hold))
            if nxt2 is None:
                continue
            nxt, nhold = nxt2
            k = (_key(nxt), nhold["D1"], nhold["D2"])
            if k in seen:
                continue
            seen.add(k)
            np = path + [a]
            if nxt.pos == goal:
                return np
            children.append((nxt, np, nhold))
        children.sort(key=lambda t: (abs(t[0].pos[0] - goal[0])
                                     + abs(t[0].pos[1] - goal[1])))
        q.extend(children)
    return None


def _r_step(state, action, config: RConfig, hold_k: int, hold: dict):
    """One-step family-R transition returning ((state, hold), events)."""
    nxt = public_next(state, action)
    if nxt is None:
        return None, []
    events = []
    for door in ("D1", "D2"):
        cond = eval_atoms(config.cond_for(door), nxt)
        if not nxt.door_open[door]:
            if cond:
                hold[door] += 1
                if hold[door] >= hold_k:
                    nxt.door_open[door] = True
                    events.append((door, "open"))
            else:
                hold[door] = 0
        else:
            if not cond and nxt.pos != _dc(door) and \
                    _occupied(nxt, _dc(door)) is None:
                nxt.door_open[door] = False
                hold[door] = 0
                events.append((door, "close"))
    return (nxt, hold), events


def _dc(door):
    from lab_env import DOOR_CELL
    return DOOR_CELL[door]


def _occupied(state, cell):
    for o, c in state.objs.items():
        if c == cell:
            return o
    return None


def _l_plan(state, config: LConfig, goal=GOAL, window=30, max_expand=4000):
    if state.pos == goal:
        return []
    q = deque([(state, [], {"D1": 0, "D2": 0},
                {"D1": None, "D2": None})])
    seen = set()
    seen.add((_key(state), 0, 0, None, None))
    while q and len(seen) < max_expand:
        st, path, phase, phase_step = q.popleft()
        children = []
        for a in ACTIONS:
            nxt, _ev, nph, nps = _l_step_full(
                st, a, config, window, dict(phase), dict(phase_step),
                len(path) + 1)
            if nxt is None:
                continue
            k = (_key(nxt), nph["D1"], nph["D2"],
                 nps["D1"], nps["D2"])
            if k in seen:
                continue
            seen.add(k)
            np = path + [a]
            if nxt.pos == goal:
                return np
            children.append((nxt, np, nph, nps))
        children.sort(key=lambda t: (abs(t[0].pos[0] - goal[0])
                                     + abs(t[0].pos[1] - goal[1])))
        q.extend(children)
    return None


class AuditOraclePolicy:
    """Privileged diagnostic comparator with perfect knowledge of the
    audited family's wiring. Excluded from deployable rankings.
    Plans once and follows the plan while it remains executable; replans
    only when the plan is invalidated (the planner's model includes
    hidden progress counters that cannot be reconstructed from a single
    observation)."""
    name = "oracle"

    def __init__(self, seed: int = 0, family: str = "r"):
        self.seed = seed
        self.rng = random.Random(seed)
        self.family = family
        self.config = None
        self.plan_cache = {}
        self.current_plan = []
        self.belief = make_state()

    def episode_start(self, config=None):
        self.config = config
        self.belief = make_state()
        self.current_plan = []

    def _sync(self, obs):
        b = make_state()
        b = b._replace(pos=(obs.x, obs.y))
        objs = dict(b.objs); objs["A"] = (obs.ax, obs.ay); objs["B"] = (obs.bx, obs.by)
        b = b._replace(objs=objs)
        sw = dict(b.switch_state); sw["S1"] = obs.s1; sw["S2"] = obs.s2
        b = b._replace(switch_state=sw)
        pl = dict(b.plate_load); pl["P1"] = obs.p1; pl["P2"] = obs.p2
        b = b._replace(plate_load=pl)
        dr = dict(b.door_open); dr["D1"] = obs.d1; dr["D2"] = obs.d2
        b = b._replace(door_open=dr)
        self.belief = b

    def act(self, obs):
        self._sync(obs)
        while self.current_plan:
            a = self.current_plan[0]
            if public_next(self.belief, a) is not None:
                self.current_plan.pop(0)
                return a
            self.current_plan.pop(0)
        sig = (_key(self.belief),
               getattr(self.config, "key", lambda: None)())
        plan = self.plan_cache.get(sig)
        if plan is None:
            if self.family == "r":
                plan = _r_plan(self.belief, self.config) or []
            else:
                plan = _l_plan(self.belief, self.config) or []
            self.plan_cache[sig] = plan
        self.current_plan = list(plan)
        if self.current_plan:
            return self.current_plan.pop(0)
        v = [a for a in ACTIONS if public_next(self.belief, a) is not None]
        return self.rng.choice(v if v else list(ACTIONS))

    def observe_outcome(self, obs_before, action, res):
        pass

    def diagnostics(self):
        return {}


# --------------------------------------------------------------------------
# Compute-budget governor (matched-compute arm)
# --------------------------------------------------------------------------
class _BudgetExceeded(Exception):
    pass


class ComputeCappedPolicy:
    """Wraps a policy and enforces an identical per-decision *expansion*
    budget. For METARGON, expansions are counted by intercepting the
    planner's `on_expand` callback chain; once the budget is exceeded,
    the current plan search aborts and the agent falls back to its own
    exploration. For the Bayesian planner, the ground-truth search is run
    with `max_expand=budget`. Used identically for every policy in the
    matched-compute arm."""

    def __init__(self, policy, budget: int = COMPUTE_BUDGET, seed: int = 0):
        self.inner = policy
        self.budget = budget
        self.rng = random.Random(seed + 7)
        self.expansions = 0
        self.fallbacks = 0
        self._last_obs = None
        self._kind = None
        if type(policy).__name__ == "MetargonV10":
            self._kind = "metargon"
        elif type(policy).__name__ == "BayesPlannerPolicy":
            self._kind = "bayes"

    @property
    def name(self):
        return getattr(self.inner, "name", "capped") + "-cap"

    def episode_start(self, *args, **kwargs):
        self.expansions = 0
        self.fallbacks = 0
        self._last_obs = None
        return self.inner.episode_start(*args, **kwargs)

    def _belief_from_obs(self, obs):
        b = make_state()
        b = b._replace(pos=(obs.x, obs.y))
        objs = dict(b.objs); objs["A"] = (obs.ax, obs.ay); objs["B"] = (obs.bx, obs.by)
        b = b._replace(objs=objs)
        sw = dict(b.switch_state); sw["S1"] = obs.s1; sw["S2"] = obs.s2
        b = b._replace(switch_state=sw)
        pl = dict(b.plate_load); pl["P1"] = obs.p1; pl["P2"] = obs.p2
        b = b._replace(plate_load=pl)
        dr = dict(b.door_open); dr["D1"] = obs.d1; dr["D2"] = obs.d2
        b = b._replace(door_open=dr)
        return b

    def _act_metargon(self, obs):
        import metargon_v10 as m10
        budget = self.budget
        counter = {"n": 0}
        orig_atoms = m10.plan_to_atoms
        orig_pos = m10.plan_to_pos

        def make_cb(orig_cb):
            def cb():
                counter["n"] += 1
                if counter["n"] > budget:
                    raise _BudgetExceeded()
                if orig_cb is not None:
                    orig_cb()
            return cb

        def capped_atoms(*a, **kw):
            cb = make_cb(kw.get("on_expand"))
            kw["on_expand"] = cb
            try:
                return orig_atoms(*a, **kw)
            except _BudgetExceeded:
                return None

        def capped_pos(*a, **kw):
            cb = make_cb(kw.get("on_expand"))
            kw["on_expand"] = cb
            try:
                return orig_pos(*a, **kw)
            except _BudgetExceeded:
                return None

        m10.plan_to_atoms = capped_atoms
        m10.plan_to_pos = capped_pos
        try:
            action = self.inner.act(obs)
            self.expansions += counter["n"]
            if counter["n"] > budget:
                self.fallbacks += 1
            return action
        finally:
            m10.plan_to_atoms = orig_atoms
            m10.plan_to_pos = orig_pos

    def _act_bayes(self, obs):
        import baselines_v09 as b09
        budget = self.budget
        orig = b09.ground_truth_plan

        def capped(s, c):
            return orig(s, c, max_expand=budget)

        b09.ground_truth_plan = capped
        try:
            action = self.inner.act(obs)
            self.fallbacks += getattr(self.inner, "plan_cache", None) is not None and 0 or 0
            return action
        finally:
            b09.ground_truth_plan = orig

    def act(self, obs):
        self._last_obs = obs
        if self._kind == "metargon":
            return self._act_metargon(obs)
        if self._kind == "bayes":
            return self._act_bayes(obs)
        return self.inner.act(obs)

    def observe_outcome(self, obs_before, action, res):
        return self.inner.observe_outcome(obs_before, action, res)

    def diagnostics(self):
        d = dict(getattr(self.inner, "diagnostics", lambda: {})())
        d["compute_expansions"] = self.expansions
        d["compute_fallbacks"] = self.fallbacks
        return d


# --------------------------------------------------------------------------
# Roster factory
# --------------------------------------------------------------------------
def make_policy(name: str, seed: int, family: str = "r"):
    from metargon_v10 import MetargonV10
    from baselines_v09 import (RandomPolicy, EpisodicMemoryPolicy,
                                QLearningPolicy, BayesPlannerPolicy)
    if name == "metargon10":
        return MetargonV10(seed)
    if name == "metargon10-cap":
        return ComputeCappedPolicy(MetargonV10(seed), seed=seed)
    if name == "random":
        return RandomPolicy(seed)
    if name == "episodic":
        return EpisodicMemoryPolicy(seed)
    if name == "qlearning":
        return QLearningPolicy(seed)
    if name == "qlearning-cap":
        return ComputeCappedPolicy(QLearningPolicy(seed), seed=seed)
    if name == "bayes":
        return BayesPlannerPolicy(seed)
    if name == "bayes-cap":
        return ComputeCappedPolicy(BayesPlannerPolicy(seed), seed=seed)
    if name == "oracle":
        return AuditOraclePolicy(seed, family=family)
    raise ValueError(name)


ALL_POLICIES = ("metargon10", "random", "episodic", "qlearning", "bayes",
                "metargon10-cap", "qlearning-cap", "bayes-cap", "oracle")
DEPLOYABLE = ("metargon10", "random", "episodic", "qlearning", "bayes",
              "metargon10-cap", "qlearning-cap", "bayes-cap")
COMPARISONS = (
    ("metargon10", "random"),
    ("metargon10", "episodic"),
    ("metargon10", "qlearning"),
    ("metargon10", "bayes"),
    ("metargon10", "metargon10-cap"),
    ("bayes", "bayes-cap"),
    ("qlearning", "qlearning-cap"),
)
