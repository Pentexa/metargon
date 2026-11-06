"""METARGON v0.9 — Phase E: strong comparison baselines.

All policies consume only the public observation stream and the public
mechanics. None accesses hidden simulator state. The oracle is privileged
and clearly labeled; it is excluded from deployable rankings.

Baselines:
  * random           — uniform random action.
  * episodic         — simple episodic memory: replay the closest memorized
                       successful trajectory by state signature; otherwise
                       act randomly (a lightweight episodic controller).
  * qlearning        — tabular model-free Q-learning with epsilon-greedy
                       exploration (standard minimal implementation; no
                       function approximation).
  * bayes           — model-based Bayesian planner: maintains a particle
                       filter posterior over the declared generative family
                       (F_TEMPLATES^2 x delays), and acts by posterior
                       sampling with full ground-truth planning under the
                       sampled hypothesis. This baseline is given the
                       generative family, i.e. a correct model class (in
                       the standard conditions) — an intentionally strong
                       comparator.
  * oracle           — privileged: knows the true wiring; optimal plan by
                       full search. Diagnostic only.
"""
from __future__ import annotations

import random
from collections import deque

from lab_env import (
    ACTIONS, DOORS, F1_TEMPLATES, F2_TEMPLATES, GOAL, LabConfig, WS,
    make_state, public_next, transition_with_hidden,
)

LAMBDA_LL = 0.02        # likelihood smoothing for model-based posteriors
K_PARTICLES = 48        # particle filter size
PLAN_MAX_EXPAND = 4000  # ground-truth plan BFS cap
QL_ALPHA = 0.30
QL_GAMMA = 0.95
QL_EPSILON0 = 0.35
QL_EPSILON_MIN = 0.05
EPI_MEM_MAX = 200


def sign(obs):
    return (obs.x, obs.y, obs.ax, obs.ay, obs.bx, obs.by,
            obs.s1, obs.s2, obs.p1, obs.p2, obs.d1, obs.d2)


def valid_actions(state: WS):
    return [a for a in ACTIONS if public_next(state, a) is not None]


def _state_key_full(st: WS):
    return (st.pos, st.objs["A"], st.objs["B"],
            tuple(st.switch_state[s] for s in ("S1", "S2")),
            tuple(st.plate_load[p] for p in ("P1", "P2")),
            tuple(st.door_open[d] for d in DOORS))


def ground_truth_plan(state: WS, config: LabConfig, goal=GOAL,
                      max_expand: int = PLAN_MAX_EXPAND):
    """Optimal plan under full hidden dynamics (used by Bayes post. sample
    and the oracle). BFS over the full state graph, with actions ordered by
    distance-to-goal to find solutions much earlier."""
    if state.pos == goal:
        return []
    start_key = _state_key_full(state)
    queue = deque([(state, [])])
    seen = {start_key}
    while queue and len(seen) < max_expand:
        st, path = queue.popleft()
        children = []
        for a in ACTIONS:
            nxt, _ = transition_with_hidden(st, a, config)
            if nxt is None:
                continue
            k = _state_key_full(nxt)
            if k in seen:
                continue
            seen.add(k)
            np = path + [a]
            if nxt.pos == goal:
                return np
            children.append((nxt, np))
        # expand closer-to-goal states first (cheap heuristic ordering)
        children.sort(key=lambda t: (abs(t[0].pos[0] - goal[0])
                                     + abs(t[0].pos[1] - goal[1])))
        queue.extend(children)
    return None


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.belief = make_state()

    def episode_start(self):
        self.belief = make_state()

    def _sync(self, obs):
        b = self.belief
        b = b._replace(pos=(obs.x, obs.y))
        objs = dict(b.objs); objs["A"] = (obs.ax, obs.ay); objs["B"] = (obs.bx, obs.by)
        b = b._replace(objs=objs)
        sw = dict(b.switch_state); sw["S1"] = obs.s1; sw["S2"] = obs.s2
        b = b._replace(switch_state=sw)
        pl = dict(b.plate_load); pl["P1"] = obs.p1; pl["P2"] = obs.p2
        b = b._replace(plate_load=pl)
        doors = dict(b.door_open); doors["D1"] = obs.d1; doors["D2"] = obs.d2
        b = b._replace(door_open=doors)
        self.belief = b

    def act(self, obs) -> str:
        self._sync(obs)
        v = valid_actions(self.belief)
        return self.rng.choice(v if v else ACTIONS)

    def observe_outcome(self, obs_before, action, res):
        pass


class EpisodicMemoryPolicy:
    name = "episodic"

    def __init__(self, seed: int = 0, max_mem: int = EPI_MEM_MAX):
        self.rng = random.Random(seed)
        self.mem = {}                 # signature -> deque of actions
        self.max_mem = max_mem
        self.belief = make_state()
        self.current_trace = []       # (signature, action) of this episode

    def episode_start(self):
        self.belief = make_state()
        self.current_trace = []

    def _sync(self, obs):
        b = self.belief
        b = b._replace(pos=(obs.x, obs.y))
        objs = dict(b.objs); objs["A"] = (obs.ax, obs.ay); objs["B"] = (obs.bx, obs.by)
        b = b._replace(objs=objs)
        sw = dict(b.switch_state); sw["S1"] = obs.s1; sw["S2"] = obs.s2
        b = b._replace(switch_state=sw)
        pl = dict(b.plate_load); pl["P1"] = obs.p1; pl["P2"] = obs.p2
        b = b._replace(plate_load=pl)
        doors = dict(b.door_open); doors["D1"] = obs.d1; doors["D2"] = obs.d2
        b = b._replace(door_open=doors)
        self.belief = b

    def act(self, obs) -> str:
        self._sync(obs)
        sig = sign(obs)
        if sig in self.mem and self.mem[sig]:
            return self.mem[sig].popleft()
        v = valid_actions(self.belief)
        act = self.rng.choice(v if v else ACTIONS)
        self.current_trace.append((sig, act))
        return act

    def observe_outcome(self, obs_before, action, res):
        if res.observation.finished and self.current_trace:
            # store the successful trajectory
            if len(self.mem) >= self.max_mem:
                # FIFO eviction: remove oldest key
                k = next(iter(self.mem))
                del self.mem[k]
            dq = deque(a for _, a in self.current_trace)
            self.mem[sign(obs_before)] = dq
            self.current_trace = []


class QLearningPolicy:
    name = "qlearning"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.Q = {}
        self.epsilon = QL_EPSILON0
        self.belief = make_state()
        self.episodes = 0
        self.last_state = None
        self.last_action = None

    def episode_start(self):
        self.belief = make_state()
        self.last_state = None
        self.last_action = None

    def _sync(self, obs):
        b = self.belief
        b = b._replace(pos=(obs.x, obs.y))
        objs = dict(b.objs); objs["A"] = (obs.ax, obs.ay); objs["B"] = (obs.bx, obs.by)
        b = b._replace(objs=objs)
        sw = dict(b.switch_state); sw["S1"] = obs.s1; sw["S2"] = obs.s2
        b = b._replace(switch_state=sw)
        pl = dict(b.plate_load); pl["P1"] = obs.p1; pl["P2"] = obs.p2
        b = b._replace(plate_load=pl)
        doors = dict(b.door_open); doors["D1"] = obs.d1; doors["D2"] = obs.d2
        b = b._replace(door_open=doors)
        self.belief = b

    def _q(self, sig, a):
        return self.Q.get((sig, a), 0.0)

    def act(self, obs) -> str:
        self._sync(obs)
        sig = sign(obs)
        self.last_state = sig
        v = valid_actions(self.belief)
        if not v:
            v = list(ACTIONS)
        if self.rng.random() < self.epsilon:
            a = self.rng.choice(v)
        else:
            a = max(v, key=lambda x: self._q(sig, x))
        self.last_action = a
        return a

    def observe_outcome(self, obs_before, action, res):
        r = 1.0 if res.observation.finished else 0.0
        if self.last_state is not None and self.last_action is not None:
            nsig = sign(res.observation)
            max_q = max((self._q(nsig, a) for a in ACTIONS), default=0.0)
            old = self._q(self.last_state, self.last_action)
            self.Q[(self.last_state, self.last_action)] = (
                old + QL_ALPHA * (r + QL_GAMMA * max_q - old))
        if res.observation.finished:
            self.episodes += 1
            self.epsilon = max(QL_EPSILON_MIN,
                               QL_EPSILON0 * (0.90 ** self.episodes))


class BayesPlannerPolicy:
    name = "bayes"

    def __init__(self, seed: int = 0, k_particles: int = K_PARTICLES):
        self.rng = random.Random(seed)
        self.k = k_particles
        # full hypothesis space over the declared generative family
        # (F1 x F2 x delays = 7 * 9 * 4 = 252 hypotheses)
        self.hyp_space = [
            LabConfig(d1_cond=c1, d2_cond=c2, d1_delay=bool(b1), d2_delay=bool(b2))
            for c1 in F1_TEMPLATES for c2 in F2_TEMPLATES
            for b1 in (0, 1) for b2 in (0, 1)
        ]
        self.particles = self.rng.choices(self.hyp_space, k=self.k)
        self.weights = [1.0 / self.k] * self.k
        self.belief = make_state()
        self.plan_cache = {}
        self.noise_q = 0.0
        self.current_plan = []        # follow-one-plan-until-invalid
        self.current_plan_config = None

    def episode_start(self, noise_q: float = 0.0):
        self.belief = make_state()
        self.noise_q = noise_q
        self.current_plan = []
        self.current_plan_config = None
        # plan cache is keyed by state + sampled hypothesis; it is kept
        # across episodes (bounded) so repeated states are not re-planned.

    def _sync(self, obs):
        b = self.belief
        b = b._replace(pos=(obs.x, obs.y))
        objs = dict(b.objs); objs["A"] = (obs.ax, obs.ay); objs["B"] = (obs.bx, obs.by)
        b = b._replace(objs=objs)
        sw = dict(b.switch_state); sw["S1"] = obs.s1; sw["S2"] = obs.s2
        b = b._replace(switch_state=sw)
        pl = dict(b.plate_load); pl["P1"] = obs.p1; pl["P2"] = obs.p2
        b = b._replace(plate_load=pl)
        doors = dict(b.door_open); doors["D1"] = obs.d1; doors["D2"] = obs.d2
        b = b._replace(door_open=doors)
        self.belief = b

    def _observed_doors(self, obs):
        return (obs.d1, obs.d2)

    def _update_posterior(self, obs_before, action, obs_after):
        before_doors = (obs_before.d1, obs_before.d2)
        after_doors = (obs_after.d1, obs_after.d2)
        w = []
        q = self.noise_q
        for c in self.particles:
            ll = 1.0
            for door_i, door in enumerate(DOORS):
                # predicted door reading under hypothesis c
                predicted = self._predict_doors_after(action, c)
                observed = after_doors[door_i]
                if predicted is None:
                    ll *= LAMBDA_LL
                    continue
                if q > 0.0:
                    p_match = 1.0 - q if predicted == observed else q
                    ll *= p_match
                else:
                    ll *= 1.0 if predicted == observed else LAMBDA_LL
            w.append(ll)
        total = sum(w)
        if total <= 0.0:
            self.weights = [1.0 / self.k] * self.k
            return
        self.weights = [x / total for x in w]
        ess = 1.0 / sum(x * x for x in self.weights)
        if ess < self.k / 2.0:
            idx = self.rng.choices(range(self.k), weights=self.weights, k=self.k)
            self.particles = [self.particles[i] for i in idx]
            self.weights = [1.0 / self.k] * self.k

    def _predict_doors_after(self, action, config):
        nxt, _ = transition_with_hidden(self.belief, action, config)
        if nxt is None:
            return None
        return (nxt.door_open["D1"], nxt.door_open["D2"])

    def act(self, obs) -> str:
        self._sync(obs)
        # follow the current plan while it remains executable under belief
        while self.current_plan:
            a = self.current_plan[0]
            if public_next(self.belief, a) is not None:
                self.current_plan.pop(0)
                return a
            self.current_plan.pop(0)
        # plan is exhausted/invalid: posterior sampling, plan under sample
        c = self.rng.choices(self.particles, weights=self.weights, k=1)[0]
        sig = (self.belief.pos, self.belief.objs["A"], self.belief.objs["B"],
               self.belief.switch_state["S1"], self.belief.switch_state["S2"],
               self.belief.plate_load["P1"], self.belief.plate_load["P2"],
               self.belief.door_open["D1"], self.belief.door_open["D2"],
               c.d1_cond, c.d2_cond, c.d1_delay, c.d2_delay)
        plan = self.plan_cache.get(sig)
        if plan is None:
            plan = ground_truth_plan(self.belief, c) or []
            self.plan_cache[sig] = plan
            if len(self.plan_cache) > 8000:
                # FIFO eviction of the cache (bounded memory)
                for key in list(self.plan_cache)[:2000]:
                    del self.plan_cache[key]
        self.current_plan = list(plan)
        self.current_plan_config = c
        if self.current_plan:
            return self.current_plan.pop(0)
        v = valid_actions(self.belief)
        return self.rng.choice(v if v else ACTIONS)

    def observe_outcome(self, obs_before, action, res):
        self._update_posterior(obs_before, action, res.observation)


class OraclePolicy:
    """Privileged diagnostic comparator. Knows the true config. Not a
    deployable baseline."""

    name = "oracle"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.config = None
        self.belief = make_state()
        self.plan_cache = {}

    def episode_start(self, config=None):
        self.config = config
        self.belief = make_state()
        self.current_plan = []
        # keep plan cache across episodes (state key includes the config)

    def _sync(self, obs):
        b = self.belief
        b = b._replace(pos=(obs.x, obs.y))
        objs = dict(b.objs); objs["A"] = (obs.ax, obs.ay); objs["B"] = (obs.bx, obs.by)
        b = b._replace(objs=objs)
        sw = dict(b.switch_state); sw["S1"] = obs.s1; sw["S2"] = obs.s2
        b = b._replace(switch_state=sw)
        pl = dict(b.plate_load); pl["P1"] = obs.p1; pl["P2"] = obs.p2
        b = b._replace(plate_load=pl)
        doors = dict(b.door_open); doors["D1"] = obs.d1; doors["D2"] = obs.d2
        b = b._replace(door_open=doors)
        self.belief = b

    def act(self, obs) -> str:
        self._sync(obs)
        while self.current_plan:
            a = self.current_plan[0]
            if public_next(self.belief, a) is not None:
                self.current_plan.pop(0)
                return a
            self.current_plan.pop(0)
        sig = (self.belief.pos, self.belief.objs["A"], self.belief.objs["B"],
               self.belief.switch_state["S1"], self.belief.switch_state["S2"],
               self.belief.plate_load["P1"], self.belief.plate_load["P2"],
               self.belief.door_open["D1"], self.belief.door_open["D2"])
        plan = self.plan_cache.get(sig)
        if plan is None:
            plan = ground_truth_plan(self.belief, self.config) or []
            self.plan_cache[sig] = plan
            if len(self.plan_cache) > 8000:
                for key in list(self.plan_cache)[:2000]:
                    del self.plan_cache[key]
        self.current_plan = list(plan)
        if self.current_plan:
            return self.current_plan.pop(0)
        v = valid_actions(self.belief)
        return self.rng.choice(v if v else ACTIONS)

    def observe_outcome(self, obs_before, action, res):
        pass
