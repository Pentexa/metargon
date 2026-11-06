"""METARGON Academy — baseline agents (Phase 5).

Matched exposure contract: every policy consumes the same episode stream,
the same public observations and the same budget. None accesses hidden
simulator state. The oracle is privileged and clearly labeled.

Policies:
  * random        — uniform random valid action.
  * episodic      — simple episodic-memory agent: store successful action
                    traces keyed by state signature; replay them.
  * qlearning     — tabular model-free Q-learning with epsilon-greedy
                    exploration (no function approximation).
  * bayes        — model-based Bayesian planner over the declared core
                    generative family (correct model class in core
                    conditions; deliberately mismatched on OOD).
  * oracle        — privileged comparator that knows the true wiring
                    (diagnostic only).
"""

from __future__ import annotations

import random
from collections import deque

from .common import belief_from_obs, sign, valid_actions
from ..envs.gridworld import (
    ACTIONS, DOORS, EnvConfig, TOPOLOGIES, make_ws, public_next,
    transition_with_hidden,
)
from ..ckpt.checkpoint import rng_from_state, rng_state_safe

LAMBDA_LL = 0.02


def _rng_state_safe(rng):
    return rng_state_safe(rng)


def _rng_from_state(rng, state):
    return rng_from_state(rng, state)
K_PARTICLES = 24
PLAN_MAX_EXPAND = 2500
QL_ALPHA = 0.30
QL_GAMMA = 0.95
QL_EPSILON0 = 0.35
QL_EPSILON_MIN = 0.05
EPI_MEM_MAX = 400


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.rng = random.Random(seed)
        self.geom = None
        self.belief = None

    def episode_start(self, spec, geom=None):
        self.geom = geom

    def act(self, obs) -> str:
        self.belief = belief_from_obs(obs, self.geom)
        v = valid_actions(self.belief, self.geom)
        return self.rng.choice(v if v else list(ACTIONS))

    def learn(self, obs_before, action, result):
        pass

    def learn_from_experience(self, exp):
        pass

    def finish_episode(self, success):
        pass

    def save_state(self) -> dict:
        return {"seed": self.seed, "rng": _rng_state_safe(self.rng)}

    def load_state(self, state: dict) -> None:
        self.seed = state["seed"]
        _rng_from_state(self.rng, state)

    def diagnostics(self) -> dict:
        return {}


class EpisodicPolicy:
    """Simple episodic-memory agent: store successful action traces keyed
    by state signature; replay them when the same signature reappears."""

    name = "episodic"

    def __init__(self, seed: int = 0, max_mem: int = EPI_MEM_MAX):
        self.seed = seed
        self.rng = random.Random(seed)
        self.max_mem = max_mem
        self.geom = None
        self.mem = {}
        self.current_trace = []
        self.episodes = 0
        self.successes = 0

    def episode_start(self, spec, geom=None):
        self.geom = geom
        self.current_trace = []

    def act(self, obs) -> str:
        sig = sign(obs)
        if sig in self.mem and self.mem[sig]:
            return self.mem[sig].popleft()
        belief = belief_from_obs(obs, self.geom)
        v = valid_actions(belief, self.geom)
        act = self.rng.choice(v if v else list(ACTIONS))
        self.current_trace.append((sig, act))
        return act

    def learn(self, obs_before, action, result):
        if result.observation.finished and self.current_trace:
            if len(self.mem) >= self.max_mem:
                oldest = next(iter(self.mem))
                del self.mem[oldest]
            dq = deque(a for _, a in self.current_trace)
            self.mem[sign(obs_before)] = dq
            self.current_trace = []

    def learn_from_experience(self, exp):
        pass

    def finish_episode(self, success):
        self.episodes += 1
        if success:
            self.successes += 1

    def save_state(self) -> dict:
        return {
            "seed": self.seed,
            "rng": _rng_state_safe(self.rng),
            "max_mem": self.max_mem,
            "mem": {str(k): list(v) for k, v in self.mem.items()},
            "episodes": self.episodes,
            "successes": self.successes,
        }

    def load_state(self, state: dict) -> None:
        self.seed = state["seed"]
        _rng_from_state(self.rng, state)
        self.max_mem = state["max_mem"]
        self.mem = {eval(k): deque(v) for k, v in state["mem"].items()}
        self.episodes = state["episodes"]
        self.successes = state["successes"]

    def diagnostics(self) -> dict:
        return {"mem_size": len(self.mem), "max_mem": self.max_mem,
                "episodes": self.episodes, "successes": self.successes}


class QLearningPolicy:
    name = "qlearning"

    def __init__(self, seed: int = 0, alpha: float = QL_ALPHA,
                 gamma: float = QL_GAMMA, eps0: float = QL_EPSILON0,
                 eps_min: float = QL_EPSILON_MIN):
        self.seed = seed
        self.rng = random.Random(seed)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon0 = eps0
        self.eps_min = eps_min
        self.epsilon = eps0
        self.geom = None
        self.Q = {}
        self.episodes = 0
        self.successes = 0
        self.last_state = None
        self.last_action = None

    def episode_start(self, spec, geom=None):
        self.geom = geom
        self.last_state = None
        self.last_action = None

    def _q(self, sig, a):
        return self.Q.get((sig, a), 0.0)

    def act(self, obs) -> str:
        belief = belief_from_obs(obs, self.geom)
        sig = sign(obs)
        self.last_state = sig
        v = valid_actions(belief, self.geom)
        if not v:
            v = list(ACTIONS)
        if self.rng.random() < self.epsilon:
            a = self.rng.choice(v)
        else:
            a = max(v, key=lambda x: self._q(sig, x))
        self.last_action = a
        return a

    def learn(self, obs_before, action, result):
        r = 1.0 if result.observation.finished else 0.0
        if self.last_state is not None and self.last_action is not None:
            nsig = sign(result.observation)
            max_q = max((self._q(nsig, a) for a in ACTIONS), default=0.0)
            old = self._q(self.last_state, self.last_action)
            self.Q[(self.last_state, self.last_action)] = (
                old + self.alpha * (r + self.gamma * max_q - old))
        if result.observation.finished:
            self.episodes += 1
            self.epsilon = max(self.eps_min, self.epsilon0 * (0.90 ** self.episodes))

    def learn_from_experience(self, exp):
        # offline Q-update on a replayed transition
        sig0 = sign(exp.obs_before)
        sig1 = sign(exp.obs_after)
        max_q = max((self._q(sig1, a) for a in ACTIONS), default=0.0)
        old = self._q(sig0, exp.action)
        self.Q[(sig0, exp.action)] = (
            old + self.alpha * (exp.reward + self.gamma * max_q - old))

    def finish_episode(self, success):
        if success:
            self.successes += 1

    def save_state(self) -> dict:
        return {
            "seed": self.seed,
            "rng": _rng_state_safe(self.rng),
            "alpha": self.alpha, "gamma": self.gamma,
            "epsilon0": self.epsilon0, "eps_min": self.eps_min,
            "epsilon": self.epsilon,
            "Q": [list(map(list, k)) + [v] for k, v in self.Q.items()],
            "episodes": self.episodes,
            "successes": self.successes,
        }

    def load_state(self, state: dict) -> None:
        self.seed = state["seed"]
        _rng_from_state(self.rng, state)
        self.alpha = state["alpha"]
        self.gamma = state["gamma"]
        self.epsilon0 = state["epsilon0"]
        self.eps_min = state["eps_min"]
        self.epsilon = state["epsilon"]
        self.Q = {}
        for row in state["Q"]:
            sig = tuple(tuple(x) for x in row[:-1])
            self.Q[(sig, row[-1][0])] = row[-1][1]
        self.episodes = state["episodes"]
        self.successes = state["successes"]

    def diagnostics(self) -> dict:
        return {"q_size": len(self.Q), "epsilon": self.epsilon,
                "episodes": self.episodes, "successes": self.successes}


class BayesPlannerPolicy:
    """Model-based Bayesian planner. Given the declared core generative
    family (the correct model class in core conditions), it keeps a
    particle-filter posterior over configurations, samples one hypothesis
    per decision and plans optimally under it. On OOD conditions this
    model class is mismatched — by design."""

    name = "bayes"

    def __init__(self, seed: int = 0, k_particles: int = K_PARTICLES,
                 plan_max_expand: int = PLAN_MAX_EXPAND):
        self.seed = seed
        self.rng = random.Random(seed)
        self.k = k_particles
        self.plan_max_expand = plan_max_expand
        self.geom = None

        from ..envs.gridworld import EnvConfig as EC
        from ..envs.tasks import CORE_RULESETS
        self._core_d1 = CORE_RULESETS["TP1"][0]
        self._core_d2 = CORE_RULESETS["TP1"][1]
        self.hyp_space = [
            EC(topology="TP1",
               d1_rule=c1, d2_rule=c2,
               d1_delay=bool(b1), d2_delay=bool(b2))
            for c1 in self._core_d1 for c2 in self._core_d2
            for b1 in (0, 1) for b2 in (0, 1)
        ]
        self.particles = self.rng.choices(self.hyp_space, k=self.k)
        self.weights = [1.0 / self.k] * self.k
        self.belief = None
        self.plan_cache = {}
        self.noise_q = 0.0
        self.episodes = 0
        self.successes = 0

    def episode_start(self, spec, geom=None):
        self.geom = geom
        self.noise_q = spec.noise_q if spec is not None else 0.0

    def _sync(self, obs):
        self.belief = belief_from_obs(obs, self.geom)

    def _predict_doors_after(self, action, config):
        nxt, _ = transition_with_hidden(self.belief, action, config,
                                         self.geom)
        if nxt is None:
            return None
        return (nxt.door_open["D1"], nxt.door_open["D2"])

    def _update_posterior(self, obs_before, action, obs_after):
        before_doors = (obs_before.d1, obs_before.d2)
        after_doors = (obs_after.d1, obs_after.d2)
        w = []
        q = self.noise_q
        for c in self.particles:
            ll = 1.0
            for door_i, door in enumerate(DOORS):
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

    def act(self, obs) -> str:
        self._sync(obs)
        c = self.rng.choices(self.particles, weights=self.weights, k=1)[0]
        sig = (self.belief.pos, self.belief.objs["A"], self.belief.objs["B"],
               self.belief.switch_state["S1"], self.belief.switch_state["S2"],
               self.belief.plate_load["P1"], self.belief.plate_load["P2"],
               self.belief.door_open["D1"], self.belief.door_open["D2"],
               c.d1_rule, c.d2_rule, c.d1_delay, c.d2_delay)
        plan = self.plan_cache.get(sig)
        if plan is None:
            plan = self._ground_truth_plan(c) or []
            self.plan_cache[sig] = plan
            if len(self.plan_cache) > 8000:
                for key in list(self.plan_cache)[:2000]:
                    del self.plan_cache[key]
        if plan:
            return plan[0]
        v = valid_actions(self.belief, self.geom)
        return self.rng.choice(v if v else list(ACTIONS))

    def _ground_truth_plan(self, config):
        from collections import deque
        from .common import state_key
        state = self.belief
        if state.pos == self.geom.goal:
            return []
        q = deque([(state, [])])
        seen = {state_key(state)}
        while q and len(seen) < self.plan_max_expand:
            st, path = q.popleft()
            children = []
            for a in ACTIONS:
                nxt, _ = transition_with_hidden(st, a, config, self.geom)
                if nxt is None:
                    continue
                k = state_key(nxt)
                if k in seen:
                    continue
                seen.add(k)
                np = path + [a]
                if nxt.pos == self.geom.goal:
                    return np
                children.append((nxt, np))
            children.sort(key=lambda t: (abs(t[0].pos[0] - self.geom.goal[0])
                                         + abs(t[0].pos[1] - self.geom.goal[1])))
            q.extend(children)
        return None

    def learn(self, obs_before, action, result):
        self._update_posterior(obs_before, action, result.observation)

    def learn_from_experience(self, exp):
        self._update_posterior(exp.obs_before, exp.action, exp.obs_after)

    def finish_episode(self, success):
        self.episodes += 1
        if success:
            self.successes += 1

    def save_state(self) -> dict:
        return {
            "seed": self.seed,
            "rng": _rng_state_safe(self.rng),
            "k": self.k,
            "plan_max_expand": self.plan_max_expand,
            "particles": [c.key for c in self.particles],
            "weights": self.weights,
            "noise_q": self.noise_q,
            "plan_cache": {str(k): list(v) for k, v in list(self.plan_cache.items())[:4000]},
            "episodes": self.episodes,
            "successes": self.successes,
        }

    def load_state(self, state: dict) -> None:
        self.seed = state["seed"]
        _rng_from_state(self.rng, state)
        self.k = state["k"]
        self.plan_max_expand = state["plan_max_expand"]
        by_key = {c.key: c for c in self.hyp_space}
        self.particles = [by_key[tuple(k)] for k in state["particles"]]
        self.weights = state["weights"]
        self.noise_q = state["noise_q"]
        self.plan_cache = {}
        for k, v in state["plan_cache"].items():
            self.plan_cache[eval(k)] = v
        self.episodes = state["episodes"]
        self.successes = state["successes"]

    def diagnostics(self) -> dict:
        return {"particles": self.k, "episodes": self.episodes,
                "successes": self.successes,
                "plan_cache_size": len(self.plan_cache)}


class OraclePolicy:
    """Privileged diagnostic comparator: knows the true wiring. Excluded
    from deployable rankings."""

    name = "oracle"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.rng = random.Random(seed)
        self.geom = None
        self.config = None
        self.plan_cache = {}

    def episode_start(self, spec, geom=None):
        self.geom = geom
        self.config = spec.config
        # keep plan cache across episodes (state key includes the config)

    def act(self, obs) -> str:
        from .common import belief_from_obs
        belief = belief_from_obs(obs, self.geom)
        sig = (belief.pos, belief.objs["A"], belief.objs["B"],
               belief.switch_state["S1"], belief.switch_state["S2"],
               belief.plate_load["P1"], belief.plate_load["P2"],
               belief.door_open["D1"], belief.door_open["D2"],
               self.config.key)
        plan = self.plan_cache.get(sig)
        if plan is None:
            plan = self._plan(belief) or []
            self.plan_cache[sig] = plan
        if plan:
            return plan[0]
        v = valid_actions(belief, self.geom)
        return self.rng.choice(v if v else list(ACTIONS))

    def _plan(self, belief):
        from collections import deque
        from .common import state_key
        state = belief
        if state.pos == self.geom.goal:
            return []
        q = deque([(state, [])])
        seen = {state_key(state)}
        while q and len(seen) < PLAN_MAX_EXPAND:
            st, path = q.popleft()
            children = []
            for a in ACTIONS:
                nxt, _ = transition_with_hidden(st, a, self.config, self.geom)
                if nxt is None:
                    continue
                k = state_key(nxt)
                if k in seen:
                    continue
                seen.add(k)
                np = path + [a]
                if nxt.pos == self.geom.goal:
                    return np
                children.append((nxt, np))
            children.sort(key=lambda t: (abs(t[0].pos[0] - self.geom.goal[0])
                                         + abs(t[0].pos[1] - self.geom.goal[1])))
            q.extend(children)
        return None

    def learn(self, obs_before, action, result):
        pass

    def learn_from_experience(self, exp):
        pass

    def finish_episode(self, success):
        pass

    def save_state(self) -> dict:
        return {"seed": self.seed, "rng": _rng_state_safe(self.rng)}

    def load_state(self, state: dict) -> None:
        self.seed = state["seed"]
        _rng_from_state(self.rng, state)

    def diagnostics(self) -> dict:
        return {}
