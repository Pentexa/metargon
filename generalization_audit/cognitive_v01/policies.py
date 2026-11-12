"""Policy harness: unmodified Academy student, baselines, and diagnostics.

The student is used exactly as shipped (``MetargonAcademyAgent`` with its
default constructor). Baselines come from ``academy.agents.baselines``.
The *oracle* is privileged (it reads hidden wiring) — diagnostic only,
excluded from all rankings.

The shared episode driver is the single loop used by every protocol, so
all policies see identical environments, noise, and budgets.
"""
from __future__ import annotations

from collections import deque

from academy.agents.baselines import EpisodicPolicy, QLearningPolicy, RandomPolicy
from academy.agents.metargon_agent import MetargonAcademyAgent
from academy.envs.gridworld import ACTIONS, MechanismGrid, make_ws, transition_with_hidden

TRACE_KEYS = (
    "random_fallback_steps", "plans_used", "plan_cache_hits",
    "planner_ops", "plan_calls", "hypothesis_candidates",
    "hypothesis_retrieved", "memory_actions", "memory_correct",
    "memory_incorrect", "skipped_tried", "skipped_infeasible",
)


def make_policy(name, seed):
    if name in ("metargon_trained", "metargon_muted", "metargon_fresh"):
        return MetargonAcademyAgent(seed=seed)
    if name == "random":
        return RandomPolicy(seed=seed)
    if name == "episodic":
        return EpisodicPolicy(seed=seed)
    if name == "qlearning":
        return QLearningPolicy(seed=seed)
    if name == "oracle":
        return OraclePolicy(seed=seed)
    raise ValueError(f"unknown policy: {name}")


# ---------------------------------------------------------------------------
# Memory / trace summaries (bounded, JSON-safe)
# ---------------------------------------------------------------------------
def memory_summary(agent):
    return [
        {
            "id": h.h_id,
            "door": h.door,
            "antecedent": sorted(h.antecedent, key=repr),
            "status": h.status,
            "sup": h.sup,
            "con": h.con,
        }
        for h in agent.memory.hypotheses
    ]


def trace_summary(agent):
    tr = getattr(agent, "trace", {})
    out = {k: tr.get(k, 0) for k in TRACE_KEYS}
    # planner counters live as attributes on the student agent
    for attr in ("planner_ops", "plan_calls", "plan_cache_hits"):
        out[attr] = out.get(attr, 0) + int(getattr(agent, attr, 0) or 0)
    return out


def wipe_memory(agent):
    """Remove declarative learned content while keeping the agent machinery.

    Used for the *memory-muted* control: identical training exposure, but
    the learned rule store is emptied before evaluation, so any remaining
    performance must come from non-declarative mechanisms.
    """
    m = agent.memory
    m.hypotheses = []
    m.evidence_bank = []
    m.bank_retrievals = 0


# ---------------------------------------------------------------------------
# Shared episode driver
# ---------------------------------------------------------------------------
def run_episode(policy, spec) -> dict:
    env = MechanismGrid(spec.config, spec.max_steps, spec.noise_q)
    obs = env.reset(spec.spec_seed)
    policy.episode_start(spec, env.geom)

    trace_before = trace_summary(policy)
    invalid = 0
    events = []
    door_open = {}
    actions = []
    while not env.finished and env.steps < env.max_steps:
        action = policy.act(obs)
        actions.append(action)
        res = env.step(action)
        if not res.valid:
            invalid += 1
        for door, ev in res.events:
            door_open[door] = True
            events.append((door, ev))
        policy.learn(obs, action, res)
        obs = res.observation

    success = bool(env.finished)
    finish = getattr(policy, "finish_episode", None)
    if finish is not None:
        finish(success)

    trace_after = trace_summary(policy)
    summary = {
        "policy": getattr(policy, "name", type(policy).__name__),
        "success": success,
        "steps": env.steps,
        "invalid": invalid,
        "door_open": door_open,
        "events": events,
        "actions": actions,
        "trace": trace_after,
        "trace_delta": {k: trace_after.get(k, 0) - trace_before.get(k, 0)
                        for k in TRACE_KEYS},
    }
    try:
        summary["memory"] = memory_summary(policy)
    except AttributeError:
        summary["memory"] = []
    if hasattr(policy, "diagnostics"):
        try:
            summary["diagnostics"] = policy.diagnostics()
        except Exception:
            summary["diagnostics"] = {}
    return summary


# ---------------------------------------------------------------------------
# Privileged diagnostic oracle
# ---------------------------------------------------------------------------
class OraclePolicy:
    """Privileged comparator: BFS over the true dynamics finds a shortest
    action sequence to the goal and replays it. Reads hidden wiring —
    diagnostic only, never used in rankings."""

    name = "oracle"

    def __init__(self, seed: int = 0, max_expand: int = 400000):
        self.seed = seed
        self.max_expand = max_expand
        self._plan = []

    def episode_start(self, spec, geom=None):
        self._plan = self._solve(spec, geom)
        self._plan = deque(self._plan)

    def _solve(self, spec, geom):
        from collections import deque
        start = make_ws(geom)
        goal = geom.goal

        def key(st):
            return (st.pos, st.objs.get("A"), st.objs.get("B"),
                    tuple(st.switch_state[s] for s in ("S1", "S2")),
                    tuple(st.plate_load[p] for p in ("P1", "P2")),
                    tuple(st.door_open[d] for d in ("D1", "D2")),
                    tuple(st.armed[d] for d in ("D1", "D2")))

        q = deque([(start, [])])
        seen = {key(start)}
        while q and len(seen) < self.max_expand:
            st, path = q.popleft()
            if st.pos == goal:
                return path
            for action in ACTIONS:
                nxt, _ = transition_with_hidden(st, action, spec.config, geom)
                if nxt is None:
                    continue
                k = key(nxt)
                if k in seen:
                    continue
                seen.add(k)
                q.append((nxt, path + [action]))
        return []

    def act(self, obs) -> str:
        if self._plan:
            return self._plan.popleft()
        return "N"

    def learn(self, obs_before, action, result):
        pass

    def finish_episode(self, success):
        pass

    def save_state(self):
        return {"seed": self.seed}

    def load_state(self, state):
        self.seed = state["seed"]
