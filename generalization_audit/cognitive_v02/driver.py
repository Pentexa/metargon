"""Shared episode driver for Cognitive Evaluation v0.2.

Runs any policy through the public Academy environment loop. All arms see
identical environments, noise, budgets and seeds. Records per-episode
telemetry, including whether a *new* verified-correct hypothesis was
created inside the episode (within-episode discovery), with the agent's
memory step counters.
"""
from __future__ import annotations

from academy.envs.gridworld import MechanismGrid

TRACE_KEYS = (
    "random_fallback_steps", "plans_used", "plan_cache_hits",
    "planner_ops", "plan_calls", "hypothesis_candidates",
    "hypothesis_retrieved", "memory_actions", "memory_correct",
    "memory_incorrect", "skipped_tried", "skipped_infeasible",
)


def trace_summary(agent):
    tr = getattr(agent, "trace", {})
    out = {k: tr.get(k, 0) for k in TRACE_KEYS}
    for attr in ("planner_ops", "plan_calls", "plan_cache_hits"):
        out[attr] = out.get(attr, 0) + int(getattr(agent, attr, 0) or 0)
    return out


def memory_summary(agent):
    """Compact hypothesis summary incl. created_step (for discovery
    attribution). antecedent is a sorted list of atoms."""
    return [
        {
            "id": h.h_id,
            "door": h.door,
            "status": h.status,
            "sup": h.sup,
            "con": h.con,
            "created_step": h.created_step,
            "antecedent": sorted(h.antecedent, key=repr),
        }
        for h in agent.memory.hypotheses
    ]


def run_episode(policy, spec, truths=None) -> dict:
    env = MechanismGrid(spec.config, spec.max_steps, spec.noise_q)
    obs = env.reset(spec.spec_seed)
    policy.episode_start(spec, env.geom)

    mem_before = _mem_step(policy)
    trace_before = trace_summary(policy)
    invalid = 0
    events = []
    first_open_step = None
    door_open = False
    actions = 0

    while not env.finished and env.steps < env.max_steps:
        action = policy.act(obs)
        actions += 1
        res = env.step(action)
        if not res.valid:
            invalid += 1
        if res.events:
            for door, ev in res.events:
                events.append((door, ev))
                if ev == "open" and first_open_step is None:
                    first_open_step = env.steps
                    door_open = True
        policy.learn(obs, action, res)
        obs = res.observation

    success = bool(env.finished)
    finish = getattr(policy, "finish_episode", None)
    if finish is not None:
        finish(success)

    try:
        mem_after = memory_summary(policy)
    except AttributeError:
        mem_after = None          # baseline policies have no memory
    mem_after_step = _mem_step(policy)
    trace_after = trace_summary(policy)
    deltas = {k: trace_after.get(k, 0) - trace_before.get(k, 0)
              for k in TRACE_KEYS}

    # within-episode discovery: a hypothesis created inside this episode
    # that matches ground truth (when truths are provided)
    discovered = False
    if truths is not None and mem_after:
        for h in mem_after:
            if mem_before < h["created_step"] <= mem_after_step:
                ante = frozenset(tuple(a) for a in h["antecedent"])
                if (h["door"], ante) in truths:
                    discovered = True
                    break

    return {
        "policy": getattr(policy, "name", type(policy).__name__),
        "success": success,
        "steps": env.steps,
        "invalid": invalid,
        "actions": actions,
        "door_open": door_open,
        "events": events,
        "first_open_step": first_open_step,
        "trace_delta": deltas,
        "mem_before_step": mem_before,
        "mem_after_step": mem_after_step,
        "discovered_this_episode": discovered,
        "mem": mem_after,
    }


def _mem_step(policy):
    try:
        return int(policy.memory.step)
    except Exception:
        return 0


def run_batch(policy, stream, truths=None):
    return [run_episode(policy, spec, truths) for spec in stream]
