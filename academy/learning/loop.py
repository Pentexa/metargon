"""METARGON Academy — the online learning loop (Phase 3).

Observe -> Decide -> Act -> Receive Feedback -> Update Memory/Model -> Continue.

A single episode stream (deterministic, generated from the training pool
and a *calibrated curriculum schedule*) is consumed identically by every
policy, which is what makes baseline comparisons fair (matched task
exposure, matched seeds, matched budgets).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from ..core.interfaces import Experience
from ..envs.gridworld import MechanismGrid
from ..envs.tasks import EpisodeSpec, make_episodes
from ..eval.protocol import spec_for
from .curriculum import BINS, CurriculumManager
from .experience_memory import BoundedExperienceMemory


@dataclass
class TrainRunResult:
    agent_name: str
    run_id: str
    episodes: int
    wall_s: float
    success_rate: float
    mean_steps: float
    curve: list = field(default_factory=list)       # per-episode records
    diagnostics_trace: list = field(default_factory=list)
    checkpoint_paths: list = field(default_factory=list)
    replay_count: int = 0


# ---------------------------------------------------------------------------
# Curriculum calibration (measured difficulty, not an arbitrary schedule).
# ---------------------------------------------------------------------------
def calibrate_curriculum(pool, episodes_per_bin: int = 40,
                         seed: int = 0, policy_factory=None,
                         max_steps: int = 80):
    """Run a reference learner on a fixed sample of each difficulty bin and
    return measured success/uncertainty per bin. This *measures* the
    difficulty of the task family before scheduling training."""
    from ..agents.metargon_agent import MetargonAcademyAgent
    from ..envs.tasks import EpisodeSpec as ES, difficulty_of
    factory = policy_factory or (lambda s: MetargonAcademyAgent(seed=s))
    rng = random.Random(seed)
    stats = {}
    for b in BINS:
        bin_pool = [c for c in pool if difficulty_of(c).bin == b]
        if not bin_pool:
            stats[b] = {"n": 0, "success_rate": 0.0, "std": 0.0}
            continue
        agent = factory(seed)
        ok, steps_total, n = 0, 0, 0
        wins = []
        for _ in range(episodes_per_bin):
            cfg = rng.choice(bin_pool)
            spec = ES(config=cfg, max_steps=max_steps,
                      noise_q=0.0,
                      spec_seed=seed * 100000 + n,
                      pool_tag="train", difficulty=difficulty_of(cfg))
            rec = run_one_episode(agent, spec)
            ok += 1 if rec["success"] else 0
            wins.append(1.0 if rec["success"] else 0.0)
            n += 1
        mean = sum(wins) / n if n else 0.0
        std = (sum((w - mean) ** 2 for w in wins) / n) ** 0.5 if n else 0.0
        stats[b] = {"n": n, "success_rate": mean, "std": std}
    return stats


def build_schedule(calib_stats: dict, total_episodes: int,
                   episodes_per_block: int = 40,
                   seed: int = 0,
                   success_target: float = 0.85,
                   harder_epsilon: float = 0.10) -> List[str]:
    """Deterministic schedule: focus blocks on measured-hardest bins,
    progressing when measured success crosses the target. The same
    schedule is then used for every policy (matched exposure)."""
    rng = random.Random(seed)
    schedule = []
    active = "easy"
    blocks = max(1, total_episodes // episodes_per_block)
    for _ in range(blocks):
        # choose active bin by measured success
        best = None
        for b in BINS:
            st = calib_stats.get(b, {"success_rate": 0.0})
            if best is None or st["success_rate"] < best[1]:
                best = (b, st["success_rate"])
        if best[1] >= success_target - harder_epsilon:
            idx = BINS.index(best[0])
            active = BINS[min(idx + 1, len(BINS) - 1)]
        else:
            active = best[0]
        # 25% exploration jitter among unvisited/underperforming bins
        for _ in range(episodes_per_block):
            if rng.random() < 0.25:
                b = rng.choice([b for b in BINS if b != active])
                schedule.append(b)
            else:
                schedule.append(active)
    return schedule[:total_episodes]


def make_train_stream(train_pool, schedule: List[str], seed: int = 0,
                      max_steps: int = 80, noise_q: float = 0.0) -> List[EpisodeSpec]:
    from ..envs.tasks import difficulty_of
    rng = random.Random(seed)
    by_bin = {}
    for c in train_pool:
        by_bin.setdefault(difficulty_of(c).bin, []).append(c)
    out = []
    for e, b in enumerate(schedule):
        pool_bin = by_bin.get(b) or train_pool
        cfg = rng.choice(pool_bin)
        diff = difficulty_of(cfg)
        out.append(EpisodeSpec(config=cfg,
                               max_steps=max_steps,
                               noise_q=noise_q,
                               spec_seed=seed * 1000 + e,
                               pool_tag="train",
                               difficulty=diff))
    return out


# ---------------------------------------------------------------------------
# Episode runner.
# ---------------------------------------------------------------------------
def run_one_episode(agent, spec: EpisodeSpec, max_steps: Optional[int] = None,
                    experience_memory=None, collect_trace: bool = False):
    """Run one episode and return a record dict."""
    env = MechanismGrid(spec.config,
                        max_steps=max_steps or spec.max_steps,
                        noise_q=spec.noise_q)
    obs = env.reset(spec.spec_seed)
    agent.episode_start(spec_for(agent, spec), env.geom)
    steps = 0
    door_events = 0
    trace = [] if collect_trace else None
    success = False
    while steps < env.max_steps:
        action = agent.act(obs)
        before = obs
        res = env.step(action)
        agent.learn(before, action, res)
        reward = 1.0 if res.terminated else 0.0
        if experience_memory is not None:
            experience_memory.push(Experience(
                obs_before=before, action=action,
                obs_after=res.observation,
                reward=reward, terminated=res.terminated,
                events=res.events, step_index=steps,
                spec_seed=spec.spec_seed, pool_tag=spec.pool_tag))
        steps += 1
        door_events += len(res.events)
        if trace is not None:
            trace.append({"action": action, "success": res.terminated})
        obs = res.observation
        if res.terminated:
            success = True
            break
    agent.finish_episode(success)
    return {
        "success": success,
        "steps": steps,
        "door_events": door_events,
        "trace": trace,
        "diag": agent.diagnostics(),
    }


# ---------------------------------------------------------------------------
# Training over a stream.
# ---------------------------------------------------------------------------
def train_agent(agent, stream: List[EpisodeSpec], cfg, run_id: str,
                experience_memory=None, on_episode=None,
                replay_enabled: bool = True,
                replay_frequency: int = 8,
                replay_batch: int = 64,
                eval_at: Optional[Callable] = None,
                checkpoint_paths: Optional[list] = None):
    t0 = time.perf_counter()
    curve = []
    diag_trace = []
    replay_count = 0
    total_success = 0
    total_steps = 0
    mem = experience_memory or BoundedExperienceMemory(capacity=cfg.get("replay_capacity", 4000))
    rng = random.Random(cfg.get("run_seed", 0) + hash(run_id) % 1000)
    for ep_i, spec in enumerate(stream):
        rec = run_one_episode(agent, spec, experience_memory=mem)
        curve.append({"episode": ep_i, "success": rec["success"],
                      "steps": rec["steps"],
                      "door_events": rec["door_events"]})
        total_success += 1 if rec["success"] else 0
        total_steps += rec["steps"]
        diag_trace.append(rec["diag"])
        if replay_enabled and (ep_i + 1) % replay_frequency == 0 and len(mem) > 0:
            for exp in mem.sample_batch(rng, replay_batch):
                agent.learn_from_experience(exp)
            replay_count += 1
        if eval_at is not None:
            eval_at(agent, ep_i, curve, diag_trace)
        if on_episode is not None:
            on_episode(agent, ep_i, rec, curve)
    wall = time.perf_counter() - t0
    return TrainRunResult(
        agent_name=agent.name,
        run_id=run_id,
        episodes=len(stream),
        wall_s=wall,
        success_rate=total_success / len(stream) if stream else 0.0,
        mean_steps=total_steps / len(stream) if stream else 0.0,
        curve=curve,
        diagnostics_trace=diag_trace,
        checkpoint_paths=checkpoint_paths or [],
        replay_count=replay_count,
    )
