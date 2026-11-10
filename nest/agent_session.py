"""METARGON NEST — agent session controller.

Wraps the REAL Academy student agent (`MetargonAcademyAgent`) and the REAL
Academy environment (`MechanismGrid`), and runs the same
observe -> plan -> act -> learn cycle as the experiment harness. All
control flow lives here; the HTTP layer only serializes snapshots.

The agent is never scripted: every action comes from `agent.act(obs)`,
every learning step from `agent.learn(...)`. Hidden rules are never
passed to the agent; they exist only in the `EnvConfig` used to
construct the environment.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from academy.agents.metargon_agent import MetargonAcademyAgent
from academy.envs.gridworld import (
    ACTIONS, DOORS, GOAL, PLATES, SWITCHES, TOPOLOGIES,
)
from academy.envs.tasks import EpisodeSpec, get_splits, make_episodes

from .storage import NestStorage
from .teaching import Teaching

STATUS_IDLE = "idle"
STATUS_RUNNING = "running"
STATUS_PAUSED = "paused"
STATUS_STOPPED = "stopped"
STATUS_FINISHED = "finished"

ROLE = "role"
VERIFIED = "verified"
CONTRADICTED = "contradicted"
HYPOTHESIS = "hypothesis"
PREDICTED = "predicted"

MAX_GLOBAL_STEPS = 4000          # safety cap across all sessions
STEP_SLEEP = 0.01               # cadence between steps while running


@dataclass
class Challenge:
    """A supported challenge. The goal is expressed over public state."""
    pool_tag: str = "train"      # train | val | test | ood | custom
    difficulty: str = "medium"   # easy | medium | hard | ood
    focus_door: Optional[str] = None  # D1 | D2 | None (reach goal)
    seed: int = 0
    description: str = ""

    def as_dict(self):
        return {"pool_tag": self.pool_tag, "difficulty": self.difficulty,
                "focus_door": self.focus_door, "seed": self.seed,
                "description": self.description}


@dataclass
class NestSession:
    """One interactive session backed by a persistent student agent."""

    storage: NestStorage
    seed: int = 20261110
    # agent configuration (fixed, sane defaults; mirrors Academy student)
    agent_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        # Memory v0.7 student configuration for NEST: episode-scoped causal
        # memory (avoids cross-task contradiction destroying everything),
        # atom-level transfer, bank reuse, and a transfer threshold of 1
        # verified epoch so rules verified in one teaching episode can
        # influence later related tasks.
        defaults = dict(
            memory_episode_scoped=True,
            memory_min_verified_epochs_for_transfer=1,
            memory_atom_transfer=True,
            probe_bank_reuse=True,
            probe_reset_per_episode=True,
        )
        defaults.update(self.agent_kwargs)
        self.agent_kwargs = defaults
        self.agent = MetargonAcademyAgent(seed=self.seed, **self.agent_kwargs)
        self.teaching = Teaching(self)
        self.env = None
        self.spec = None
        self.challenge: Optional[Challenge] = None
        self.status = STATUS_IDLE
        self.steps_this_episode = 0
        self.global_steps = 0
        self.episode_index = 0
        self.episode_successes = 0
        self.episode_door_opened: dict = {}
        self.last_action = None
        self.last_result_valid = None
        self.last_events = ()
        self.last_obs = None
        self.last_action_reason = ""
        self._lock = threading.RLock()
        self._wake = threading.Condition(self._lock)
        self._thread = None
        self._stop_requested = False
        self._run_requested = False
        self._step_requested = False
        self._single_step = False
        self._active_episode = False
        self.messages = []          # bounded conversational log (client may also keep history)
        self.notice = ""
        self._last_error = None
        self.last_guidance = False

    # ------------------------------------------------------------------ #
    # Episode / environment lifecycle
    # ------------------------------------------------------------------ #
    def _build_spec(self, challenge: Challenge, splits) -> EpisodeSpec:
        pool = splits.get(challenge.pool_tag)
        if not pool:
            raise ValueError(f"unknown pool: {challenge.pool_tag}")
        # pick a configuration matching difficulty when requested
        if challenge.difficulty and challenge.difficulty != "any":
            from academy.envs.tasks import difficulty_of
            candidates = [c for c in pool
                          if difficulty_of(c).bin == challenge.difficulty]
            pool = candidates or pool
        episodes = make_episodes(pool, challenge.pool_tag,
                                 seed=challenge.seed, episodes=1)
        return episodes[0]

    def _start_episode(self, challenge: Challenge):
        splits = get_splits()
        spec = self._build_spec(challenge, splits)
        from academy.envs.gridworld import MechanismGrid
        env = MechanismGrid(spec.config, spec.max_steps, spec.noise_q)
        obs = env.reset(spec.spec_seed)
        self.agent.episode_start(spec, env.geom)
        self.teaching.begin_episode_snapshot()
        self.env = env
        self.spec = spec
        self.challenge = challenge
        self.steps_this_episode = 0
        self.episode_door_opened = {}
        self.last_action = None
        self.last_events = ()
        self.last_obs = obs
        self.last_action_reason = ""
        self._active_episode = True
        self.trace(f"Starting a new episode · {challenge.description} · "
                   f"topology {spec.config.topology}, max {spec.max_steps} steps.")
        return self.snapshot()

    def _one_step(self):
        """Execute exactly one observe->plan->act->learn cycle."""
        if self.env is None or not self._active_episode:
            return
        if self.env.finished or self.env.steps >= self.env.max_steps:
            self._finish_episode()
            return
        if self.global_steps >= MAX_GLOBAL_STEPS:
            self._request_stop("Global step budget exhausted.")
            return
        obs_before = self.last_obs or self.env.observe()
        # The student remains the sole action authority. Teaching only
        # injects unverified candidate hypotheses that the student's own
        # planner/probe machinery may or may not act on.
        action = self.agent.act(obs_before)
        self.last_guidance = self.teaching.has_teacher_suggestion()
        self.last_action = action
        self.last_action_reason = self._reason_for_action(action, obs_before)
        result = self.env.step(action)
        self.agent.learn(obs_before, action, result)
        self.teaching.record_step(
            action, result,
            note="teacher-suggestion active" if self.last_guidance else "")
        self.steps_this_episode += 1
        self.global_steps += 1
        self.last_result_valid = result.valid
        self.last_events = result.events
        for door, ev in result.events:
            if ev == "open":
                self.episode_door_opened[door] = True
        self.last_obs = result.observation
        if result.terminated:
            self._finish_episode(success=True)

    def _finish_episode(self, success: bool = False):
        self.agent.finish_episode(success)
        self.episode_index += 1
        if success:
            self.episode_successes += 1
        outcome = "succeeded" if success else "ended without reaching the goal"
        self.trace(f"Episode finished — {outcome} · learned memory summarized below.")
        self._active_episode = False
        self.status = STATUS_FINISHED if not self._run_requested else self.status

    def _reason_for_action(self, action: str, obs) -> str:
        """Explain the last action from REAL agent telemetry."""
        tr = self.agent.trace
        if action in ACTIONS[:4]:
            # movement: part of a plan for a probe, a door crossing, the
            # goal, or a random fallback
            if tr.get("random_fallback_steps", 0) and \
                    self.last_action_reason.startswith("random"):
                return "random fallback (no feasible plan found)"
            return "step along the planner's plan (goal / probe / door crossing)"
        if action == "Toggle":
            return "toggle a switch probe (testing a hypothesis or a novel single)"
        if action.startswith("Push"):
            return f"push an object ({action}) to set up a plate/absence probe"
        return f"execute {action}"

    # ------------------------------------------------------------------ #
    # Worker loop (runs in a background thread; UI never blocks)
    # ------------------------------------------------------------------ #
    def _worker(self):
        with self._lock:
            while not self._stop_requested:
                if self._step_requested:
                    self._step_requested = False
                    try:
                        self._one_step()
                    except Exception as exc:  # surface worker errors for diagnosis
                        self._last_error = f"{type(exc).__name__}: {exc}"
                        self._step_requested = False
                        self._run_requested = False
                        self.status = STATUS_STOPPED
                    if self.steps_this_episode >= (self.env.max_steps if self.env else 0):
                        self._finish_episode()
                elif self._run_requested and self.status == STATUS_RUNNING:
                    try:
                        self._one_step()
                    except Exception as exc:
                        self._last_error = f"{type(exc).__name__}: {exc}"
                        self._run_requested = False
                        self.status = STATUS_STOPPED
                    self._wake.wait(STEP_SLEEP)
                else:
                    self._wake.wait(0.2)

    def start(self):
        with self._lock:
            if self.env is None:
                self._start_episode(self.challenge or self._default_challenge())
            self.status = STATUS_RUNNING
            self._run_requested = True
            self._ensure_thread()

    def pause(self):
        with self._lock:
            self.status = STATUS_PAUSED
            self._run_requested = False

    def continue_running(self):
        with self._lock:
            self.status = STATUS_RUNNING
            self._run_requested = True
            self._ensure_thread()
            self._wake.notify_all()

    def single_step(self):
        with self._lock:
            self.status = STATUS_PAUSED
            self._run_requested = False
            if self.env is None:
                self._start_episode(self.challenge or self._default_challenge())
            self._step_requested = True
            self._ensure_thread()
            self._wake.notify_all()

    def run_steps(self, n: int = 1):
        """Synchronously execute up to n steps (tests/demos). Never blocks
        the UI thread because it is called only from tests/demo scripts."""
        with self._lock:
            self.status = STATUS_PAUSED
            self._run_requested = False
            self._step_requested = False
            if self.env is None:
                self._start_episode(self.challenge or self._default_challenge())
            done = 0
            while done < n:
                if self._stop_requested:
                    break
                if self.env is None or not self._active_episode:
                    break
                if self.env.finished or self.env.steps >= self.env.max_steps:
                    self._finish_episode()
                    break
                if self.global_steps >= MAX_GLOBAL_STEPS:
                    self._request_stop("Global step budget exhausted.")
                    break
                self._one_step()
                done += 1
            return done

    def stop(self):
        with self._lock:
            self._run_requested = False
            self._step_requested = False
            self.status = STATUS_STOPPED

    def reset_episode(self):
        with self._lock:
            self.stop()
            if self.challenge:
                self._start_episode(self.challenge)

    def new_challenge(self, challenge: Challenge):
        with self._lock:
            self.stop()
            self.challenge = challenge
            self._start_episode(challenge)
            self.status = STATUS_PAUSED

    def _request_stop(self, notice: str = ""):
        self.notice = notice
        self.stop()

    def _ensure_thread(self):
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def _default_challenge(self):
        return Challenge(description="reach the goal in the training academy")

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self) -> dict:
        with self._lock:
            self.stop()
            return self.storage.save(self.agent, {
                "challenge": self.challenge.as_dict() if self.challenge else None,
                "episode_index": self.episode_index,
                "episode_successes": self.episode_successes,
                "global_steps": self.global_steps,
                "seed": self.seed,
            })

    def restore(self) -> dict:
        with self._lock:
            self.stop()
            meta = self.storage.restore(self.agent)
            if meta:
                self.seed = meta.get("seed", self.seed)
                self.episode_index = meta.get("episode_index", 0)
                self.episode_successes = meta.get("episode_successes", 0)
                self.global_steps = meta.get("global_steps", 0)
                c = meta.get("challenge")
                if c:
                    self.challenge = Challenge(**c)
                self.trace("Restored saved progress. Learned memory is intact.")
            return meta

    def fresh(self):
        with self._lock:
            self.stop()
            self.storage.clear()
            self.agent = MetargonAcademyAgent(seed=self.seed, **self.agent_kwargs)
            self.env = None
            self.spec = None
            self.challenge = None
            self.episode_index = 0
            self.episode_successes = 0
            self.global_steps = 0
            self.status = STATUS_IDLE
            self.trace("Fresh start. No memory.")

    # ------------------------------------------------------------------ #
    # Telemetry / rendering
    # ------------------------------------------------------------------ #
    def trace(self, text: str):
        self.messages.append({"kind": "system", "text": text})
        if len(self.messages) > 400:
            del self.messages[:100]

    def memory_snapshot(self) -> dict:
        mem = self.agent.memory
        hyps = []
        current_epoch = mem._current_epoch
        for h in mem.hypotheses:
            hyps.append({
                "id": h.h_id,
                "door": h.door,
                "antecedent": h.antecedent,
                "antecedent_text": " ∧ ".join(sorted(
                    f"{a[0]}({','.join(map(str, a[1:]))})" for a in h.antecedent)),
                "status": h.status,
                "sup": h.sup,
                "con": h.con,
                "confidence": round(h.confidence, 3),
                "created_step": h.created_step,
                "last_update": h.last_update,
                "verified_epochs": h.verified_epochs,
                "from_earlier_episodes": bool(
                    h.verified_epochs and max(h.verified_epochs) < (current_epoch or 0)),
            })
        diag = mem.diagnostics() if hasattr(mem, "diagnostics") else {}
        stats = diag.get("status_counts", {})
        # always ensure the four keys are present
        for k in (HYPOTHESIS, PREDICTED, VERIFIED, CONTRADICTED):
            stats.setdefault(k, 0)
        return {
            "hypotheses": hyps,
            "stats": stats,
            "generated": mem.generated,
            "step": mem.step,
            "evidence_bank_size": len(mem.evidence_bank),
            "bank_retrievals": mem.bank_retrievals,
            "current_epoch": current_epoch,
            "verified_hypotheses": diag.get("verified_hypotheses", 0),
        }

    def world_snapshot(self) -> dict:
        if self.env is None:
            return {"present": False, "goal": list(GOAL)}
        obs = self.last_obs or self.env.observe()
        geom = self.env.geom
        return {
            "present": True,
            "pos": list(obs.pos),
            "objs": obs.objs,
            "plates": obs.plate_load,
            "switches": obs.switch_state,
            "doors": obs.door_open,
            "goal": list(geom.goal),
            "steps": self.env.steps,
            "max_steps": self.env.max_steps,
            "finished": self.env.finished,
            "last_events": list(self.last_events),
            "last_action": self.last_action,
            "last_valid": self.last_result_valid,
            "door_cell": {d: list(c) for d, c in geom.door_cell.items()},
            "switch_cell": {s: list(c) for s, c in geom.switch_cell.items()},
            "plate_cell": {p: list(c) for p, c in geom.plate_cell.items()},
            "walls": [list(c) for c in geom.walls],
            "topology": geom.__class__.__name__,
        }

    def trace_snapshot(self) -> dict:
        t = dict(self.agent.trace)
        return t

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "status": self.status,
                "episode_index": self.episode_index,
                "episode_successes": self.episode_successes,
                "global_steps": self.global_steps,
                "steps_this_episode": self.steps_this_episode,
                "challenge": self.challenge.as_dict() if self.challenge else None,
                "world": self.world_snapshot(),
                "memory": self.memory_snapshot(),
                "trace": self.trace_snapshot(),
                "notice": self.notice,
                "worker_error": self._last_error,
                "last_guidance": self.last_guidance,
                "teaching": self.teaching.as_dict(),
                "messages_tail": self.messages[-20:],
            }
