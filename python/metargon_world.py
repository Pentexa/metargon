#!/usr/bin/env python3
"""METARGON v0.8 — stateful, partially observed rule-switching workshop.

Independent simulator from v0.1-0.7. Agents never receive hidden mode, latent
transition probabilities, or pre-generated RNG. The geometry and actuator
semantics are public; the gate's mechanism is unknown and changes by regime.

Research prototype; none of these policies learns a general world model.
The learned component is a small latent gate-transition predictor.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import random
import statistics
from typing import NamedTuple

RECIPES = ("key", "switch")
ACTIONS = ("right", "left", "take", "drop", "toggle", "open")
CONDITIONS = ("stable", "shift", "rapid_shift", "iid", "novel", "noisy_gate", "tight_budget", "sham")
POLICIES = ("always_key", "always_switch", "random", "win_stay", "bayes", "metargon", "oracle")
MAX_STEPS = 17


@dataclass(frozen=True)
class Observation:
    position: int
    has_key: bool
    switch_on: bool
    gate_open: bool
    finished: bool


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    gate_result: bool | None
    valid: bool


class Workshop:
    """A five-position environment with an irreversible episode success state.

    Key can be picked/dropped at 1; switch toggled at 2; gate opens at 3.
    Either key-only or switch-only opens it in a hidden regime. Opening is
    probabilistic, with random numbers pre-generated for both recipes, so
    policy comparisons see matched gate-outcome potential outcomes.
    """
    def __init__(self, hidden_mode: str, uniforms: dict[str, tuple[float, ...]],
                 sham: bool = False, max_steps: int = MAX_STEPS,
                 match_success: float = 0.93, mismatch_success: float = 0.07):
        if hidden_mode not in RECIPES or max_steps < 1:
            raise ValueError("invalid world configuration")
        self._mode = hidden_mode
        self._uniforms = uniforms
        self._sham = sham
        self._match_success = match_success
        self._mismatch_success = mismatch_success
        self._used = {x: 0 for x in RECIPES}
        self.position = 0
        self.has_key = False
        self.switch_on = False
        self.gate_open = False
        self.finished = False
        self.steps = 0
        self.max_steps = max_steps
        self.gate_attempts = 0

    def observe(self) -> Observation:
        return Observation(self.position, self.has_key, self.switch_on,
                           self.gate_open, self.finished)

    def step(self, action: str) -> StepResult:
        if action not in ACTIONS:
            raise ValueError("unknown action")
        if self.finished or self.steps >= self.max_steps:
            raise RuntimeError("episode terminated")
        self.steps += 1
        valid = True
        gate_result = None
        if action == "right":
            if self.position < 4 and (self.position != 3 or self.gate_open):
                self.position += 1
                if self.position == 4:
                    self.finished = True
            else:
                valid = False
        elif action == "left":
            if self.position > 0:
                self.position -= 1
            else:
                valid = False
        elif action == "take":
            if self.position == 1 and not self.has_key:
                self.has_key = True
            else:
                valid = False
        elif action == "drop":
            if self.position == 1 and self.has_key:
                self.has_key = False
            else:
                valid = False
        elif action == "toggle":
            if self.position == 2:
                self.switch_on = not self.switch_on
            else:
                valid = False
        elif action == "open":
            if self.position != 3 or self.gate_open:
                valid = False
            else:
                choice = "key" if self.has_key and not self.switch_on else (
                    "switch" if self.switch_on and not self.has_key else "none")
                if choice not in RECIPES:
                    valid = False
                else:
                    self.gate_attempts += 1
                    idx = self._used[choice]
                    self._used[choice] += 1
                    seq = self._uniforms[choice]
                    if idx >= len(seq):
                        raise RuntimeError("insufficient pre-generated random numbers")
                    success_prob = 0.55 if self._sham else (self._match_success if choice == self._mode else self._mismatch_success)
                    gate_result = seq[idx] < success_prob
                    self.gate_open = gate_result
        return StepResult(self.observe(), gate_result, valid)


class State(NamedTuple):
    position: int
    key: bool
    switch: bool


def deterministic_transition(state: State, action: str) -> State | None:
    """Public movement and control semantics, *not* the hidden lock rule."""
    p, k, s = state
    if action == "right" and p < 3:
        return State(p + 1, k, s)
    if action == "left" and p > 0:
        return State(p - 1, k, s)
    if action == "take" and p == 1 and not k:
        return State(p, True, s)
    if action == "drop" and p == 1 and k:
        return State(p, False, s)
    if action == "toggle" and p == 2:
        return State(p, k, not s)
    return None


def plan_to_gate(observation: Observation, recipe: str) -> list[str]:
    """BFS over public transition mechanics; independent of latent mode."""
    if recipe not in RECIPES:
        raise ValueError(recipe)
    start = State(observation.position, observation.has_key, observation.switch_on)
    goal = State(3, recipe == "key", recipe == "switch")
    queue = deque([(start, [])])
    seen = {start}
    while queue:
        state, path = queue.popleft()
        if state == goal:
            return path
        for action in ("right", "left", "take", "drop", "toggle"):
            next_state = deterministic_transition(state, action)
            if next_state is not None and next_state not in seen:
                seen.add(next_state)
                queue.append((next_state, path + [action]))
    raise RuntimeError("goal cannot be reached")


class Policy:
    """Policies receive only context and their own gate observations."""
    def __init__(self, name: str, seed: int):
        self.name = name
        self.rng = random.Random(seed)
        self.posteriors = [0.5, 0.5, 0.5]
        self.last_choice: dict[int, str] = {}
        self.last_success: dict[int, bool] = {}
        self.surprises: list[float] = [0.0, 0.0, 0.0]
        self.learning_events = 0
        self.predictions: list[tuple[float, int]] = []
        self.episode_predictions: list[float] = []

    def start(self, context: int):
        if self.name in ("bayes", "metargon"):
            hazard = 0.045 if self.name == "bayes" else (
                0.018 + min(0.23, 0.24 * self.surprises[context]))
            p = self.posteriors[context]
            self.posteriors[context] = p * (1.0 - hazard) + (1.0 - p) * hazard

    def choose(self, context: int, first: bool = True) -> str:
        if self.name == "always_key":
            return "key"
        if self.name == "always_switch":
            return "switch"
        if self.name == "random":
            return self.rng.choice(RECIPES)
        if self.name == "win_stay":
            previous = self.last_choice.get(context, "key")
            if self.last_success.get(context) is False:
                return "switch" if previous == "key" else "key"
            return previous
        p = self.posteriors[context]
        return "switch" if p > 0.5 else "key"

    def predict(self, context: int, choice: str) -> float:
        if self.name in ("bayes", "metargon"):
            belief = self.posteriors[context]
            return (0.93 if choice == "switch" else 0.07) * belief + (
                0.07 if choice == "switch" else 0.93) * (1.0 - belief)
        return 0.5

    def feedback(self, context: int, choice: str, opened: bool, predicted: float):
        self.predictions.append((predicted, int(opened)))
        self.learning_events += 1
        if self.name in ("bayes", "metargon"):
            belief = self.posteriors[context]
            surprise = 1.0 - predicted if opened else predicted
            if self.name == "metargon":
                # Surprise-gated forgetting: a hypothetical change detector,
                # pre-registered before benchmark. Can overreact to random failures.
                self.surprises[context] = 0.80 * self.surprises[context] + 0.20 * surprise
                if surprise > 0.72:
                    belief = 0.5 * belief + 0.5 * 0.5
            like_sw = 0.93 if choice == "switch" else 0.07
            like_key = 0.07 if choice == "switch" else 0.93
            if not opened:
                like_sw, like_key = 1.0 - like_sw, 1.0 - like_key
            denominator = belief * like_sw + (1.0 - belief) * like_key
            self.posteriors[context] = (belief * like_sw / denominator) if denominator else 0.5
        if self.name == "win_stay":
            self.last_choice[context] = choice
            self.last_success[context] = opened


@dataclass(frozen=True)
class Episode:
    context: int
    mode: str
    sham: bool
    uniforms: dict[str, tuple[float, ...]]
    match_success: float = 0.93
    mismatch_success: float = 0.07
    max_steps: int = MAX_STEPS


def generate_worlds(condition: str, seed: int, episodes: int) -> list[Episode]:
    if condition not in CONDITIONS or episodes < 4:
        raise ValueError("invalid condition or episode count")
    rng = random.Random(seed)
    modes = [rng.choice(RECIPES) for _ in range(3)]
    out = []
    for e in range(episodes):
        if condition == "novel":
            context = rng.choice((0, 1)) if e < episodes // 2 else 2
        else:
            context = rng.randrange(3)
        if condition in ("shift", "tight_budget") and e == episodes // 2:
            modes[0] = "switch" if modes[0] == "key" else "key"
            modes[1] = "switch" if modes[1] == "key" else "key"
            modes[2] = "switch" if modes[2] == "key" else "key"
        if condition in ("rapid_shift", "noisy_gate") and e > 0 and e % 24 == 0:
            modes = ["switch" if m == "key" else "key" for m in modes]
        mode = rng.choice(RECIPES) if condition == "iid" else modes[context]
        # Gate outcomes are drawn independently for both candidate recipes and
        # stored before policy decisions, preventing policy-dependent RNG streams.
        uniforms = {c: tuple(rng.random() for _ in range(4)) for c in RECIPES}
        out.append(Episode(context, mode, condition == "sham", uniforms,
                           0.72 if condition == "noisy_gate" else 0.93,
                           0.28 if condition == "noisy_gate" else 0.07,
                           7 if condition == "tight_budget" else MAX_STEPS))
    return out


def play_episode(spec: Episode, policy: Policy, max_steps: int | None = None) -> dict:
    world = Workshop(spec.mode, spec.uniforms, spec.sham,
                     max_steps=spec.max_steps if max_steps is None else max_steps,
                     match_success=spec.match_success, mismatch_success=spec.mismatch_success)
    policy.start(spec.context)
    first = policy.choose(spec.context)
    planned = [first, "switch" if first == "key" else "key"]
    attempts = []
    invalid_actions = 0
    trace = []
    for recipe in planned:
        route = plan_to_gate(world.observe(), recipe)
        for action in route:
            if world.steps >= world.max_steps:
                break
            result = world.step(action)
            trace.append(action)
            invalid_actions += int(not result.valid)
        if world.steps >= world.max_steps:
            break
        forecast = policy.predict(spec.context, recipe)
        result = world.step("open")
        trace.append("open")
        invalid_actions += int(not result.valid)
        if result.gate_result is None:
            raise AssertionError("planned gate operation was invalid")
        policy.feedback(spec.context, recipe, result.gate_result, forecast)
        attempts.append({"recipe": recipe, "opened": result.gate_result, "forecast": forecast})
        if result.gate_result:
            if world.steps < world.max_steps:
                result = world.step("right")
                trace.append("right")
                invalid_actions += int(not result.valid)
            break
    return {
        "success": world.finished,
        "steps": world.steps,
        "attempts": attempts,
        "invalid": invalid_actions,
        "trace": trace,
    }


def score_policy(condition: str, seed: int, name: str, episodes: int) -> dict:
    worlds = generate_worlds(condition, seed, episodes)
    policy = Policy(name, seed + 1000003)
    successes = 0
    total_steps = 0
    total_invalid = 0
    post_shift_steps = []
    first_choice_correct = 0
    for i, spec in enumerate(worlds):
        if name == "oracle":
            # Explicitly PRIVILEGED upper bound, never considered a deployable baseline.
            # Mode is used only here, not in any trainable policy.
            oracle_p = Policy("always_key" if spec.mode == "key" else "always_switch", seed)
            result = play_episode(spec, oracle_p)
        else:
            result = play_episode(spec, policy)
        successes += int(result["success"])
        total_steps += result["steps"]
        total_invalid += result["invalid"]
        first_choice_correct += int(bool(result["attempts"]) and result["attempts"][0]["recipe"] == spec.mode)
        if i >= episodes // 2:
            post_shift_steps.append(result["steps"])
    brier = (statistics.fmean((p - y) ** 2 for p, y in policy.predictions)
             if policy.predictions else None)
    return {"success_rate": successes / episodes,
            "mean_steps": total_steps / episodes,
            "post_half_steps": statistics.fmean(post_shift_steps),
            "first_choice_matches_hidden": first_choice_correct / episodes,
            "invalid_actions": total_invalid,
            "brier_gate": brier,
            "learning_events": policy.learning_events}


def paired_bootstrap(diffs: list[float], seed: int = 4301, rounds: int = 1500) -> tuple[float, float]:
    rng = random.Random(seed)
    values = []
    for _ in range(rounds):
        values.append(statistics.fmean(diffs[rng.randrange(len(diffs))] for _ in diffs))
    values.sort()
    return values[int(0.025 * rounds)], values[int(0.975 * rounds)]


def benchmark(seeds: int = 30, episodes: int = 240) -> dict:
    if seeds < 1 or episodes < 4:
        raise ValueError("seeds must be >= 1, episodes >= 4")
    scores: dict = {}
    summary = {}
    for condition in CONDITIONS:
        per_seed = []
        for i in range(seeds):
            results = {p: score_policy(condition, i + 11800, p, episodes) for p in POLICIES}
            per_seed.append(results)
        scores[condition] = per_seed
        summary[condition] = {}
        for p in POLICIES:
            summary[condition][p] = {
                k: statistics.fmean(row[p][k] for row in per_seed)
                for k in ("success_rate", "mean_steps", "post_half_steps", "first_choice_matches_hidden")}
        delta = [row["metargon"]["mean_steps"] - row["bayes"]["mean_steps"] for row in per_seed]
        success_delta = [row["metargon"]["success_rate"] - row["bayes"]["success_rate"]
                         for row in per_seed]
        summary[condition]["comparison"] = {
            "metargon_minus_bayes_steps": statistics.fmean(delta),
            "paired_bootstrap_steps_95pct": paired_bootstrap(delta),
            "metargon_minus_bayes_success": statistics.fmean(success_delta),
            "paired_bootstrap_success_95pct": paired_bootstrap(success_delta, seed=4302)}
    return {"metadata": {"name": "METARGON", "version": "0.8", "seeds": seeds,
                          "episodes_per_seed": episodes, "max_steps_per_episode": MAX_STEPS,
                          "tight_budget_steps": 7,
                          "same_pre_generated_worlds": True,
                          "oracle_has_hidden_mode": True},
            "summary": summary, "raw": scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=240)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    result = benchmark(args.seeds, args.episodes)
    print("METARGON v0.8 — stateful hidden-rule transfer evaluation")
    for scenario, row in result["summary"].items():
        print(f"\n{scenario}:")
        for name in POLICIES:
            x = row[name]
            print(f"  {name:>13}: success {x['success_rate']:.2%}, steps {x['mean_steps']:.2f}, "
                  f"first mechanism match {x['first_choice_matches_hidden']:.2%}")
        d = row["comparison"]
        print(f"  delta steps Metargon–Bayes: {d['metargon_minus_bayes_steps']:+.3f}, "
              f"paired CI {d['paired_bootstrap_steps_95pct']}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
