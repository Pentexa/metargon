#!/usr/bin/env python3
"""CORVID preliminary experiment. Python stdlib, no GPU or API.

NOT a novel AI architecture, nor proof of biological fidelity.
Tests value of working memory, episodic action values, and event gating.
"""
from __future__ import annotations

import argparse
import random
import statistics
from dataclasses import dataclass


@dataclass
class Metrics:
    successes: list[int]
    compute_calls: int
    steps: int

    def first_half(self) -> float:
        return 100 * sum(self.successes[:len(self.successes)//2]) / (len(self.successes)//2)

    def second_half(self) -> float:
        return 100 * sum(self.successes[len(self.successes)//2:]) / (len(self.successes) - len(self.successes)//2)


class Agent:
    def __init__(self, mode: str, seed: int):
        self.mode = mode
        self.rng = random.Random(seed)
        self.working_memory: int | None = None
        # The cue is visible only at trial start. Estimated reward per (cue, action).
        self.episodic_values = [[0.5, 0.5], [0.5, 0.5]]
        self.compute_calls = 0
        self.steps = 0
        self.trial_count = 0

    def observe(self, event: str, value: int | None = None) -> None:
        self.steps += 1
        # All agents can only access the input event; no hidden rule is supplied.
        # Event-gated mode avoids executive updates on unchanged empty ticks.
        if self.mode == "always_on" or event != "idle":
            self.compute_calls += 1
        if event == "cue":
            if self.mode != "memoryless":
                self.working_memory = value

    def choose(self) -> int:
        self.steps += 1
        self.compute_calls += 1
        if self.mode == "memoryless" or self.working_memory is None:
            return self.rng.randrange(2)
        cue = self.working_memory
        if self.rng.random() < 0.12:
            return self.rng.randrange(2)
        scores = self.episodic_values[cue]
        # Deterministic tie break with seed to avoid a fixed preferred tool.
        if scores[0] == scores[1]:
            return self.rng.randrange(2)
        return 0 if scores[0] > scores[1] else 1

    def feedback(self, cue: int, action: int, reward: int) -> None:
        self.observe("feedback", reward)
        if self.mode != "memoryless":
            old = self.episodic_values[cue][action]
            self.episodic_values[cue][action] = old + 0.35 * (reward - old)
        self.working_memory = None
        self.trial_count += 1


def run(mode: str, seed: int, trials: int, max_delay: int) -> Metrics:
    env_rng = random.Random(seed)
    agent = Agent(mode, seed + 137)
    successes = []
    for trial in range(trials):
        cue = env_rng.randrange(2)
        # Unannounced rule reversal halfway through the test.
        reversal = 0 if trial < trials // 2 else 1
        agent.observe("cue", cue)
        for _ in range(env_rng.randint(2, max_delay)):
            agent.observe("idle")
        action = agent.choose()
        reward = int(action == (cue ^ reversal))
        agent.feedback(cue, action, reward)
        successes.append(reward)
    return Metrics(successes, agent.compute_calls, agent.steps)


def main() -> None:
    p = argparse.ArgumentParser(description="CORVID exploratory delayed-cue/reversal benchmark")
    p.add_argument("--trials", type=int, default=400)
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--max-delay", type=int, default=8)
    args = p.parse_args()
    if args.trials < 20 or args.seeds < 1 or args.max_delay < 2:
        p.error("Require trials>=20, seeds>=1, max-delay>=2")
    print("CORVID PRELIMINARY TASK / illustrative, not neuroscience validation")
    print(f"trials={args.trials} seeds={args.seeds} delay=2..{args.max_delay} reversal=halfway")
    print(f"{'mode':<14} {'pre %':>9} {'post %':>9} {'updates / step':>17}")
    for mode in ("memoryless", "always_on", "event_gated"):
        metrics = [run(mode, s, args.trials, args.max_delay) for s in range(args.seeds)]
        early = statistics.mean(m.first_half() for m in metrics)
        late = statistics.mean(m.second_half() for m in metrics)
        ratio = statistics.mean(m.compute_calls/m.steps for m in metrics)
        print(f"{mode:<14} {early:9.1f} {late:9.1f} {ratio:17.3f}")
    print("\nUpdate ratio is an artificial work proxy, NOT energy or real CPU time.")
    print("Real discovery requires stronger equal-budget baselines and held-out tasks.")


if __name__ == "__main__":
    main()
