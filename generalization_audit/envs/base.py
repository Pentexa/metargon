#!/usr/bin/env python3
"""Independent generalization-audit environment scaffolding.

This package does NOT modify the shared research definitions in python/
(lab_env.py, policies, benchmarks). It reuses only the *public physics*
primitives (movement, pushing, toggling, atom evaluation) and defines new
world classes with the same public interface as lab_env.LabWorld
(observe/step/observe/finished/steps/max_steps + Observation/StepResult),
so that unmodified student and baseline policies can be evaluated against
structurally novel hidden-rule families.

Two new families are provided:

* Family R — "sustained-hold" gates. A door opens only after its condition
  has held continuously for HOLD_K consecutive post-states, stays open
  while the condition continues to hold, and closes as soon as the
  condition is lost (the hold counter then resets). This introduces a
  temporal-threshold axis absent from families F and H (which open
  instantaneously on satisfaction).

* Family L — "ordered-sequence" gates. A door opens only when its two
  condition atoms become true in a specific order (atom A first, then
  atom B) within a sliding window; the reverse order never opens it.
  This introduces an order/serial dependency absent from the unordered
  conjunctive families F and H.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from lab_env import (
    ACTIONS, DOORS, GOAL, PLATES, SWITCHES, WALLS,
    Observation, StepResult, WS,
    clone_ws, in_bounds, make_state, occupied_by_object, public_next,
)
from lab_env import LabEpisode  # reused as the frozen episode container


MAX_STEPS = 80
TIGHT_STEPS = 44
NOISE_Q = 0.08

# --------------------------------------------------------------------------
# Family R configuration
# --------------------------------------------------------------------------
HOLD_K = 3  # consecutive states the condition must hold before opening


@dataclass(frozen=True)
class RConfig:
    """Hidden wiring for a family-R world: a single (or conjunctive)
    condition per door plus the hold threshold."""
    d1_cond: frozenset
    d2_cond: frozenset

    def cond_for(self, door: str) -> frozenset:
        return self.d1_cond if door == "D1" else self.d2_cond

    def key(self) -> tuple:
        return (tuple(sorted(self.d1_cond)), tuple(sorted(self.d2_cond)))


R1_ATOMS = (
    frozenset({("sw", "S1", True)}),
    frozenset({("sw", "S1", False)}),
    frozenset({("load", "P1", "A")}),
    frozenset({("loadany", "P1")}),
)
R2_ATOMS = (
    frozenset({("sw", "S2", True)}),
    frozenset({("sw", "S2", False)}),
    frozenset({("load", "P2", "B")}),
    frozenset({("loadany", "P2")}),
    frozenset({("sw", "S1", True), ("loadany", "P2")}),
)


def make_r_config(d1, d2) -> RConfig:
    return RConfig(d1_cond=frozenset(d1), d2_cond=frozenset(d2))


# --------------------------------------------------------------------------
# Family L configuration
# --------------------------------------------------------------------------
L_WINDOW = 30  # steps allowed between first and second atom of the sequence


@dataclass(frozen=True)
class LConfig:
    """Hidden wiring for a family-L world: each door has an *ordered*
    pair of atoms (first, second); the door opens only when the first
    becomes true before the second within the window."""
    d1_first: frozenset
    d1_second: frozenset
    d2_first: frozenset
    d2_second: frozenset

    def first_for(self, door: str) -> frozenset:
        return self.d1_first if door == "D1" else self.d2_first

    def second_for(self, door: str) -> frozenset:
        return self.d1_second if door == "D1" else self.d2_second

    def key(self) -> tuple:
        return (tuple(sorted(self.d1_first)), tuple(sorted(self.d1_second)),
                tuple(sorted(self.d2_first)), tuple(sorted(self.d2_second)))


L1_FIRST = (frozenset({("sw", "S1", True)}), frozenset({("load", "P1", "A")}))
L1_SECOND = (frozenset({("sw", "S1", False)}), frozenset({("loadany", "P1")}))
L2_FIRST = (frozenset({("sw", "S2", True)}), frozenset({("load", "P2", "B")}))
L2_SECOND = (frozenset({("sw", "S2", False)}), frozenset({("loadany", "P2")}))


# --------------------------------------------------------------------------
# Shared hidden-rule evaluation helpers
# --------------------------------------------------------------------------
def eval_atoms(atoms: frozenset, state: WS) -> bool:
    """True when every atom in the set holds in the physical state."""
    from lab_env import eval_atoms as _lab_eval
    return _lab_eval(atoms, state)


def _door_cell(door: str) -> tuple:
    from lab_env import DOOR_CELL
    return DOOR_CELL[door]


def _door_blocked_by_occupant(state: WS, door: str) -> bool:
    """Door cannot close while the agent stands on it or an object does."""
    cell = _door_cell(door)
    return state.pos == cell or occupied_by_object(cell, state) is not None


# --------------------------------------------------------------------------
# Family R world (sustained-hold)
# --------------------------------------------------------------------------
class RWorld:
    def __init__(self, config: RConfig, ep_seed: int = 0, noise_q: float = 0.0,
                 max_steps: int = MAX_STEPS, hold_k: int = HOLD_K):
        self._config = config
        self._noise_q = noise_q
        self.max_steps = max_steps
        self._ep_seed = ep_seed
        self._hold_k = hold_k
        self.steps = 0
        self.finished = False
        self._state = make_state()
        self.last_events = ()
        self._hold = {"D1": 0, "D2": 0}

    # -- hidden dynamics (family R) ------------------------------------
    def _transition(self, state: WS, action: str):
        nxt = public_next(state, action)
        if nxt is None:
            return None, []
        events = []
        for door in DOORS:
            cond = eval_atoms(self._config.cond_for(door), nxt)
            if not nxt.door_open[door]:
                if cond:
                    self._hold[door] += 1
                    if self._hold[door] >= self._hold_k:
                        nxt.door_open[door] = True
                        events.append((door, "open"))
                else:
                    self._hold[door] = 0
            else:
                if not cond:
                    if not _door_blocked_by_occupant(nxt, door):
                        nxt.door_open[door] = False
                        self._hold[door] = 0
                        events.append((door, "close"))
        return nxt, events

    # -- public surface mirrors LabWorld --------------------------------
    def _readings(self):
        s = self._state
        return {
            "s1": s.switch_state["S1"], "s2": s.switch_state["S2"],
            "d1": s.door_open["D1"], "d2": s.door_open["D2"],
            "p1": s.plate_load["P1"], "p2": s.plate_load["P2"],
        }

    def _corrupt(self, readings):
        if self._noise_q <= 0.0:
            return dict(readings)
        out = dict(readings)
        nrng = random.Random((self._ep_seed * 1000003 + self.steps) & 0xFFFFFFFF)
        for key in ("s1", "s2", "d1", "d2"):
            if nrng.random() < self._noise_q:
                out[key] = not out[key]
        for key in ("p1", "p2"):
            if nrng.random() < self._noise_q:
                out[key] = nrng.choice(("none", "A", "B"))
        return out

    def observe(self) -> Observation:
        s = self._state
        r = self._corrupt(self._readings())
        return Observation(
            s.pos[0], s.pos[1],
            s.objs["A"][0], s.objs["A"][1],
            s.objs["B"][0], s.objs["B"][1],
            r["s1"], r["s2"], r["p1"], r["p2"], r["d1"], r["d2"],
            self.finished,
        )

    def step(self, action: str) -> StepResult:
        if action not in ACTIONS:
            raise ValueError(f"unknown action: {action}")
        if self.finished or self.steps >= self.max_steps:
            raise RuntimeError("episode terminated")
        self.steps += 1
        nxt, events = self._transition(self._state, action)
        if nxt is None:
            return StepResult(self.observe(), False, ())
        self._state = nxt
        self.last_events = tuple(events)
        if self._state.pos == GOAL:
            self.finished = True
        return StepResult(self.observe(), True, tuple(events))


# --------------------------------------------------------------------------
# Family L world (ordered-sequence)
# --------------------------------------------------------------------------
class LWorld:
    def __init__(self, config: LConfig, ep_seed: int = 0, noise_q: float = 0.0,
                 max_steps: int = MAX_STEPS, window: int = L_WINDOW):
        self._config = config
        self._noise_q = noise_q
        self.max_steps = max_steps
        self._ep_seed = ep_seed
        self._window = window
        self.steps = 0
        self.finished = False
        self._state = make_state()
        self.last_events = ()
        self._phase = {"D1": 0, "D2": 0}
        self._phase_step = {"D1": None, "D2": None}

    def _transition(self, state: WS, action: str):
        nxt = public_next(state, action)
        if nxt is None:
            return None, []
        events = []
        for door in DOORS:
            first = eval_atoms(self._config.first_for(door), nxt)
            second = eval_atoms(self._config.second_for(door), nxt)
            # advance phase: 0 -> 1 when first holds AND second is not yet
            # true (strict order), 1 -> 2 when second becomes true
            if self._phase[door] == 0:
                if first and not second:
                    self._phase[door] = 1
                    self._phase_step[door] = self.steps
            elif self._phase[door] == 1:
                # window expiry resets the sequence
                if self._phase_step[door] is not None and \
                        self.steps - self._phase_step[door] > self._window:
                    self._phase[door] = 0
                    self._phase_step[door] = None
                elif second:
                    if not nxt.door_open[door]:
                        nxt.door_open[door] = True
                        events.append((door, "open"))
                    self._phase[door] = 2
            # door stays open once opened (stability), unless... keep open
        return nxt, events

    # public surface identical to RWorld
    def _readings(self):
        s = self._state
        return {
            "s1": s.switch_state["S1"], "s2": s.switch_state["S2"],
            "d1": s.door_open["D1"], "d2": s.door_open["D2"],
            "p1": s.plate_load["P1"], "p2": s.plate_load["P2"],
        }

    def _corrupt(self, readings):
        if self._noise_q <= 0.0:
            return dict(readings)
        out = dict(readings)
        nrng = random.Random((self._ep_seed * 1000003 + self.steps) & 0xFFFFFFFF)
        for key in ("s1", "s2", "d1", "d2"):
            if nrng.random() < self._noise_q:
                out[key] = not out[key]
        for key in ("p1", "p2"):
            if nrng.random() < self._noise_q:
                out[key] = nrng.choice(("none", "A", "B"))
        return out

    def observe(self) -> Observation:
        s = self._state
        r = self._corrupt(self._readings())
        return Observation(
            s.pos[0], s.pos[1],
            s.objs["A"][0], s.objs["A"][1],
            s.objs["B"][0], s.objs["B"][1],
            r["s1"], r["s2"], r["p1"], r["p2"], r["d1"], r["d2"],
            self.finished,
        )

    def step(self, action: str) -> StepResult:
        if action not in ACTIONS:
            raise ValueError(f"unknown action: {action}")
        if self.finished or self.steps >= self.max_steps:
            raise RuntimeError("episode terminated")
        self.steps += 1
        nxt, events = self._transition(self._state, action)
        if nxt is None:
            return StepResult(self.observe(), False, ())
        self._state = nxt
        self.last_events = tuple(events)
        if self._state.pos == GOAL:
            self.finished = True
        return StepResult(self.observe(), True, tuple(events))


# --------------------------------------------------------------------------
# World factory for the audit benchmark
# --------------------------------------------------------------------------
WORLD_CLASSES = {"r": RWorld, "l": LWorld}
CONFIG_CLASSES = {"r": RConfig, "l": LConfig}
