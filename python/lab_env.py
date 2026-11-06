#!/usr/bin/env python3
"""METARGON v0.9 — "Mechanism Lab": an independent research environment.

This simulator is structurally independent from the v0.8 five-position
workshop (python/metargon_world.py). It is a 7x5 grid world containing:

  * two movable objects (A, B) that can be pushed (Sokoban-style),
  * two pressure plates (P1, P2) that report which object is on them,
  * two switches (S1, S2) that can be toggled by standing on them,
  * two door barriers (D1, D2) that block movement until a hidden
    conjunctive condition is satisfied,
  * an optional one-step delayed effect for each door,
  * an optional noisy observation channel (readings flipped w.p. q),
  * procedural hidden-rule configurations sampled from a declared template
    pool (the *generative family*), with disjoint training / validation /
    held-out test pools.

Hidden state: the wiring of each door (which atoms must hold), the delay
flags, and the episode's noise stream. None of this is ever exposed through
observations. The agent receives a fixed-size public Observation only.

Public mechanics (known to every agent and planner): movement, pushing,
switch toggling, and the fact that doors block movement while closed.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import NamedTuple, Optional

# --------------------------------------------------------------------------
# World geometry (fixed topology for the headline benchmark; procedural
# topologies are exercised in an auxiliary robustness check only).
# --------------------------------------------------------------------------
W, H = 7, 5
START = (0, 4)
GOAL = (6, 0)
OBJECTS = ("A", "B")
PLATES = ("P1", "P2")
SWITCHES = ("S1", "S2")
DOORS = ("D1", "D2")
WALLS = frozenset({
    (0, 2), (1, 0), (3, 0), (3, 2), (3, 3), (3, 4),
    (5, 0), (5, 1), (5, 3), (5, 4),
})
OBJECT_HOME = {"A": (2, 3), "B": (4, 3)}
PLATE_CELL = {"P1": (2, 2), "P2": (4, 4)}
SWITCH_CELL = {"S1": (0, 1), "S2": (4, 0)}
DOOR_CELL = {"D1": (3, 1), "D2": (5, 2)}

ACTIONS = ("N", "E", "S", "W", "Toggle", "PushN", "PushE", "PushS", "PushW")
MOVE_ACTIONS = ("N", "E", "S", "W")
DIRS = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}
MAX_STEPS = 80
TIGHT_STEPS = 44

# Noise: probability that a single public reading (switch / plate / door)
# is corrupted in the noisy-observation condition.
NOISE_Q = 0.08

# --------------------------------------------------------------------------
# Hidden-rule vocabulary.
# Atoms are tuples:
#   ("load", plate, obj)   plate is loaded with exactly that object
#   ("loadany", plate)     plate loaded with any object
#   ("sw", switch, value)  switch is in that state (True = on)
# A door condition is a frozenset of atoms (conjunction). Configurations are
# sampled from two *solvability-guaranteed* families:
#
#   F1_TEMPLATES — D1 conditions. Left-room only (P1 / S1), so D1 can
#                  always be opened from the starting side.
#   F2_TEMPLATES — D2 conditions. May reference any component; the agent
#                  can satisfy them after D1 opens (S2 sits in the middle
#                  room, reachable once D1 is open).
#
# This makes every configuration in the family solvable: D1 first, then
# D2. The families are the declared generative model given to the Bayesian
# baseline (a correct model class in standard conditions).
# --------------------------------------------------------------------------
F1_TEMPLATES = (
    frozenset({("sw", "S1", True), ("load", "P1", "A")}),
    frozenset({("sw", "S1", False), ("load", "P1", "A")}),
    frozenset({("sw", "S1", True), ("loadany", "P1")}),
    frozenset({("sw", "S1", False), ("loadany", "P1")}),
    frozenset({("sw", "S1", True)}),
    frozenset({("sw", "S1", False)}),
    frozenset({("load", "P1", "A")}),
)

F2_TEMPLATES = (
    frozenset({("sw", "S2", True), ("loadany", "P2")}),
    frozenset({("sw", "S2", False), ("loadany", "P2")}),
    frozenset({("sw", "S2", True)}),
    frozenset({("sw", "S2", False)}),
    frozenset({("sw", "S1", True), ("sw", "S2", False)}),
    frozenset({("load", "P1", "A"), ("load", "P2", "B")}),
    frozenset({("sw", "S1", True), ("load", "P1", "A")}),
    frozenset({("sw", "S1", True), ("loadany", "P1")}),
    frozenset({("load", "P2", "B")}),
)

# *Structural novelty* families used in the structured-OOD diagnostic: the
# template vocabulary itself differs from F1/F2 (novel polarity/different
# conjunctive shapes). The Bayesian baseline's model class excludes these.
H1_TEMPLATES = (
    frozenset({("sw", "S1", False), ("load", "P1", "A")}),
    frozenset({("sw", "S1", True), ("load", "P1", "B")}),
    frozenset({("sw", "S1", False), ("loadany", "P1")}),
)

H2_TEMPLATES = (
    frozenset({("sw", "S1", False), ("sw", "S2", True)}),
    frozenset({("sw", "S1", True), ("load", "P2", "B")}),
    frozenset({("load", "P2", "B"), ("loadany", "P1")}),
    frozenset({("sw", "S2", True), ("load", "P1", "B")}),
)


def sample_atoms(rng: random.Random, pool):
    return rng.choice(pool)


# --------------------------------------------------------------------------
# Public observation (immutable, fixed size, no hidden fields).
# --------------------------------------------------------------------------
class Observation(NamedTuple):
    x: int
    y: int
    ax: int
    ay: int
    bx: int
    by: int
    s1: bool
    s2: bool
    p1: str          # "none" | "A" | "B"
    p2: str
    d1: bool         # open
    d2: bool
    finished: bool


class StepResult(NamedTuple):
    observation: Observation
    valid: bool
    door_events: tuple = ()   # ("D1","open") / ("D2","close") observed changes


@dataclass(frozen=True)
class LabConfig:
    """Hidden wiring: conditions and delay flags for the two doors.
    Distractor components are implicit: any plate/switch not referenced by
    either condition is a correlated decoy."""
    d1_cond: frozenset
    d2_cond: frozenset
    d1_delay: bool
    d2_delay: bool

    def cond_for(self, door: str) -> frozenset:
        return self.d1_cond if door == "D1" else self.d2_cond

    def delay_for(self, door: str) -> bool:
        return self.d1_delay if door == "D1" else self.d2_delay


def sample_config(rng: random.Random, d1_pool=F1_TEMPLATES,
                  d2_pool=F2_TEMPLATES) -> LabConfig:
    return LabConfig(
        d1_cond=sample_atoms(rng, d1_pool),
        d2_cond=sample_atoms(rng, d2_pool),
        d1_delay=bool(rng.getrandbits(1)),
        d2_delay=bool(rng.getrandbits(1)),
    )


# --------------------------------------------------------------------------
# Internal world state and public physics.
# --------------------------------------------------------------------------
class WS(NamedTuple):
    """Full physical state. `door_open` and `armed` are public in the sense
    that door cell is observable; the *wiring* is not."""
    pos: tuple
    objs: dict       # obj -> (x, y)
    plate_load: dict  # plate -> "none"/obj
    switch_state: dict  # switch -> bool
    door_open: dict   # door -> bool
    armed: dict       # door -> bool (delay latch)


def make_state() -> WS:
    return WS(
        pos=START,
        objs=dict(OBJECT_HOME),
        plate_load={p: "none" for p in PLATES},
        switch_state={s: False for s in SWITCHES},
        door_open={d: False for d in DOORS},
        armed={d: False for d in DOORS},
    )


def in_bounds(cell) -> bool:
    x, y = cell
    return 0 <= x < W and 0 <= y < H


def cell_blocked_physical(cell, state: WS) -> bool:
    """A cell is impassable to movement/pushes if wall, or closed door."""
    if cell in WALLS:
        return True
    for door, dc in DOOR_CELL.items():
        if dc == cell and not state.door_open[door]:
            return True
    return False


def occupied_by_object(cell, state: WS) -> Optional[str]:
    for o, c in state.objs.items():
        if c == cell:
            return o
    return None


def clone_ws(state: WS) -> WS:
    """Return a deep-enough copy: all inner dictionaries are copied so a
    later mutation of `armed`/`door_open` etc. never leaks into the source
    state (critical for search/planning/particle use)."""
    return WS(
        pos=state.pos,
        objs=dict(state.objs),
        plate_load=dict(state.plate_load),
        switch_state=dict(state.switch_state),
        door_open=dict(state.door_open),
        armed=dict(state.armed),
    )


def public_next(state: WS, action: str) -> Optional[WS]:
    """Deterministic public mechanics: movement, pushing, switch toggling.
    Doors are treated as static cells here (their opening/closing is the
    hidden part). Returns None for an invalid action. The returned state
    is always fully independent of `state` (deep-copied containers)."""
    if action in MOVE_ACTIONS:
        dx, dy = DIRS[action]
        target = (state.pos[0] + dx, state.pos[1] + dy)
        if not in_bounds(target) or cell_blocked_physical(target, state):
            return None
        if occupied_by_object(target, state) is not None:
            return None
        return clone_ws(state)._replace(pos=target)
    if action == "Toggle":
        for sw, sc in SWITCH_CELL.items():
            if sc == state.pos:
                ns = clone_ws(state)
                ns.switch_state[sw] = not ns.switch_state[sw]
                return ns
        return None
    if action.startswith("Push"):
        dx, dy = DIRS[action[4:]]
        c1 = (state.pos[0] + dx, state.pos[1] + dy)
        obj = occupied_by_object(c1, state)
        if obj is None:
            return None
        c2 = (c1[0] + dx, c1[1] + dy)
        if (not in_bounds(c2) or cell_blocked_physical(c2, state)
                or occupied_by_object(c2, state) is not None
                or c2 == state.pos):
            return None
        ns = clone_ws(state)
        ns.objs[obj] = c2
        for p, pc in PLATE_CELL.items():
            if pc == c1:
                ns.plate_load[p] = "none"
            if pc == c2:
                ns.plate_load[p] = obj
        return ns._replace(pos=c1)
    return None


def eval_atoms(atoms, state: WS) -> bool:
    for atom in atoms:
        k = atom[0]
        if k == "load":
            _, plate, obj = atom
            if state.plate_load.get(plate) != obj:
                return False
        elif k == "loadany":
            _, plate = atom
            if state.plate_load.get(plate) == "none":
                return False
        elif k == "sw":
            _, sw, val = atom
            if state.switch_state.get(sw) != val:
                return False
        else:
            raise ValueError(f"unknown atom {atom}")
    return True


def door_condition_satisfied(config: LabConfig, door: str, state: WS) -> bool:
    return eval_atoms(config.cond_for(door), state)


def transition_with_hidden(state: WS, action: str, config: LabConfig):
    """Full step model: public mechanics plus hidden door dynamics.
    Returns (next_state_or_None, events). Shared by the environment and by
    model-based planners (a single source of truth for the wiring)."""
    nxt = public_next(state, action)
    if nxt is None:
        return None, []
    events = []
    delay = (config.d1_delay, config.d2_delay)
    for i, door in enumerate(DOORS):
        cond = door_condition_satisfied(config, door, nxt)
        if not nxt.door_open[door]:
            if cond:
                if not delay[i]:
                    nxt.door_open[door] = True
                    nxt.armed[door] = False
                    events.append((door, "open"))
                elif nxt.armed[door]:
                    nxt.door_open[door] = True
                    nxt.armed[door] = False
                    events.append((door, "open"))
                else:
                    nxt.armed[door] = True
            else:
                nxt.armed[door] = False
        else:
            if not cond:
                if nxt.pos != DOOR_CELL[door] and occupied_by_object(DOOR_CELL[door], nxt) is None:
                    nxt.door_open[door] = False
                    nxt.armed[door] = False
                    events.append((door, "close"))
    return nxt, events


# --------------------------------------------------------------------------
# The simulation world.
# --------------------------------------------------------------------------
class LabWorld:
    """The environment. Holds the hidden config; exposes only `observe()`
    and `step(action)`. Nothing in the public surface references the wiring.
    """

    def __init__(self, config: LabConfig, ep_seed: int = 0, noise_q: float = 0.0,
                 max_steps: int = MAX_STEPS):
        self._config = config
        self._noise_q = noise_q
        self.max_steps = max_steps
        self._ep_seed = ep_seed
        self.steps = 0
        self.finished = False
        self._state = make_state()
        self.last_events = ()

    # -- hidden noise stream (used to match readings across policies) ----
    def _readings(self):
        """Return the true readings dict for the current physical state."""
        s = self._state
        return {
            "s1": s.switch_state["S1"], "s2": s.switch_state["S2"],
            "d1": s.door_open["D1"], "d2": s.door_open["D2"],
            "p1": s.plate_load["P1"], "p2": s.plate_load["P2"],
        }

    def _corrupt(self, readings: dict) -> dict:
        if self._noise_q <= 0.0:
            return dict(readings)
        out = dict(readings)
        # Deterministic per (episode, step) noise stream, keyed only by the
        # step index, so every policy sees *identical* corrupted readings
        # regardless of how many steps it has taken.
        nrng = random.Random((self._ep_seed * 1000003 + self.steps) & 0xFFFFFFFF)
        for key in ("s1", "s2", "d1", "d2"):
            if nrng.random() < self._noise_q:
                out[key] = not out[key]
        for key in ("p1", "p2"):
            if nrng.random() < self._noise_q:
                out[key] = nrng.choice(("none", "A", "B"))
        return out

    # -- public surface --------------------------------------------------
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
        nxt, events = transition_with_hidden(self._state, action, self._config)
        if nxt is None:
            return StepResult(self.observe(), False, ())
        self._state = nxt
        self.last_events = tuple(events)
        if self._state.pos == GOAL:
            self.finished = True
        return StepResult(self.observe(), True, tuple(events))


# --------------------------------------------------------------------------
# Episode specifications and world generation (paired across policies).
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LabEpisode:
    config: LabConfig
    max_steps: int
    noise_q: float
    seed: int          # per-episode seed for the noise stream
    context: int = 0   # optional symbolic context id (unused by policies)

CONDITIONS = ("stable", "shift", "rapid_shift", "iid", "novel", "noisy_obs", "tight_budget")


def is_solvable(config: LabConfig, max_plan: int = 60) -> bool:
    """Ground-truth solvability from the initial state: a plan to the goal
    must exist under the full hidden dynamics. This is an environment
    property (all studied configurations are solvable); it is never passed
    to any policy."""
    from collections import deque
    start = make_state()
    if start.pos == GOAL:
        return True
    key0 = (start.pos, start.objs["A"], start.objs["B"],
            tuple(start.switch_state[s] for s in SWITCHES),
            tuple(start.plate_load[p] for p in PLATES),
            tuple(start.door_open[d] for d in DOORS))
    q = deque([(start, 0)])
    seen = {key0}
    while q:
        st, depth = q.popleft()
        if depth >= max_plan:
            continue
        for a in ACTIONS:
            nxt, _ = transition_with_hidden(st, a, config)
            if nxt is None:
                continue
            k = (nxt.pos, nxt.objs["A"], nxt.objs["B"],
                 tuple(nxt.switch_state[s] for s in SWITCHES),
                 tuple(nxt.plate_load[p] for p in PLATES),
                 tuple(nxt.door_open[d] for d in DOORS))
            if k in seen:
                continue
            seen.add(k)
            if nxt.pos == GOAL:
                return True
            q.append((nxt, depth + 1))
    return False


def split_pools(seed_pools: int = 20260901):
    """Deterministic *disjoint* training/validation/held-out-test pools of
    *solvable* configurations. Enumerates the full distinct config space
    (|F|^2 * 2 * 2 = 256), keeps only ground-truth solvable ones, and
    splits them 60/20/20. Returns (train, val, test) lists of LabConfig."""
    prng = random.Random(seed_pools)
    all_configs = [
        LabConfig(d1_cond=c1, d2_cond=c2, d1_delay=bool(b1), d2_delay=bool(b2))
        for c1 in F1_TEMPLATES for c2 in F2_TEMPLATES
        for b1 in (0, 1) for b2 in (0, 1)
    ]
    solvable = [c for c in all_configs if is_solvable(c)]
    prng.shuffle(solvable)
    n = len(solvable)
    train_n = int(n * 0.60)
    val_n = int(n * 0.20)
    train = solvable[:train_n]
    val = solvable[train_n:train_n + val_n]
    test = solvable[train_n + val_n:]
    return train, val, test, len(solvable), n


TRAIN_POOL, VAL_POOL, TEST_POOL, N_SOLVABLE, N_TOTAL = split_pools()


def generate_worlds(condition: str, seed: int, episodes: int,
                    train_pool=None, val_pool=None, test_pool=None) -> list:
    if condition not in CONDITIONS or episodes < 4:
        raise ValueError("invalid condition or episode count")
    if train_pool is None:
        train_pool, val_pool, test_pool = TRAIN_POOL, VAL_POOL, TEST_POOL
    rng = random.Random(seed)
    train = list(train_pool)
    val = list(val_pool)
    test = list(test_pool)
    out = []
    cfg = rng.choice(train)
    for e in range(episodes):
        if condition == "iid":
            cfg = rng.choice(train)
        elif condition == "rapid_shift" and e > 0 and e % 30 == 0:
            # alternate between train- and val-pool configurations
            cfg = rng.choice(val if rng.getrandbits(1) else train)
        elif condition in ("shift", "tight_budget") and e == episodes // 2:
            cfg = rng.choice(val)
        elif condition == "novel" and e == episodes // 2:
            cfg = rng.choice(test)
        elif condition in ("novel", "shift", "tight_budget") and e >= episodes // 2:
            pass  # keep the switched-in config stable for the second half
        elif e == 0:
            pass
        max_steps = TIGHT_STEPS if condition == "tight_budget" else MAX_STEPS
        noise_q = NOISE_Q if condition == "noisy_obs" else 0.0
        out.append(LabEpisode(cfg, max_steps, noise_q, seed=seed * 1000 + e, context=0))
    return out
