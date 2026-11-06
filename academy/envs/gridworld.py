"""METARGON Academy — internal grid-world environment ("MechanismGrid").

A lightweight, CPU-only world used by the Academy. It is an independent
implementation (authored for the Academy) of a grid world with:

  * two movable objects (A, B) that can be pushed (Sokoban-style),
  * up to two pressure plates (P1, P2) which report what is on them,
  * up to two switches (S1, S2) toggled by standing on them,
  * up to two door barriers (D1, D2) that block movement until a hidden
    rule is satisfied,
  * hidden door rules that are either conjunctive (all atoms must hold) or
    disjunctive (any of several clauses may hold; used only by the OOD
    family), with an optional one-step delay and an optional noisy
    observation channel.

The hidden state is the rule configuration, delay flags and the noise
stream. None of it is exposed through observations.

The *public world spec* (walls, plate/switch/door cells, start, goal,
object homes) is static per topology and is handed to every agent at
episode start — it is public physics, like the board of a game. Only the
rule wiring, delays and noise are hidden.

No external dependencies. Deterministic given seeds.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import NamedTuple, Optional, Union

# ---------------------------------------------------------------------------
# Geometry constants (two authored topologies; both are 7x5).
# ---------------------------------------------------------------------------
W, H = 7, 5

START = (0, 4)
GOAL = (6, 0)

OBJECTS = ("A", "B")
PLATES = ("P1", "P2")
SWITCHES = ("S1", "S2")
DOORS = ("D1", "D2")

ACTIONS = ("N", "E", "S", "W", "Toggle", "PushN", "PushE", "PushS", "PushW")
MOVE_ACTIONS = ("N", "E", "S", "W")
DIRS = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}

MAX_STEPS = 80
TIGHT_STEPS = 44

NOISE_Q = 0.08


class Geom(NamedTuple):
    """Public, static world geometry for one topology."""
    walls: frozenset
    door_cell: dict          # door -> cell
    object_home: dict        # object -> cell
    plate_cell: dict         # plate -> cell
    switch_cell: dict        # switch -> cell
    start: tuple
    goal: tuple

    def cell_blocked(self, cell, door_open: dict) -> bool:
        if cell in self.walls:
            return True
        for door, dc in self.door_cell.items():
            if dc == cell and not door_open[door]:
                return True
        return False

    def occupied(self, cell, objs: dict) -> Optional[str]:
        for o, c in objs.items():
            if c == cell:
                return o
        return None

    def refresh_plates(self, plate_load: dict, objs: dict) -> None:
        for p, pc in self.plate_cell.items():
            obj = self.occupied(pc, objs)
            plate_load[p] = obj if obj is not None else "none"


TP1 = Geom(
    walls=frozenset({(0, 2), (1, 2), (2, 2), (4, 2), (5, 2),
                     (6, 2), (5, 0)}),
    door_cell={"D1": (3, 2), "D2": (5, 1)},
    object_home={"A": (2, 4), "B": (4, 3)},
    plate_cell={"P1": (3, 4), "P2": (1, 1)},
    switch_cell={"S1": (0, 1), "S2": (4, 4)},
    start=(0, 4),
    goal=(6, 0),
)

TP2 = Geom(
    walls=frozenset({(0, 2), (1, 2), (3, 2), (4, 2), (5, 2),
                     (6, 2), (3, 0)}),
    door_cell={"D1": (2, 2), "D2": (3, 1)},
    object_home={"A": (1, 3), "B": (2, 4)},
    plate_cell={"P1": (2, 1), "P2": (4, 1)},
    switch_cell={"S1": (1, 0), "S2": (4, 3)},
    start=(0, 4),
    goal=(6, 0),
)

TOPOLOGIES = {"TP1": TP1, "TP2": TP2}


# ---------------------------------------------------------------------------
# Atom vocabulary.
#   ("load", plate, obj)   plate loaded with exactly that object
#   ("loadany", plate)     plate loaded with any object
#   ("empty", plate)       plate unloaded
#   ("sw", switch, value)  switch in that state (True = on)
# A *conjunctive* rule is a frozenset of atoms. A *disjunctive* rule is a
# tuple of conjunctive clauses (any clause true => rule true).
# ---------------------------------------------------------------------------
def atoms_from_state(plate_load: dict, switch_state: dict) -> frozenset:
    out = set()
    for p in PLATES:
        load = plate_load[p]
        if load in OBJECTS:
            out.add(("load", p, load))
            out.add(("loadany", p))
        else:
            out.add(("empty", p))
    for s in SWITCHES:
        out.add(("sw", s, bool(switch_state[s])))
    return frozenset(out)


def _atom_holds(atom, plate_load: dict, switch_state: dict) -> bool:
    k = atom[0]
    if k == "load":
        return plate_load.get(atom[1]) == atom[2]
    if k == "loadany":
        return plate_load.get(atom[1]) in OBJECTS
    if k == "empty":
        return plate_load.get(atom[1]) not in OBJECTS
    if k == "sw":
        return switch_state.get(atom[1]) == atom[2]
    raise ValueError(f"unknown atom {atom}")


def rule_holds(rule, plate_load: dict, switch_state: dict) -> bool:
    if isinstance(rule, tuple):          # disjunctive
        return any(rule_holds(clause, plate_load, switch_state)
                   for clause in rule)
    return all(_atom_holds(a, plate_load, switch_state) for a in rule)


# ---------------------------------------------------------------------------
# Public observation and step results (no hidden fields).
# ---------------------------------------------------------------------------
class Observation(NamedTuple):
    x: int
    y: int
    ax: int
    ay: int
    bx: int
    by: int
    s1: bool
    s2: bool
    p1: str            # "none" | "A" | "B"
    p2: str
    d1: bool           # open
    d2: bool
    finished: bool

    @property
    def pos(self):
        return (self.x, self.y)

    @property
    def objs(self):
        return {"A": (self.ax, self.ay), "B": (self.bx, self.by)}

    @property
    def switch_state(self):
        return {"S1": self.s1, "S2": self.s2}

    @property
    def plate_load(self):
        return {"P1": self.p1, "P2": self.p2}

    @property
    def door_open(self):
        return {"D1": self.d1, "D2": self.d2}


class StepResult(NamedTuple):
    observation: Observation
    valid: bool
    events: tuple = ()     # ("D1","open") / ("D2","close") observed changes
    terminated: bool = False


# ---------------------------------------------------------------------------
# Environment configuration (hidden wiring).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EnvConfig:
    topology: str                       # "TP1" | "TP2"
    d1_rule: Union[frozenset, tuple]
    d2_rule: Union[frozenset, tuple]
    d1_delay: bool
    d2_delay: bool

    @property
    def kind(self) -> str:
        return ("disjunctive"
                if isinstance(self.d1_rule, tuple) or isinstance(self.d2_rule, tuple)
                else "conjunctive")

    def rule_for(self, door: str):
        return self.d1_rule if door == "D1" else self.d2_rule

    def delay_for(self, door: str) -> bool:
        return self.d1_delay if door == "D1" else self.d2_delay

    def n_atoms(self) -> int:
        def count(r):
            if isinstance(r, tuple):
                return sum(len(c) for c in r)
            return len(r)
        return count(self.d1_rule) + count(self.d2_rule)

    def has_delay(self) -> bool:
        return self.d1_delay or self.d2_delay

    @property
    def key(self):
        return (self.topology, self.d1_rule, self.d2_rule,
                self.d1_delay, self.d2_delay)


# ---------------------------------------------------------------------------
# Internal world state.
# ---------------------------------------------------------------------------
class WS(NamedTuple):
    pos: tuple
    objs: dict            # object -> (x, y)
    plate_load: dict      # plate -> "none" | object
    switch_state: dict    # switch -> bool
    door_open: dict       # door -> bool
    armed: dict           # door -> bool (delay latch)


def make_ws(geom: Geom) -> WS:
    return WS(
        pos=geom.start,
        objs=dict(geom.object_home),
        plate_load={p: "none" for p in PLATES},
        switch_state={s: False for s in SWITCHES},
        door_open={d: False for d in DOORS},
        armed={d: False for d in DOORS},
    )


def clone_ws(state: WS) -> WS:
    return WS(
        pos=state.pos,
        objs=dict(state.objs),
        plate_load=dict(state.plate_load),
        switch_state=dict(state.switch_state),
        door_open=dict(state.door_open),
        armed=dict(state.armed),
    )


def in_bounds(cell) -> bool:
    x, y = cell
    return 0 <= x < W and 0 <= y < H


def public_next(state: WS, action: str, geom: Geom) -> Optional[WS]:
    """Deterministic public mechanics. Returns None for invalid actions.
    Doors act as static cells here; opening/closing is the hidden part."""
    if action in MOVE_ACTIONS:
        dx, dy = DIRS[action]
        target = (state.pos[0] + dx, state.pos[1] + dy)
        if not in_bounds(target) or geom.cell_blocked(target, state.door_open):
            return None
        if geom.occupied(target, state.objs) is not None:
            return None
        return clone_ws(state)._replace(pos=target)
    if action == "Toggle":
        for sw, sc in geom.switch_cell.items():
            if sc == state.pos:
                ns = clone_ws(state)
                ns.switch_state[sw] = not ns.switch_state[sw]
                return ns
        return None
    if action.startswith("Push"):
        dx, dy = DIRS[action[4:]]
        c1 = (state.pos[0] + dx, state.pos[1] + dy)
        obj = geom.occupied(c1, state.objs)
        if obj is None:
            return None
        c2 = (c1[0] + dx, c1[1] + dy)
        if (not in_bounds(c2) or geom.cell_blocked(c2, state.door_open)
                or geom.occupied(c2, state.objs) is not None
                or c2 == state.pos):
            return None
        ns = clone_ws(state)
        ns.objs[obj] = c2
        geom.refresh_plates(ns.plate_load, ns.objs)
        return ns._replace(pos=c1)
    return None


def transition_with_hidden(state: WS, action: str, config: EnvConfig,
                           geom: Geom):
    """Full step model: public mechanics plus hidden door dynamics.
    Returns (next_state_or_None, events)."""
    nxt = public_next(state, action, geom)
    if nxt is None:
        return None, []
    events = []
    for door in DOORS:
        rule = config.rule_for(door)
        hold = rule_holds(rule, nxt.plate_load, nxt.switch_state)
        if not nxt.door_open[door]:
            if hold:
                if not config.delay_for(door):
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
            if not hold:
                dc = geom.door_cell[door]
                if nxt.pos != dc and geom.occupied(dc, nxt.objs) is None:
                    nxt.door_open[door] = False
                    nxt.armed[door] = False
                    events.append((door, "close"))
    return nxt, events


# ---------------------------------------------------------------------------
# The environment.
# ---------------------------------------------------------------------------
class MechanismGrid:
    """Public surface: `reset(ep_seed)`, `step(action)`, `observe()`.
    The hidden config is never exposed through this API."""

    def __init__(self, config: EnvConfig, max_steps: int = MAX_STEPS,
                 noise_q: float = 0.0):
        self._config = config
        self._geom = TOPOLOGIES[config.topology]
        self.max_steps = max_steps
        self._noise_q = noise_q
        self.steps = 0
        self.finished = False
        self.last_events = ()
        self._ep_seed = 0
        self._state = make_ws(self._geom)

    @property
    def geom(self) -> Geom:
        return self._geom

    def reset(self, ep_seed: int = 0) -> Observation:
        self._ep_seed = ep_seed
        self.steps = 0
        self.finished = False
        self._state = make_ws(self._geom)
        self.last_events = ()
        return self.observe()

    # -- hidden noise stream --------------------------------------------
    def _corrupt(self, readings: dict) -> dict:
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

    def _readings(self) -> dict:
        s = self._state
        return {
            "s1": s.switch_state["S1"], "s2": s.switch_state["S2"],
            "d1": s.door_open["D1"], "d2": s.door_open["D2"],
            "p1": s.plate_load["P1"], "p2": s.plate_load["P2"],
        }

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
        nxt, events = transition_with_hidden(
            self._state, action, self._config, self._geom)
        if nxt is None:
            return StepResult(self.observe(), False, (), False)
        self._state = nxt
        self.last_events = tuple(events)
        if self._state.pos == self._geom.goal:
            self.finished = True
        return StepResult(self.observe(), True, tuple(events), self.finished)


# ---------------------------------------------------------------------------
# Solvability check (environment property; never passed to policies).
# ---------------------------------------------------------------------------
def is_solvable(config: EnvConfig, max_plan: int = 80) -> bool:
    """BFS over the full hidden dynamics from the initial state."""
    from collections import deque
    geom = TOPOLOGIES[config.topology]
    start = make_ws(geom)
    if start.pos == geom.goal:
        return True

    def key(st: WS):
        return (st.pos, st.objs["A"], st.objs["B"],
                tuple(st.switch_state[s] for s in SWITCHES),
                tuple(st.plate_load[p] for p in PLATES),
                tuple(st.door_open[d] for d in DOORS))

    q = deque([(start, 0)])
    seen = {key(start)}
    while q:
        st, depth = q.popleft()
        if depth >= max_plan:
            continue
        for a in ACTIONS:
            nxt, _ = transition_with_hidden(st, a, config, geom)
            if nxt is None:
                continue
            k = key(nxt)
            if k in seen:
                continue
            seen.add(k)
            if nxt.pos == geom.goal:
                return True
            q.append((nxt, depth + 1))
    return False
