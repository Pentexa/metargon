"""METARGON v0.9 — Phase D: adaptive planning with a compositional causal
world model.

The agent combines:

  * a belief state maintained from the public observation stream,
  * the compositional causal memory (Phase C),
  * a bounded, hypothesis-driven planner that never enumerates complete
    wirings. It builds candidate *interventions* (subgoals) from its own
    hypotheses; for each candidate it runs a bounded BFS over the public
    physics to generate a concrete action plan, predicts the outcome, and
    executes the first action. If no hypothesis-backed plan exists, it
    probes the cheapest untried single intervention for each closed door.

Computational budget:
  * memory: hard-capped (MAX_HYPOTHESES),
  * planner: at most PROBE_MAX_PLANS BFS calls per decision, each bounded
    by MAX_EXPANSIONS node expansions (bounded depth by construction).
"""
from __future__ import annotations

import random
from collections import deque

from lab_env import (
    ACTIONS, DOORS, DOOR_CELL, GOAL, PLATES, PLATE_CELL, SWITCH_CELL,
    SWITCHES, WALLS, WS, make_state, public_next,
)


def _is_closed_door(cell, belief) -> bool:
    for door, dc in DOOR_CELL.items():
        if dc == cell and not belief.door_open[door]:
            return True
    return False
from compositional_memory import (
    CompositionalMemory, atoms_from_observation, opens_key,
)

# pre-registered planner budgets
MAX_EXPANSIONS = 1500        # node expansions per probe BFS
PROBE_MAX_PLANS = 4          # probe plans evaluated per decision
MAX_HYPOTHESES_AGENT = 128   # memory budget (matches memory default)

ALL_SINGLES = (
    frozenset({("loadany", "P1")}),
    frozenset({("loadany", "P2")}),
    frozenset({("load", "P1", "A")}),
    frozenset({("load", "P1", "B")}),
    frozenset({("load", "P2", "A")}),
    frozenset({("load", "P2", "B")}),
    frozenset({("sw", "S1", True)}),
    frozenset({("sw", "S1", False)}),
    frozenset({("sw", "S2", True)}),
    frozenset({("sw", "S2", False)}),
)

# conjunctive probes: pairs of a plate-load atom and a switch atom; these
# are tried only after their members have been tried as singles without
# effect, which bounds the combinatorial cost.
LOAD_ATOMS = (
    ("loadany", "P1"), ("loadany", "P2"),
    ("load", "P1", "A"), ("load", "P1", "B"),
    ("load", "P2", "A"), ("load", "P2", "B"),
)
SW_ATOMS = (
    ("sw", "S1", True), ("sw", "S1", False),
    ("sw", "S2", True), ("sw", "S2", False),
)
ALL_PAIRS = tuple(
    frozenset({a, b}) for a in LOAD_ATOMS for b in SW_ATOMS
)


def state_key(st: WS):
    return (st.pos, st.objs["A"], st.objs["B"],
            tuple(st.switch_state[s] for s in SWITCHES),
            tuple(st.plate_load[p] for p in PLATES),
            tuple(st.door_open[d] for d in DOORS))


def atoms_of_state(st: WS) -> frozenset:
    out = set()
    for p in PLATES:
        load = st.plate_load[p]
        if load in ("A", "B"):
            out.add(("load", p, load))
            out.add(("loadany", p))
    for s in SWITCHES:
        out.add(("sw", s, bool(st.switch_state[s])))
    return frozenset(out)


def plan_to_atoms(state: WS, target: frozenset, max_expand: int = MAX_EXPANSIONS,
                  on_expand=None):
    """Shortest plan (public physics only) to a state whose atoms are a
    superset of `target`. Returns the action list, or None."""
    if target <= atoms_of_state(state):
        return []
    start_key = state_key(state)
    queue = deque([(state, [])])
    seen = {start_key}
    while queue and len(seen) < max_expand:
        st, path = queue.popleft()
        for a in ACTIONS:
            nxt = public_next(st, a)
            if nxt is None:
                continue
            k = state_key(nxt)
            if k in seen:
                continue
            seen.add(k)
            if on_expand is not None:
                on_expand()
            np = path + [a]
            if target <= atoms_of_state(nxt):
                return np
            queue.append((nxt, np))
    return None


def plan_to_pos(state: WS, target_pos, max_expand: int = MAX_EXPANSIONS,
                on_expand=None):
    """Shortest plan (public physics only) to occupy target_pos."""
    if state.pos == target_pos:
        return []
    start_key = state_key(state)
    queue = deque([(state, [])])
    seen = {start_key}
    while queue and len(seen) < max_expand:
        st, path = queue.popleft()
        for a in ACTIONS:
            nxt = public_next(st, a)
            if nxt is None:
                continue
            k = state_key(nxt)
            if k in seen:
                continue
            seen.add(k)
            if on_expand is not None:
                on_expand()
            np = path + [a]
            if nxt.pos == target_pos:
                return np
            queue.append((nxt, np))
    return None


class MetargonV09:
    name = "metargon"

    def __init__(self, seed: int = 0, max_hypotheses: int = MAX_HYPOTHESES_AGENT):
        self.rng = random.Random(seed)
        self.memory = CompositionalMemory(max_hypotheses=max_hypotheses, seed=seed)
        self.belief = make_state()
        self.tried = {d: set() for d in DOORS}   # antecedents already probed
        self.planner_ops = 0                      # BFS node expansions
        self.plan_calls = 0                       # probe BFS calls
        self.steps = 0
        self.predictions = []                     # (predicted, actual) per door
        self.last_obs = None
        self._comp_cache = (None, None)           # (door_pattern, comp map)
        self._goal_infeasible = {}                # door_pattern -> True
        self._step_since_goal_check = 0
        self._passed = {"D1": False, "D2": False}
        self._far_anchor = {"D1": (4, 1), "D2": (6, 2)}
        self._start_anchor = {"D1": (0, 4), "D2": (4, 1)}
        self._pending_probe = {}                  # door -> (antecedent, deadline)
        self._PROBE_TIMEOUT = 10                  # steps before a probe is
                                                  # deemed failed (no event)

    # -- perception ------------------------------------------------------
    def _sync(self, obs):
        b = self.belief
        b = b._replace(pos=(obs.x, obs.y))
        objs = dict(b.objs)
        objs["A"] = (obs.ax, obs.ay)
        objs["B"] = (obs.bx, obs.by)
        b = b._replace(objs=objs)
        sw = dict(b.switch_state)
        sw["S1"] = obs.s1
        sw["S2"] = obs.s2
        b = b._replace(switch_state=sw)
        pl = dict(b.plate_load)
        pl["P1"] = obs.p1
        pl["P2"] = obs.p2
        b = b._replace(plate_load=pl)
        doors = dict(b.door_open)
        doors["D1"] = obs.d1
        doors["D2"] = obs.d2
        b = b._replace(door_open=doors)
        self.belief = b
        # mark doors passed when we are on their far side
        comp, _ = self._components()
        for door in DOORS:
            if not doors[door]:
                continue
            far = self._far_anchor[door]
            if far in comp and comp.get(self.belief.pos) == comp[far]:
                self._passed[door] = True

    # -- planner ---------------------------------------------------------
    def _novel_singles(self, door):
        tried = self.tried[door]
        return [s for s in ALL_SINGLES if s not in tried]

    def _conjunctive_probes(self, door):
        """Pair probes built from atoms whose singleton probes were already
        tried for this door without producing an event."""
        tried = self.tried[door]
        singles = [s for s in ALL_SINGLES if s in tried]
        # atoms that participated in any tried singleton
        atoms_pool = set()
        for s in singles:
            atoms_pool.update(s)
        out = []
        for p in ALL_PAIRS:
            if p in tried or p in out:
                continue
            if p <= frozenset(atoms_pool):
                out.append(p)
        return out

    def _components(self):
        """Connected components of walkable cells under the current door
        belief (objects ignored). Cached per door-open pattern."""
        pattern = (self.belief.door_open["D1"], self.belief.door_open["D2"])
        if self._comp_cache[0] == pattern:
            return self._comp_cache[1], pattern
        comp = {}
        cid = 0
        walkable = [(x, y) for x in range(7) for y in range(5)
                    if (x, y) not in WALLS and not _is_closed_door((x, y), self.belief)]
        for cell in walkable:
            if cell in comp:
                continue
            stack = [cell]
            comp[cell] = cid
            while stack:
                x, y = stack.pop()
                for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    n = (x + dx, y + dy)
                    if n in walkable and n not in comp:
                        comp[n] = cid
                        stack.append(n)
            cid += 1
        self._comp_cache = (pattern, comp)
        return comp, pattern

    def _target_feasible(self, target: frozenset) -> bool:
        """Cheap component-based feasibility: reject antecedents whose
        components are mutually unreachable under the current belief."""
        comp, pattern = self._components()
        agent = self.belief.pos
        if agent not in comp:
            return False
        c_agent = comp[agent]
        for atom in target:
            k = atom[0]
            if k in ("load", "loadany"):
                plate = atom[1]
                pc = PLATE_CELL[plate]
                if pc not in comp:
                    return False
                if k == "load":
                    obj = atom[2]
                    oc = self.belief.objs[obj]
                    if oc not in comp or comp[oc] != comp[pc] or comp[oc] != c_agent:
                        return False
                elif comp[pc] != c_agent:
                    return False
            elif k == "sw":
                sc = SWITCH_CELL[atom[1]]
                if sc not in comp or comp[sc] != c_agent:
                    return False
        return True

    def _choose_probe(self):
        """Return (plan, door) for the best candidate probe, or (None, None)."""
        def on_expand():
            self.planner_ops += 1
        pattern = (self.belief.door_open["D1"], self.belief.door_open["D2"])
        # 0. If a door is open and we have not yet passed through it, go
        # through it: this is how the agent reaches unexplored mechanisms.
        for door in DOORS:
            if self.belief.door_open[door] and not self._passed[door]:
                plan = plan_to_pos(self.belief, self._far_anchor[door],
                                   on_expand=on_expand)
                if plan is not None:
                    return plan, door
        # 1. If the goal is reachable under the current belief, just go.
        self._step_since_goal_check += 1
        if (pattern not in self._goal_infeasible
                or self._step_since_goal_check >= 8
                or self.belief.pos == GOAL):
            plan = plan_to_pos(self.belief, GOAL, on_expand=on_expand)
            self._step_since_goal_check = 0
            if plan is not None:
                return plan, None
            self._goal_infeasible[pattern] = True
        # 2. Gather candidate probes for each closed door: hypothesis-backed
        # antecedents first, then untried singles, then conjunctive pairs.
        candidates = []
        for door in DOORS:
            if self.belief.door_open[door]:
                continue
            hyps = self.memory.query(door, opens=True)
            for h in hyps[:3]:
                candidates.append((door, h.antecedent, 0))   # priority 0
            probes = self._novel_singles(door) + self._conjunctive_probes(door)
            for s in probes[:PROBE_MAX_PLANS]:
                candidates.append((door, s, 1))              # priority 1
        # 3. feasibility filter (cheap), then plan the cheapest feasible one
        best = None
        evaluated = 0
        for door, ante, prio in candidates:
            if ante in self.tried[door]:
                continue
            if not self._target_feasible(ante):
                self.tried[door].add(ante)
                continue
            plan = plan_to_atoms(self.belief, ante, on_expand=on_expand)
            self.plan_calls += 1
            evaluated += 1
            if plan:
                if best is None or len(plan) < len(best[0]):
                    best = (plan, door, ante)
            elif prio == 0:
                # a hypothesis-backed probe that cannot be executed right now
                # is not marked failed; it may become feasible later.
                pass
            if evaluated >= PROBE_MAX_PLANS:
                break
        if best is not None:
            plan, door, ante = best
            self._pending_probe[door] = (ante, self.memory.step
                                         + self._PROBE_TIMEOUT)
            return plan, door
        return None, None

    def _random_valid(self):
        valid = [a for a in ACTIONS if public_next(self.belief, a) is not None]
        if not valid:
            return self.rng.choice(ACTIONS)
        return self.rng.choice(valid)

    # -- interface -------------------------------------------------------
    def episode_start(self):
        self.belief = make_state()
        self.last_obs = None
        # keep memory and tried-set across episodes (within a run)

    def act(self, obs) -> str:
        self._sync(obs)
        plan, _ = self._choose_probe()
        if plan:
            return plan[0]
        return self._random_valid()

    def observe_outcome(self, obs_before, action, res):
        self.steps += 1
        before_atoms = atoms_from_observation(obs_before)
        after_obs = res.observation
        after_atoms = atoms_from_observation(after_obs)
        # infer door events from the observation stream (no privileged channel)
        for door in DOORS:
            before_open = getattr(obs_before, "d1" if door == "D1" else "d2")
            after_open = getattr(after_obs, "d1" if door == "D1" else "d2")
            if not before_open and after_open:
                self.memory.observe_event(before_atoms, door, True)
                self.tried[door] = set()   # new info resets probe memory
                self._passed[door] = False
                self._pending_probe.pop(door, None)
            elif before_open and not after_open:
                self.memory.observe_event(before_atoms, door, False)
                self.tried[door] = set()
        # nonevents for doors that did not change as expected
        for door in DOORS:
            after_open = getattr(after_obs, "d1" if door == "D1" else "d2")
            if not after_open:
                self.memory.observe_nonevent(after_atoms, door, True)
            else:
                self.memory.observe_nonevent(after_atoms, door, False)
        self.memory.step_advance()
        # probes that produced no door event within their timeout are marked
        # failed so the agent moves on to other candidates
        for door, (ante, deadline) in list(self._pending_probe.items()):
            if self.memory.step >= deadline:
                self.tried[door].add(ante)
                del self._pending_probe[door]

    def diagnostics(self) -> dict:
        return {
            "memory": self.memory.diagnostics(),
            "planner_ops": self.planner_ops,
            "plan_calls": self.plan_calls,
            "steps": self.steps,
        }
