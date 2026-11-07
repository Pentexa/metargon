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
    ACTIONS, DOORS, DOOR_CELL, GOAL, MOVE_ACTIONS, PLATES, PLATE_CELL,
    SWITCH_CELL, SWITCHES, WALLS, WS, make_state, public_next,
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
MAX_EXPANSIONS = 700         # node expansions per probe BFS
PROBE_MAX_PLANS = 3          # probe plans evaluated per decision
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


def _violates_protected(nxt: WS, protected: frozenset, target: frozenset) -> bool:
    """True if the transition destroys a protected atom (a condition that is
    keeping a door open) that is not itself being re-established by the
    plan target."""
    if not protected:
        return False
    after = atoms_of_state(nxt)
    for p in protected:
        if p in target:
            continue
        if p not in after:
            return True
    return False


def plan_to_atoms(state: WS, target: frozenset, max_expand: int = MAX_EXPANSIONS,
                  on_expand=None, cache=None, protected=frozenset()):
    """Shortest plan (public physics only) to a state whose atoms are a
    superset of `target`. Plans are constrained not to destroy `protected`
    atoms (conditions keeping doors open) unless the target itself needs
    to change them. Returns the action list, or None. `cache` is an
    optional policy-level {(state_key, target) -> plan | None} bounded
    dictionary."""
    if target <= atoms_of_state(state):
        return []
    key = (state_key(state), target, protected) if cache is not None else None
    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return list(hit) if hit else None
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
                if cache is not None:
                    cache[key] = tuple(np)
                    if len(cache) > 3000:
                        for kk in list(cache)[:500]:
                            del cache[kk]
                return np
            if _violates_protected(nxt, protected, target):
                continue
            queue.append((nxt, np))
    if cache is not None:
        cache[key] = None
        if len(cache) > 3000:
            for kk in list(cache)[:500]:
                del cache[kk]
    return None


def plan_to_pos(state: WS, target_pos, max_expand: int = MAX_EXPANSIONS,
                on_expand=None, cache=None, protected=frozenset()):
    """Shortest plan (public physics only) to occupy target_pos, without
    destroying `protected` atoms."""
    if state.pos == target_pos:
        return []
    key = (state_key(state), target_pos, protected) if cache is not None else None
    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return list(hit) if hit else None
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
                if cache is not None:
                    cache[key] = tuple(np)
                    if len(cache) > 3000:
                        for kk in list(cache)[:500]:
                            del cache[kk]
                return np
            if _violates_protected(nxt, protected, frozenset()):
                continue
            queue.append((nxt, np))
    if cache is not None:
        cache[key] = None
        if len(cache) > 3000:
            for kk in list(cache)[:500]:
                del cache[kk]
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
        # static far-region test: east of the barrier column
        self._far_x = {"D1": 3, "D2": 5}
        self._pending_probe = {}                  # door -> (antecedent, deadline)
        self._PROBE_TIMEOUT = 10                  # steps before a probe is
                                                  # deemed failed (no event)
        self._plan_cache = {}                     # bounded plan cache
                                                  # {(state_key, target) -> plan}
        self._current_plan = []                   # follow plan while valid
        self._recent_protect = {}                 # door -> (atoms, expiry_step)

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
        # mark doors passed only when the agent is physically on the far
        # side (east of the barrier); component equality is not usable
        # because an open door merges both sides into one component
        for door in DOORS:
            if not doors[door]:
                continue
            if self.belief.pos[0] > self._far_x[door]:
                self._passed[door] = True

    # -- planner ---------------------------------------------------------
    def _protected_atoms(self) -> frozenset:
        """Atoms that currently satisfy a usable hypothesis for any door
        (open or closed/armed), plus atoms that were true when a door just
        opened (recent causal credit). Planning must not destroy these
        inadvertently."""
        out = set()
        now = atoms_of_state(self.belief)
        for door in DOORS:
            h = self.memory.best_for_door(door, opens=True)
            if h is not None and h.antecedent <= now:
                out.update(h.antecedent)
        for door, (atoms, expiry) in self._recent_protect.items():
            if self.memory.step <= expiry:
                out.update(atoms)
        return frozenset(out)

    def _conflicts_with_protected(self, ante: frozenset) -> bool:
        """A non-hypothesis probe should not flip an atom that is currently
        load-bearing for another door (protected)."""
        protected = self._protected_atoms()
        if not protected:
            return False
        for p in protected:
            k = p[0]
            if k == "sw":
                for a in ante:
                    if a[0] == "sw" and a[1] == p[1] and a[2] != p[2]:
                        return True
            elif k == "load":
                for a in ante:
                    if a[0] == "load" and a[1] == p[1] and a[2] != p[2]:
                        return True
        return False

    def _novel_singles(self, door):
        tried = self.tried[door]
        return [s for s in ALL_SINGLES if s not in tried]

    def _conjunctive_probes(self, door):
        """Pair probes built from atoms that are either currently true or
        whose singleton probe was already tried for this door."""
        tried = self.tried[door]
        singles = [s for s in ALL_SINGLES if s in tried]
        atoms_pool = set()
        for s in singles:
            atoms_pool.update(s)
        atoms_pool.update(atoms_of_state(self.belief))
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
        protected = self._protected_atoms()
        # 0. If a door is open and we have not yet passed through it, go
        # through it: this is how the agent reaches unexplored mechanisms.
        for door in DOORS:
            if self.belief.door_open[door] and not self._passed[door]:
                plan = plan_to_pos(self.belief, self._far_anchor[door],
                                   on_expand=on_expand, cache=self._plan_cache,
                                   protected=protected)
                if plan is not None:
                    return plan, door
        # 1. If the goal is reachable under the current belief, just go.
        self._step_since_goal_check += 1
        if (pattern not in self._goal_infeasible
                or self._step_since_goal_check >= 8
                or self.belief.pos == GOAL):
            plan = plan_to_pos(self.belief, GOAL, on_expand=on_expand,
                               cache=self._plan_cache, protected=protected)
            self._step_since_goal_check = 0
            if plan is not None:
                return plan, None
            self._goal_infeasible[pattern] = True
        # 2. Gather candidate probes for each closed door: hypothesis-backed
        # antecedents first (by confidence), then untried singles, then
        # conjunctive pairs. Singles/pairs are ranked by "needs action"
        # and distance so the most relevant probe is tried first.
        candidates = []
        best_ante = {}
        now_atoms = atoms_of_state(self.belief)
        for door in DOORS:
            if self.belief.door_open[door]:
                continue
            hyps = self.memory.query(door, opens=True)
            for h in hyps[:3]:
                candidates.append((door, h.antecedent, 0, h.confidence))
            if hyps:
                best_ante[door] = hyps[0].antecedent
            probes = self._novel_singles(door) + self._conjunctive_probes(door)
            probes = [s for s in probes if not self._conflicts_with_protected(s)]

            def rank_key(ante):
                needs = 0 if ante <= now_atoms else 1
                d = min(
                    (abs(self.belief.pos[0] - cell[0])
                     + abs(self.belief.pos[1] - cell[1]))
                    for atom in ante
                    for cell in [PLATE_CELL[atom[1]] if atom[0] in ("load", "loadany")
                                 else SWITCH_CELL[atom[1]]]
                )
                return (needs, d)
            probes.sort(key=rank_key)
            for s in probes[:PROBE_MAX_PLANS]:
                candidates.append((door, s, 1, 0.0))
        # 2b. compositional cross-door probe: satisfy verified antecedents
        # for two closed doors simultaneously (keeps both doors open at once)
        if len(best_ante) >= 2:
            doors_c = [d for d in DOORS if d in best_ante]
            merged = frozenset().union(*(best_ante[d] for d in doors_c))
            if merged not in self.tried[doors_c[0]] and merged not in self.tried[doors_c[1]]:
                candidates.append((doors_c[0], merged, 0, 1.0))
        # 3. feasibility filter (cheap), then plan the best feasible one:
        # hypothesis-backed (priority 0) candidates are preferred, by
        # confidence, with plan length as the final tie-breaker
        best = None
        evaluated = 0
        now_atoms = atoms_of_state(self.belief)
        for door, ante, prio, conf in candidates:
            if ante in self.tried[door]:
                continue
            # prefer candidates that still require action (not already true)
            if prio == 1 and ante <= now_atoms:
                # already arranged: mark as probed; the missing piece is a
                # different atom, so do not pick this one
                self.tried[door].add(ante)
                continue
            if not self._target_feasible(ante):
                self.tried[door].add(ante)
                continue
            plan = plan_to_atoms(self.belief, ante, on_expand=on_expand,
                                 cache=self._plan_cache, protected=protected)
            self.plan_calls += 1
            evaluated += 1
            if plan:
                ck = (prio, -conf)
                if (best is None or ck < best[3]
                        or (ck == best[3] and len(plan) < len(best[0]))):
                    best = (plan, door, ante, ck)
            elif prio == 0:
                # a hypothesis-backed probe that cannot be executed right now
                # is not marked failed; it may become feasible later.
                pass
            else:
                # single/pair probe: no executable plan found; move on
                self.tried[door].add(ante)
            if evaluated >= PROBE_MAX_PLANS + 2:
                break
        if best is not None:
            plan, door, ante, _ = best
            self._pending_probe[door] = (ante, self.memory.step
                                         + len(plan) + self._PROBE_TIMEOUT)
            return plan, door
        return None, None

    def _random_valid(self):
        valid = [a for a in ACTIONS if public_next(self.belief, a) is not None]
        # avoid blind switch toggling in fallback (breaks open-door states)
        moves = [a for a in valid if a in MOVE_ACTIONS or a.startswith("Push")]
        if moves:
            return self.rng.choice(moves)
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
        protected = self._protected_atoms()
        # follow the current plan while its actions remain executable and
        # do not destroy currently protected atoms
        while self._current_plan:
            a = self._current_plan[0]
            nxt = public_next(self.belief, a)
            if nxt is not None and not _violates_protected(
                    nxt, protected, frozenset()):
                self._current_plan.pop(0)
                return a
            self._current_plan.pop(0)
        plan, door = self._choose_probe()
        if plan:
            self._current_plan = list(plan)
            return self._current_plan.pop(0)
        return self._random_valid()

    def observe_outcome(self, obs_before, action, res):
        self.steps += 1
        before_atoms = atoms_from_observation(obs_before)
        after_obs = res.observation
        after_atoms = atoms_from_observation(after_obs)
        # infer door events from the observation stream (no privileged channel)
        any_event = False
        for door in DOORS:
            before_open = getattr(obs_before, "d1" if door == "D1" else "d2")
            after_open = getattr(after_obs, "d1" if door == "D1" else "d2")
            if not before_open and after_open:
                # The event happened *during* this transition: the context
                # both before and after the action may contain the cause.
                ctx = before_atoms | after_atoms
                self.memory.observe_event(ctx, door, True)
                self._passed[door] = False
                self._pending_probe.pop(door, None)
                # causal credit: keep the post-transition context protected
                # for a short window (protects cold-start discoveries)
                self._recent_protect[door] = (ctx,
                                              self.memory.step + 12)
                any_event = True
            elif before_open and not after_open:
                ctx = before_atoms | after_atoms
                self.memory.observe_event(ctx, door, False)
                any_event = True
        if any_event:
            # component structure changed: every door's probe memory may be
            # stale, so clear all of them, plus any in-flight plan
            for d in DOORS:
                self.tried[d] = set()
            self._goal_infeasible.clear()
            self._current_plan = []
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
