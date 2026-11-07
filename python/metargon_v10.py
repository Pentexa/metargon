"""METARGON v0.10 — adaptive planning with robust perception, probe
cooldowns, and prioritized information gathering.

Three families of changes vs v0.9 (each independently ablatable for the
paired study; see docs/RESEARCH_V10.md):

  A. RobustBelief          — physics-reconciled belief from noisy public
                             observations (movement proof, action-explained
                             readings, debounce of unexplained changes).
  B. Probe cooldowns       — failed probes are retried after a cooldown,
                             not permanently marked dead (fixes the v0.9
                             discovery freeze).
  C. Information-gathering fallback — when no probe/plan is executable,
                             cycle through interaction targets (plates /
                             switches) instead of drifting randomly.

Public-physics planning (BFS, budgets, protected atoms, causal credit) is
reused unchanged from v0.9 to keep planner costs identical.
"""
from __future__ import annotations

import random
from collections import deque

from lab_env import (
    ACTIONS, DOORS, DOOR_CELL, GOAL, MOVE_ACTIONS, PLATES, PLATE_CELL,
    SWITCH_CELL, SWITCHES, WALLS, WS, make_state, public_next,
)
from metargon_v09 import (
    ALL_PAIRS, ALL_SINGLES, MAX_EXPANSIONS, PROBE_MAX_PLANS, atoms_of_state,
    plan_to_atoms, plan_to_pos, state_key, _violates_protected,
)
from robust_belief import RobustBelief
from compositional_memory_v10 import CompositionalMemoryV10, atoms_from_belief, opens_key

# pre-registered budgets (identical to v0.9 where they exist)
PROBE_COOLDOWN = 40          # steps before a failed probe may be retried
PROBE_TIMEOUT = 10           # steps before an executed probe is judged failed
EXPLORE_TARGETS = ("P1", "P2", "S1", "S2")


def _is_closed_door(cell, belief) -> bool:
    for door, dc in DOOR_CELL.items():
        if dc == cell and not belief.door_open[door]:
            return True
    return False


class MetargonV10:
    name = "metargon"

    def __init__(self, seed: int = 0, max_hypotheses: int = 128,
                 use_robust_belief: bool = True,
                 use_cooldowns: bool = True,
                 use_explore_fallback: bool = True,
                 use_replay: bool = True):
        self.seed = seed
        self.rng = random.Random(seed)
        self.memory = CompositionalMemoryV10(max_hypotheses=max_hypotheses,
                                            seed=seed,
                                            enable_replay=use_replay)
        self.belief = RobustBelief(seed)
        self.probed = {d: {} for d in DOORS}      # ante -> last probe step
        self.tried = {d: set() for d in DOORS}    # v0.9-style permanent set (abl. B off)
        self.planner_ops = 0
        self.plan_calls = 0
        self.explore_actions = 0
        self.hyp_probe_actions = 0
        self.steps = 0
        self.last_obs = None
        self._comp_cache = (None, None)
        self._goal_infeasible = {}
        self._step_since_goal_check = 0
        self._passed = {"D1": False, "D2": False}
        self._far_anchor = {"D1": (4, 1), "D2": (6, 2)}
        self._start_anchor = {"D1": (0, 4), "D2": (4, 1)}
        self._far_x = {"D1": 3, "D2": 5}
        self._pending_probe = {}
        self._plan_cache = {}
        self._current_plan = []
        self._recent_protect = {}
        self._explore_idx = 0
        self._first_episode = True
        # ablation switches
        self.use_robust_belief = use_robust_belief
        self.use_cooldowns = use_cooldowns
        self.use_explore_fallback = use_explore_fallback

    # -- perception ------------------------------------------------------
    def _sync_raw(self, obs):
        """v0.9-exact raw sync (used only when ablated)."""
        b = self.belief.belief
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
        self.belief.belief = b

    def _observe_public(self, obs):
        if self.use_robust_belief:
            return self.belief.observe(obs)
        self._sync_raw(obs)
        return self.belief.belief

    # -- planner (reused from v0.9, identical budgets) -------------------
    def _protected_atoms(self) -> frozenset:
        out = set()
        b = self.belief.belief
        now = atoms_of_state(b)
        for door in DOORS:
            h = self.memory.best_for_door(door, opens=True)
            if h is not None and h.antecedent <= now:
                out.update(h.antecedent)
            # an open door we have not crossed has a load-bearing state
            # right now: protect it so exploration cannot destroy it
            if b.door_open[door] and not self._passed[door]:
                out.update(now)
        for door, (atoms, expiry) in self._recent_protect.items():
            if self.memory.step <= expiry:
                out.update(atoms)
        return frozenset(out)

    def _conflicts_with_protected(self, ante: frozenset) -> bool:
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

    def _probe_eligible(self, door, ante) -> bool:
        if not self.use_cooldowns:
            return ante not in self.tried[door]
        last = self.probed[door].get(ante)
        return last is None or (self.memory.step - last) >= PROBE_COOLDOWN

    def _mark_probe_failed(self, door, ante):
        if self.use_cooldowns:
            self.probed[door][ante] = self.memory.step
        else:
            self.tried[door].add(ante)

    def _novel_singles(self, door):
        return [s for s in ALL_SINGLES if self._probe_eligible(door, s)]

    def _conjunctive_probes(self, door):
        """Pair probes built from atoms that are either currently true or
        whose members were already probed (cooldown-aware)."""
        pool = set()
        if self.use_cooldowns:
            for ante, last in self.probed[door].items():
                if (self.memory.step - last) < PROBE_COOLDOWN:
                    pool.update(ante)
            pending = self._pending_probe.get(door)
            if pending is not None and pending[0] is not None:
                pool.update(pending[0])
        else:
            for ante in self.tried[door]:
                pool.update(ante)
        pool.update(atoms_of_state(self.belief.belief))
        out = []
        for p in ALL_PAIRS:
            if not self._probe_eligible(door, p):
                continue
            if p <= frozenset(pool):
                out.append(p)
        return out

    def _components(self):
        b = self.belief.belief
        pattern = (b.door_open["D1"], b.door_open["D2"])
        if self._comp_cache[0] == pattern:
            return self._comp_cache[1], pattern
        comp = {}
        cid = 0
        walkable = [(x, y) for x in range(7) for y in range(5)
                    if (x, y) not in WALLS and not _is_closed_door((x, y), b)]
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
        comp, pattern = self._components()
        b = self.belief.belief
        agent = b.pos
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
                    oc = b.objs[obj]
                    if oc not in comp or comp[oc] != comp[pc] or comp[oc] != c_agent:
                        return False
                elif comp[pc] != c_agent:
                    return False
            elif k == "sw":
                sc = SWITCH_CELL[atom[1]]
                if sc not in comp or comp[sc] != c_agent:
                    return False
        return True

    # -- probe selection --------------------------------------------------
    def _choose_probe(self):
        def on_expand():
            self.planner_ops += 1
        b = self.belief.belief
        pattern = (b.door_open["D1"], b.door_open["D2"])
        protected = self._protected_atoms()
        # 0. pass through an open door we have not yet crossed
        for door in DOORS:
            if b.door_open[door] and not self._passed[door]:
                plan = plan_to_pos(b, self._far_anchor[door],
                                   on_expand=on_expand, cache=self._plan_cache,
                                   protected=protected)
                if plan is not None:
                    return plan, door
        # 1. if the goal is reachable under the current belief, go
        self._step_since_goal_check += 1
        if (pattern not in self._goal_infeasible
                or self._step_since_goal_check >= 8
                or b.pos == GOAL):
            plan = plan_to_pos(b, GOAL, on_expand=on_expand,
                               cache=self._plan_cache, protected=protected)
            self._step_since_goal_check = 0
            if plan is not None:
                return plan, None
            self._goal_infeasible[pattern] = True
        # 2. candidate probes
        candidates = []
        best_ante = {}
        now_atoms = atoms_of_state(b)
        for door in DOORS:
            if b.door_open[door]:
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
                    (abs(b.pos[0] - cell[0]) + abs(b.pos[1] - cell[1]))
                    for atom in ante
                    for cell in [PLATE_CELL[atom[1]] if atom[0] in ("load", "loadany")
                                 else SWITCH_CELL[atom[1]]]
                )
                return (needs, d)
            probes.sort(key=rank_key)
            for s in probes[:PROBE_MAX_PLANS]:
                candidates.append((door, s, 1, 0.0))
        # cross-door conjunctive probe (keep both doors satisfiable at once)
        if len(best_ante) >= 2:
            doors_c = [d for d in DOORS if d in best_ante]
            merged = frozenset().union(*(best_ante[d] for d in doors_c))
            if self._probe_eligible(doors_c[0], merged) and \
               self._probe_eligible(doors_c[1], merged):
                candidates.append((doors_c[0], merged, 0, 1.0))
        # 3. evaluate
        best = None
        evaluated = 0
        for door, ante, prio, conf in candidates:
            if door in self._pending_probe and self._pending_probe[door][0] == ante:
                continue  # already under evaluation; do not re-select
            if prio == 1 and ante <= now_atoms:
                # already arranged and it did not produce the effect:
                # record the failed trial (cooldown) without executing.
                self._mark_probe_failed(door, ante)
                continue
            if not self._target_feasible(ante):
                continue
            plan = plan_to_atoms(b, ante, on_expand=on_expand,
                                 cache=self._plan_cache, protected=protected)
            self.plan_calls += 1
            evaluated += 1
            if plan:
                ck = (prio, -conf)
                if (best is None or ck < best[3]
                        or (ck == best[3] and len(plan) < len(best[0]))):
                    best = (plan, door, ante, ck, prio)
            if evaluated >= PROBE_MAX_PLANS + 2:
                break
        if best is not None:
            plan, door, ante, _, prio = best
            if door not in self._pending_probe:
                self._pending_probe[door] = (ante, self.memory.step
                                             + len(plan) + PROBE_TIMEOUT)
            if prio == 0:
                self.hyp_probe_actions += 1
            return plan, door
        return None, None

    # -- fallback exploration --------------------------------------------
    def _random_valid(self):
        valid = [a for a in ACTIONS if public_next(self.belief.belief, a) is not None]
        moves = [a for a in valid if a in MOVE_ACTIONS or a.startswith("Push")]
        if moves:
            return self.rng.choice(moves)
        if not valid:
            return self.rng.choice(ACTIONS)
        return self.rng.choice(valid)

    def _random_move_safe(self, protected):
        valid = [a for a in ACTIONS if public_next(self.belief.belief, a) is not None]
        moves = [a for a in valid if a in MOVE_ACTIONS or a.startswith("Push")]
        safe = []
        for a in moves:
            nxt = public_next(self.belief.belief, a)
            if nxt is not None and not _violates_protected(nxt, protected, frozenset()):
                safe.append(a)
        if safe:
            return self.rng.choice(safe)
        if (self.rng.random() < 0.25
                and not self._conflicts_with_protected(
                    frozenset({("sw", "S1", True)}))
                and not self._conflicts_with_protected(
                    frozenset({("sw", "S2", True)}))):
            return "Toggle"
        return moves[0] if moves else valid[0]

    def _action_at_target(self, target, protected):
        b = self.belief.belief
        if target in ("S1", "S2"):
            ante = frozenset({("sw", target, not b.switch_state[target])})
            if not self._conflicts_with_protected(ante):
                return "Toggle"
            return self._random_move_safe(protected)
        # plate: push any object onto it
        ante = frozenset({("loadany", target)})
        plan = plan_to_atoms(b, ante, on_expand=lambda: setattr(self, "planner_ops", self.planner_ops + 1),
                             cache=self._plan_cache, protected=protected)
        if plan:
            return plan[0]
        return self._random_move_safe(protected)

    def _explore_action(self):
        if not self.use_explore_fallback:
            return self._random_valid()
        b = self.belief.belief
        protected = self._protected_atoms()
        targets = EXPLORE_TARGETS
        for _ in range(len(targets)):
            t = targets[self._explore_idx % len(targets)]
            cell = PLATE_CELL[t] if t in ("P1", "P2") else SWITCH_CELL[t]
            plan = plan_to_pos(b, cell, on_expand=lambda: setattr(self, "planner_ops", self.planner_ops + 1),
                               cache=self._plan_cache, protected=protected)
            if plan is not None:
                self._explore_idx = (self._explore_idx + 1) % len(targets)
                if plan:
                    return plan[0]
                return self._action_at_target(t, protected)
            self._explore_idx = (self._explore_idx + 1) % len(targets)
        return self._random_move_safe(protected)

    # -- event context cleaning ------------------------------------------
    @staticmethod
    def _clean_ctx(pre, post) -> frozenset:
        """Union of pre/post event atoms, minus post atoms that contradict a
        pre atom on the same sensor (e.g. S2=True vs S2=False when the event
        itself flipped the switch). Protecting both polarities would make
        every plan 'violate' a protected atom (v0.9 bug, now fixed)."""
        out = set(pre)
        for a in post:
            k = a[0]
            conflict = False
            if k == "sw":
                conflict = any(x[0] == "sw" and x[1] == a[1] and x[2] != a[2]
                               for x in pre)
            elif k == "load":
                conflict = any(x[0] == "load" and x[1] == a[1] and x[2] != a[2]
                               for x in pre)
            if not conflict:
                out.add(a)
        return frozenset(out)

    # -- interface -------------------------------------------------------
    def episode_start(self):
        if not self._first_episode:
            self.memory.episode_end()
        self._first_episode = False
        self.belief = RobustBelief(self.seed)
        self.last_obs = None
        self._current_plan = []
        self._passed = {"D1": False, "D2": False}
        self._goal_infeasible.clear()
        self._pending_probe.clear()
        self._recent_protect.clear()
        self._comp_cache = (None, None)

    def act(self, obs) -> str:
        b = self._observe_public(obs)
        # mark doors passed when the agent is physically on the far side
        for door in DOORS:
            if b.door_open[door] and b.pos[0] > self._far_x[door]:
                self._passed[door] = True
        protected = self._protected_atoms()
        while self._current_plan:
            a = self._current_plan[0]
            nxt = public_next(b, a)
            if nxt is not None and not _violates_protected(
                    nxt, protected, frozenset()):
                self._current_plan.pop(0)
                return a
            self._current_plan.pop(0)
        plan, door = self._choose_probe()
        if plan:
            self._current_plan = list(plan)
            return self._current_plan.pop(0)
        self.explore_actions += 1
        return self._explore_action()

    def observe_outcome(self, obs_before, action, res):
        self.steps += 1
        if self.use_robust_belief:
            pre_atoms = atoms_from_belief(self.belief.belief)
            prev_doors = dict(self.belief.belief.door_open)
            self.belief.observe(res.observation)
            self.belief.apply_transition(action, res.valid, res.observation)
            post_atoms = atoms_from_belief(self.belief.belief)
            now_doors = self.belief.belief.door_open
        else:
            pre_atoms = atoms_from_belief(self.belief.belief)
            prev_doors = dict(self.belief.belief.door_open)
            self._sync_raw(res.observation)
            post_atoms = atoms_from_belief(self.belief.belief)
            now_doors = self.belief.belief.door_open
        any_event = False
        for door in DOORS:
            before_open = prev_doors[door]
            after_open = now_doors[door]
            if not before_open and after_open:
                # generate candidates from both contexts (the cause may
                # hold before the transition or appear during it)
                self.memory.observe_event(pre_atoms, door, True)
                self.memory.observe_event(post_atoms, door, True)
                self._passed[door] = False
                self._pending_probe.pop(door, None)
                # protect the causal pre-state (and any currently load-
                # bearing atoms via _protected_atoms) for a short window
                self._recent_protect[door] = (pre_atoms, self.memory.step + 12)
                if self.use_cooldowns:
                    self.probed[door].clear()
                else:
                    self.tried[door].clear()
                any_event = True
            elif before_open and not after_open:
                self.memory.observe_event(pre_atoms, door, False)
                self.memory.observe_event(post_atoms, door, False)
                if self.use_cooldowns:
                    self.probed[door].clear()
                else:
                    self.tried[door].clear()
                any_event = True
        if any_event:
            self._goal_infeasible.clear()
            self._current_plan = []
        for door in DOORS:
            after_open = now_doors[door]
            if not after_open:
                self.memory.observe_nonevent(post_atoms, door, True)
            else:
                self.memory.observe_nonevent(post_atoms, door, False)
        self.memory.step_advance()
        for door, (ante, deadline) in list(self._pending_probe.items()):
            if self.memory.step >= deadline:
                self._mark_probe_failed(door, ante)
                del self._pending_probe[door]

    def diagnostics(self) -> dict:
        return {
            "memory": self.memory.diagnostics(),
            "belief": self.belief.diagnostics(),
            "planner_ops": self.planner_ops,
            "plan_calls": self.plan_calls,
            "explore_actions": self.explore_actions,
            "hyp_probe_actions": self.hyp_probe_actions,
            "steps": self.steps,
            "distinct_probed": sum(len(s) for s in self.probed.values()),
            "probed_D1": len(self.probed["D1"]),
            "probed_D2": len(self.probed["D2"]),
        }
