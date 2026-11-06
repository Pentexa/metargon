"""METARGON Academy — the student agent (full configuration).

Combines:
  * a belief state maintained from the public observation stream,
  * a bounded compositional hypothesis memory (persistent across episodes),
  * a public-physics planner that never accesses hidden rules,
  * an explicit "passed door" tracker and fallback random exploration,
  * optional offline reinforcement from replayed experiences.

The agent's persistent learning state (memory, tried probes, counters,
RNG) can be serialized and restored, which is what makes training
resumable and checkpointable.
"""

from __future__ import annotations

import random
from collections import deque

from .common import atoms_from_observation, belief_from_obs, valid_actions
from .memory import CompositionalMemory, consequence_key
from .planner import (
    atoms_of_state, plan_to_atoms, plan_to_pos, target_feasible,
)

MAX_EXPANSIONS = 1500
PROBE_MAX_PLANS = 4


def _all_singles(obs_atoms: frozenset) -> list:
    """Singleton antecedents from the observed atom vocabulary (bounded,
    deterministically ordered)."""
    return [frozenset({a}) for a in sorted(obs_atoms, key=str)]


def _all_pairs(obs_atoms: frozenset, tried: set) -> list:
    """Conjunctive pairs built from atoms whose singletons were already
    tried (bounded combinatorial probe space)."""
    base = sorted(obs_atoms, key=str)
    out = []
    for i in range(len(base)):
        for j in range(i + 1, len(base)):
            p = frozenset({base[i], base[j]})
            if p not in tried:
                out.append(p)
    return out


class MetargonAcademyAgent:
    """Full METARGON Academy configuration (with episodic replay)."""

    name = "metargon_full"

    def __init__(self, seed: int = 0,
                 replay_enabled: bool = True,
                 max_hypotheses: int = 128,
                 verify_support: int = 3,
                 max_contradict: int = 2,
                 predict_threshold: int = 1,
                 contradictict_window: int = 3,
                 candidates_per_event: int = 10,
                 pair_base: int = 5,
                 planner_max_expand: int = MAX_EXPANSIONS,
                 probe_max_plans: int = PROBE_MAX_PLANS,
                 probe_timeout: int = 10):
        self.seed = seed
        self.rng = random.Random(seed)
        self.replay_enabled = replay_enabled
        self.memory = CompositionalMemory(
            max_hypotheses=max_hypotheses,
            verify_support=verify_support,
            max_contradict=max_contradict,
            predict_threshold=predict_threshold,
            contradictict_window=contradict_window,
            candidates_per_event=candidates_per_event,
            pair_base=pair_base,
        )
        self.planner_max_expand = planner_max_expand
        self.probe_max_plans = probe_max_plans
        self.probe_timeout = probe_timeout
        self.geom = None
        self.belief = None
        self.tried = {}
        self.far_anchor = {}
        self.passed = {}
        self.pending_probe = {}
        self.planner_ops = 0
        self.plan_calls = 0
        self.steps = 0
        self.episodes = 0
        self.successes = 0
        self.last_obs = None
        self._comp_cache = (None, None)
        self._goal_infeasible = {}
        self._steps_since_goal_check = 0

    # -- perception ------------------------------------------------------
    def episode_start(self, spec, geom):
        self.geom = geom
        self.belief = None
        self.last_obs = None
        self.pending_probe = {}
        self._comp_cache = (None, None)
        self._goal_infeasible = {}
        self._steps_since_goal_check = 0
        # derive door anchors once per topology
        if not self.far_anchor:
            self._derive_anchors(geom)
        for door in geom.door_cell:
            self.passed[door] = False

    def _derive_anchors(self, geom):
        from collections import deque
        start = geom.start
        for door, dc in geom.door_cell.items():
            # neighbours of the door cell that are not walls
            nbrs = []
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                n = (dc[0] + dx, dc[1] + dy)
                if 0 <= n[0] < 7 and 0 <= n[1] < 5 and n not in geom.walls:
                    nbrs.append(n)
            # reachable-from-start side, ignoring closed doors
            walkable = {(x, y) for x in range(7) for y in range(5)
                        if (x, y) not in geom.walls}
            seen = {start}
            q = deque([start])
            while q:
                cur = q.popleft()
                for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    n = (cur[0] + dx, cur[1] + dy)
                    if n in walkable and n not in seen:
                        # treat door cells as walls while exploring start side
                        if n in geom.door_cell.values():
                            continue
                        seen.add(n)
                        q.append(n)
            far = [n for n in nbrs if n not in seen]
            self.far_anchor[door] = far[0] if far else nbrs[-1] if nbrs else dc

    def _sync(self, obs):
        from .common import belief_from_obs
        self.belief = belief_from_obs(obs, self.geom)
        # mark doors passed when we are on their far side
        comp, _ = self._components()
        for door in self.geom.door_cell:
            if not self.belief.door_open[door]:
                continue
            far = self.far_anchor.get(door)
            if far is not None and far in comp and comp.get(self.belief.pos) == comp[far]:
                self.passed[door] = True

    # -- planner ---------------------------------------------------------
    def _components(self):
        from .planner import components
        pattern = (self.belief.door_open["D1"], self.belief.door_open["D2"])
        if self._comp_cache[0] == pattern:
            return self._comp_cache[1], pattern
        comp = components(self.belief, self.geom)
        self._comp_cache = (pattern, comp)
        return comp, pattern

    def _novel_singles(self, door):
        atoms = atoms_of_state(self.belief)
        return [s for s in _all_singles(atoms) if s not in self.tried[door]]

    def _conjunctive_probes(self, door):
        tried = self.tried[door]
        atoms = atoms_of_state(self.belief)
        return [p for p in _all_pairs(atoms, tried) if p not in tried]

    def _choose_probe(self):
        def on_expand():
            self.planner_ops += 1
        pattern = (self.belief.door_open["D1"], self.belief.door_open["D2"])
        # 0. if a door is open and not yet passed, go through it
        for door in self.geom.door_cell:
            if self.belief.door_open[door] and not self.passed[door]:
                plan = plan_to_pos(self.belief, self.far_anchor[door],
                                   self.geom, max_expand=self.planner_max_expand,
                                   on_expand=on_expand)
                if plan is not None:
                    return plan, door
        # 1. goal reachable under belief?
        self._steps_since_goal_check += 1
        if (pattern not in self._goal_infeasible
                or self._steps_since_goal_check >= 8
                or self.belief.pos == self.geom.goal):
            plan = plan_to_pos(self.belief, self.geom.goal,
                               self.geom, max_expand=self.planner_max_expand,
                               on_expand=on_expand)
            self._steps_since_goal_check = 0
            if plan is not None:
                return plan, None
            self._goal_infeasible[pattern] = True
        # 2. candidates: hypotheses first, then novel singles, then pairs
        candidates = []
        for door in self.geom.door_cell:
            if self.belief.door_open[door]:
                continue
            hyps = self.memory.query(door, opens=True)
            for h in hyps[:3]:
                candidates.append((door, h.antecedent, 0))
            singles = self._novel_singles(door)
            pairs = self._conjunctive_probes(door)
            for s in (singles + pairs)[:self.probe_max_plans]:
                candidates.append((door, s, 1))
        best = None
        evaluated = 0
        for door, ante, prio in candidates:
            if ante in self.tried[door]:
                continue
            if not target_feasible(self.belief, ante, self.geom):
                self.tried[door].add(ante)
                continue
            plan = plan_to_atoms(self.belief, ante, self.geom,
                                 max_expand=self.planner_max_expand,
                                 on_expand=on_expand)
            self.plan_calls += 1
            evaluated += 1
            if plan:
                if best is None or len(plan) < len(best[0]):
                    best = (plan, door, ante)
            if evaluated >= self.probe_max_plans:
                break
        if best is not None:
            plan, door, ante = best
            self.pending_probe[door] = (ante, self.memory.step + self.probe_timeout)
            return plan, door
        return None, None

    # -- interface -------------------------------------------------------
    def act(self, obs) -> str:
        self._sync(obs)
        plan, _ = self._choose_probe()
        if plan:
            return plan[0]
        valid = valid_actions(self.belief, self.geom)
        return self.rng.choice(valid if valid else ("N",))

    def learn(self, obs_before, action, result):
        self.steps += 1
        before_atoms = atoms_from_observation(obs_before)
        after_obs = result.observation
        after_atoms = atoms_from_observation(after_obs)
        # door events from the public observation stream
        for door in self.geom.door_cell:
            before_open = getattr(obs_before, "d1" if door == "D1" else "d2")
            after_open = getattr(after_obs, "d1" if door == "D1" else "d2")
            if not before_open and after_open:
                self.memory.observe_event(before_atoms, door, True)
                self.tried[door] = set()
                self.passed[door] = False
                self.pending_probe.pop(door, None)
            elif before_open and not after_open:
                self.memory.observe_event(before_atoms, door, False)
                self.tried[door] = set()
        # nonevents for doors that did not change as expected
        for door in self.geom.door_cell:
            after_open = getattr(after_obs, "d1" if door == "D1" else "d2")
            if not after_open:
                self.memory.observe_nonevent(after_atoms, door, True)
            else:
                self.memory.observe_nonevent(after_atoms, door, False)
        self.memory.step_advance()
        for door, (ante, deadline) in list(self.pending_probe.items()):
            if self.memory.step >= deadline:
                self.tried[door].add(ante)
                del self.pending_probe[door]

    def learn_from_experience(self, exp):
        """Offline reinforcement from replayed evidence: strengthens existing
        hypotheses; does not generate new candidates (the original online
        pass already did)."""
        if not self.replay_enabled:
            return
        before_atoms = atoms_from_observation(exp.obs_before)
        after_atoms = atoms_from_observation(exp.obs_after)
        for door, event in exp.events:
            opened = event == "open"
            consequence = consequence_key(door, opened)
            for h in self.memory.hypotheses:
                if h.consequence == consequence and h.antecedent <= before_atoms and h.usable:
                    self.memory._support(h)
        # nonevent reinforcement
        for door in exp.obs_after.door_open:
            wants_open = not exp.obs_after.door_open[door]
            consequence = consequence_key(door, wants_open)
            for h in self.memory.hypotheses:
                if (h.door == door and h.consequence == consequence
                        and h.usable and h.antecedent <= after_atoms):
                    self.memory._contradict(h)

    def finish_episode(self, success: bool):
        self.episodes += 1
        if success:
            self.successes += 1

    # -- checkpointing ---------------------------------------------------
    def save_state(self) -> dict:
        return {
            "seed": self.seed,
            "rng": self.rng.getstate(),
            "replay_enabled": self.replay_enabled,
            "memory": self.memory.save_state(),
            "planner_max_expand": self.planner_max_expand,
            "probe_max_plans": self.probe_max_plans,
            "probe_timeout": self.probe_timeout,
            "tried": {d: sorted(t) for d, t in self.tried.items()},
            "passed": dict(self.passed),
            "far_anchor": dict(self.far_anchor),
            "pending_probe": {d: list(v) for d, v in self.pending_probe.items()},
            "planner_ops": self.planner_ops,
            "plan_calls": self.plan_calls,
            "steps": self.steps,
            "episodes": self.episodes,
            "successes": self.successes,
            "goal_infeasible": {str(k): v for k, v in self._goal_infeasible.items()},
        }

    def load_state(self, state: dict) -> None:
        self.seed = state["seed"]
        self.rng.setstate(tuple(state["rng"]))
        self.replay_enabled = state["replay_enabled"]
        self.memory.load_state(state["memory"])
        self.planner_max_expand = state["planner_max_expand"]
        self.probe_max_plans = state["probe_max_plans"]
        self.probe_timeout = state["probe_timeout"]
        self.tried = {d: set(map(tuple, t)) for d, t in state["tried"].items()}
        self.passed = dict(state["passed"])
        self.far_anchor = {d: tuple(a) for d, a in state["far_anchor"].items()}
        self.pending_probe = {d: (tuple(a), b) for d, (a, b) in state["pending_probe"].items()}
        self.planner_ops = state["planner_ops"]
        self.plan_calls = state["plan_calls"]
        self.steps = state["steps"]
        self.episodes = state["episodes"]
        self.successes = state["successes"]
        self._goal_infeasible = {eval(k): v for k, v in state["goal_infeasible"].items()}

    def diagnostics(self) -> dict:
        return {
            "memory": self.memory.diagnostics(),
            "planner_ops": self.planner_ops,
            "plan_calls": self.plan_calls,
            "steps": self.steps,
            "episodes": self.episodes,
            "successes": self.successes,
            "replay_enabled": self.replay_enabled,
        }

