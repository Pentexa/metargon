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
from typing import Optional

from .common import atoms_from_observation, belief_from_obs, state_key, valid_actions
from .memory import CompositionalMemory, consequence_key
from .planner import (
    atoms_of_state, plan_to_atoms, plan_to_pos, target_feasible,
)
from ..envs.gridworld import public_next

MAX_EXPANSIONS = 1500
PROBE_MAX_PLANS = 4


def _vocabulary_singles() -> list:
    """Fixed probe vocabulary over the public atoms (both polarities for
    switches, all load/absence combos for plates). The agent can propose
    to *set* any of these atoms true, not only the ones currently true."""
    from ..envs.gridworld import PLATES, SWITCHES, OBJECTS
    out = []
    for s in SWITCHES:
        out.append(frozenset({("sw", s, True)}))
        out.append(frozenset({("sw", s, False)}))
    for p in PLATES:
        out.append(frozenset({("empty", p)}))
        out.append(frozenset({("loadany", p)}))
        for o in OBJECTS:
            out.append(frozenset({("load", p, o)}))
    return out


VOCAB_SINGLES = _vocabulary_singles()


def _vocabulary_pairs(tried: set) -> list:
    """Conjunctive pairs over the fixed vocabulary (bounded, deterministic
    order)."""
    out = []
    base = VOCAB_SINGLES
    for i in range(len(base)):
        for j in range(i + 1, len(base)):
            p = frozenset(base[i] | base[j])
            if p not in tried:
                out.append(p)
            if len(out) >= 64:
                return out
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
                 probe_timeout: int = 10,
                 memory_recycle_contradicted: bool = False,
                 memory_contradiction_ratio: float = 0.0,
                 memory_evidence_window: int = 0,
                 probe_adaptive: bool = False,
                 protect_episode_local: bool = False,
                 probe_episode_local: bool = False,
                 memory_context_scoped: bool = False,
                 memory_task_change_reset: bool = False,
                 memory_context_gate_transfer: bool = False,
                 memory_evidence_bank_cap: int = 0,
                 memory_atom_transfer: bool = False,
                 memory_bank_refresh_on_contradiction: bool = False,
                 probe_reset_per_episode: bool = False,
                 probe_min_effective_confidence: float = 0.0,
                 probe_bank_reuse: bool = False,
                 memory_episode_scoped: bool = False,
                 memory_min_verified_epochs_for_transfer: int = 2,
                 memory_factory=None):
        self.seed = seed
        self.rng = random.Random(seed)
        self.replay_enabled = replay_enabled
        _memory_cls = memory_factory or CompositionalMemory
        self.memory = _memory_cls(
            max_hypotheses=max_hypotheses,
            verify_support=verify_support,
            max_contradict=max_contradict,
            predict_threshold=predict_threshold,
            contradictict_window=contradictict_window,
            candidates_per_event=candidates_per_event,
            pair_base=pair_base,
            recycle_contradicted=memory_recycle_contradicted,
            contradiction_ratio=memory_contradiction_ratio,
            evidence_window=memory_evidence_window,
            context_scoped=memory_context_scoped,
            task_change_reset=memory_task_change_reset,
            context_gate_transfer=memory_context_gate_transfer,
            evidence_bank_cap=memory_evidence_bank_cap,
            atom_transfer=memory_atom_transfer,
            bank_refresh_on_contradiction=memory_bank_refresh_on_contradiction,
            episode_scoped=memory_episode_scoped,
            min_verified_epochs_for_transfer=memory_min_verified_epochs_for_transfer,
        )
        self.memory_atom_transfer = bool(memory_atom_transfer)
        self.memory_bank_refresh_on_contradiction = bool(
            memory_bank_refresh_on_contradiction)
        self.memory_episode_scoped = bool(memory_episode_scoped)
        self.memory_min_verified_epochs_for_transfer = int(
            memory_min_verified_epochs_for_transfer)
        self.probe_reset_per_episode = bool(probe_reset_per_episode)
        self.probe_min_effective_confidence = float(
            probe_min_effective_confidence)
        self.probe_bank_reuse = bool(probe_bank_reuse)
        self.probe_adaptive = bool(probe_adaptive)
        # Diagnostic control: disable candidate probing entirely while
        # keeping door-pass and goal-planning steps (isolates the value
        # of vocabulary probing vs planning).
        self.probe_vocab_disabled = False
        self.protect_episode_local = bool(protect_episode_local)
        self.probe_episode_local = bool(probe_episode_local)
        self._episode_mem_step = 0
        self._ctx_signature = None
        self.frozen_policy = False
        self.planner_max_expand = planner_max_expand
        self.probe_max_plans = probe_max_plans
        self.probe_timeout = probe_timeout
        self.plan_cache = {}
        self.plan_cache_hits = 0
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
        self._collect_probe_log = False
        # v0.6 diagnostic trace (ephemeral; not part of persistent state)
        self.trace = {
            "source_hypothesis": 0, "source_bank": 0,
            "source_atom_single": 0, "source_atom_pair": 0,
            "source_novel_single": 0, "source_conjunctive_probe": 0,
            "source_other": 0,
            "skipped_tried": 0, "skipped_infeasible": 0,
            "probes_evaluated": 0, "plans_used": 0,
            "random_fallback_steps": 0,
            "door_open_after_probe": 0,
            "door_open_after_hypothesis": 0,
            "door_open_after_bank": 0,
            "door_open_after_atom_single": 0,
            "door_open_after_atom_pair": 0,
            "door_open_after_novel_single": 0,
            "door_open_after_conjunction": 0,
            "door_open_after_other": 0,
            "query_calls": 0, "query_useful": 0,
            "hypothesis_candidates": 0, "hypothesis_skipped_lowconf": 0,
            # v0.7 memory-to-action attribution
            "hypothesis_retrieved": 0,
            "cross_epoch_transfer_retrieved": 0,
            "memory_actions": 0,
            "memory_correct": 0,
            "memory_incorrect": 0,
        }

    # -- perception ------------------------------------------------------
    def episode_start(self, spec, geom):
        self.geom = geom
        self.belief = None
        self.last_obs = None
        self.pending_probe = {}
        self._comp_cache = (None, None)
        self._goal_infeasible = {}
        self._steps_since_goal_check = 0
        self._episode_mem_step = self.memory.step
        self._ctx_signature = None
        # v0.7: each episode is an evidence epoch (task boundary); the
        # memory resets consecutive-nonevent windows and scopes evidence
        # per epoch so cross-task contradictions do not destroy verified
        # hypotheses.
        self.memory.begin_episode(self.episodes)
        # derive door anchors once per topology
        if not self.far_anchor:
            self._derive_anchors(geom)
        if not self.tried:
            self.tried = {d: set() for d in geom.door_cell}
        if self.probe_reset_per_episode:
            # v0.6: probe bookkeeping is per-episode — a new task must
            # be able to re-explore the vocabulary (the tried set from a
            # previous task silently blocks reuse and exploration).
            self.tried = {d: set() for d in geom.door_cell}
            self.pending_probe = {}
            self._goal_infeasible = {}
            self.plan_cache = {}
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
        if self._ctx_signature is None:
            b = self.belief
            self._ctx_signature = (
                b.pos,
                tuple(b.switch_state[s] for s in ("S1", "S2")),
                tuple(b.plate_load[p] for p in ("P1", "P2")),
                b.objs["A"], b.objs["B"],
                tuple(b.door_open[d] for d in ("D1", "D2")),
            )
            self.memory.set_context(self._ctx_signature)
        # `passed` is updated in _choose_probe by physically standing on the
        # far cell of an open door (component-based labels are unreliable
        # once doors merge regions).

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
        return [s for s in VOCAB_SINGLES if s not in self.tried[door]]

    def _conjunctive_probes(self, door):
        tried = self.tried[door]
        return [p for p in _vocabulary_pairs(tried) if p not in tried]

    def _plan_cached(self, state, target, on_expand, to_pos=None):
        """Planning with a bounded cache keyed by state+target (BFS results
        are deterministic; repeated probes are common)."""
        key = (state_key(state), to_pos, tuple(sorted(target)))
        hit = self.plan_cache.get(key)
        if hit is not None:
            self.plan_cache_hits += 1
            return hit or None
        if to_pos is not None:
            plan = plan_to_pos(state, to_pos, self.geom,
                               max_expand=self.planner_max_expand,
                               on_expand=on_expand)
        else:
            plan = plan_to_atoms(state, target, self.geom,
                                 max_expand=self.planner_max_expand,
                                 on_expand=on_expand)
        if len(self.plan_cache) > 2000:
            for k in list(self.plan_cache)[:400]:
                del self.plan_cache[k]
        self.plan_cache[key] = plan or []
        return plan

    def _target_cells(self, target) -> list:
        """Cells relevant to a probe antecedent (switch/plate cells, object
        positions), used to bias preserving actions toward progress."""
        cells = []
        for atom in target:
            k = atom[0]
            if k == "sw":
                cells.append(self.geom.switch_cell[atom[1]])
            elif k in ("load", "loadany", "empty"):
                cells.append(self.geom.plate_cell[atom[1]])
                if k == "load" and atom[2] in self.belief.objs:
                    cells.append(self.belief.objs[atom[2]])
        return cells

    def _first_preserving_action(self, state, target) -> Optional[str]:
        """When a probe antecedent already holds, the door can open on the
        next transition. Return an action whose resulting public state
        still satisfies the antecedent (so the condition is not broken),
        preferring actions that make progress toward the relevant cells."""
        from .planner import atoms_of_state
        valid = valid_actions(state, self.geom)
        targets = self._target_cells(target)
        best = None
        best_d = None
        for a in valid:
            nxt = public_next(state, a, self.geom)
            if nxt is None or not (target <= atoms_of_state(nxt)):
                continue
            if not targets:
                return a
            d = min(abs(nxt.pos[0] - c[0]) + abs(nxt.pos[1] - c[1])
                    for c in targets)
            if best_d is None or d < best_d:
                best, best_d = a, d
        return best if best is not None else (valid[0] if valid else None)

    def _protected_atoms(self) -> frozenset:
        """Atoms believed to keep an open, *not-yet-passed* door open.
        Only verified hypotheses count (weak co-occurrence evidence is not
        enough to freeze exploration); after a door is crossed, breaking
        its condition is harmless, so protection stops."""
        from ..envs.gridworld import OBJECTS
        from .memory import VERIFIED
        out = set()
        for door in self.geom.door_cell:
            if not self.belief.door_open[door] or self.passed[door]:
                continue
            ref_switches = set()
            ref_plates = set()
            for h in self.memory.query(door):
                if h.status != VERIFIED:
                    continue
                if self.protect_episode_local and (
                        h.last_update < self._episode_mem_step):
                    continue
                for a in h.antecedent:
                    if a[0] == "sw":
                        ref_switches.add(a[1])
                    else:
                        ref_plates.add(a[1])
            for s in ref_switches:
                out.add(("sw", s, bool(self.belief.switch_state[s])))
            for p in ref_plates:
                ld = self.belief.plate_load[p]
                if ld in OBJECTS:
                    out.add(("load", p, ld))
                else:
                    out.add(("empty", p))
        return frozenset(out)

    def _plan_breaks_protection(self, plan, protected) -> bool:
        if not protected:
            return False
        st = self.belief
        for a in plan:
            nxt = public_next(st, a, self.geom)
            if nxt is None:
                return True
            st = nxt
        return not (protected <= atoms_of_state(st))

    def _adaptive_probe_candidates(self, door, current) -> list:
        """Adaptive probe candidate construction:
        * hypothesis-backed antecedents (verbatim reuse gated by
          context-conditioned confidence; unchanged priority, optionally
          restricted to hypotheses evidenced in the current episode);
        * v0.5 evidence-bank retrieval (high-priority reuse);
        * v0.6 atom-transfer candidates (causal structure matching —
          singles and compact conjunctions of cross-task-robust atoms);
        * conjunctions of the most-supported atoms (composite targets);
        * remaining singles ordered by support-weighted popularity.
        Candidates are tagged with a source for diagnostics and are
        bounded and deterministic (ordering by (count, str))."""
        hyps = self.memory.query(door, opens=True)
        if self.probe_episode_local:
            hyps = [h for h in hyps
                    if h.last_update >= self._episode_mem_step]
        out = []
        for h in hyps[:3]:
            eff_conf = self.memory.effective_confidence(h)
            if eff_conf < self.probe_min_effective_confidence:
                self.trace["hypothesis_skipped_lowconf"] += 1
                continue
            self.trace["hypothesis_candidates"] += 1
            prio = 0 if (len(h.antecedent) <= 2 and eff_conf >= 0.75) else 1
            out.append((door, h.antecedent, prio, "hypothesis"))
        # v0.5: evidence-bank retrieval (high-priority reuse)
        if self.memory.evidence_bank_cap > 0:
            for ante, cons in self.memory.bank_entries():
                if cons.split(":")[1] != door:
                    continue
                if ante not in self.tried[door]:
                    out.append((door, ante, 0, "bank"))
                    self.memory.mark_retrieval()
        tried = self.tried[door]
        # v0.6: atom transfer (causal structure matching across tasks)
        if self.memory_atom_transfer:
            atoms = self.memory.transfer_atoms(door, k=6)
            for a in atoms[:4]:
                p = frozenset({a})
                if p not in tried:
                    out.append((door, p, 1, "atom_single"))
            # pair co-occurrence from verified hypotheses (conjunctive
            # structure transfer) — ordered before single probes.
            for p in self.memory.transfer_pairs(door, k=4):
                if p not in tried:
                    out.append((door, p, 1, "atom_pair"))
        atom_count = {}
        for h in hyps[:8]:
            for a in h.antecedent:
                atom_count[a] = atom_count.get(a, 0) + 1
        popular = sorted(atom_count, key=lambda a: (-atom_count[a], str(a)))
        comps = []
        for i in range(len(popular)):
            for j in range(i + 1, len(popular)):
                p = frozenset({popular[i], popular[j]})
                if p not in tried:
                    comps.append(p)
                if len(comps) >= 8:
                    break
            if len(comps) >= 8:
                break
        for p in comps[:4]:
            out.append((door, p, 1, "conjunction"))
        singles = [frozenset({a}) for a in popular]
        singles = [s for s in singles if s not in tried]
        for s in singles[:self.probe_max_plans]:
            out.append((door, s, 1, "novel_single"))
        return out

    def _choose_probe(self):
        def on_expand():
            self.planner_ops += 1
        pattern = (self.belief.door_open["D1"], self.belief.door_open["D2"])
        # 0. if a door is open and not yet passed, go through it
        for door in self.geom.door_cell:
            if self.belief.door_open[door] and not self.passed[door]:
                if self.belief.pos == self.far_anchor[door]:
                    self.passed[door] = True
                    continue
                plan = self._plan_cached(self.belief, (), on_expand,
                                         to_pos=self.far_anchor[door])
                if plan is not None:
                    return plan, door
        # 1. goal reachable under belief?
        self._steps_since_goal_check += 1
        if (pattern not in self._goal_infeasible
                or self._steps_since_goal_check >= 8
                or self.belief.pos == self.geom.goal):
            plan = self._plan_cached(self.belief, (), on_expand,
                                     to_pos=self.geom.goal)
            self._steps_since_goal_check = 0
            if plan is not None:
                return plan, None
            self._goal_infeasible[pattern] = True
        if self.probe_vocab_disabled:
            return None, None
        # 2. candidates grouped by priority:
        #    prio 0: hypothesis-backed antecedents (strongest information)
        #    prio 1: probes that require changing the world (novel info)
        #    prio 2: probes already satisfied (weakest; fallback)
        current = atoms_of_state(self.belief)
        candidates = []
        for door in self.geom.door_cell:
            if self.belief.door_open[door]:
                continue
            if self.probe_adaptive:
                candidates.extend(self._adaptive_probe_candidates(door, current))
                continue
            hyps = self.memory.query(door, opens=True)
            self.trace["query_calls"] += 1
            if hyps:
                self.trace["query_useful"] += 1
            self.trace["hypothesis_retrieved"] += len(hyps)
            if self.memory.episode_scoped:
                self.trace["cross_epoch_transfer_retrieved"] += sum(
                    1 for h in hyps
                    if h.verified_epoch_count
                    >= self.memory.min_verified_epochs_for_transfer)
            if self.probe_episode_local:
                hyps = [h for h in hyps
                        if h.last_update >= self._episode_mem_step]
            for h in hyps[:3]:
                eff_conf = self.memory.effective_confidence(h)
                if eff_conf < self.probe_min_effective_confidence:
                    self.trace["hypothesis_skipped_lowconf"] += 1
                    continue
                self.trace["hypothesis_candidates"] += 1
                prio = 0 if (len(h.antecedent) <= 2
                             and eff_conf >= 0.75) else 1
                candidates.append((door, h.antecedent, prio, "hypothesis"))
            if self.probe_bank_reuse and self.memory.evidence_bank_cap > 0:
                for ante, cons in self.memory.bank_entries():
                    if cons.split(":")[1] != door:
                        continue
                    if ante not in self.tried[door]:
                        candidates.append((door, ante, 0, "bank"))
                        self.memory.mark_retrieval()
            if self.memory_atom_transfer:
                atoms = self.memory.transfer_atoms(door, k=6)
                for a in atoms[:4]:
                    p = frozenset({a})
                    if p not in self.tried[door]:
                        candidates.append((door, p, 1, "atom_single"))
                for p in self.memory.transfer_pairs(door, k=4):
                    if p not in self.tried[door]:
                        candidates.append((door, p, 1, "atom_pair"))
            probes = self._novel_singles(door) + self._conjunctive_probes(door)
            need_change = [s for s in probes if not (s <= current)]
            already_true = [s for s in probes if s <= current]
            for s in need_change[:self.probe_max_plans]:
                candidates.append((door, s, 1, "novel_single"))
            for s in already_true[:2]:
                candidates.append((door, s, 2, "other"))
        candidates.sort(key=lambda x: x[2])
        protected = self._protected_atoms()
        best = None
        evaluated = 0
        for door, ante, prio, src in candidates:
            if ante in self.tried[door]:
                self.trace["skipped_tried"] += 1
                continue
            if not target_feasible(self.belief, ante, self.geom):
                self.tried[door].add(ante)
                self.trace["skipped_infeasible"] += 1
                continue
            plan = self._plan_cached(self.belief, ante, on_expand)
            self.plan_calls += 1
            evaluated += 1
            if plan:
                if self._plan_breaks_protection(plan, protected):
                    continue
                if (best is None or prio < best[3]
                        or (prio == best[3] and len(plan) < len(best[0]))):
                    best = (plan, door, ante, prio, src)
            elif ante <= atoms_of_state(self.belief):
                # antecedent already true: any preserving action will trigger
                # the door transition; use it as the plan.
                pres = self._first_preserving_action(self.belief, ante)
                if pres is not None and not self._plan_breaks_protection([pres], protected):
                    if (best is None or prio < best[3]
                            or (prio == best[3] and len(best[0]) > 1)):
                        best = ([pres], door, ante, prio, src)
                else:
                    self.tried[door].add(ante)
            if evaluated >= self.probe_max_plans:
                break
        if best is not None:
            plan, door, ante, _, src = best
            key = f"source_{src}"
            if key not in self.trace:
                key = "source_other"
            self.trace[key] += 1
            self.trace["plans_used"] += 1
            if src == "hypothesis":
                self.trace["memory_actions"] += 1
            self.pending_probe[door] = (ante, self.memory.step + self.probe_timeout, src)
            if self._collect_probe_log:
                self.trace.setdefault("planned_probes", []).append(
                    [door, sorted(list(ante)), src])
            return plan, door
        return None, None

    # -- interface -------------------------------------------------------
    def act(self, obs) -> str:
        self._sync(obs)
        plan, _ = self._choose_probe()
        if plan:
            return plan[0]
        self.trace["random_fallback_steps"] += 1
        valid = valid_actions(self.belief, self.geom)
        return self.rng.choice(valid if valid else ("N",))

    def learn(self, obs_before, action, result):
        if self.frozen_policy:
            return
        self.steps += 1
        before_atoms = atoms_from_observation(obs_before)
        after_obs = result.observation
        after_atoms = atoms_from_observation(after_obs)
        # door events from the public observation stream
        event_happened = False
        for door in self.geom.door_cell:
            before_open = getattr(obs_before, "d1" if door == "D1" else "d2")
            after_open = getattr(after_obs, "d1" if door == "D1" else "d2")
            if not before_open and after_open:
                self.memory.observe_event(before_atoms, door, True, after_atoms)
                event_happened = True
                self.passed[door] = False
                if door in self.pending_probe:
                    _, _, src = self.pending_probe[door]
                    self.trace["door_open_after_probe"] += 1
                    self.trace[f"door_open_after_{src}"] += 1
                    if src == "hypothesis":
                        self.trace["memory_correct"] += 1
                self.pending_probe.pop(door, None)
            elif before_open and not after_open:
                self.memory.observe_event(before_atoms, door, False, after_atoms)
                event_happened = True
        if event_happened:
            # connectivity/rule information changed: probe feasibility may
            # have changed for every door, so reset all tried sets.
            for d in self.geom.door_cell:
                self.tried[d] = set()
        # nonevents for doors that did not change as expected
        for door in self.geom.door_cell:
            after_open = getattr(after_obs, "d1" if door == "D1" else "d2")
            if not after_open:
                self.memory.observe_nonevent(after_atoms, door, True)
            else:
                self.memory.observe_nonevent(after_atoms, door, False)
                self.memory.observe_maintain(after_atoms, door)
        self.memory.step_advance()
        for door, (ante, deadline, _src) in list(self.pending_probe.items()):
            if self.memory.step >= deadline:
                self.tried[door].add(ante)
                if _src == "hypothesis":
                    self.trace["memory_incorrect"] += 1
                del self.pending_probe[door]

    def learn_from_experience(self, exp):
        """Offline reinforcement from replayed evidence: existing
        hypotheses — including ones created *after* this transition was
        recorded — are adjudicated against the stored transition. This
        gives replay genuine retroactive value (old data validates new
        hypotheses)."""
        if not self.replay_enabled or self.frozen_policy:
            return
        before_atoms = atoms_from_observation(exp.obs_before)
        after_atoms = atoms_from_observation(exp.obs_after)
        evts = {}
        for door, event in exp.events:
            evts[door] = event == "open"
        for h in self.memory.hypotheses:
            if not h.usable:
                continue
            door = h.door
            wants_open = h.predicted_opens
            if wants_open is True:
                if door in evts:
                    if evts[door] and h.antecedent <= before_atoms:
                        self.memory._support(h, record_ctx=False)
                    elif not evts[door] and h.antecedent <= before_atoms:
                        self.memory._contradict(h, record_ctx=False)
                elif exp.obs_after.door_open[door]:
                    if h.antecedent <= after_atoms:
                        self.memory._support(h, record_ctx=False)
                else:
                    if h.antecedent <= after_atoms:
                        self.memory._contradict(h, record_ctx=False)
            elif wants_open is False:
                if door in evts:
                    if not evts[door] and h.antecedent <= before_atoms:
                        self.memory._support(h, record_ctx=False)
                    elif evts[door] and h.antecedent <= before_atoms:
                        self.memory._contradict(h, record_ctx=False)
                elif not exp.obs_after.door_open[door]:
                    if h.antecedent <= after_atoms:
                        self.memory._support(h, record_ctx=False)
                else:
                    if h.antecedent <= after_atoms:
                        self.memory._contradict(h, record_ctx=False)

    def finish_episode(self, success: bool):
        self.episodes += 1
        if success:
            self.successes += 1

    # -- checkpointing ---------------------------------------------------
    def save_state(self) -> dict:
        rng_state = self.rng.getstate()
        return {
            "seed": self.seed,
            "rng": {
                "version": rng_state[0],
                "internal": list(rng_state[1]),
                "gauss": rng_state[2],
            },
            "replay_enabled": self.replay_enabled,
            "memory": self.memory.save_state(),
            "probe_adaptive": self.probe_adaptive,
            "probe_episode_local": self.probe_episode_local,
            "protect_episode_local": self.protect_episode_local,
            "memory_atom_transfer": self.memory_atom_transfer,
            "memory_bank_refresh_on_contradiction":
                self.memory_bank_refresh_on_contradiction,
            "memory_episode_scoped": self.memory_episode_scoped,
            "memory_min_verified_epochs_for_transfer":
                self.memory_min_verified_epochs_for_transfer,
            "probe_reset_per_episode": self.probe_reset_per_episode,
            "probe_min_effective_confidence":
                self.probe_min_effective_confidence,
            "probe_bank_reuse": self.probe_bank_reuse,
            "planner_max_expand": self.planner_max_expand,
            "probe_max_plans": self.probe_max_plans,
            "probe_timeout": self.probe_timeout,
            "tried": {
                d: [sorted(list(ante)) for ante in t]
                for d, t in self.tried.items()
            },
            "passed": dict(self.passed),
            "far_anchor": dict(self.far_anchor),
            "_ctx_signature": self._ctx_signature,
            "frozen_policy": self.frozen_policy,
            "pending_probe": {
                d: [sorted(list(ante)), deadline, src]
                for d, (ante, deadline, src) in self.pending_probe.items()
            },
            "planner_ops": self.planner_ops,
            "plan_calls": self.plan_calls,
            "plan_cache": {str(k): list(v) for k, v in list(self.plan_cache.items())[:2000]},
            "plan_cache_hits": self.plan_cache_hits,
            "steps": self.steps,
            "episodes": self.episodes,
            "successes": self.successes,
            "goal_infeasible": {str(k): v for k, v in self._goal_infeasible.items()},
        }

    def load_state(self, state: dict) -> None:
        self.seed = state["seed"]
        rng = state["rng"]
        if isinstance(rng, dict):          # JSON-safe form
            self.rng.setstate((rng["version"], tuple(rng["internal"]),
                               rng["gauss"]))
        else:                              # legacy tuple form
            self.rng.setstate(tuple(rng))
        self.replay_enabled = state["replay_enabled"]
        self.memory.load_state(state["memory"])
        self.probe_adaptive = bool(state.get("probe_adaptive", False))
        self.probe_episode_local = bool(state.get("probe_episode_local", False))
        self.protect_episode_local = bool(state.get("protect_episode_local", False))
        self.memory_atom_transfer = bool(state.get("memory_atom_transfer", False))
        self.memory_bank_refresh_on_contradiction = bool(state.get(
            "memory_bank_refresh_on_contradiction", False))
        self.memory_episode_scoped = bool(state.get(
            "memory_episode_scoped", False))
        self.memory_min_verified_epochs_for_transfer = int(state.get(
            "memory_min_verified_epochs_for_transfer", 2))
        self.probe_reset_per_episode = bool(state.get(
            "probe_reset_per_episode", False))
        self.probe_min_effective_confidence = float(state.get(
            "probe_min_effective_confidence", 0.0))
        self.probe_bank_reuse = bool(state.get("probe_bank_reuse", False))
        self.planner_max_expand = state["planner_max_expand"]
        self.probe_max_plans = state["probe_max_plans"]
        self.probe_timeout = state["probe_timeout"]
        self.tried = {
            d: {frozenset(map(tuple, atoms)) for atoms in t}
            for d, t in state["tried"].items()
        }
        self.passed = dict(state["passed"])
        self._ctx_signature = state.get("_ctx_signature")
        self.frozen_policy = state.get("frozen_policy", False)
        self.far_anchor = {d: tuple(a) for d, a in state["far_anchor"].items()}
        self.pending_probe = {}
        for d, row in state["pending_probe"].items():
            atoms, deadline = row[0], row[1]
            src = row[2] if len(row) > 2 else "other"
            self.pending_probe[d] = (
                frozenset(map(tuple, atoms)), deadline, src)
        self.planner_ops = state["planner_ops"]
        self.plan_calls = state["plan_calls"]
        self.plan_cache = {}
        for k, v in state.get("plan_cache", {}).items():
            self.plan_cache[eval(k)] = v
        self.plan_cache_hits = state.get("plan_cache_hits", 0)
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

    def probe_trace(self) -> dict:
        """Ephemeral probe-source diagnostics (not part of persistent
        state; excluded from diagnostics() so checkpoint-restore
        equality is preserved)."""
        return dict(self.trace)

