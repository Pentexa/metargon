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
    atoms_of_state, maximal_feasible_subset, plan_to_atoms, plan_to_pos,
    target_feasible,
)
from ..envs.gridworld import public_next

MAX_EXPANSIONS = 1500
PROBE_MAX_PLANS = 4

# ---- cognitive exploration (v0.1) tuning constants ----------------------
RECOVERY_BUDGET = 12          # steps of directed anti-loop movement
RECOVERY_STALL_WINDOW = 8     # revisit window used to detect loops
RECOVERY_MAX_TRIGGERS_PER_EPISODE = 4
MAX_PLANNER_BUDGET_PER_ACT = 4000   # global BFS budget per act() call
HOLD_BUDGET = 10              # steps a held probe waits for a delayed open


class _BudgetExhausted(Exception):
    """Internal: the per-act() planner budget ran out (bounded act()
    cost under cognitive exploration)."""


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
                 memory_factory=None,
                 cognitive_exploration: bool = False,
                 cog_knowledge_gaps: Optional[bool] = None,
                 cog_info_gain: Optional[bool] = None,
                 cog_recovery: Optional[bool] = None,
                 cog_decompose: Optional[bool] = None,
                 cog_probe_hold: Optional[bool] = None,
                 recovery_budget: int = RECOVERY_BUDGET,
                 hold_budget: int = HOLD_BUDGET,
                 planner_budget_per_act: int = MAX_PLANNER_BUDGET_PER_ACT):
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
        # ---- cognitive exploration (v0.1) flags ---------------------
        # Independent, all off by default (baseline behavior is unchanged
        # when every flag is off). ``cognitive_exploration`` is a master
        # switch: each sub-mechanism defaults to the master value unless
        # set explicitly (used for ablations).
        self.cognitive_exploration = bool(cognitive_exploration)
        self.cog_knowledge_gaps = bool(
            cog_knowledge_gaps if cog_knowledge_gaps is not None
            else cognitive_exploration)
        self.cog_info_gain = bool(
            cog_info_gain if cog_info_gain is not None
            else cognitive_exploration)
        self.cog_recovery = bool(
            cog_recovery if cog_recovery is not None
            else cognitive_exploration)
        self.cog_decompose = bool(
            cog_decompose if cog_decompose is not None
            else cognitive_exploration)
        self.cog_probe_hold = bool(
            cog_probe_hold if cog_probe_hold is not None
            else cognitive_exploration)
        self.recovery_budget = int(recovery_budget)
        self.hold_budget = int(hold_budget)
        self.planner_budget_per_act = int(planner_budget_per_act)
        # runtime state for the cognitive mechanisms (ephemeral unless
        # checkpointed; seeded deterministically)
        self._recent_states = deque(maxlen=24)
        self._recent_actions = deque(maxlen=16)
        self._invalid_streak = 0
        self._last_state_key = None
        self._steps_since_state_change = 0
        self._loop_steps = 0
        self._recovery_steps_left = 0
        self._stall_triggers = 0
        self._active_hold = None
        self._pending_hold_target = None
        self._pending_hold_since = 0
        self._local_atom_evidence = {}
        self._attempts_per_antecedent = {}
        self._act_planner_budget = 0
        # probe-attempt instrumentation (ephemeral, used by the research
        # harness to separate fresh from repeated experiments)
        self._trace_last_probe = None
        self._trace_completed_probes = set()
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
            # v0.1 cognitive exploration trace
            "stalls_detected": 0,
            "recoveries_triggered": 0,
            "recovery_steps": 0,
            "decomposed_candidates": 0,
            "holds_started": 0,
            "holds_resolved_open": 0,
            "holds_resolved_fail": 0,
            "holds_interrupted": 0,
            "hold_preserve_steps": 0,
            "loop_steps": 0,
            "gap_guided_candidates": 0,
            "probes_new": 0,
            "probes_repeat": 0,
            "skipped_search_failed": 0,
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
        # cognitive-exploration per-episode state
        self._recent_states.clear()
        self._recent_actions.clear()
        self._invalid_streak = 0
        self._last_state_key = None
        self._steps_since_state_change = 0
        self._loop_steps = 0
        self._recovery_steps_left = 0
        self._stall_triggers = 0
        self._active_hold = None
        self._pending_hold_target = None
        self._pending_hold_since = 0
        self._attempts_per_antecedent = {}
        self._trace_last_probe = None
        self._trace_completed_probes = set()
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

    def _goal_component_reachable(self) -> bool:
        """Cheap component pre-filter: if the goal is not even in the same
        connected region as the agent under the current door belief, the
        full goal-plan BFS cannot succeed — skip it entirely."""
        comp = self._components()[0]
        pos = self.belief.pos
        goal = self.geom.goal
        return (pos in comp and goal in comp and comp[pos] == comp[goal])

    def _novel_singles(self, door):
        return [s for s in VOCAB_SINGLES if s not in self.tried[door]]

    def _conjunctive_probes(self, door):
        tried = self.tried[door]
        return [p for p in _vocabulary_pairs(tried) if p not in tried]

    def _plan_cached(self, state, target, on_expand, to_pos=None,
                     max_expand=None):
        """Planning with a bounded cache keyed by state+target (BFS results
        are deterministic; repeated probes are common).

        ``max_expand`` overrides the default per-plan expansion cap
        (used by cognitive exploration to keep low-tier probes cheap)."""
        if max_expand is None:
            max_expand = self.planner_max_expand
        key = (state_key(state), to_pos, tuple(sorted(target)))
        hit = self.plan_cache.get(key)
        if hit is not None:
            self.plan_cache_hits += 1
            return hit or None
        if to_pos is not None:
            plan = plan_to_pos(state, to_pos, self.geom,
                               max_expand=max_expand,
                               on_expand=on_expand)
        else:
            plan = plan_to_atoms(state, target, self.geom,
                                 max_expand=max_expand,
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

    def _first_preserving_action(self, state, target,
                                 strict: bool = False) -> Optional[str]:
        """When a probe antecedent already holds, the door can open on the
        next transition. Return an action whose resulting public state
        still satisfies the antecedent (so the condition is not broken),
        preferring actions that make progress toward the relevant cells.

        With ``strict=True`` (used by the hold mechanism) the fallback to
        an arbitrary valid action is disabled: None is returned when no
        preserving action exists at all (so the caller knows the condition
        cannot be sustained)."""
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
        if best is not None:
            return best
        return None if strict else (valid[0] if valid else None)

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

    # -- cognitive exploration (v0.1) -----------------------------------
    # Bounded mechanisms for deciding *which* experiment to run when the
    # current beliefs are insufficient: local knowledge-gap coverage,
    # information-value scoring, stagnation detection with recovery, and
    # hold-based probe attribution (fixes mid-plan expiry and delayed
    # doors). All are independent, gated by the cog_* flags, and never
    # touch hidden state.

    def _bump_attempts(self, door, ante):
        key = (door, frozenset(ante))
        self._attempts_per_antecedent[key] = \
            self._attempts_per_antecedent.get(key, 0) + 1

    def _probe_key(self, door, ante, src):
        return (door, frozenset(ante), src)

    def _log_probe_attempt(self, door, ante, src):
        """Record a probe attempt (fresh vs repeated) for the research
        harness. A selection is only logged when it starts a new attempt
        (differs from the previous selection) — continuation steps of the
        same plan are not attempts."""
        if not self._collect_probe_log:
            return
        key = self._probe_key(door, ante, src)
        self.trace.setdefault("planned_probes", []).append(
            [door, sorted(list(ante)), src])
        if key == self._trace_last_probe:
            return
        self._trace_last_probe = key
        if key in self._trace_completed_probes:
            self.trace["probes_repeat"] = \
                self.trace.get("probes_repeat", 0) + 1
        else:
            self.trace["probes_new"] = self.trace.get("probes_new", 0) + 1

    def _mark_probe_completed(self, door, ante, src):
        """A probe attempt reached a definite outcome (door opened, budget
        exhausted, or the condition could no longer be sustained)."""
        if not self._collect_probe_log:
            return
        self._trace_completed_probes.add(
            self._probe_key(door, ante, src))

    def _local_vocab(self) -> frozenset:
        """Vocabulary atoms for the local knowledge-gap model (the fixed
        public probe vocabulary; both switch polarities and all plate
        load combos)."""
        return frozenset().union(*VOCAB_SINGLES) if VOCAB_SINGLES \
            else frozenset()

    def _uncovered_atoms(self, door) -> set:
        """Atoms with no decisive local evidence for ``door``: not yet
        classified by this task's own observations. Bounded and small."""
        vocab = self._local_vocab()
        ev = self._local_atom_evidence.get(door, {})
        out = set()
        for a in vocab:
            p, n = ev.get(a, (0, 0))
            if not (p >= 2 and p > n) and not (n >= 2 and n > p):
                out.add(a)
        return out

    def _note_local_evidence(self, door, atoms, opened: bool):
        """Update the local per-door atom evidence profile from this step's
        observation (positive: atoms present before an opening event;
        negative: atoms present while the door stayed closed)."""
        ev = self._local_atom_evidence.setdefault(door, {})
        for a in atoms:
            p, n = ev.get(a, (0, 0))
            if opened:
                ev[a] = (p + 1, n)
            else:
                ev[a] = (p, n + 1)

    def _candidate_score(self, door, ante, prio, src, plan_len) -> float:
        """Deterministic information-value score for a candidate probe.

        * knowledge-gap term: novelty over uncovered atoms;
        * discrimination term: weighted overlap with live hypotheses
          (bounded to keep the score stable);
        * attempt penalty: demote antecedents that already failed/incon-
          clusively resolved this episode;
        * cost term: shorter plans win ties (matches baseline tie-break).
        """
        score = 0.0
        if self.cog_knowledge_gaps:
            unc = self._uncovered_atoms(door)
            novelty = sum(1 for a in ante if a in unc)
            score += 2.0 * novelty
        if self.cog_info_gain:
            disc = 0.0
            for h in self.memory.query(door, opens=True)[:8]:
                if h.antecedent & ante:
                    disc += h.confidence
            score += 1.5 * min(disc, 4.0)
            attempts = self._attempts_per_antecedent.get(
                (door, frozenset(ante)), 0)
            score -= 1.0 * attempts
        score -= 0.05 * plan_len
        return score

    def _gap_guided_candidates(self, door) -> list:
        """Pairs of currently-uncovered atoms (never in the fixed pair
        list because of its 64-entry bound). Used when exploration is
        escalated after repeated stalls."""
        uncovered = sorted(self._uncovered_atoms(door), key=str)
        tried = self.tried[door]
        out = []
        for i in range(len(uncovered)):
            for j in range(i + 1, len(uncovered)):
                p = frozenset({uncovered[i], uncovered[j]})
                if p not in tried:
                    # prio 2: fallback tier — only used once ordinary
                    # probes are exhausted or infeasible
                    out.append((door, p, 2, "gap"))
                if len(out) >= 8:
                    return out
        return out

    def _note_step(self):
        """Per-step bookkeeping for stagnation detection (bounded)."""
        key = state_key(self.belief)
        if self._last_state_key is None:
            self._last_state_key = key
            self._recent_states.append(key)
            return
        if key != self._last_state_key:
            self._steps_since_state_change = 0
            self._last_state_key = key
        else:
            self._steps_since_state_change += 1
        self._recent_states.append(key)
        recent = list(self._recent_states)
        if len(recent) >= 5 and recent.count(key) >= 3:
            self._loop_steps += 1

    def _detect_stall(self) -> bool:
        """True when the student is making no progress: repeated invalid
        actions, no state change for many steps, or revisiting the same
        state within a short window."""
        if self._active_hold is not None or self._pending_hold_target:
            return False            # a held probe is productive waiting
        if self._invalid_streak >= 3:
            return True
        if self._steps_since_state_change >= 10:
            return True
        rec = list(self._recent_states)
        if len(rec) >= 6 and rec.count(rec[-1]) >= 3:
            return True
        return False

    def _recovery_plan(self) -> Optional[list]:
        """Directed anti-loop movement: pick a valid action whose resulting
        state is least recently visited, preferring movement away from the
        current cell. Bounded (<= 9 actions), deterministic, public-only."""
        st = self.belief
        valid = valid_actions(st, self.geom)
        if not valid:
            return None
        recent = set(self._recent_states)
        best = None
        best_score = None
        pos = st.pos
        for a in valid:
            nxt = public_next(st, a, self.geom)
            if nxt is None:
                continue
            k = state_key(nxt)
            score = 10.0 if k not in recent else -5.0
            score += 0.5 * (abs(nxt.pos[0] - pos[0])
                             + abs(nxt.pos[1] - pos[1]))
            if best_score is None or score > best_score:
                best_score = score
                best = a
        return [best] if best is not None else None

    def _maybe_start_hold(self, door, ante, src) -> bool:
        """Begin holding a model-backed probe once its antecedent currently
        holds. Holding waits through delayed openings and prevents the
        agent from cancelling a probe mid-attempt."""
        if not self.cog_probe_hold:
            return False
        if src not in ("hypothesis", "bank"):
            return False
        if self._active_hold is not None:
            return False
        if ante <= atoms_of_state(self.belief):
            self._active_hold = {
                "door": door, "ante": frozenset(ante), "src": src,
                "held_steps": 0,
            }
            self.trace["holds_started"] += 1
            return True
        return False

    def _resolve_hold(self):
        """Advance an active held probe. Returns (plan, door):

        * ([preserving_action], door) — keep waiting for a delayed open;
        * (None, door) — the hold resolved (door opened); caller falls
          through to the normal open-door pass-through;
        * (None, None) — no hold / hold ended without opening.
        """
        h = self._active_hold
        if h is None:
            return None, None
        door = h["door"]
        ante = h["ante"]
        if self.belief.door_open.get(door, False):
            # success: the door opened while the condition was held
            self._mark_probe_completed(door, ante, h["src"])
            self._active_hold = None
            self._pending_hold_target = None
            self.trace["holds_resolved_open"] += 1
            return None, door
        if not (ante <= atoms_of_state(self.belief)):
            # the condition was broken before the door opened: inconclusive
            self._mark_probe_completed(door, ante, h["src"])
            self._active_hold = None
            self.trace["holds_interrupted"] += 1
            self._bump_attempts(door, ante)
            return None, None
        if h["held_steps"] < self.hold_budget:
            pres = self._first_preserving_action(self.belief, ante,
                                                 strict=True)
            if pres is not None:
                h["held_steps"] += 1
                self.trace["hold_preserve_steps"] += 1
                return [pres], door
            self._mark_probe_completed(door, ante, h["src"])
            self._active_hold = None
            self.trace["holds_interrupted"] += 1
            self._bump_attempts(door, ante)
            return None, None
        # held for the full budget without opening: genuine failure
        self._mark_probe_completed(door, ante, h["src"])
        self._active_hold = None
        self._pending_hold_target = None
        self.tried.setdefault(door, set()).add(ante)
        self.trace["holds_resolved_fail"] += 1
        if h["src"] == "hypothesis":
            self.trace["memory_incorrect"] += 1
        return None, None

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
            if self._goal_component_reachable():
                plan = self._plan_cached(self.belief, (), on_expand,
                                         to_pos=self.geom.goal)
                if plan is not None:
                    self._steps_since_goal_check = 0
                    return plan, None
            self._steps_since_goal_check = 0
            self._goal_infeasible[pattern] = True
        if self.probe_vocab_disabled:
            return None, None
        # hold-based probe attribution (cognitive path): keep a model-backed
        # probe's condition true while waiting for a (possibly delayed)
        # opening, instead of the fixed selection-time deadline.
        if self.cog_probe_hold:
            plan_h, door_h = self._resolve_hold()
            if plan_h:
                return plan_h, door_h
            if self._pending_hold_target is not None:
                pd, pa, ps = self._pending_hold_target
                if pa <= atoms_of_state(self.belief):
                    if self._maybe_start_hold(pd, pa, ps):
                        self._pending_hold_target = None
                        plan_h, door_h = self._resolve_hold()
                        if plan_h:
                            return plan_h, door_h
                elif self.memory.step - self._pending_hold_since \
                        > self.probe_timeout + 4:
                    self._pending_hold_target = None
        if not self.cognitive_exploration:
            return self._choose_probe_baseline()
        return self._choose_probe_cognitive()

    def _choose_probe_baseline(self):
        """The original probe-selection path (identical to the pre-v0.1
        behavior; used when cognitive exploration is off)."""
        def on_expand():
            self.planner_ops += 1
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
            self._log_probe_attempt(door, ante, src)
            return plan, door
        return None, None

    def _choose_probe_cognitive(self):
        """Cognitive probe selection (v0.1): bounded information-value
        scoring within priority tiers, decomposition of infeasible
        conjunctive antecedents, gap-guided escalation after stalls, and
        a global planner budget per act() call."""
        def on_expand():
            self.planner_ops += 1
            self._act_planner_budget -= 1
            if self._act_planner_budget <= 0:
                raise _BudgetExhausted()
        current = atoms_of_state(self.belief)
        candidates = []
        for door in self.geom.door_cell:
            if self.belief.door_open[door]:
                continue
            if self.probe_adaptive:
                candidates.extend(self._adaptive_probe_candidates(door, current))
                if self._stall_triggers >= 2:
                    gap = self._gap_guided_candidates(door)
                    self.trace["gap_guided_candidates"] += len(gap)
                    candidates.extend(gap)
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
            if self._stall_triggers >= 2:
                gap = self._gap_guided_candidates(door)
                self.trace["gap_guided_candidates"] += len(gap)
                candidates.extend(gap)
        candidates.sort(key=lambda x: x[2])
        protected = self._protected_atoms()
        best_by_prio = {}
        evaluated = 0
        try:
            for door, ante, prio, src in candidates:
                if ante in self.tried[door]:
                    self.trace["skipped_tried"] += 1
                    continue
                if not target_feasible(self.belief, ante, self.geom):
                    # Infeasible under the current belief: never burn the
                    # antecedent (connectivity changes cheaply re-check it
                    # next time). Decompose into a feasible subset when
                    # enabled so partial evidence is still collected.
                    if self.cog_decompose:
                        sub = maximal_feasible_subset(
                            self.belief, ante, self.geom)
                        if sub and sub != ante and sub not in self.tried[door]:
                            ante = sub
                            self.trace["decomposed_candidates"] += 1
                        else:
                            self.trace["skipped_infeasible"] += 1
                            continue
                    else:
                        self.trace["skipped_infeasible"] += 1
                        continue
                # Model-backed probes get the full planning budget; low-
                # tier vocabulary/gap probes use a cheaper cap so wasted
                # searches cannot dominate the action budget.
                me = 1500 if src in ("hypothesis", "bank") else 700
                plan = self._plan_cached(self.belief, ante, on_expand,
                                         max_expand=me)
                self.plan_calls += 1
                evaluated += 1
                if plan:
                    if self._plan_breaks_protection(plan, protected):
                        continue
                    score = self._candidate_score(
                        door, ante, prio, src, len(plan))
                    cur = best_by_prio.get(prio)
                    if cur is None or score > cur[0]:
                        best_by_prio[prio] = (score, plan, door, ante, src)
                elif ante <= atoms_of_state(self.belief):
                    pres = self._first_preserving_action(self.belief, ante,
                                                        strict=True)
                    if pres is not None and not self._plan_breaks_protection(
                            [pres], protected):
                        score = self._candidate_score(
                            door, ante, prio, src, 1)
                        cur = best_by_prio.get(prio)
                        if cur is None or score > cur[0]:
                            best_by_prio[prio] = (
                                score, [pres], door, ante, src)
                    else:
                        self.tried[door].add(ante)
                else:
                    # no plan within the expansion cap: the full search is
                    # expensive, so do not re-run it on every step. Mark
                    # the antecedent as attempted (bounded); decomposition
                    # and gap-guided candidates keep alternative routes
                    # available.
                    self.tried[door].add(ante)
                    self.trace["skipped_search_failed"] += 1
                if evaluated >= self.probe_max_plans:
                    break
        except _BudgetExhausted:
            pass
        if best_by_prio:
            prio = min(best_by_prio)
            _, plan, door, ante, src = best_by_prio[prio]
            key = f"source_{src}"
            if key not in self.trace:
                key = "source_other"
            self.trace[key] += 1
            self.trace["plans_used"] += 1
            if src == "hypothesis":
                self.trace["memory_actions"] += 1
            if self.cog_probe_hold and src in ("hypothesis", "bank"):
                if ante <= atoms_of_state(self.belief):
                    if self._maybe_start_hold(door, ante, src):
                        self._log_probe_attempt(door, ante, src)
                        return plan, door
                    self._pending_hold_target = (door, ante, src)
                    self._pending_hold_since = self.memory.step
                else:
                    self._pending_hold_target = (door, ante, src)
                    self._pending_hold_since = self.memory.step
            else:
                self.pending_probe[door] = (
                    ante, self.memory.step + self.probe_timeout, src)
            self._log_probe_attempt(door, ante, src)
            return plan, door
        return None, None

    # -- interface -------------------------------------------------------
    def act(self, obs) -> str:
        self._sync(obs)
        if self.cognitive_exploration:
            self._note_step()
            self._act_planner_budget = self.planner_budget_per_act
            if self.cog_recovery:
                if self._recovery_steps_left > 0:
                    plan = self._recovery_plan()
                    if plan:
                        self._recovery_steps_left -= 1
                        self.trace["recovery_steps"] += 1
                        return plan[0]
                    self._recovery_steps_left = 0
                if self._detect_stall():
                    self._stall_triggers += 1
                    self.trace["stalls_detected"] += 1
                    if self._stall_triggers \
                            <= RECOVERY_MAX_TRIGGERS_PER_EPISODE:
                        self._recovery_steps_left = self.recovery_budget
                        self.trace["recoveries_triggered"] += 1
                        self._recent_states.clear()
                        plan = self._recovery_plan()
                        if plan:
                            self._recovery_steps_left -= 1
                            self.trace["recovery_steps"] += 1
                            return plan[0]
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
        if self.cognitive_exploration:
            if result.valid:
                self._invalid_streak = 0
            else:
                self._invalid_streak += 1
            self._recent_actions.append(action)
        # door events from the public observation stream
        event_happened = False
        event_doors = set()
        for door in self.geom.door_cell:
            before_open = getattr(obs_before, "d1" if door == "D1" else "d2")
            after_open = getattr(after_obs, "d1" if door == "D1" else "d2")
            if not before_open and after_open:
                self.memory.observe_event(before_atoms, door, True, after_atoms)
                event_happened = True
                event_doors.add(door)
                self.passed[door] = False
                if self.cog_knowledge_gaps:
                    self._note_local_evidence(door, before_atoms, True)
                if self._active_hold is not None \
                        and self._active_hold["door"] == door:
                    if self._active_hold["src"] == "hypothesis":
                        self.trace["memory_correct"] += 1
                    self._mark_probe_completed(
                        door, self._active_hold["ante"],
                        self._active_hold["src"])
                    self._active_hold = None
                    self._pending_hold_target = None
                    self.trace["holds_resolved_open"] += 1
                if door in self.pending_probe:
                    _, _, src = self.pending_probe[door]
                    self._mark_probe_completed(door, self.pending_probe[door][0], src)
                    self.trace["door_open_after_probe"] += 1
                    self.trace[f"door_open_after_{src}"] += 1
                    if src == "hypothesis":
                        self.trace["memory_correct"] += 1
                self.pending_probe.pop(door, None)
            elif before_open and not after_open:
                self.memory.observe_event(before_atoms, door, False, after_atoms)
                event_happened = True
                event_doors.add(door)
        if event_happened:
            # connectivity/rule information changed: probe feasibility may
            # have changed. The baseline resets every door's tried set on
            # ANY event (a diagnosed source of repeated probes). The
            # cognitive path never resets tried sets: content-feasible
            # antecedents are re-checked cheaply every time, and executed-
            # failed antecedents stay excluded (they were genuinely tried),
            # so no budget is re-burned on the same experiment.
            if not self.cognitive_exploration:
                for d in self.geom.door_cell:
                    self.tried[d] = set()
        # nonevents for doors that did not change as expected
        for door in self.geom.door_cell:
            after_open = getattr(after_obs, "d1" if door == "D1" else "d2")
            if self.cog_knowledge_gaps and not after_open:
                self._note_local_evidence(door, after_atoms, False)
            if not after_open:
                self.memory.observe_nonevent(after_atoms, door, True)
            else:
                self.memory.observe_nonevent(after_atoms, door, False)
                self.memory.observe_maintain(after_atoms, door)
        self.memory.step_advance()
        for door, (ante, deadline, _src) in list(self.pending_probe.items()):
            if self.memory.step >= deadline:
                self._mark_probe_completed(door, ante, _src)
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
            "cognitive_exploration": self.cognitive_exploration,
            "cog_knowledge_gaps": self.cog_knowledge_gaps,
            "cog_info_gain": self.cog_info_gain,
            "cog_recovery": self.cog_recovery,
            "cog_decompose": self.cog_decompose,
            "cog_probe_hold": self.cog_probe_hold,
            "recovery_budget": self.recovery_budget,
            "hold_budget": self.hold_budget,
            "planner_budget_per_act": self.planner_budget_per_act,
            "local_atom_evidence": {
                door: {tuple(a): list(e)
                       for a, e in ev.items()}
                for door, ev in self._local_atom_evidence.items()
            },
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
        self.cognitive_exploration = bool(state.get(
            "cognitive_exploration", False))
        self.cog_knowledge_gaps = bool(state.get(
            "cog_knowledge_gaps", self.cognitive_exploration))
        self.cog_info_gain = bool(state.get(
            "cog_info_gain", self.cognitive_exploration))
        self.cog_recovery = bool(state.get(
            "cog_recovery", self.cognitive_exploration))
        self.cog_decompose = bool(state.get(
            "cog_decompose", self.cognitive_exploration))
        self.cog_probe_hold = bool(state.get(
            "cog_probe_hold", self.cognitive_exploration))
        self.recovery_budget = int(state.get(
            "recovery_budget", RECOVERY_BUDGET))
        self.hold_budget = int(state.get("hold_budget", HOLD_BUDGET))
        self.planner_budget_per_act = int(state.get(
            "planner_budget_per_act", MAX_PLANNER_BUDGET_PER_ACT))
        self._local_atom_evidence = {}
        for door, ev in state.get("local_atom_evidence", {}).items():
            self._local_atom_evidence[door] = {
                tuple(a): tuple(e) for a, e in ev.items()}
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

