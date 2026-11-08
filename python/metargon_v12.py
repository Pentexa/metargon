"""METARGON v0.12 — robust general causal learning.

Mission A (reduce clean regressions while keeping noise robustness):
  1. dual-context candidate generation (`event_context_mode="dual"`):
     the event-flipped atom is a likely cause, never dropped from candidate
     contexts (v0.11 cleaned contexts could not learn toggle-induced rules).
  2. adaptive evidence confirmation: the confirmation threshold tracks the
     perceived corruption rate (1 extra agreeing reading when calm, 2 when
     noisy).
  3. short recent-causal-credit protection (6 steps): after a door opens,
     protect the atoms that held at opening while the door is open and
     uncrossed, but never long enough to block necessary toggles.
  4. nonevent gating also skips the few steps right after a transition
     (no contradiction signal during the event itself).

Mission B (temporal ordered-sequence hypotheses, prototype):
  optional `TemporalMemory` fed by public belief transitions; hypotheses
  <first becomes true> THEN <second becomes true> within a window => door
  opens; integrated into probing as an additional candidate source.
"""
from __future__ import annotations

from lab_env import DOORS

from metargon_v11 import MetargonV11
from metargon_v09 import atoms_of_state, plan_to_atoms, PROBE_MAX_PLANS
from compositional_memory_v10 import atoms_from_belief
from temporal_memory import TemporalMemory, VERIFIED

RECENT_CREDIT_STEPS = 6
NONEVENT_SKIP_AFTER_TRANSITION = 2


class MetargonV12(MetargonV11):
    name = "metargon"

    def __init__(self, seed: int = 0, max_hypotheses: int = 128,
                 use_robust_belief: bool = True,
                 use_cooldowns: bool = True,
                 use_explore_fallback: bool = True,
                 use_replay: bool = True,
                 replay_mode: str = "filtered",
                 use_confirmed_events: bool = True,
                 use_certainty: bool = True,
                 use_dual_context: bool = True,
                 use_adaptive_confirm: bool = True,
                 use_recent_credit: bool = True,
                 use_temporal: bool = False,
                 temporal_max: int = 64):
        super().__init__(
            seed=seed, max_hypotheses=max_hypotheses,
            use_robust_belief=use_robust_belief,
            use_cooldowns=use_cooldowns,
            use_explore_fallback=use_explore_fallback,
            use_replay=use_replay,
            replay_mode=replay_mode,
            use_confirmed_events=use_confirmed_events,
            use_certainty=use_certainty,
        )
        self.use_dual_context = use_dual_context
        self.use_adaptive_confirm = use_adaptive_confirm
        self.use_recent_credit = use_recent_credit
        self.use_temporal = use_temporal
        if use_dual_context:
            self.memory.event_context_mode = "dual"
        if not use_adaptive_confirm:
            self.belief.adaptive_confirm_persist = 1
        self.temporal = TemporalMemory(max_hypotheses=temporal_max, seed=seed) \
            if use_temporal else None
        self._temporal_probes = 0
        self._prev_atoms = frozenset()

    # -- Mission A: recent-causal-credit protection -----------------------
    def _protected_atoms(self) -> frozenset:
        out = set()
        b = self.belief.belief
        now = atoms_of_state(b)
        for door in DOORS:
            h = self.memory.best_for_door(door, opens=True)
            if h is not None and h.antecedent <= now:
                out.update(h.antecedent)
            if b.door_open[door] and not self._passed[door]:
                out.update(now)
            if self.use_recent_credit and b.door_open[door] and \
                    not self._passed[door]:
                recent = self._recent_protect.get(door)
                if recent is not None:
                    atoms, expiry = recent
                    if self.memory.step <= expiry:
                        out.update(atoms)
        return frozenset(out)

    # -- temporal integration ---------------------------------------------
    def _feed_temporal(self, pre_atoms, post_atoms):
        if self.temporal is None:
            return
        step = self.memory.step
        newly_true = post_atoms - pre_atoms
        for atom in newly_true:
            if atom[0] in ("door_open",):
                continue
            comp = atom[1]
            door = "D1" if comp in ("P1", "S1") else "D2"
            self.temporal.transitions[door].append((step, atom))
        for door in DOORS:
            sub_new = frozenset(a for a in newly_true
                                if (a[1] in ("P1", "S1") and door == "D1")
                                or (a[1] in ("P2", "S2") and door == "D2"))
            self.temporal.update(door, step, sub_new)

    # -- observation pipeline ---------------------------------------------
    def _observe_outcome_v11(self, obs_before, action, res):
        pre_atoms = atoms_from_belief(self.belief.belief)
        if not self.use_confirmed_events:
            prev_doors = dict(self.belief.belief.door_open)
        if self.use_robust_belief:
            self.belief.observe(res.observation)
            self.belief.apply_transition(action, res.valid, res.observation)
        else:
            self._sync_raw(res.observation)
        post_atoms = atoms_from_belief(self.belief.belief)

        if self.use_temporal:
            self._feed_temporal(pre_atoms, post_atoms)

        if self.use_confirmed_events:
            confirmed = self.belief.poll_confirmed_events()
            self._confirmed_total += len(confirmed)
            self._dropped_total += (
                self.belief.dropped_candidates - self._prev_dropped)
            self._prev_dropped = self.belief.dropped_candidates
            events = []
            seen = set()
            for door, opened in reversed(confirmed):
                if door in seen:
                    continue
                seen.add(door)
                events.append((door, opened))
            events.reverse()
        else:
            events = []
            for door in DOORS:
                cur = self.belief.belief.door_open[door]
                if cur != prev_doors[door]:
                    events.append((door, cur))

        any_event = False
        for door, opened in events:
            pre, post = pre_atoms, post_atoms
            if self.use_certainty:
                certainty = self.belief.context_certainty(pre | post)
            else:
                certainty = 1.0
            self._certainty_sum += certainty
            self.memory.observe_event_ex(pre, post, door, opened,
                                         certainty=certainty)
            if self.temporal is not None:
                self.temporal.observe_open(door, opened, self.memory.step)
            self._passed[door] = False
            self._pending_probe.pop(door, None)
            # short recent-credit protection (6 steps)
            self._recent_protect[door] = (
                pre, self.memory.step + RECENT_CREDIT_STEPS)
            if self.use_cooldowns:
                self.probed[door].clear()
            else:
                self.tried[door].clear()
            any_event = True

        if any_event:
            self._goal_infeasible.clear()
            self._current_plan = []

        for door in DOORS:
            if self.use_confirmed_events:
                if not self.belief.door_is_stable(door):
                    self._nonevent_gated += 1
                    continue
                last_trans = self.belief._door_transition_step.get(door, -10**6)
                if self.memory.step - last_trans <= NONEVENT_SKIP_AFTER_TRANSITION:
                    self._nonevent_gated += 1
                    continue
            if not self.belief.belief.door_open[door]:
                self.memory.observe_nonevent(post_atoms, door, True)
                if self.temporal is not None:
                    self.temporal.observe_nonevent(door, self.memory.step)
            else:
                self.memory.observe_nonevent(post_atoms, door, False)
        self.memory.step_advance()
        if self.temporal is not None:
            self.temporal.step_advance()
        for door, (ante, deadline) in list(self._pending_probe.items()):
            if self.memory.step >= deadline:
                self._mark_probe_failed(door, ante)
                del self._pending_probe[door]

    # -- temporal-augmented probing ---------------------------------------
    def _choose_probe(self):
        plan, door = super()._choose_probe()
        if plan is not None:
            return plan, door
        if self.temporal is None:
            return None, None
        b = self.belief.belief
        protected = self._protected_atoms()

        def on_expand():
            self.planner_ops += 1
        # temporal hypothesis: probe its second atom (the completion step)
        best = None
        for d in DOORS:
            if b.door_open[d]:
                continue
            h = self.temporal.best_for_door(d)
            if h is None or h.status not in (VERIFIED, "predicted"):
                continue
            # temporal objective: satisfy first, then second (ordered)
            now = atoms_of_state(b)
            if not h.first <= now:
                target = h.first
            else:
                target = h.second
            if target <= now:
                continue
            if any(a in protected for a in target):
                continue
            p = plan_to_atoms(b, target, on_expand=on_expand,
                              cache=self._plan_cache, protected=protected)
            if p is not None:
                self.plan_calls += 1
                self._temporal_probes += 1
                if best is None or len(p) < len(best[0]):
                    best = (p, d, target)
        if best is not None:
            p, d, target = best
            self._pending_probe[d] = (target, self.memory.step
                                      + len(p) + 10)
            return p, d
        return None, None

    def episode_start(self):
        super().episode_start()
        if self.temporal is not None:
            self.temporal.episode_end()
        self._prev_atoms = frozenset()

    def diagnostics(self) -> dict:
        d = super().diagnostics()
        d["dual_context"] = self.use_dual_context
        d["adaptive_confirm"] = self.use_adaptive_confirm
        d["recent_credit"] = self.use_recent_credit
        d["temporal_probes"] = self._temporal_probes
        if self.temporal is not None:
            d["temporal"] = self.temporal.diagnostics()
        return d
