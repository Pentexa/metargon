"""METARGON v0.11 — noise-aware learning (confirmed events, certainty-
weighted evidence, replay filtering/weighting).

v0.10 measured pathology under noisy observations (see docs/RESEARCH_V10.md
and results/v11/diagnostics_noise_v10.json):
  * the agent believed ~4.3x more door events than exist (single-bit raw
    flips become events), generating a polluted causal pool;
  * ~96% of \"verified\" hypotheses were false under noise;
  * unreplayered replay amplified that polluted pool (approx. 8.8k
    replay applications per episode).

v0.11 keeps exactly v0.10's planning, budgets, memory cap and action
interface, and adds three family-agnostic, public-information-only
mechanisms:

  A. Confirmed events — door events enter the learning stream only when
     confirmed by (i) two consecutive agreeing raw readings or (ii)
     movement proof (successful move through an open door, failed move
     into a closed door). Single corrupted flips are dropped. The
     *planning* belief still adopts readings immediately (v0.10
     behavior), so no planning lag is introduced.
  B. Certainty-weighted evidence — contexts are built only from stable
     sensors (not mid-debounce, not mid-event); each supporting update is
     weighted by `certainty` = the fraction of context atoms whose
     sensors were stable (floor 0.25). This is the family-agnostic
     observation-likelihood proxy.
  C. Replay modes — \"filtered\" (only confirmed high-certainty events are
     replayed) and \"weighted\" (priority and support scaled by
     certainty) are compared against v0.10's legacy \"on\" and \"off\".
"""
from __future__ import annotations

from lab_env import DOORS

from metargon_v10 import MetargonV10
from metargon_v09 import atoms_of_state
from compositional_memory_v10 import atoms_from_belief, opens_key
from robust_belief import RobustBelief


class MetargonV11(MetargonV10):
    name = "metargon"

    def __init__(self, seed: int = 0, max_hypotheses: int = 128,
                 use_robust_belief: bool = True,
                 use_cooldowns: bool = True,
                 use_explore_fallback: bool = True,
                 use_replay: bool = True,
                 replay_mode: str = "filtered",
                 use_confirmed_events: bool = True,
                 use_certainty: bool = True):
        if replay_mode not in ("on", "off", "filtered", "weighted"):
            raise ValueError(f"unknown replay_mode {replay_mode}")
        self.replay_mode = replay_mode
        self.use_confirmed_events = use_confirmed_events
        self.use_certainty = use_certainty
        # memory flags: enable_replay mirrors legacy toggle; replay_mode is
        # forwarded for filtered/weighted behavior.
        super().__init__(
            seed=seed, max_hypotheses=max_hypotheses,
            use_robust_belief=use_robust_belief,
            use_cooldowns=use_cooldowns,
            use_explore_fallback=use_explore_fallback,
            use_replay=use_replay,
        )
        self.memory.replay_mode = replay_mode
        self._confirmed_total = 0
        self._dropped_total = 0
        self._prev_dropped = 0
        self._certainty_sum = 0.0
        self._nonevent_gated = 0

    # -- noise-aware event/context helpers -------------------------------
    def _learn_contexts(self):
        """Pre/post atom sets from the reconciled belief (membership is
        never filtered; certainty handles noise)."""
        return (atoms_from_belief(self.belief.belief),
                atoms_from_belief(self.belief.belief))

    def _observe_outcome_v11(self, obs_before, action, res):
        """Override of the event-driven learning block."""
        pre_atoms = atoms_from_belief(self.belief.belief)
        if not self.use_confirmed_events:
            prev_doors = dict(self.belief.belief.door_open)
        if self.use_robust_belief:
            self.belief.observe(res.observation)
            self.belief.apply_transition(action, res.valid, res.observation)
        else:
            self._sync_raw(res.observation)
        post_atoms = atoms_from_belief(self.belief.belief)

        if self.use_confirmed_events:
            confirmed = self.belief.poll_confirmed_events()
            self._confirmed_total += len(confirmed)
            self._dropped_total += (
                self.belief.dropped_candidates - self._prev_dropped)
            self._prev_dropped = self.belief.dropped_candidates
            events = []
            # dedupe per door (keep the latest in a step)
            seen = set()
            for door, opened in reversed(confirmed):
                if door in seen:
                    continue
                seen.add(door)
                events.append((door, opened))
            events.reverse()
        else:
            # replicate v0.10 belief-diff events
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
            self._passed[door] = False
            self._pending_probe.pop(door, None)
            self._recent_protect[door] = (pre, self.memory.step + 12)
            if self.use_cooldowns:
                self.probed[door].clear()
            else:
                self.tried[door].clear()
            any_event = True

        if any_event:
            self._goal_infeasible.clear()
            self._current_plan = []

        # nonevents are gated on door stability when confirmed-event mode is
        # on: a door with an open candidate is not believed; feeding a
        # spurious state into contradiction counting would corrupt verified
        # hypotheses.
        for door in DOORS:
            if self.use_confirmed_events and \
                    not self.belief.door_is_stable(door):
                self._nonevent_gated += 1
                continue
            if not self.belief.belief.door_open[door]:
                self.memory.observe_nonevent(post_atoms, door, True)
            else:
                self.memory.observe_nonevent(post_atoms, door, False)
        self.memory.step_advance()
        for door, (ante, deadline) in list(self._pending_probe.items()):
            if self.memory.step >= deadline:
                self._mark_probe_failed(door, ante)
                del self._pending_probe[door]

    def _protected_atoms(self) -> frozenset:
        """v0.11 protection: hypothesis-satisfying atoms plus the current
        load-bearing state of open, uncrossed doors. The v0.10
        "recent causal credit" pre-event-atom protection is removed: it
        protected stale low-value literals (e.g. an old S2=False) and made
        the required toggles illegal (measured failure mode)."""
        out = set()
        b = self.belief.belief
        now = atoms_of_state(b)
        for door in DOORS:
            h = self.memory.best_for_door(door, opens=True)
            if h is not None and h.antecedent <= now:
                out.update(h.antecedent)
            if b.door_open[door] and not self._passed[door]:
                out.update(now)
        return frozenset(out)

    def episode_start(self):
        super().episode_start()
        self._prev_dropped = 0

    def observe_outcome(self, obs_before, action, res):
        self.steps += 1
        self._observe_outcome_v11(obs_before, action, res)

    def diagnostics(self) -> dict:
        d = super().diagnostics()
        d["confirmed_events_total"] = self._confirmed_total
        d["dropped_candidates_total"] = self.belief.dropped_candidates
        d["certainty_sum"] = round(self._certainty_sum, 3)
        d["nonevent_gated"] = self._nonevent_gated
        d["replay_mode"] = self.replay_mode
        return d
