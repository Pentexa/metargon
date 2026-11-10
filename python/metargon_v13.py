"""METARGON v0.13 — robust causal generalization.

Fixes (all family-agnostic, public information only):

Mission A — make temporal ordered-rule discovery reliable across seeds
  (v0.12.1 exact D1 on family L: 16.9% stable / 23.1% novel, driven by a
  single seed per condition; see `diagnose_v13.py`):

  *Adopted* — rolling completion generation
    (`temporal_completion_mode="rolling"`): v0.12 kept a SINGLE "active
    first" atom per door, so when one step flips several atoms at once
    (e.g. loading a plate turns on BOTH ("load", P, o) and ("loadany",
    P)), a spurious completion consumes the slot and the true ordered
    pair (e.g. (load,P1,A) then (sw,S1,False)) is never generated.
    v0.13 keeps a per-door buffer of all recent first atoms and generates
    one candidate per different buffered first on every flip — the true
    pair is generated whenever it is observed, instead of only when it
    happens to be adjacent in the single-slot stream.

  *Evaluated, neutral on L, OFF by default (preserved as ablations):*
    - trigger-proximity support weighting (`temporal_support_weighting=
      "proximity"`): weights support by the gap between the completion's
      second atom and the confirmed open. Exactly identical L results
      (39.4/33.1 exact D1) with and without it in the 4×40 pilot.
    - temporal probe exemption (`temporal_target_exempt=True`): lets a
      predicted/verified temporal target atom bypass hypothesis-level
      protection. Diagnostics showed 0 blocked temporal probes on L, so
      the exemption never engaged; results identical.
    - persistence-gated dual context (`dual_persist_gate`): switches event
      learning from dual to clean context when the belief state persisted
      for N steps before an event. Fired on ~half of R events but did not
      recover R-stable (60.62 both ways) while full-clean (-nodc) does
      recover it (71.88) at the cost of F-D2. Rejected.

Mission B — kept as v0.12.1 defaults (dual context, recent credit off,
  adaptive confirm, confirmed-event gating); R-stable remains an honest
  negative (dual-context cost), documented with reproduction data in
  `docs/RESEARCH_V13.md`.
"""
from __future__ import annotations

from lab_env import DOORS

from metargon_v12 import MetargonV12
from metargon_v09 import atoms_of_state, plan_to_atoms
from temporal_memory import VERIFIED, PREDICTED


class MetargonV13(MetargonV12):
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
                 dual_context_mode: str = "dual",
                 use_adaptive_confirm: bool = True,
                 use_recent_credit: bool = False,
                 use_temporal: bool = False,
                 temporal_max: int = 64,
                 adaptive_confirmed_events: bool = False,
                 anomaly_threshold: float = 0.03,
                 temporal_completion_mode: str = "rolling",
                 temporal_support_weighting: str = "flat",
                 temporal_target_exempt: bool = False,
                 dual_persist_gate: int = 0):
        super().__init__(
            seed=seed, max_hypotheses=max_hypotheses,
            use_robust_belief=use_robust_belief,
            use_cooldowns=use_cooldowns,
            use_explore_fallback=use_explore_fallback,
            use_replay=use_replay,
            replay_mode=replay_mode,
            use_confirmed_events=use_confirmed_events,
            use_certainty=use_certainty,
            use_dual_context=use_dual_context,
            dual_context_mode=dual_context_mode,
            use_adaptive_confirm=use_adaptive_confirm,
            use_recent_credit=use_recent_credit,
            use_temporal=use_temporal,
            temporal_max=temporal_max,
            adaptive_confirmed_events=adaptive_confirmed_events,
            anomaly_threshold=anomaly_threshold,
            dual_persist_gate=dual_persist_gate,
        )
        self.dual_persist_gate = dual_persist_gate
        self.temporal_completion_mode = temporal_completion_mode
        self.temporal_support_weighting = temporal_support_weighting
        self.temporal_target_exempt = temporal_target_exempt
        if self.temporal is not None:
            self.temporal.completion_mode = temporal_completion_mode
            self.temporal.support_weighting = temporal_support_weighting

    # -- protection with provenance --------------------------------------
    def _protected_atoms_split(self):
        """Split protection into:
          * hold — atoms that keep an open, uncrossed door open (plus
            v0.10-style recent-causal-credit atoms): never exempted;
          * hyp — atoms that are antecedents of currently-satisfied best
            hypotheses: exemptable for temporal targeting.
        """
        hold = set()
        hyp = set()
        b = self.belief.belief
        now = atoms_of_state(b)
        for door in DOORS:
            h = self.memory.best_for_door(door, opens=True)
            if h is not None and h.antecedent <= now:
                hyp.update(h.antecedent)
            if b.door_open[door] and not self._passed[door]:
                hold.update(now)
            if self.use_recent_credit and b.door_open[door] and \
                    not self._passed[door]:
                recent = self._recent_protect.get(door)
                if recent is not None:
                    atoms, expiry = recent
                    if self.memory.step <= expiry:
                        hold.update(atoms)
        return frozenset(hold), frozenset(hyp)

    # -- temporal-augmented probing (with exemption) ----------------------
    def _choose_temporal_probe(self):
        if self.temporal is None:
            return None, None
        b = self.belief.belief
        hold_protected, hyp_protected = self._protected_atoms_split()

        def on_expand():
            self.planner_ops += 1
        # temporal hypothesis: probe its second atom (the completion step)
        best = None
        for d in DOORS:
            if b.door_open[d]:
                continue
            h = self.temporal.best_for_door(d)
            if h is None or h.status not in (VERIFIED, PREDICTED):
                continue
            # temporal objective: satisfy first, then second (ordered)
            now = atoms_of_state(b)
            if not h.first <= now:
                target = h.first
            else:
                target = h.second
            if target <= now:
                continue
            if self.temporal_target_exempt:
                protected = hold_protected | (hyp_protected - set(target))
            else:
                protected = hold_protected | hyp_protected
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

    def diagnostics(self) -> dict:
        d = super().diagnostics()
        d["temporal_completion_mode"] = self.temporal_completion_mode
        d["temporal_support_weighting"] = self.temporal_support_weighting
        d["temporal_target_exempt"] = self.temporal_target_exempt
        d["dual_persist_gate"] = self.dual_persist_gate
        return d
