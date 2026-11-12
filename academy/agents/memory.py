"""METARGON Academy — compositional hypothesis memory.

The persistent learning state of the METARGON agent. It stores *candidate
causal rules* (conjunctive antecedents over the public atom vocabulary)
mapped to predicted door outcomes, and accumulates evidence from the
observable event stream:

    "with S1 on and P2 loaded, door D2 opens."

Evidence is bounded and hypothesis entries are evicted by a deterministic
score. The memory can be fully serialized/restored (checkpointing).

This is an Academy-native implementation: it shares only the design idea
of bounded compositional hypothesis memory with the v0.9 research line.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

MAX_HYPOTHESES = 128
VERIFY_SUPPORT = 3
MAX_CONTRADICT = 2
PREDICT_THRESHOLD = 1
CONTRADICT_WINDOW = 3
CANDIDATES_PER_EVENT = 10
PAIR_BASE = 5

# -- Cognitive Memory v0.2 (protect exploration) --------------------------
# v0.1 (PR #50, re-ranking) is treated as a negative finding: re-ranking
# all hypotheses by reliability starved exact novel-single discovery and
# broke repeated learning. v0.2 therefore changes *nothing* about the
# retrieval order (the ``epoch_reset`` order is kept). It adds instead a
# minimal, context-sensitive eligibility filter on hypothesis *reuse*:
#   * a hypothesis may be reused only when it has at least
#     ``protection_min_active_support`` support events in the *active*
#     evidence basis (per-epoch / per-context / cumulative) — otherwise it
#     is treated as unexplored and the agent explores;
#   * hypotheses with more than ``protection_max_failures`` repeated
#     failure windows are excluded from reuse (evidence kept; revived by
#     fresh support or a task boundary);
#   * at most ``protection_max_reuse_per_door`` hypotheses are returned
#     for probing per door, so remaining candidate slots always go to
#     exploration (novel singles, bank, atom transfer).
# All flags default off ⇒ baseline behavior, evidence and checkpoints are
# untouched.
PROTECT_EXPLORATION_DEFAULT = False
PROTECTION_MIN_ACTIVE_SUPPORT = 1
PROTECTION_MAX_FAILURES = 1
PROTECTION_MAX_REUSE_PER_DOOR = 1
PROTECTION_ALLOW_TRUSTED_TRANSFER_DEFAULT = False

HYPOTHESIS = "hypothesis"
PREDICTED = "predicted"
VERIFIED = "verified"
CONTRADICTED = "contradicted"

OPENS = "opens"
CLOSES = "closes"


def consequence_key(door: str, opened: bool) -> str:
    return f"door:{door}:{OPENS if opened else CLOSES}"


def ctx_key(ctx) -> Optional[str]:
    """Canonical string key for a context signature (None-safe)."""
    return None if ctx is None else repr(tuple(ctx))


def atom_text(atom) -> str:
    k = atom[0]
    if k == "load":
        return f"{atom[2]} on {atom[1]}"
    if k == "loadany":
        return f"anything on {atom[1]}"
    if k == "empty":
        return f"{atom[1]} empty"
    if k == "sw":
        return f"{atom[1]}={'on' if atom[2] else 'off'}"
    return str(atom)


def antecedent_text(ante) -> str:
    return " ∧ ".join(sorted(atom_text(a) for a in ante)) if ante else "(empty)"


@dataclass
class Hypothesis:
    antecedent: frozenset
    consequence: str
    sup: int = 0
    con: int = 0
    status: str = HYPOTHESIS
    created_step: int = 0
    last_update: int = 0
    h_id: int = 0
    # v0.4: context-scoped evidence (mapped by context-signature string)
    sup_by_ctx: dict = field(default_factory=dict)
    con_by_ctx: dict = field(default_factory=dict)
    # v0.7: episode-scoped evidence (mapped by episode index)
    sup_by_epoch: dict = field(default_factory=dict)
    con_by_epoch: dict = field(default_factory=dict)
    verified_epochs: list = field(default_factory=list)   # sorted ints
    # CM v0.2: consecutive completed failure windows since last support in
    # the current context (reset at task boundaries). Used only by the
    # protect-exploration eligibility filter; default 0 keeps legacy
    # checkpoints byte-compatible.
    consecutive_failures: int = 0

    @property
    def door(self) -> str:
        return self.consequence.split(":")[1]

    @property
    def predicted_opens(self) -> Optional[bool]:
        if ":opens" in self.consequence:
            return True
        if ":closes" in self.consequence:
            return False
        return None

    @property
    def confidence(self) -> float:
        return (self.sup + 1.0) / (self.sup + self.con + 2.0)

    @property
    def usable(self) -> bool:
        return self.status != CONTRADICTED

    def score(self) -> float:
        if self.status == CONTRADICTED:
            return -1.0 + self.con * 0.01
        base = self.confidence
        if self.status == VERIFIED:
            base += 10.0
        elif self.status == PREDICTED:
            base += 2.0
        return base - 0.0001 * self.last_update

    def ctx_support(self, ctx: str) -> int:
        return self.sup_by_ctx.get(ctx, 0)

    def ctx_contradiction(self, ctx: str) -> int:
        return self.con_by_ctx.get(ctx, 0)

    # -- v0.7 epoch-scoped evidence -------------------------------------
    def epoch_support(self, epoch: int) -> int:
        return self.sup_by_epoch.get(epoch, 0)

    def epoch_contradiction(self, epoch: int) -> int:
        return self.con_by_epoch.get(epoch, 0)

    @property
    def verified_epoch_count(self) -> int:
        return len(self.verified_epochs)

    def as_dict(self) -> dict:
        return {
            "antecedent": sorted(self.antecedent),
            "consequence": self.consequence,
            "sup": self.sup, "con": self.con,
            "status": self.status,
            "created_step": self.created_step,
            "last_update": self.last_update,
            "h_id": self.h_id,
            "sup_by_ctx": self.sup_by_ctx,
            "con_by_ctx": self.con_by_ctx,
            "sup_by_epoch": {str(k): v for k, v in self.sup_by_epoch.items()},
            "con_by_epoch": {str(k): v for k, v in self.con_by_epoch.items()},
            "verified_epochs": self.verified_epochs,
            "consecutive_failures": self.consecutive_failures,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Hypothesis":
        return cls(
            antecedent=frozenset(map(tuple, d["antecedent"])),
            consequence=d["consequence"],
            sup=d["sup"], con=d["con"],
            status=d["status"],
            created_step=d["created_step"],
            last_update=d["last_update"],
            h_id=d["h_id"],
            sup_by_ctx=dict(d.get("sup_by_ctx", {})),
            con_by_ctx=dict(d.get("con_by_ctx", {})),
            sup_by_epoch={int(k): v for k, v in d.get("sup_by_epoch", {}).items()},
            con_by_epoch={int(k): v for k, v in d.get("con_by_epoch", {}).items()},
            verified_epochs=list(d.get("verified_epochs", [])),
            consecutive_failures=int(d.get("consecutive_failures", 0)),
        )


class CompositionalMemory:
    """Bounded, evidence-driven hypothesis memory with checkpointing.

    v0.3 improvements (each independent and off by default):
      * ``recycle_contradicted`` — remove a hypothesis as soon as it is
        contradicted (instead of retaining it until score-eviction), and
        record its antecedent in a bounded per-door "dead" set so it is
        not regenerated immediately (contradiction-aware recycling).
      * ``contradiction_ratio`` — gate contradiction transitions by the
        support evidence: a hypothesis may only be *status*-contradicted
        once its contradiction count (within the evidence window, when
        enabled) reaches ``max_contradict`` AND either ``ratio <= 0`` or
        ``con >= ratio * sup``. This stops rare coincidental
        non-openings in other configs from flipping heavily-supported
        (true) hypotheses.
      * ``evidence_window`` — when >0, support/contradict counts used
        for status are recomputed over the last ``evidence_window``
        evidence events (sliding recency), instead of being cumulative
        for the lifetime of the hypothesis.

    v0.4 context-aware evidence (each independent and off by default):
      * ``context_scoped`` — evidence is recorded per task-context
        signature (set via ``set_context``). Status uses the *active*
        context's support/contradiction counts only, so a hypothesis
        that is correct in the context where it was learned is not
        destroyed by contradictions observed in other contexts
        (reversible confidence; no irreversible destruction).
      * ``task_change_reset`` — when the context changes, reset the
        consecutive-nonevent windows (``_continuous_true``). Prevents
        a task switch from counting as evidence against hypotheses
        from the previous task.
      * ``context_gate_transfer`` — only hypotheses with support in the
        *active* context may be considered usable/queryable. This is a
        conservative transfer gate: new contexts start with no borrowed
        knowledge until evidence arrives.

    v0.5 evidence bank (off by default):
      * ``evidence_bank_cap`` (default 0 = disabled) — maintain a
        compact bank of *verified* hypotheses (antecedent +
        consequence). Bank entries are used as high-priority probe
        candidates in new tasks (useful evidence reuse). The bank
        stores only antecedent/consequence pairs, is bounded, and is
        fully checkpointed. It never accesses hidden configuration.

    v0.6 transfer mechanisms (each independent and off by default):
      * ``atom_transfer`` — maintain a bounded per-door evidence
        profile over *atoms* derived from the antecedents of verified
        and contradicted hypotheses. Atoms that recur across tasks in
        verified rules accumulate support; atoms unique to contradicted
        rules accumulate contradiction. ``transfer_atoms`` returns the
        most cross-task-robust atoms for a door, which the agent can
        probe singly or as compact conjunctions — causal structure
        matching across tasks without replaying verbatim (fragile)
        antecedents.
      * ``bank_refresh_on_contradiction`` — when a banked hypothesis is
        contradicted, its bank entry is removed immediately (adaptive
        forgetting of genuinely invalid evidence; the bank only keeps
        currently verified entries).
      * ``effective_confidence`` — context-conditioned confidence for a
        hypothesis: per-context counts when context-scoped, otherwise
        cumulative; returns 0.0 when the active context has no local
        evidence (so stale cross-task hypotheses are not trusted
        verbatim in a new task unless they have support there).

    CM v0.2 (protect exploration; each off by default):
      * ``protect_exploration`` — do **not** re-rank hypotheses.
        Retrieval order is unchanged (the ``epoch_reset`` order).
        Instead, hypothesis *reuse* is gated by a minimal context-sensitive
        eligibility filter that guarantees exploration slots:
        - ``protection_min_active_support``: a hypothesis is eligible for
          reuse only once it has this many support events in the active
          evidence basis (per-epoch when ``episode_scoped``, per-context
          when ``context_scoped``, else cumulative). Without local
          evidence the hypothesis is treated as unexplored and the agent
          explores.
        - ``protection_max_failures``: hypotheses with more repeated
          failure windows than this are excluded from reuse (evidence
          kept; revived by support or a task boundary).
        - ``protection_allow_trusted_transfer``: hypotheses verified in
          >= ``min_verified_epochs_for_transfer`` past epochs and not
          currently under repeated-failure attack are also eligible
          (bounded trusted-transfer tier — the v0.7 cross-epoch
          retrieval, but capped by ``protection_max_reuse_per_door``).
        - ``protection_max_reuse_per_door``: at most this many reusable
          hypotheses are returned per door for probing, so the remaining
          candidate slots always go to exploration (novel singles, bank,
          atom transfer).
        This prevents conflicting or repeatedly failing memories from
        monopolizing the action budget without suppressing exploration.
      * ``CognitiveMemoryV02`` is a ready-made subclass: episode-scoped
        evidence + protect-exploration on (including the trusted-transfer
        tier).
    """

    def __init__(self, max_hypotheses: int = MAX_HYPOTHESES,
                 verify_support: int = VERIFY_SUPPORT,
                 max_contradict: int = MAX_CONTRADICT,
                 predict_threshold: int = PREDICT_THRESHOLD,
                 contradictict_window: int = CONTRADICT_WINDOW,
                 candidates_per_event: int = CANDIDATES_PER_EVENT,
                 pair_base: int = PAIR_BASE,
                 recycle_contradicted: bool = False,
                 contradiction_ratio: float = 0.0,
                 evidence_window: int = 0,
                 dead_set_cap: int = 96,
                 context_scoped: bool = False,
                 task_change_reset: bool = False,
                 context_gate_transfer: bool = False,
                 evidence_bank_cap: int = 0,
                 atom_transfer: bool = False,
                 bank_refresh_on_contradiction: bool = False,
                 episode_scoped: bool = False,
                 min_verified_epochs_for_transfer: int = 2,
                 protect_exploration: bool = PROTECT_EXPLORATION_DEFAULT,
                 protection_min_active_support: int = PROTECTION_MIN_ACTIVE_SUPPORT,
                 protection_max_failures: int = PROTECTION_MAX_FAILURES,
                 protection_max_reuse_per_door: int = PROTECTION_MAX_REUSE_PER_DOOR,
                 protection_allow_trusted_transfer: bool = PROTECTION_ALLOW_TRUSTED_TRANSFER_DEFAULT):
        self.max_hypotheses = max_hypotheses
        self.verify_support = verify_support
        self.max_contradict = max_contradict
        self.predict_threshold = predict_threshold
        self.contradict_window = contradictict_window
        self.candidates_per_event = candidates_per_event
        self.pair_base = pair_base
        self.recycle_contradicted = bool(recycle_contradicted)
        self.contradiction_ratio = float(contradiction_ratio)
        self.evidence_window = int(evidence_window)
        self.dead_set_cap = int(dead_set_cap)
        self.context_scoped = bool(context_scoped)
        self.task_change_reset = bool(task_change_reset)
        self.context_gate_transfer = bool(context_gate_transfer)
        self.evidence_bank_cap = int(evidence_bank_cap)
        self.atom_transfer = bool(atom_transfer)
        self.bank_refresh_on_contradiction = bool(
            bank_refresh_on_contradiction)
        self.episode_scoped = bool(episode_scoped)
        self.min_verified_epochs_for_transfer = int(
            min_verified_epochs_for_transfer)
        # CM v0.2
        self.protect_exploration = bool(protect_exploration)
        self.protection_min_active_support = max(
            0, int(protection_min_active_support))
        self.protection_max_failures = max(0, int(protection_max_failures))
        self.protection_max_reuse_per_door = max(
            1, int(protection_max_reuse_per_door))
        self.protection_allow_trusted_transfer = bool(
            protection_allow_trusted_transfer)
        self.reuse_filtered = 0
        self.reuse_eligible = 0
        self.current_ctx = None
        self._current_epoch = None
        self.hypotheses: list[Hypothesis] = []
        self.next_id = 1
        self.step = 0
        self.evictions = 0
        self.generated = 0
        self._continuous_true: dict[int, int] = {}
        # bounded per-door dead-antecedent set (contradiction recycling)
        self._dead: dict[str, deque] = {}
        # evidence timelines for windowed counts: h_id -> deque of 0/1
        self._sup_timeline: dict[int, deque] = {}
        self._con_timeline: dict[int, deque] = {}
        # v0.5 evidence bank: list of (antecedent, consequence_key)
        self.evidence_bank: list = []
        self.bank_retrievals = 0
        # v0.6 atom-transfer profile: door -> atom -> [sup, con, n_ctxs,
        # last_update]; bounded per door (FIFO by lowest score).
        self._atom_stats: dict = {}
        self._atom_ctxs: dict = {}
        self._ATOM_STATS_CAP = 32
        # v0.6 pair-co-occurrence profile: door -> pair -> [sup, con];
        # built from hypotheses with >=2 antecedent atoms (bounded).
        self._pair_stats: dict = {}
        self._PAIR_STATS_CAP = 16
        # v0.6 event counts per door (for conditional atom probability).
        self._door_event_count: dict = {}
        # v0.6 event-level support per (door, atom): counts only the
        # directly observed opening events, not hypothesis-derived credit.
        self._atom_event_support: dict = {}
        # v0.6 nonevent-level support per (door, atom): atoms present at
        # steps where the door stayed closed (used for lift filtering).
        self._atom_nonevent_support: dict = {}
        self._door_nonevent_count: dict = {}

    # -- context (v0.4) -----------------------------------------------------
    def set_context(self, ctx: Optional[tuple]) -> None:
        """Declare the current task-context signature (public-observation
        derived). With ``task_change_reset``, a change resets consecutive-
        nonevent windows so a task switch cannot count as evidence."""
        key = ctx_key(ctx)
        if self.current_ctx == key:
            return
        if self.current_ctx is not None and self.task_change_reset:
            self._continuous_true.clear()
        self.current_ctx = key

    def begin_episode(self, epoch: int) -> None:
        """Declare a new episode (task) boundary.

        With ``episode_scoped``, each episode is an evidence epoch:
        * the consecutive-nonevent windows are reset, so a task switch
          can never be read as latent contradiction;
        * support / contradiction are recorded per epoch and status is
          computed relative to the active epoch only (cross-epoch
          contradictions demote but do not destroy a hypothesis);
        * every hypothesis status is recomputed against the fresh epoch
          (no evidence yet => neutral), so a hypothesis contradicted in
          a past epoch can be retrieved again as a cross-epoch transfer
          candidate in the new task.
        """
        self._current_epoch = int(epoch)
        if self.protect_exploration:
            # CM v0.2: a task boundary resets *current-context* failure
            # streaks. Historical verification is untouched, so old
            # hypotheses start neutral in the new task.
            for h in self.hypotheses:
                h.consecutive_failures = 0
        if self.episode_scoped:
            self._continuous_true.clear()
            for h in self.hypotheses:
                self._recompute_status(h)

    def _epoch_counts(self, h: Hypothesis) -> tuple:
        """(sup, con) used for status in episode-scoped mode."""
        ep = self._current_epoch
        if ep is None:
            return h.sup, h.con
        return (h.epoch_support(ep), h.epoch_contradiction(ep))

    def _ctx_counts(self, h: Hypothesis) -> tuple:
        """(sup, con) used for status: per-context when context_scoped,
        per-epoch when episode_scoped, otherwise global cumulative (with
        evidence-window override)."""
        if self.episode_scoped:
            return self._epoch_counts(h)
        if self.context_scoped and self.current_ctx is not None:
            return (h.ctx_support(self.current_ctx),
                    h.ctx_contradiction(self.current_ctx))
        return self._windowed_counts(h)

    # -- introspection ----------------------------------------------------
    def __len__(self) -> int:
        return len(self.hypotheses)

    def query(self, door: str, opens: Optional[bool] = None) -> list[Hypothesis]:
        out = []
        for h in self.hypotheses:
            if h.door != door or not h.usable:
                continue
            if opens is not None and h.predicted_opens != opens:
                continue
            if (self.context_gate_transfer and self.context_scoped
                    and self.current_ctx is not None
                    and h.ctx_support(self.current_ctx) <= 0):
                continue
            out.append(h)
        if self.protect_exploration and opens is not None:
            # CM v0.2: eligibility filter on *probe retrieval* only
            # (``opens`` specified). Protection/other queries (opens=None)
            # are untouched.
            eligible = []
            for h in out:
                if self._reuse_eligible(h):
                    eligible.append(h)
                else:
                    self.reuse_filtered += 1
            out = eligible
            self.reuse_eligible += len(out)
        if self.episode_scoped:
            out.sort(key=lambda h: (-self._retrieval_key(h), h.h_id))
        else:
            out.sort(key=lambda h: (-h.confidence, -h.sup))
        if self.protect_exploration and opens is not None:
            # cap AFTER sorting: return the top-N reusable hypotheses by
            # the baseline order (never a ranking change).
            out = out[:self.protection_max_reuse_per_door]
        return out

    def _active_support(self, h: Hypothesis) -> int:
        """Support in the *active* evidence basis (epoch > context >
        cumulative windowed), used by the v0.2 eligibility filter."""
        sup, _ = self._ctx_counts(h)
        return sup

    def _reuse_eligible(self, h: Hypothesis) -> bool:
        """Context-sensitive eligibility for hypothesis reuse (v0.2).

        A hypothesis is reusable only if it has accumulated at least
        ``protection_min_active_support`` support events in the active
        evidence basis AND is not under repeated-failure attack. This is
        a *gate*, not a ranking change: eligible hypotheses keep the
        baseline retrieval order.
        """
        if h.consecutive_failures > self.protection_max_failures:
            return False
        if self._active_support(h) >= self.protection_min_active_support:
            return True
        if (self.protection_allow_trusted_transfer
                and h.verified_epoch_count
                >= self.min_verified_epochs_for_transfer):
            # bounded trusted-transfer tier: historically verified across
            # >= K epochs, not under repeated-failure attack, and untested
            # in the active epoch -> eligible as a *transfer candidate*
            # (still subject to the per-door cap).
            return True
        return False

    def _retrieval_key(self, h: Hypothesis) -> float:
        """Retrieval priority in episode-scoped mode.

        1. VERIFIED in the active epoch;
        2. PREDICTED in the active epoch;
        3. cross-epoch transfer candidates — verified in >=
           ``min_verified_epochs_for_transfer`` past epochs and untested
           in the active epoch (uncertainty-aware reuse);
        4. everything else usable.
        """
        ep = self._current_epoch
        if h.status == VERIFIED:
            base = 100.0
        elif h.status == PREDICTED:
            base = 50.0
        else:
            base = 0.0
        if ep is not None and h.epoch_support(ep) == 0 \
                and h.epoch_contradiction(ep) == 0 \
                and h.verified_epoch_count >= self.min_verified_epochs_for_transfer:
            # untested cross-epoch verified hypothesis
            base = max(base, 20.0)
        return base + h.confidence

    def best_for_door(self, door: str, opens: Optional[bool] = None) -> Optional[Hypothesis]:
        q = self.query(door, opens)
        return q[0] if q else None

    # -- evidence update --------------------------------------------------
    def _find(self, antecedent, consequence) -> Optional[Hypothesis]:
        for h in self.hypotheses:
            if h.antecedent == antecedent and h.consequence == consequence:
                return h
        return None

    def _add(self, antecedent, consequence) -> Optional[Hypothesis]:
        existing = self._find(antecedent, consequence)
        if existing is not None:
            return existing
        if len(self.hypotheses) >= self.max_hypotheses:
            self._evict_one()
        h = Hypothesis(antecedent, consequence, created_step=self.step,
                       last_update=self.step, h_id=self.next_id)
        self.next_id += 1
        self.hypotheses.append(h)
        self.generated += 1
        return h

    # -- contradiction-aware dead set --------------------------------------
    def _dead_set(self, door: str) -> deque:
        dq = self._dead.get(door)
        if dq is None:
            dq = deque(maxlen=self.dead_set_cap)
            self._dead[door] = dq
        return dq

    def _is_dead(self, antecedent, door: str) -> bool:
        return antecedent in self._dead_set(door)

    def _mark_dead(self, h: Hypothesis) -> None:
        """Recycle a contradicted hypothesis: remove it and record its
        antecedent in the per-door dead set (bounded, FIFO)."""
        dq = self._dead_set(h.door)
        if h.antecedent not in dq:
            dq.append(h.antecedent)
        self._continuous_true.pop(h.h_id, None)
        self._sup_timeline.pop(h.h_id, None)
        self._con_timeline.pop(h.h_id, None)
        if h in self.hypotheses:
            self.hypotheses.remove(h)

    # -- evidence timeline (windowed counts) -------------------------------
    def _record_evidence(self, h: Hypothesis, kind: str) -> None:
        if self.episode_scoped and self._current_epoch is not None:
            ep = self._current_epoch
            if kind == "s":
                h.sup_by_epoch[ep] = h.sup_by_epoch.get(ep, 0) + 1
            else:
                h.con_by_epoch[ep] = h.con_by_epoch.get(ep, 0) + 1
        if self.context_scoped and self.current_ctx is not None:
            if kind == "s":
                h.sup_by_ctx[self.current_ctx] = h.sup_by_ctx.get(
                    self.current_ctx, 0) + 1
            else:
                h.con_by_ctx[self.current_ctx] = h.con_by_ctx.get(
                    self.current_ctx, 0) + 1
            return
        if self.episode_scoped:
            return
        if self.evidence_window <= 0:
            return
        tl = (self._sup_timeline if kind == "s" else self._con_timeline)
        dq = tl.get(h.h_id)
        if dq is None:
            dq = deque(maxlen=self.evidence_window)
            tl[h.h_id] = dq
        dq.append(1 if kind == "s" else 0)

    def _windowed_counts(self, h: Hypothesis) -> tuple:
        if self.evidence_window <= 0 or (h.h_id not in self._sup_timeline
                                         and h.h_id not in self._con_timeline):
            return h.sup, h.con
        return (sum(self._sup_timeline.get(h.h_id, ())),
                sum(self._con_timeline.get(h.h_id, ())))

    def _contradiction_gated(self, h: Hypothesis) -> bool:
        """Whether a transition to contradicted is *allowed* by the ratio
        gate (contradiction_ratio). Uses per-context or windowed counts."""
        sup, con = self._ctx_counts(h)
        if self.contradiction_ratio <= 0.0:
            return True
        return con > 0 and con >= self.contradiction_ratio * max(sup, 0)

    # -- v0.6: context-conditioned confidence -----------------------------
    def effective_confidence(self, h: Hypothesis) -> float:
        """Context-conditioned confidence for a hypothesis.

        With ``context_scoped``, the *active* context's counts decide;
        a context with zero local evidence yields confidence 0.0 (the
        hypothesis is not trusted verbatim in a new task). Otherwise
        the global cumulative counts are used.
        """
        if self.context_scoped and self.current_ctx is not None:
            sup = h.ctx_support(self.current_ctx)
            con = h.ctx_contradiction(self.current_ctx)
            if sup == 0 and con == 0:
                return 0.0
            return (sup + 1.0) / (sup + con + 2.0)
        sup, con = self._ctx_counts(h)
        return (sup + 1.0) / (sup + con + 2.0)

    # -- v0.6: atom-level cross-task evidence ------------------------------
    def _note_atom_evidence(self, h: Hypothesis, sign: int) -> None:
        """Update the cross-task atom profile from a verified (+1) or
        contradicted (−1) hypothesis. Bounded per door."""
        if h.antecedent is None:
            return
        stats = self._atom_stats.setdefault(h.door, {})
        ctx = self.current_ctx
        atoms = sorted(h.antecedent)
        for i, a in enumerate(atoms):
            e = stats.get(a)
            if e is None:
                if len(stats) >= self._ATOM_STATS_CAP:
                    victim = min(
                        stats,
                        key=lambda k: (self.atom_score(stats[k]), str(k)))
                    del stats[victim]
                    self._atom_ctxs.pop((h.door, victim), None)
                e = [0, 0, 0, self.step]
                stats[a] = e
            if sign > 0:
                e[0] += 1
            else:
                e[1] += 1
            if ctx is not None:
                key = (h.door, a)
                s = self._atom_ctxs.setdefault(key, set())
                if len(s) < 8 and ctx not in s:
                    s.add(ctx)
                e[2] = len(s)
            e[3] = self.step
        # pair co-occurrence (for conjunctive structure transfer)
        if len(atoms) >= 2:
            pairs = self._pair_stats.setdefault(h.door, {})
            for i in range(len(atoms)):
                for j in range(i + 1, len(atoms)):
                    pair = frozenset({atoms[i], atoms[j]})
                    pe = pairs.get(pair)
                    if pe is None:
                        if len(pairs) >= self._PAIR_STATS_CAP:
                            victim = min(pairs, key=lambda p: (pairs[p][1]
                                                               - pairs[p][0],
                                                               p))
                            del pairs[victim]
                        pe = [0, 0]
                        pairs[pair] = pe
                    if sign > 0:
                        pe[0] += 1
                    else:
                        pe[1] += 1

    def transfer_pairs(self, door: str, k: int = 4) -> list:
        """Top-k atom pairs for a door by verified co-occurrence (sup > con).
        These are compact conjunctive structure candidates. Pairs that are
        self-contradictory (e.g. S2 on ∧ S2 off) are never proposed."""
        pairs = self._pair_stats.get(door, {})
        pos = {}
        for p, e in pairs.items():
            if e[0] <= e[1]:
                continue
            atoms = list(p)
            if len(atoms) == 2 and atoms[0][0] == atoms[1][0] \
                    and atoms[0][1] == atoms[1][1] \
                    and atoms[0][2] != atoms[1][2]:
                continue
            pos[p] = e
        ranked = sorted(pos.items(),
                        key=lambda kv: (-kv[1][0], kv[0]))
        return [p for p, _ in ranked[:k]]

    def atom_score(self, e) -> float:
        """Cross-task robustness of an atom: confidence × context spread."""
        sup, con, n_ctxs, _ = e
        conf = (sup + 1.0) / (sup + con + 2.0)
        return conf * (1.0 + 0.25 * min(n_ctxs, 8))

    def _note_event_atoms(self, atoms: frozenset, door: str) -> None:
        """Event-level positive atom evidence: every atom present *before*
        a door-opening event is supported for that door. This is much
        denser than hypothesis-verification-derived evidence (which is
        nearly absent because few hypotheses reach VERIFIED).
        ``transfer_atoms`` converts these counts to a conditional
        probability (support / door event count) so that atoms that are
        merely often co-present at openings are not over-trusted."""
        if not atoms:
            return
        self._door_event_count[door] = self._door_event_count.get(door, 0) + 1
        stats = self._atom_stats.setdefault(door, {})
        ctx = self.current_ctx
        for a in atoms:
            e = stats.get(a)
            if e is None:
                if len(stats) >= self._ATOM_STATS_CAP:
                    victim = min(
                        stats,
                        key=lambda k: (self.atom_score(stats[k]), str(k)))
                    del stats[victim]
                    self._atom_ctxs.pop((door, victim), None)
                e = [0, 0, 0, self.step]
                stats[a] = e
            e[0] += 1
            key_e = (door, a)
            self._atom_event_support[key_e] = \
                self._atom_event_support.get(key_e, 0) + 1
            if ctx is not None:
                key = (door, a)
                s = self._atom_ctxs.setdefault(key, set())
                if len(s) < 8 and ctx not in s:
                    s.add(ctx)
                e[2] = len(s)
            e[3] = self.step

    def transfer_atoms(self, door: str, k: int = 6,
                       min_prob: float = 0.25) -> list:
        """Top-k atoms for a door by cross-task evidence (causal structure
        matching). With event-level data, an atom must be present at
        openings at least ``min_prob`` and strictly more often than at
        closed-door steps (lift over the standing-state baseline), with
        net positive hypothesis-derived evidence. Without event-level
        data, only the net positive hypothesis-derived evidence is used
        (fallback for small/hypothesis-only profiles)."""
        stats = self._atom_stats.get(door, {})
        n_events = self._door_event_count.get(door, 0)
        n_nonevents = self._door_nonevent_count.get(door, 0)
        pos = {}
        for a, e in stats.items():
            if e[0] <= e[1]:
                continue
            if n_events > 0:
                ev_sup = self._atom_event_support.get((door, a), 0)
                p_open = ev_sup / n_events
                if p_open < min_prob:
                    continue
                if n_nonevents > 0:
                    p_closed = self._atom_nonevent_support.get((door, a), 0) \
                        / n_nonevents
                    if p_open <= p_closed:
                        continue
            pos[a] = e
        ranked = sorted(pos.items(),
                        key=lambda kv: (-kv[1][0] / max(1, n_events),
                                        kv[0]))
        return [a for a, _ in ranked[:k]]

    def atom_profile_size(self) -> int:
        return sum(len(s) for s in self._atom_stats.values())

    def atom_profile(self) -> dict:
        """Diagnostics for the atom-transfer profile (bounded)."""
        return {
            door: {str(sorted(list(a))): list(e)
                   for a, e in stats.items()}
            for door, stats in self._atom_stats.items()
        }

    def _epoch_eviction_score(self, h: Hypothesis) -> float:
        """Eviction priority in episode-scoped mode: hypotheses verified
        across more epochs and with higher confidence are retained; a
        hypothesis contradicted in the active epoch is evicted first."""
        if h.status == CONTRADICTED:
            con_active = h.epoch_contradiction(self._current_epoch) \
                if self._current_epoch is not None else h.con
            return -10.0 + con_active * 0.01
        return (h.verified_epoch_count * 5.0 + h.confidence
                - 0.0001 * h.last_update)

    def _evict_one(self):
        if not self.hypotheses:
            return
        if self.episode_scoped:
            victim = min(self.hypotheses,
                         key=lambda h: (self._epoch_eviction_score(h),
                                        -h.last_update, -h.h_id))
        else:
            victim = min(self.hypotheses,
                         key=lambda h: (h.score(), -h.last_update, -h.h_id))
        self.hypotheses.remove(victim)
        self._continuous_true.pop(victim.h_id, None)
        self._sup_timeline.pop(victim.h_id, None)
        self._con_timeline.pop(victim.h_id, None)
        self.evictions += 1

    def _gated_con(self, h: Hypothesis) -> int:
        """Contradiction count used for status (ratio-gated, per-context
        when context_scoped)."""
        sup, con = self._ctx_counts(h)
        if self.contradiction_ratio <= 0.0:
            return con
        if con > 0 and con >= self.contradiction_ratio * max(sup, 0):
            return con
        return 0

    def _recompute_status(self, h: Hypothesis):
        con = self._gated_con(h)
        sup, _ = self._ctx_counts(h)
        before = h.status
        if con > self.max_contradict:
            h.status = CONTRADICTED
        elif sup >= self.verify_support and con <= self.max_contradict:
            h.status = VERIFIED
        elif sup >= self.predict_threshold:
            h.status = PREDICTED
        else:
            h.status = HYPOTHESIS
        if h.status == VERIFIED and before != VERIFIED:
            # record the epoch in which this hypothesis became verified
            # (episode-scoped mode); bounded, sorted, deduplicated.
            if self._current_epoch is not None:
                ep = self._current_epoch
                if ep not in h.verified_epochs:
                    h.verified_epochs.append(ep)
                    h.verified_epochs.sort()
                    if len(h.verified_epochs) > 512:
                        h.verified_epochs = h.verified_epochs[-512:]
        if self.atom_transfer:
            if before != VERIFIED and h.status == VERIFIED:
                self._note_atom_evidence(h, +1)
            elif before != CONTRADICTED and h.status == CONTRADICTED:
                self._note_atom_evidence(h, -1)
        if self.evidence_bank_cap > 0 and h.status == VERIFIED:
            self._bank_verified(h)
        elif (self.evidence_bank_cap > 0
                and self.bank_refresh_on_contradiction
                and h.status == CONTRADICTED):
            self._bank_remove(h)

    def _bank_verified(self, h: Hypothesis) -> None:
        """Add a verified hypothesis to the compact evidence bank
        (bounded, deduplicated by (antecedent, consequence))."""
        entry = (h.antecedent, h.consequence)
        for i, (ante, cons) in enumerate(self.evidence_bank):
            if ante == entry[0] and cons == entry[1]:
                return
        self.evidence_bank.append(entry)
        if len(self.evidence_bank) > self.evidence_bank_cap:
            self.evidence_bank.pop(0)   # FIFO eviction

    def _bank_remove(self, h: Hypothesis) -> None:
        """Remove a bank entry (adaptive forgetting of invalid evidence)."""
        self.evidence_bank = [
            (ante, cons) for ante, cons in self.evidence_bank
            if not (ante == h.antecedent and cons == h.consequence)
        ]

    def bank_entries(self) -> list:
        return list(self.evidence_bank)

    def mark_retrieval(self) -> None:
        self.bank_retrievals += 1

    def _support(self, h: Hypothesis, record_ctx: bool = True):
        h.sup += 1
        h.last_update = self.step
        self._continuous_true.pop(h.h_id, None)
        if self.protect_exploration:
            # any support ends the repeated-failure streak
            h.consecutive_failures = 0
        if record_ctx:
            self._record_evidence(h, "s")
        self._recompute_status(h)

    def _contradict(self, h: Hypothesis, record_ctx: bool = True):
        h.con += 1
        h.last_update = self.step
        self._continuous_true.pop(h.h_id, None)
        if self.protect_exploration:
            # one more completed failure window (repeated prediction failure)
            h.consecutive_failures += 1
        if record_ctx:
            self._record_evidence(h, "c")
        self._recompute_status(h)
        if self.recycle_contradicted and h.status == CONTRADICTED:
            self._mark_dead(h)

    def _candidate_antecedents(self, atoms: frozenset) -> list:
        base = sorted(atoms, key=str)[:self.pair_base]
        cands = [frozenset({a}) for a in base]
        for i in range(len(base)):
            for j in range(i + 1, len(base)):
                if len(cands) >= self.candidates_per_event:
                    break
                cands.append(frozenset({base[i], base[j]}))
            if len(cands) >= self.candidates_per_event:
                break
        return cands

    def observe_event(self, before_atoms: frozenset, door: str, opened: bool,
                      after_atoms: Optional[frozenset] = None):
        """Called when a door's state visibly changed this step. Candidate
        hypotheses are generated from the context *before* the event and
        from the *after* state (for persistent doors the true cause often
        only appears in the after-state)."""
        consequence = consequence_key(door, opened)
        if opened and self.atom_transfer:
            # event-level positive atom evidence (dense; not dependent on
            # hypothesis verification, which rarely happens)
            self._note_event_atoms(before_atoms, door)
        contexts = [before_atoms]
        if after_atoms is not None:
            contexts.append(after_atoms)
        seen = set()
        for atoms in contexts:
            for ante in self._candidate_antecedents(atoms):
                if ante in seen:
                    continue
                seen.add(ante)
                if self._is_dead(ante, door):
                    continue
                h = self._add(ante, consequence)
                if h is not None and ante <= atoms:
                    self._support(h)

    def observe_maintain(self, atoms: frozenset, door: str) -> None:
        """Called every step a door stays open. Hypotheses predicting the
        open state whose antecedent continuously holds are supported
        (persistent-cause inductive bias)."""
        consequence = consequence_key(door, True)
        for h in self.hypotheses:
            if h.door != door or h.consequence != consequence or not h.usable:
                continue
            if h.antecedent <= atoms:
                c = self._continuous_true.get(h.h_id, 0) + 1
                self._continuous_true[h.h_id] = c
                if c >= 1:
                    self._support(h)
            else:
                self._continuous_true.pop(h.h_id, None)

    def observe_nonevent(self, atoms: frozenset, door: str, wants_open: bool):
        consequence = consequence_key(door, wants_open)
        if wants_open and self.atom_transfer and atoms:
            self._door_nonevent_count[door] = \
                self._door_nonevent_count.get(door, 0) + 1
            for a in atoms:
                key = (door, a)
                self._atom_nonevent_support[key] = \
                    self._atom_nonevent_support.get(key, 0) + 1
        for h in list(self.hypotheses):
            if h.door != door or h.consequence != consequence or not h.usable:
                continue
            if h.antecedent <= atoms:
                c = self._continuous_true.get(h.h_id, 0) + 1
                self._continuous_true[h.h_id] = c
                if c >= self.contradict_window:
                    self._contradict(h)
            else:
                self._continuous_true.pop(h.h_id, None)

    def step_advance(self):
        self.step += 1

    # -- checkpointing ----------------------------------------------------
    def save_state(self) -> dict:
        return {
            "max_hypotheses": self.max_hypotheses,
            "verify_support": self.verify_support,
            "max_contradict": self.max_contradict,
            "predict_threshold": self.predict_threshold,
            "contradict_window": self.contradict_window,
            "candidates_per_event": self.candidates_per_event,
            "pair_base": self.pair_base,
            "recycle_contradicted": self.recycle_contradicted,
            "contradiction_ratio": self.contradiction_ratio,
            "evidence_window": self.evidence_window,
            "dead_set_cap": self.dead_set_cap,
            "context_scoped": self.context_scoped,
            "task_change_reset": self.task_change_reset,
            "context_gate_transfer": self.context_gate_transfer,
            "evidence_bank_cap": self.evidence_bank_cap,
            "atom_transfer": self.atom_transfer,
            "bank_refresh_on_contradiction": self.bank_refresh_on_contradiction,
            "episode_scoped": self.episode_scoped,
            "min_verified_epochs_for_transfer":
                self.min_verified_epochs_for_transfer,
            "protect_exploration": self.protect_exploration,
            "protection_min_active_support": self.protection_min_active_support,
            "protection_max_failures": self.protection_max_failures,
            "protection_max_reuse_per_door":
                self.protection_max_reuse_per_door,
            "protection_allow_trusted_transfer":
                self.protection_allow_trusted_transfer,
            "current_epoch": self._current_epoch,
            "current_ctx": self.current_ctx,
            "evidence_bank": [
                [sorted(list(ante)), cons] for ante, cons in self.evidence_bank
            ],
            "bank_retrievals": self.bank_retrievals,
            "atom_stats": {
                door: {tuple(a): e for a, e in stats.items()}
                for door, stats in self._atom_stats.items()
            },
            "atom_ctxs": {
                f"{door}|{tuple(a)}": sorted(s)
                for (door, a), s in self._atom_ctxs.items()
            },
            "pair_stats": {
                door: {tuple(sorted(p)): e for p, e in pairs.items()}
                for door, pairs in self._pair_stats.items()
            },
            "door_event_count": dict(self._door_event_count),
            "atom_event_support": {
                f"{door}|{tuple(a)}": c
                for (door, a), c in self._atom_event_support.items()
            },
            "atom_nonevent_support": {
                f"{door}|{tuple(a)}": c
                for (door, a), c in self._atom_nonevent_support.items()
            },
            "door_nonevent_count": dict(self._door_nonevent_count),
            "hypotheses": [h.as_dict() for h in self.hypotheses],
            "next_id": self.next_id,
            "step": self.step,
            "evictions": self.evictions,
            "generated": self.generated,
            "continuous_true": {str(k): v for k, v in self._continuous_true.items()},
            "dead": {door: [sorted(list(a)) for a in dq]
                     for door, dq in self._dead.items()},
            "sup_timeline": {str(k): list(v) for k, v in self._sup_timeline.items()},
            "con_timeline": {str(k): list(v) for k, v in self._con_timeline.items()},
        }

    def load_state(self, state: dict) -> None:
        self.max_hypotheses = state["max_hypotheses"]
        self.verify_support = state["verify_support"]
        self.max_contradict = state["max_contradict"]
        self.predict_threshold = state["predict_threshold"]
        self.contradict_window = state["contradict_window"]
        self.candidates_per_event = state["candidates_per_event"]
        self.pair_base = state["pair_base"]
        self.recycle_contradicted = bool(state.get("recycle_contradicted", False))
        self.contradiction_ratio = float(state.get("contradiction_ratio", 0.0))
        self.evidence_window = int(state.get("evidence_window", 0))
        self.dead_set_cap = int(state.get("dead_set_cap", 96))
        self.context_scoped = bool(state.get("context_scoped", False))
        self.task_change_reset = bool(state.get("task_change_reset", False))
        self.context_gate_transfer = bool(state.get("context_gate_transfer", False))
        self.episode_scoped = bool(state.get("episode_scoped", False))
        self.min_verified_epochs_for_transfer = int(state.get(
            "min_verified_epochs_for_transfer", 2))
        self.protect_exploration = bool(state.get(
            "protect_exploration", PROTECT_EXPLORATION_DEFAULT))
        self.protection_min_active_support = max(0, int(state.get(
            "protection_min_active_support", PROTECTION_MIN_ACTIVE_SUPPORT)))
        self.protection_max_failures = max(0, int(state.get(
            "protection_max_failures", PROTECTION_MAX_FAILURES)))
        self.protection_max_reuse_per_door = max(1, int(state.get(
            "protection_max_reuse_per_door",
            PROTECTION_MAX_REUSE_PER_DOOR)))
        self.protection_allow_trusted_transfer = bool(state.get(
            "protection_allow_trusted_transfer",
            PROTECTION_ALLOW_TRUSTED_TRANSFER_DEFAULT))
        self.reuse_filtered = 0
        self.reuse_eligible = 0
        self._current_epoch = state.get("current_epoch")
        self.current_ctx = state.get("current_ctx")
        self.evidence_bank_cap = int(state.get("evidence_bank_cap", 0))
        self.atom_transfer = bool(state.get("atom_transfer", False))
        self.bank_refresh_on_contradiction = bool(state.get(
            "bank_refresh_on_contradiction", False))
        self.evidence_bank = [
            (frozenset(map(tuple, ante)), cons)
            for ante, cons in state.get("evidence_bank", [])
        ]
        self.bank_retrievals = int(state.get("bank_retrievals", 0))
        self._atom_stats = {
            door: {tuple(a): list(e) for a, e in stats.items()}
            for door, stats in state.get("atom_stats", {}).items()
        }
        self._atom_ctxs = {
            (door, tuple(a)): set(s)
            for k, s in state.get("atom_ctxs", {}).items()
            for door, a in [k.split("|", 1)]
        }
        self._pair_stats = {
            door: {frozenset(p): list(e) for p, e in pairs.items()}
            for door, pairs in state.get("pair_stats", {}).items()
        }
        self._door_event_count = {
            str(d): int(c) for d, c in state.get("door_event_count", {}).items()
        }
        self._atom_event_support = {
            (door, tuple(a)): int(c)
            for k, c in state.get("atom_event_support", {}).items()
            for door, a in [k.split("|", 1)]
        }
        self._atom_nonevent_support = {
            (door, tuple(a)): int(c)
            for k, c in state.get("atom_nonevent_support", {}).items()
            for door, a in [k.split("|", 1)]
        }
        self._door_nonevent_count = {
            str(d): int(c)
            for d, c in state.get("door_nonevent_count", {}).items()
        }
        self.hypotheses = [Hypothesis.from_dict(d) for d in state["hypotheses"]]
        self.next_id = state["next_id"]
        self.step = state["step"]
        self.evictions = state["evictions"]
        self.generated = state["generated"]
        self._continuous_true = {int(k): v for k, v in state["continuous_true"].items()}
        self._dead = {}
        for door, rows in state.get("dead", {}).items():
            self._dead[door] = deque(
                (frozenset(map(tuple, a)) for a in rows),
                maxlen=self.dead_set_cap)
        self._sup_timeline = {
            int(k): deque(v, maxlen=self.evidence_window)
            for k, v in state.get("sup_timeline", {}).items()}
        self._con_timeline = {
            int(k): deque(v, maxlen=self.evidence_window)
            for k, v in state.get("con_timeline", {}).items()}

    def diagnostics(self) -> dict:
        return {
            "size": len(self.hypotheses),
            "max": self.max_hypotheses,
            "evictions": self.evictions,
            "generated": self.generated,
            "step": self.step,
            "status_counts": {
                s: sum(1 for h in self.hypotheses if h.status == s)
                for s in (HYPOTHESIS, PREDICTED, VERIFIED, CONTRADICTED)
            },
            "bank_size": len(self.evidence_bank),
            "bank_retrievals": self.bank_retrievals,
            "atom_profile_size": sum(
                len(stats) for stats in self._atom_stats.values()),
            "current_epoch": self._current_epoch,
            "episode_scoped": self.episode_scoped,
            "verified_hypotheses": sum(
                1 for h in self.hypotheses if h.status == VERIFIED),
            "cross_epoch_verified": sum(
                1 for h in self.hypotheses
                if h.verified_epoch_count >= self.min_verified_epochs_for_transfer),
            "protect_exploration": self.protect_exploration,
            "reuse_eligible": self.reuse_eligible,
            "reuse_filtered": self.reuse_filtered,
            "reuse_filtered_streak": sum(
                1 for h in self.hypotheses
                if h.consecutive_failures > self.protection_max_failures),
        }


class CognitiveMemoryV02(CompositionalMemory):
    """Cognitive Memory v0.2: protect exploration.

    Episode-scoped evidence (v0.7) + the v0.2 eligibility filter/cap for
    hypothesis reuse. Construct via ``MetargonAcademyAgent(memory_factory=
    CognitiveMemoryV02, memory_episode_scoped=True,
    probe_reset_per_episode=True)`` — no agent changes required.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("protect_exploration", True)
        kwargs.setdefault("episode_scoped", True)
        kwargs.setdefault("protection_allow_trusted_transfer", True)
        super().__init__(*args, **kwargs)
