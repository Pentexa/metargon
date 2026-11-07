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
from dataclasses import dataclass
from typing import Optional

MAX_HYPOTHESES = 128
VERIFY_SUPPORT = 3
MAX_CONTRADICT = 2
PREDICT_THRESHOLD = 1
CONTRADICT_WINDOW = 3
CANDIDATES_PER_EVENT = 10
PAIR_BASE = 5

HYPOTHESIS = "hypothesis"
PREDICTED = "predicted"
VERIFIED = "verified"
CONTRADICTED = "contradicted"

OPENS = "opens"
CLOSES = "closes"


def consequence_key(door: str, opened: bool) -> str:
    return f"door:{door}:{OPENS if opened else CLOSES}"


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

    def as_dict(self) -> dict:
        return {
            "antecedent": sorted(self.antecedent),
            "consequence": self.consequence,
            "sup": self.sup, "con": self.con,
            "status": self.status,
            "created_step": self.created_step,
            "last_update": self.last_update,
            "h_id": self.h_id,
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
                 dead_set_cap: int = 96):
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
            out.append(h)
        out.sort(key=lambda h: (-h.confidence, -h.sup))
        return out

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
        gate (contradiction_ratio). Uses windowed counts when enabled."""
        sup, con = self._windowed_counts(h)
        if self.contradiction_ratio <= 0.0:
            return True
        return con > 0 and con >= self.contradiction_ratio * max(sup, 0)

    def _evict_one(self):
        if not self.hypotheses:
            return
        victim = min(self.hypotheses,
                     key=lambda h: (h.score(), -h.last_update, -h.h_id))
        self.hypotheses.remove(victim)
        self._continuous_true.pop(victim.h_id, None)
        self._sup_timeline.pop(victim.h_id, None)
        self._con_timeline.pop(victim.h_id, None)
        self.evictions += 1

    def _gated_con(self, h: Hypothesis) -> int:
        """Contradiction count used for status (ratio-gated)."""
        sup, con = self._windowed_counts(h)
        if self.contradiction_ratio <= 0.0:
            return con
        if con > 0 and con >= self.contradiction_ratio * max(sup, 0):
            return con
        return 0

    def _recompute_status(self, h: Hypothesis):
        con = self._gated_con(h)
        sup, _ = self._windowed_counts(h)
        if con > self.max_contradict:
            h.status = CONTRADICTED
        elif sup >= self.verify_support and con <= self.max_contradict:
            h.status = VERIFIED
        elif sup >= self.predict_threshold:
            h.status = PREDICTED
        else:
            h.status = HYPOTHESIS

    def _support(self, h: Hypothesis):
        h.sup += 1
        h.last_update = self.step
        self._continuous_true.pop(h.h_id, None)
        self._record_evidence(h, "s")
        self._recompute_status(h)

    def _contradict(self, h: Hypothesis):
        h.con += 1
        h.last_update = self.step
        self._continuous_true.pop(h.h_id, None)
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
        }
