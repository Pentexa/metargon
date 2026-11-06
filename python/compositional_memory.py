"""METARGON v0.9 — Phase C: compositional causal memory.

A compact, bounded memory that stores *hypotheses* about the hidden wiring
of the mechanism lab, not raw action-success memories. A hypothesis is a
conjunctive antecedent (a set of grounded atoms over plates/switches) mapped
to a predicted outcome for a door:

    "Moving object A onto mechanism P1 while S1 is on opens door D1."

The memory distinguishes, explicitly:

    OBSERVATION     a raw event observed this step
    HYPOTHESIS      a candidate causal rule, generated from a co-occurrence
    PREDICTED       the hypothesis is used by the planner (sup >= 1)
    VERIFIED        sufficient supporting evidence, no contradiction
    CONTRADICTED    contradicting evidence; not used for planning

Evidence is accumulated from experience. Hypotheses do not automatically
become facts: they require >= VERIFY_SUPPORT supporting events and <=
MAX_CONTRADICT contradicting events. Memory is bounded (MAX_HYPOTHESES)
with a deterministic score-based eviction policy.

The memory is deliberately **hypothesis-agnostic about the environment's
generative family**: it composes atom-level fragments and never enumerates
complete wirings (unlike the Bayesian baseline, which is given the family).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from lab_env import DOORS, PLATES, SWITCHES

# --------------------------------------------------------------------------
# Pre-registered constants (hyperparameters fixed before evaluation; not
# tuned on held-out data).
# --------------------------------------------------------------------------
MAX_HYPOTHESES = 128        # hard memory budget
VERIFY_SUPPORT = 3          # supporting events for a hypothesis to be verified
MAX_CONTRADICT = 2          # contradicting events invalidate (<=) or demote
PREDICT_THRESHOLD = 1       # support needed to become PREDICTED
CONTRADICT_WINDOW = 3       # consecutive antecedent-true steps without the
                            # predicted effect count as one contradiction
CANDIDATES_PER_EVENT = 10   # max new hypotheses generated per observed event
PAIR_BASE = 5               # atoms considered for pairwise candidate generation

# statuses
HYPOTHESIS = "hypothesis"
PREDICTED = "predicted"
VERIFIED = "verified"
CONTRADICTED = "contradicted"

# consequence keys
OPENS = "opens"
CLOSES = "closes"


def opens_key(door: str, opened: bool) -> str:
    return f"door:{door}:{OPENS if opened else CLOSES}"


# --------------------------------------------------------------------------
# Atom utilities (same grammar as lab_env).
# --------------------------------------------------------------------------
def atoms_from_observation(obs) -> frozenset:
    """Extract the set of true grounded atoms from a (possibly noisy)
    observation. `loadany` atoms are included alongside `load` atoms."""
    out = set()
    for plate, load in (("P1", obs.p1), ("P2", obs.p2)):
        if load in ("A", "B"):
            out.add(("load", plate, load))
            out.add(("loadany", plate))
    for sw, val in (("S1", obs.s1), ("S2", obs.s2)):
        out.add(("sw", sw, bool(val)))
    return frozenset(out)


def atom_text(atom) -> str:
    k = atom[0]
    if k == "load":
        return f"{atom[2]} on {atom[1]}"
    if k == "loadany":
        return f"anything on {atom[1]}"
    if k == "sw":
        return f"{atom[1]}={'on' if atom[2] else 'off'}"
    return str(atom)


def antecedent_text(ante: frozenset) -> str:
    return " ∧ ".join(sorted(atom_text(a) for a in ante)) if ante else "(empty)"


# --------------------------------------------------------------------------
# Hypothesis record.
# --------------------------------------------------------------------------
@dataclass
class Hypothesis:
    antecedent: frozenset
    consequence: str            # e.g. "door:D1:opens"
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
        """Empirical P(outcome | antecedent) with a weak prior."""
        return (self.sup + 1.0) / (self.sup + self.con + 2.0)

    @property
    def usable(self) -> bool:
        return self.status != CONTRADICTED

    def score(self) -> float:
        """Deterministic eviction score."""
        if self.status == CONTRADICTED:
            return -1.0 + self.con * 0.01
        base = self.confidence
        if self.status == VERIFIED:
            base += 10.0
        elif self.status == PREDICTED:
            base += 2.0
        return base - 0.0001 * self.last_update  # tie-break: older is lower


# --------------------------------------------------------------------------
# The memory.
# --------------------------------------------------------------------------
class CompositionalMemory:
    def __init__(self, max_hypotheses: int = MAX_HYPOTHESES, seed: int = 0):
        self.max_hypotheses = max_hypotheses
        self.hypotheses: list[Hypothesis] = []
        self.next_id = 1
        self.step = 0
        self.evictions = 0
        self.generated = 0
        self._continuous_true: dict[int, int] = {}

    # -- introspection ----------------------------------------------------
    def __len__(self) -> int:
        return len(self.hypotheses)

    def query(self, door: str, opens: Optional[bool] = None) -> list[Hypothesis]:
        """Return usable hypotheses for a door, sorted by confidence."""
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

    def _evict_one(self):
        """Deterministic policy: remove the lowest-score hypothesis; on
        ties remove the oldest (smallest last_update, then smallest id)."""
        if not self.hypotheses:
            return
        victim = min(self.hypotheses, key=lambda h: (h.score(), -h.last_update, -h.h_id))
        self.hypotheses.remove(victim)
        self._continuous_true.pop(victim.h_id, None)
        self.evictions += 1

    def _recompute_status(self, h: Hypothesis):
        if h.con > MAX_CONTRADICT:
            h.status = CONTRADICTED
        elif h.sup >= VERIFY_SUPPORT and h.con <= MAX_CONTRADICT:
            h.status = VERIFIED
        elif h.sup >= PREDICT_THRESHOLD:
            h.status = PREDICTED
        else:
            h.status = HYPOTHESIS

    def _support(self, h: Hypothesis):
        h.sup += 1
        h.last_update = self.step
        self._continuous_true.pop(h.h_id, None)
        self._recompute_status(h)

    def _contradict(self, h: Hypothesis):
        h.con += 1
        h.last_update = self.step
        self._continuous_true.pop(h.h_id, None)
        self._recompute_status(h)

    # -- candidate generation --------------------------------------------
    def _candidate_antecedents(self, atoms: frozenset, consequence: str) -> list:
        """From the set of atoms true just before an event, propose
        singleton and pair antecedents (bounded)."""
        base = sorted(atoms, key=str)[:PAIR_BASE]
        cands = [frozenset({a}) for a in base]
        for i in range(len(base)):
            for j in range(i + 1, len(base)):
                if len(cands) >= CANDIDATES_PER_EVENT:
                    break
                cands.append(frozenset({base[i], base[j]}))
            if len(cands) >= CANDIDATES_PER_EVENT:
                break
        return cands

    def observe_event(self, before_atoms: frozenset, door: str, opened: bool):
        """Called when a door's state visibly changed this step. Generates
        candidate hypotheses from the context that preceded the event and
        supports any existing hypothesis whose antecedent was true."""
        consequence = opens_key(door, opened)
        for ante in self._candidate_antecedents(before_atoms, consequence):
            h = self._add(ante, consequence)
            if h is not None and ante <= before_atoms:
                self._support(h)

    def observe_nonevent(self, atoms: frozenset, door: str, wants_open: bool):
        """Called each step for a closed (resp. open) door that did not
        open (resp. close): any hypothesis whose antecedent is continuously
        satisfied for CONTRADICT_WINDOW steps without the predicted effect
        accumulates one contradiction."""
        consequence = opens_key(door, wants_open)
        for h in self.hypotheses:
            if h.door != door or h.consequence != consequence or not h.usable:
                continue
            if h.antecedent <= atoms:
                c = self._continuous_true.get(h.h_id, 0) + 1
                self._continuous_true[h.h_id] = c
                if c >= CONTRADICT_WINDOW:
                    self._contradict(h)
            else:
                self._continuous_true.pop(h.h_id, None)

    def step_advance(self):
        self.step += 1

    # -- diagnostics ------------------------------------------------------
    def diagnostics(self) -> dict:
        return {
            "size": len(self.hypotheses),
            "max": self.max_hypotheses,
            "evictions": self.evictions,
            "generated": self.generated,
            "status_counts": {
                s: sum(1 for h in self.hypotheses if h.status == s)
                for s in (HYPOTHESIS, PREDICTED, VERIFIED, CONTRADICTED)
            },
        }
