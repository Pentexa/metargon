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
    """Bounded, evidence-driven hypothesis memory with checkpointing."""

    def __init__(self, max_hypotheses: int = MAX_HYPOTHESES,
                 verify_support: int = VERIFY_SUPPORT,
                 max_contradict: int = MAX_CONTRADICT,
                 predict_threshold: int = PREDICT_THRESHOLD,
                 contradictict_window: int = CONTRADICT_WINDOW,
                 candidates_per_event: int = CANDIDATES_PER_EVENT,
                 pair_base: int = PAIR_BASE):
        self.max_hypotheses = max_hypotheses
        self.verify_support = verify_support
        self.max_contradict = max_contradict
        self.predict_threshold = predict_threshold
        self.contradict_window = contradictict_window
        self.candidates_per_event = candidates_per_event
        self.pair_base = pair_base
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
        if not self.hypotheses:
            return
        victim = min(self.hypotheses,
                     key=lambda h: (h.score(), -h.last_update, -h.h_id))
        self.hypotheses.remove(victim)
        self._continuous_true.pop(victim.h_id, None)
        self.evictions += 1

    def _recompute_status(self, h: Hypothesis):
        if h.con > self.max_contradict:
            h.status = CONTRADICTED
        elif h.sup >= self.verify_support and h.con <= self.max_contradict:
            h.status = VERIFIED
        elif h.sup >= self.predict_threshold:
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
        for h in self.hypotheses:
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
            "hypotheses": [h.as_dict() for h in self.hypotheses],
            "next_id": self.next_id,
            "step": self.step,
            "evictions": self.evictions,
            "generated": self.generated,
            "continuous_true": {str(k): v for k, v in self._continuous_true.items()},
        }

    def load_state(self, state: dict) -> None:
        self.max_hypotheses = state["max_hypotheses"]
        self.verify_support = state["verify_support"]
        self.max_contradict = state["max_contradict"]
        self.predict_threshold = state["predict_threshold"]
        self.contradict_window = state["contradict_window"]
        self.candidates_per_event = state["candidates_per_event"]
        self.pair_base = state["pair_base"]
        self.hypotheses = [Hypothesis.from_dict(d) for d in state["hypotheses"]]
        self.next_id = state["next_id"]
        self.step = state["step"]
        self.evictions = state["evictions"]
        self.generated = state["generated"]
        self._continuous_true = {int(k): v for k, v in state["continuous_true"].items()}

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
