"""METARGON v0.10 — compositional causal memory with evidence decay and
prioritized replay.

Changes vs v0.9 (documented in docs/RESEARCH_V10.md, §failure-modes):

  1. Evidence is float EWMA support/contradiction weight with a weak
     prior; every episode end decays all evidence (EVENT_DECAY) so that
     mechanism changes (shift / rapid_shift / novel) slowly unlearn old
     rules instead of accumulating permanent contradictions.
  2. Candidate generation is complete for the event context (all
     singletons + all pairs, capped at CANDIDATES_PER_EVENT) — v0.9
     truncated by a string-sorted base, which skewed discovery toward
     alphabetical atoms.
  3. Informative causal events (door opens/closes) are stored in a
     bounded replay buffer with priority = 1/(1+current support of the
     matched hypothesis), i.e. novel/under-supported rules are replayed
     first. Replay is applied immediately after the event (bounded per
     event), strengthening sparse evidence without unbounded memory.
  4. hypotheses survive with weighted evidence; contradicted means the
     weighted contradiction count exceeds MAX_CONTRADICT *after decay*
     — a transient false start does not permanently poison a rule.

Memory remains hard-capped (MAX_HYPOTHESES) with deterministic score-based
eviction. Family-agnostic: hypotheses are grounded atoms, never complete
wirings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from lab_env import DOORS, PLATES, SWITCHES

MAX_HYPOTHESES = 128
VERIFY_SUPPORT = 3.0          # weighted support for VERIFIED
MAX_CONTRADICT = 2.0          # weighted contradictions for CONTRADICTED
PREDICT_THRESHOLD = 1.0       # weighted support for PREDICTED
CONTRADICT_WINDOW = 3         # consecutive believed-true steps without effect
CANDIDATES_PER_EVENT = 32
EVENT_DECAY = 0.70            # per-episode evidence decay
REPLAY_MAX = 64               # replay buffer cap
REPLAY_PER_EVENT = 8          # replay applications per event
REPLAY_CERT_MIN = 0.5         # min certainty for "filtered" replay

HYPOTHESIS = "hypothesis"
PREDICTED = "predicted"
VERIFIED = "verified"
CONTRADICTED = "contradicted"

OPENS = "opens"
CLOSES = "closes"


def opens_key(door: str, opened: bool) -> str:
    return f"door:{door}:{OPENS if opened else CLOSES}"


def atoms_from_belief(belief) -> frozenset:
    """Extract true grounded atoms from a reconciled physical belief."""
    out = set()
    for plate, load in (("P1", belief.plate_load["P1"]),
                        ("P2", belief.plate_load["P2"])):
        if load in ("A", "B"):
            out.add(("load", plate, load))
            out.add(("loadany", plate))
    for sw, val in (("S1", belief.switch_state["S1"]),
                    ("S2", belief.switch_state["S2"])):
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


@dataclass
class Hypothesis:
    antecedent: frozenset
    consequence: str
    sup: float = 0.0           # weighted support
    con: float = 0.0           # weighted contradiction
    n_sup: int = 0             # raw support count (diagnostics)
    n_con: int = 0             # raw contradiction count (diagnostics)
    n_replay: int = 0          # support events that came from replay
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
        if self.status == CONTRADICTED:
            return -1.0 + min(self.con, 10.0) * 0.01
        base = self.confidence
        if self.status == VERIFIED:
            base += 10.0
        elif self.status == PREDICTED:
            base += 2.0
        return base - 0.0001 * self.last_update


class CompositionalMemoryV10:
    def __init__(self, max_hypotheses: int = MAX_HYPOTHESES, seed: int = 0,
                 decay: float = EVENT_DECAY, enable_replay: bool = True,
                 replay_mode: str = "on"):
        self.max_hypotheses = max_hypotheses
        self.decay = decay
        self.enable_replay = enable_replay
        self.replay_mode = replay_mode
        self.hypotheses: list[Hypothesis] = []
        self.next_id = 1
        self.step = 0
        self.evictions = 0
        self.generated = 0
        self.replay: list[tuple] = []  # (priority, ante, cons, certainty)
        self.replay_applied = 0
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

    def _support(self, h: Hypothesis, from_replay: bool = False,
                 weight: float = 1.0):
        h.sup += max(0.0, weight)
        h.n_sup += 1
        if from_replay:
            h.n_replay += 1
        h.last_update = self.step
        self._continuous_true.pop(h.h_id, None)
        self._recompute_status(h)

    def _contradict(self, h: Hypothesis):
        h.con += 1.0
        h.n_con += 1
        h.last_update = self.step
        self._continuous_true.pop(h.h_id, None)
        self._recompute_status(h)

    def _decay_evidence(self):
        for h in self.hypotheses:
            h.sup *= self.decay
            h.con *= self.decay
            self._recompute_status(h)
        # also decay pending continuous-true counters
        for k in list(self._continuous_true):
            self._continuous_true[k] = max(0, self._continuous_true[k] - 1)

    # -- candidate generation (complete for the context) ------------------
    def _candidate_antecedents(self, atoms: frozenset, consequence: str) -> list:
        base = sorted(atoms, key=str)
        cands = [frozenset({a}) for a in base]
        for i in range(len(base)):
            for j in range(i + 1, len(base)):
                if len(cands) >= CANDIDATES_PER_EVENT:
                    break
                cands.append(frozenset({base[i], base[j]}))
            if len(cands) >= CANDIDATES_PER_EVENT:
                break
        return cands

    # -- replay -----------------------------------------------------------
    def _replay_priority(self, antecedent, consequence) -> float:
        h = self._find(antecedent, consequence)
        if h is None:
            return 1.0
        return 1.0 / (1.0 + max(h.sup, 0.0))

    def _store_replay(self, antecedent, consequence, certainty=1.0):
        prio = self._replay_priority(antecedent, consequence)
        self.replay.append((prio, antecedent, consequence, certainty))
        if len(self.replay) > REPLAY_MAX:
            # drop the lowest-priority item
            self.replay.sort(key=lambda t: -t[0])
            del self.replay[-1]

    def _apply_replay(self, items=None):
        if items is None:
            items = self.replay
        if not items:
            return
        mode = getattr(self, "replay_mode", "on")
        if mode == "off":
            return
        if mode == "filtered":
            items = [it for it in items if it[3] >= REPLAY_CERT_MIN]
        items = sorted(items, key=lambda t: -t[0] * (t[3] if mode == "weighted" else 1.0))
        taken = 0
        for prio, ante, cons, certainty in items:
            if taken >= REPLAY_PER_EVENT:
                break
            h = self._find(ante, cons)
            if h is None:
                h = self._add(ante, cons)
            if h is not None:
                weight = certainty if mode == "weighted" else 1.0
                self._support(h, from_replay=True, weight=weight)
                self.replay_applied += 1
            taken += 1

    def _clear_replay_for(self, consequence: str):
        self.replay = [t for t in self.replay if t[2] != consequence]

    # -- event handling ---------------------------------------------------
    def observe_event(self, before_atoms: frozenset, door: str, opened: bool):
        self.observe_event_ex(before_atoms, before_atoms, door, opened,
                               certainty=1.0)

    def observe_event_ex(self, pre_atoms: frozenset, post_atoms: frozenset,
                         door: str, opened: bool, certainty: float = 1.0):
        """Combined context event with an evidence-quality weight. Contexts
        are unioned and cleaned of contradictory polarities; support is
        added with weight `certainty` (a family-agnostic observation-
        likelihood proxy)."""
        consequence = opens_key(door, opened)
        ctx = self._clean_ctx(pre_atoms, post_atoms)
        # replay reinforces *older* evidence; the just-processed event is
        # already counted directly, so exclude it from this replay pass.
        old_replay = self.replay[:] if self.enable_replay else []
        cands = self._candidate_antecedents(ctx, consequence)
        for ante in cands:
            h = self._add(ante, consequence)
            if h is not None and ante <= ctx:
                self._support(h, weight=certainty)
            if self.enable_replay:
                self._store_replay(ante, consequence, certainty)
        if self.enable_replay:
            self._apply_replay(old_replay)

    @staticmethod
    def _clean_ctx(pre, post) -> frozenset:
        out = set(pre)
        for a in post:
            k = a[0]
            conflict = False
            if k == "sw":
                conflict = any(x[0] == "sw" and x[1] == a[1] and x[2] != a[2]
                               for x in pre)
            elif k == "load":
                conflict = any(x[0] == "load" and x[1] == a[1] and x[2] != a[2]
                               for x in pre)
            if not conflict:
                out.add(a)
        return frozenset(out)

    def observe_nonevent(self, atoms: frozenset, door: str, wants_open: bool):
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

    def episode_end(self):
        self._decay_evidence()

    # -- diagnostics ------------------------------------------------------
    def diagnostics(self) -> dict:
        return {
            "size": len(self.hypotheses),
            "max": self.max_hypotheses,
            "evictions": self.evictions,
            "generated": self.generated,
            "replay_applied": self.replay_applied,
            "replay_buffer": len(self.replay),
            "status_counts": {
                s: sum(1 for h in self.hypotheses if h.status == s)
                for s in (HYPOTHESIS, PREDICTED, VERIFIED, CONTRADICTED)
            },
            "total_support_weight": sum(h.sup for h in self.hypotheses),
            "total_weight_sup": sum(h.n_sup for h in self.hypotheses),
        }
