"""METARGON v0.12 — bounded temporal (ordered-sequence) causal memory
(prototype, v2).

Motivation (independent audit, family L): the unordered conjunctive
hypothesis language cannot express ordered rules such as "atom A becomes
true, then atom B becomes true, and the door opens". In the audit's 3×20
run, v0.10 exact D1 discovery on L was 0.0% on every condition.

Hypothesis language (bounded, interpretable):

    <atom A becomes true>  THEN  <atom B becomes true>  (within WINDOW
    steps)  =>  door opens.

Evidence model (public observations only):
  * candidates are generated whenever an *ordered completion* is observed:
    some atom A newly became true, then later a different atom B newly
    became true within WINDOW, while the door was closed. This does NOT
    require the door to have opened yet (fixes the chicken-and-egg of
    generating from open events only).
  * a completed order that is followed by a confirmed door-open within a
    credit window SUPPORTS the matching hypothesis;
  * a completed order that is followed by continued closed-door nonevents
    for CONTRA_WINDOW steps CONTRADICTS the matching hypothesis;
  * reverse-order completions (second true before first) also contradict.

Bounds: per-door transition ring (TRANS_RING=30), hypothesis cap
TEMPORAL_MAX=64 with deterministic score eviction, at most one candidate
per completion, completion bookkeeping bounded per door.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

from lab_env import DOORS
from compositional_memory_v10 import opens_key, VERIFY_SUPPORT, MAX_CONTRADICT

TRANS_RING = 30            # per-door transition buffer length
TEMPORAL_MAX = 64          # hypothesis store cap
WINDOW = 30                # max steps allowed between first and second atom
OPEN_CREDIT = 4            # steps after completion to credit a door-open
CONTRA_WINDOW = 4          # steps of closed-door nonevents => contradiction

HYPOTHESIS = "hypothesis"
PREDICTED = "predicted"
VERIFIED = "verified"
CONTRADICTED = "contradicted"


@dataclass
class TemporalHypothesis:
    first: frozenset
    second: frozenset
    consequence: str
    sup: float = 0.0
    con: float = 0.0
    n_sup: int = 0
    n_con: int = 0
    status: str = HYPOTHESIS
    created_step: int = 0
    last_update: int = 0
    h_id: int = 0

    @property
    def door(self) -> str:
        return self.consequence.split(":")[1]

    @property
    def confidence(self) -> float:
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


@dataclass
class _Completion:
    first: frozenset
    second: frozenset
    step: int


class TemporalMemory:
    def __init__(self, max_hypotheses: int = TEMPORAL_MAX, seed: int = 0,
                 window: int = WINDOW):
        self.max_hypotheses = max_hypotheses
        self.window = window
        self.hypotheses: list[TemporalHypothesis] = []
        self.transitions: dict[str, deque] = {
            d: deque(maxlen=TRANS_RING) for d in DOORS}
        self.next_id = 1
        self.step = 0
        self.evictions = 0
        self.generated = 0
        # per-door active "first atom" waiting for completion
        self._active_first: dict[str, tuple] = {}   # door -> (atom, step)
        # pending completions awaiting open credit / contradiction
        self._pending: dict[str, list[_Completion]] = {
            d: [] for d in DOORS}

    # -- introspection ----------------------------------------------------
    def __len__(self) -> int:
        return len(self.hypotheses)

    def query(self, door: str, opens: Optional[bool] = True) -> list:
        out = []
        for h in self.hypotheses:
            if h.door != door or not h.usable:
                continue
            if opens is not None and ((opens and ":opens" not in h.consequence)
                                      or (not opens and ":opens" in h.consequence)):
                continue
            out.append(h)
        out.sort(key=lambda h: (-h.confidence, -h.sup))
        return out

    def best_for_door(self, door: str) -> Optional[TemporalHypothesis]:
        q = self.query(door, True)
        return q[0] if q else None

    # -- evidence ---------------------------------------------------------
    def _find(self, first, second, consequence) -> Optional[TemporalHypothesis]:
        for h in self.hypotheses:
            if h.consequence == consequence and h.first == first and \
                    h.second == second:
                return h
        return None

    def _add(self, first, second, consequence) -> Optional[TemporalHypothesis]:
        existing = self._find(first, second, consequence)
        if existing is not None:
            return existing
        if len(self.hypotheses) >= self.max_hypotheses:
            self._evict_one()
        h = TemporalHypothesis(first, second, consequence,
                               created_step=self.step,
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
        self.evictions += 1

    def _recompute_status(self, h: TemporalHypothesis):
        if h.con > MAX_CONTRADICT:
            h.status = CONTRADICTED
        elif h.sup >= VERIFY_SUPPORT:
            h.status = VERIFIED
        elif h.sup >= 1.0:
            h.status = PREDICTED
        else:
            h.status = HYPOTHESIS

    # -- per-step update --------------------------------------------------
    def update(self, door: str, step: int, newly_true: frozenset):
        """Consume newly-true atoms: arm firsts, detect completions, prune
        expired pendings."""
        # expire stale pendings and active firsts
        pend = self._pending[door]
        self._pending[door] = [p for p in pend
                               if step - p.step <= CONTRA_WINDOW + 2]
        active = self._active_first.get(door)
        if active is not None and step - active[1] > self.window:
            self._active_first.pop(door, None)
        for atom in sorted(newly_true, key=str):
            if atom[0] in ("door_open",):
                continue
            if active is None:
                self._active_first[door] = (atom, step)
                active = (atom, step)
            else:
                first_atom, s1 = active
                if first_atom != atom and step - s1 <= self.window:
                    # ordered completion: first -> atom
                    self._pending[door].append(
                        _Completion(frozenset({first_atom}),
                                    frozenset({atom}), step))
                    self._get_or_create(door, first_atom, atom)
                    self._active_first.pop(door, None)
                    active = None
                else:
                    self._active_first[door] = (atom, step)
                    active = (atom, step)

    def _get_or_create(self, door, first, second):
        consequence = opens_key(door, True)
        self._add(frozenset({first}), frozenset({second}), consequence)

    def observe_open(self, door: str, opened: bool, step: int):
        """Confirmed door event: credit pending completions of the same door
        (and only that door) as support."""
        if not opened:
            return
        consequence = opens_key(door, True)
        keep = []
        for p in self._pending[door]:
            h = self._find(p.first, p.second, consequence)
            if h is not None and step - p.step <= OPEN_CREDIT:
                self._support(h, step)
            else:
                keep.append(p)
        self._pending[door] = keep

    def observe_nonevent(self, door: str, step: int):
        """Closed-door step: completions past their credit window count as
        contradiction."""
        consequence = opens_key(door, True)
        keep = []
        for p in self._pending[door]:
            h = self._find(p.first, p.second, consequence)
            if h is not None and step - p.step > OPEN_CREDIT:
                self._contradict(h)
            else:
                keep.append(p)
        self._pending[door] = keep

    def _support(self, h: TemporalHypothesis, step: int):
        h.sup += 1.0
        h.n_sup += 1
        h.last_update = step
        self._recompute_status(h)

    def _contradict(self, h: TemporalHypothesis):
        h.con += 1.0
        h.n_con += 1
        h.last_update = self.step
        self._recompute_status(h)

    def step_advance(self):
        self.step += 1

    def episode_end(self):
        for h in self.hypotheses:
            h.sup *= 0.70
            h.con *= 0.70
            self._recompute_status(h)
        self._active_first.clear()
        for d in DOORS:
            self._pending[d] = []

    # -- diagnostics ------------------------------------------------------
    def diagnostics(self) -> dict:
        return {
            "size": len(self.hypotheses),
            "max": self.max_hypotheses,
            "generated": self.generated,
            "evictions": self.evictions,
            "status_counts": {
                s: sum(1 for h in self.hypotheses if h.status == s)
                for s in (HYPOTHESIS, PREDICTED, VERIFIED, CONTRADICTED)
            },
            "pending": {d: len(v) for d, v in self._pending.items()},
            "ring_sizes": {d: len(q) for d, q in self.transitions.items()},
        }
