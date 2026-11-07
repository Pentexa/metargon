"""EVREN curriculum — adaptive task selection.

The v0.1 experiment dropped teacher-approved tasks into the training
stream at a fixed probability (``p_teacher``). That is *task-set
augmentation*, not a curriculum: the stream was order-blind and ignored
what the student actually did. This module makes selection adaptive:

  * **learning progress** — tasks whose *family* is measured to be
    improving fastest (|Δ success| between the two halves of a bounded
    sliding window) are preferred, which keeps the student working at
    the frontier of its competence;
  * **weakness targeting** — families and atom classes where measured
    success is furthest below target receive a stable boost, so
    chronically failing structures are not starved by their own low
    success rate;
  * **novelty recovery** — families visited rarely get a bonus until
    they have enough evidence to be scored.

Everything is deterministic under a seed: scores derive only from
recorded outcomes, and all randomness flows through one seeded
``random.Random``. The selector never sees evaluation pools; callers
pass exactly the candidates that are legal to train on (isolation is
enforced upstream by the validator and re-asserted by the experiment).
"""

from __future__ import annotations

import json
import random
from collections import deque
from typing import Optional

from ..ckpt.checkpoint import rng_state_safe, rng_from_state
from ..envs.gridworld import EnvConfig
from ..envs.tasks import EpisodeSpec

DEFAULTS = {    "window": 10,             # sliding-window length per family
    "min_visits": 3,          # visits before a family is "scored"
    "lp_weight": 1.0,
    "weakness_weight": 1.2,
    "novelty_weight": 0.8,
    "noise_weight": 0.15,     # exploration floor (uniform slot probability)
    "soft_temperature": 0.15, # softmax temperature over scores
    "weakness_target": 0.75,  # success target for weakness deficit
}

# ---------------------------------------------------------------------------
# Structural signatures (families for learning progress)
# ---------------------------------------------------------------------------
def atom_class(atom) -> tuple:
    """Abstract an atom to its structural class.

    Switches collapse to a polarity class (which switch is irrelevant);
    plates keep their *kind* (load / loadany / empty); ``load`` keeps the
    object identity because "load P1 with A" and "with B" are different
    skills for the student.
    """
    kind = atom[0]
    if kind == "sw":
        return ("sw", bool(atom[2]))
    if kind == "load":
        return ("load", atom[2])
    return (kind,)            # loadany / empty


def _classes(clause) -> tuple:
    return tuple(sorted(atom_class(a) for a in clause))


def rule_structure(rule) -> tuple:
    """Structural form of a rule: clauses become atom-class tuples, in
    canonical order (conjunctive single clause stays a single clause)."""
    if isinstance(rule, tuple):
        return tuple(sorted(_classes(clause) for clause in rule))
    return _classes(rule)


def family_key(config: EnvConfig) -> tuple:
    """Canonical family signature of a task configuration.

    Two configs share a family when their two door rules have the same
    abstract structure pair (atom classes, arity, conjunctive/
    disjunctive shape) and the same topology. The door roles (D1/D2)
    are unordered — swapping which door carries which rule does not make
    it a different *skill* — and the delay flags are excluded (they
    modulate observability, not the causal structure to be learned).
    Family-level stats pool evidence that any single singleton config
    would split too thin."""
    return (
        config.topology,
        frozenset({rule_structure(config.d1_rule),
                   rule_structure(config.d2_rule)}),
    )


def atom_classes_of(config: EnvConfig) -> set:
    """All atom classes occurring in either rule (weakness tracking)."""
    def classes(rule):
        clauses = rule if isinstance(rule, tuple) else (rule,)
        return {atom_class(a) for clause in clauses for a in clause}
    return classes(config.d1_rule) | classes(config.d2_rule)


# ---------------------------------------------------------------------------
# Family statistics
# ---------------------------------------------------------------------------
class FamilyStats:
    """Bounded sliding-window success record for one family."""

    def __init__(self, window: int, min_visits: int):
        self.window = window
        self.min_visits = min_visits
        self.results = deque(maxlen=window)
        self.visits = 0
        self.last_probe_step = 0

    def record(self, outcome: bool, step: int) -> None:
        self.results.append(1.0 if outcome else 0.0)
        self.visits += 1
        self.last_probe_step = step

    # -- measurements ------------------------------------------------------
    def success_rate(self) -> Optional[float]:
        """Success rate over the window (None before ``min_visits``)."""
        if self.visits < self.min_visits:
            return None
        return sum(self.results) / len(self.results)

    def learning_progress(self) -> Optional[float]:
        """|Δ success| between the two halves of the window.

        Absolute learning progress: a family whose success rate is
        *moving* — up or down — is an active frontier. Families with
        stable success (already mastered, or hopeless) score low and are
        de-prioritized in favour of frontiers."""
        if len(self.results) < 4:
            return None
        half = len(self.results) // 2
        first = list(self.results)[:half]
        second = list(self.results)[half:]
        if not first or not second:
            return None
        delta = (sum(second) / len(second)) - (sum(first) / len(first))
        return abs(delta)

    def weakness(self, target: float) -> Optional[float]:
        """Deficit below the measured-success target (>= 0)."""
        sr = self.success_rate()
        if sr is None:
            return None
        return max(0.0, target - sr)

    def as_dict(self) -> dict:
        return {
            "visits": self.visits,
            "results": list(self.results),
            "last_probe_step": self.last_probe_step,
        }

    @classmethod
    def from_dict(cls, d: dict, window: int, min_visits: int) -> "FamilyStats":
        st = cls(window=window, min_visits=min_visits)
        st.visits = int(d["visits"])
        for x in d["results"]:
            st.results.append(x)
        st.last_probe_step = int(d.get("last_probe_step", 0))
        return st


class WeaknessTracker:
    """Per-atom-class success statistics across all recorded episodes.

    A class is *weak* when episodes whose tasks contain it succeed well
    below target — e.g. absence atoms (``empty``) or object-specific
    loads. The selector boosts candidates containing weak classes, so
    the curriculum adapts to *student weaknesses* rather than to a fixed
    syllabus."""

    def __init__(self, window: int, min_visits: int, target: float):
        self.window = window
        self.min_visits = min_visits
        self.target = target
        self.stats: dict = {}

    def record(self, config: EnvConfig, outcome: bool, step: int) -> None:
        for cls in atom_classes_of(config):
            s = self.stats.setdefault(
                cls, FamilyStats(self.window, self.min_visits))
            s.record(outcome, step)

    def weakness_of(self, cls) -> float:
        s = self.stats.get(cls)
        if s is None:
            return 0.0                    # unknown class: no boost (yet)
        w = s.weakness(self.target)
        return w if w is not None else 0.0

    def summary(self) -> list:
        rows = []
        for cls, s in sorted(self.stats.items(), key=lambda kv: str(kv[0])):
            w = s.weakness(self.target)
            rows.append({
                "atom_class": list(cls),
                "visits": s.visits,
                "success_rate": s.success_rate(),
                "weakness": w if w is not None else 0.0,
            })
        rows.sort(key=lambda r: -r["weakness"])
        return rows

    def as_dict(self) -> dict:
        return {"target": self.target, "stats": {
            json.dumps(list(cls)): st.as_dict()
            for cls, st in self.stats.items()}}

    @classmethod
    def from_dict(cls, d: dict, window: int, min_visits: int) -> "WeaknessTracker":
        tr = cls(window=window, min_visits=min_visits,
                 target=d["target"])
        for k, v in d["stats"].items():
            tr.stats[_json_key_to_class(k)] = FamilyStats.from_dict(
                v, window, min_visits)
        return tr


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------
def family_key_json(fk: tuple) -> str:
    """Canonical JSON string form of a family key (for records)."""
    return json.dumps(_lists(fk))


def _json_key_to_family(s: str) -> tuple:
    """Rebuild a canonical family key from its JSON serialization."""
    return _tuples(json.loads(s))


def _family_to_json(fk: tuple) -> str:
    return json.dumps(_lists(fk))


def _lists(x):
    if isinstance(x, frozenset):
        # canonical, deterministic order for JSON
        return {"$set": [_lists(v) for v in sorted(x, key=str)]}
    if isinstance(x, (tuple, list)):
        return [_lists(v) for v in x]
    return x


def _tuples(x):
    if isinstance(x, dict) and set(x) == {"$set"}:
        return frozenset(_tuples(v) for v in x["$set"])
    if isinstance(x, (tuple, list)):
        return tuple(_tuples(v) for v in x)
    return x


def _json_key_to_class(s: str) -> tuple:
    return _tuples(json.loads(s))


class LearningProgressSelector:
    """Adaptive curriculum selector over a fixed candidate pool.

    Score of a candidate = lp_weight * |ΔPR(family)| + weakness_weight *
    Σ weakness(atom classes of the family) (+ novelty bonus for young
    families), resolved by softmax over candidates with a uniform
    exploration floor.
    """

    def __init__(self, candidates: list, seed: int = 0,
                 **overrides):
        unknown = set(overrides) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown selector params: {sorted(unknown)}")
        self.params = dict(DEFAULTS)
        self.params.update(overrides)
        self.seed = seed
        self.rng = random.Random(seed)
        self.candidates = list(candidates)
        self._family_of = [family_key(c) for c in self.candidates]
        self.family_stats: dict = {}
        self.weakness = WeaknessTracker(
            window=self.params["window"],
            min_visits=self.params["min_visits"],
            target=self.params["weakness_target"])
        self._episodes = 0
        self.choice_history = []
        self.last_reason = None            # introspection for audit

    # -- recording --------------------------------------------------------
    def record(self, spec: EpisodeSpec, outcome: bool) -> None:
        """Record the outcome of an episode (any source)."""
        self._episodes += 1
        step = self._episodes
        fk = family_key(spec.config)
        st = self.family_stats.setdefault(
            fk, FamilyStats(self.params["window"], self.params["min_visits"]))
        st.record(outcome, step)
        self.weakness.record(spec.config, outcome, step)

    # -- scoring -----------------------------------------------------------
    def family_score(self, fk) -> float:
        """Composite score for a family (higher = select more)."""
        st = self.family_stats.get(fk)
        if st is None or st.visits < self.params["min_visits"]:
            return self.params["novelty_weight"]      # young family bonus
        score = 0.0
        lp = st.learning_progress()
        if lp is not None:
            score += self.params["lp_weight"] * lp
        wk = st.weakness(self.params["weakness_target"])
        if wk is not None:
            score += self.params["weakness_weight"] * wk
        return score

    def candidate_score(self, idx: int) -> float:
        """Score of candidate index, including a tiny deterministic
        jitter to break within-family ties."""
        base = self.family_score(self._family_of[idx])
        return base + self.params["soft_temperature"] * self.rng.random() * 1e-3

    score = candidate_score          # backwards-friendly alias

    # -- selection ---------------------------------------------------------
    def select(self, rng: Optional[random.Random] = None) -> tuple:
        """Pick the next candidate index: softmax over scores with an
        exploration floor. Returns (index, info)."""
        rng = rng or self.rng
        if not self.candidates:
            raise ValueError("selector has no candidates")
        if rng.random() < self.params["noise_weight"]:
            idx = rng.randrange(len(self.candidates))
            self.last_reason = "explore"
        else:
            scores = [self.candidate_score(i) for i in range(len(self.candidates))]
            m = max(scores)
            t = max(self.params["soft_temperature"], 1e-9)
            weights = [pow(2.718281828459045, (s - m) / t) for s in scores]
            total = sum(weights)
            r = rng.random() * total
            acc = 0.0
            idx = len(scores) - 1
            for i, w in enumerate(weights):
                acc += w
                if r <= acc:
                    idx = i
                    break
            self.last_reason = "scored"
        self.choice_history.append(idx)
        fk = self._family_of[idx]
        info = {"family": fk, "score": self.family_score(fk),
                "reason": self.last_reason}
        return idx, info

    # -- introspection -----------------------------------------------------
    def progress_report(self) -> dict:
        rows = []
        for fk, st in sorted(self.family_stats.items(),
                             key=lambda kv: _family_to_json(kv[0])):
            rows.append({
                "family_json": _family_to_json(fk),
                "visits": st.visits,
                "success_rate": st.success_rate(),
                "learning_progress": st.learning_progress(),
            })
        rows.sort(key=lambda r: -r["visits"])
        return {
            "episodes": self._episodes,
            "families": rows,
            "weakness": self.weakness.summary(),
        }

    # -- checkpointing -----------------------------------------------------
    def save_state(self) -> dict:
        return {
            "params": self.params,
            "seed": self.seed,
            "rng": rng_state_safe(self.rng),
            "family_stats": {
                _family_to_json(fk): st.as_dict()
                for fk, st in self.family_stats.items()},
            "weakness": self.weakness.as_dict(),
            "episodes": self._episodes,
            "choice_history": list(self.choice_history),
        }

    def load_state(self, state: dict) -> None:
        self.params = dict(state["params"])
        self.seed = state["seed"]
        rng_from_state(self.rng, state)
        self.family_stats = {
            _json_key_to_family(k): FamilyStats.from_dict(
                v, self.params["window"], self.params["min_visits"])
            for k, v in state["family_stats"].items()}
        self.weakness = WeaknessTracker.from_dict(
            state["weakness"], self.params["window"],
            self.params["min_visits"])
        self._episodes = int(state["episodes"])
        self.choice_history = list(state["choice_history"])
        self.last_reason = None
