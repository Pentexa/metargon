"""EVREN curriculum — diversity and generalization metrics.

Task-set diversity is measured over the *canonical atom vocabulary* (not
strings): Jaccard distances over atom sets of the two door rules, plus
structural coverage (topology, clause shape, delayed flags, atom-class
multisets). Novelty is measured against a reference pool (usually the
classic training pool).

All functions are pure and deterministic; no environment interaction.
"""

from __future__ import annotations

import json
import math
from itertools import combinations

from ..envs.gridworld import EnvConfig
from .selection import atom_class, atom_classes_of, family_key, rule_structure


def atoms_of(config: EnvConfig) -> frozenset:
    """Canonical atom multiset of both door rules (polarity intact)."""
    def atoms(rule):
        clauses = rule if isinstance(rule, tuple) else (rule,)
        return {tuple(a) for clause in clauses for a in clause}
    return frozenset(atoms(config.d1_rule) | atoms(config.d2_rule))


def signature(config: EnvConfig) -> dict:
    """Canonical JSON-able signature of a configuration."""
    def ser_rule(r):
        if isinstance(r, tuple):
            return [sorted(map(list, c)) for c in r]
        return [sorted(map(list, r))]
    return {
        "topology": config.topology,
        "d1_rule": ser_rule(config.d1_rule),
        "d2_rule": ser_rule(config.d2_rule),
        "d1_delay": config.d1_delay,
        "d2_delay": config.d2_delay,
    }


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return 1.0 - (inter / union if union else 1.0)


def pairwise_jaccard_stats(configs: list, cap: int = 60) -> dict:
    """Mean/min/max pairwise Jaccard distance over atom sets.

    ``cap`` bounds the worst-case pairwise cost deterministically (first
    ``cap`` configs in pool order)."""
    pool = list(configs)[:cap]
    if len(pool) < 2:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "pairs": max(0, len(pool) - 1)}
    sets = [atoms_of(c) for c in pool]
    dists = [jaccard(sets[i], sets[j]) for i, j in combinations(range(len(pool)), 2)]
    return {
        "mean": sum(dists) / len(dists),
        "min": min(dists),
        "max": max(dists),
        "pairs": len(dists),
    }


def _normalized_entropy(counts: list) -> float:
    n = sum(counts)
    if n <= 0 or len(counts) <= 1:
        return 0.0
    h = sum(-c / n * math.log(c / n) for c in counts if c > 0)
    max_h = math.log(len(counts))
    return h / max_h if max_h > 0 else 0.0


def family_distribution(configs: list) -> dict:
    """family-key -> count over a config list."""
    out = {}
    for c in configs:
        fk = family_key(c)
        k = _key_str(fk)
        out[k] = out.get(k, 0) + 1
    return out


def coverage_dims(configs: list) -> dict:
    """Structural coverage across interpretable dimensions."""
    topologies = {}
    shapes = {}          # conjunctive vs disjunctive per door
    delays = {}
    n_atoms = {}
    classes = {}
    for c in configs:
        topologies[c.topology] = topologies.get(c.topology, 0) + 1
        for door, rule in (("d1", c.d1_rule), ("d2", c.d2_rule)):
            shape = "conjunctive" if not isinstance(rule, tuple) else "disjunctive"
            k = f"{door}:{shape}"
            shapes[k] = shapes.get(k, 0) + 1
        delay = (c.d1_delay, c.d2_delay)
        delays[str(delay)] = delays.get(str(delay), 0) + 1
        n_atoms[c.n_atoms()] = n_atoms.get(c.n_atoms(), 0) + 1
        for cls in atom_classes_of(c):
            k = str(cls)
            classes[k] = classes.get(k, 0) + 1
    return {
        "topology": topologies,
        "clause_shape": shapes,
        "delay_flags": delays,
        "n_atoms": {str(k): v for k, v in sorted(n_atoms.items())},
        "atom_classes": classes,
    }


def _key_str(fk: tuple) -> str:
    def lst(x):
        if isinstance(x, frozenset):
            return {"$set": [lst(v) for v in sorted(x, key=str)]}
        if isinstance(x, (tuple, list)):
            return [lst(v) for v in x]
        return x
    return json.dumps(lst(fk))


def diversity_metrics(configs: list, reference=None, cap: int = 60) -> dict:
    """Full diversity report for a candidate/curriculum pool.

    ``reference`` (optional): pool for novelty measurement (e.g. the
    classic train pool — measuring how far the teacher/curriculum tasks
    sit from everything the procedural pipeline already offers)."""
    n = len(configs)
    fams = family_distribution(configs)
    keys = [c.key for c in configs]
    unique_sigs = len(set(keys)) if n else 0
    fam_ent = _normalized_entropy(list(fams.values())) if fams else 0.0
    out = {
        "n": n,
        "unique_configs": unique_sigs,
        "unique_ratio": (unique_sigs / n) if n else 0.0,
        "families": fams,
        "family_entropy_normalized": fam_ent,
        "pairwise_jaccard": pairwise_jaccard_stats(configs, cap=cap),
        "coverage": coverage_dims(configs),
    }
    if reference:
        ref_sets = [atoms_of(c) for c in list(reference)[:cap]]
        nov = []
        for c in configs:
            if not ref_sets:
                nov.append(1.0)
                continue
            nov.append(min(jaccard(atoms_of(c), rs) for rs in ref_sets) if ref_sets else 0.0)
        out["novelty_vs_reference"] = {
            "mean": (sum(nov) / len(nov)) if nov else 0.0,
            "min": min(nov) if nov else 0.0,
            "max": max(nov) if nov else 0.0,
            "reference_n": len(reference),
        }
    return out
