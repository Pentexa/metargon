"""Cognitive dimension metrics.

All metrics are computed from raw per-episode records produced by the
shared driver. Aggregates are reported per seed, then summarized across
seeds as mean ± 95% CI (Student-t for small samples).

Metrics deliberately separate *what the agent knows* (memory content vs
ground truth) from *what it does* (task success / steps / actions), and
from *how much it searched* (planner/fallback counters), so improvements
are not conflated with test-passing or memory growth.
"""
from __future__ import annotations

import math
import statistics

from .protocols import config_signature, _clause_key, _rule_clauses

# ---------------------------------------------------------------------------
# Small-sample statistics
# ---------------------------------------------------------------------------
_T_TABLE = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
            6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def _t(n):
    return _T_TABLE.get(n, 1.96)


def summarize(values):
    """mean / sem / ci95 from a list of per-seed values (None filtered).

    Used for *unbounded* metrics (steps, episodes-to-event, planner
    counters). For bounded probability metrics use ``summarize_proportion``.
    """
    if isinstance(values, (int, float)):
        values = [values]
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "sem": None, "ci": None, "n": 0,
                "method": "t"}
    n = len(vals)
    mean = statistics.mean(vals)
    if n < 2:
        return {"mean": mean, "sem": None, "ci": None, "n": n,
                "method": "t"}
    sem = statistics.stdev(vals) / math.sqrt(n)
    half = _t(n) * sem
    return {"mean": mean, "sem": sem, "ci": (mean - half, mean + half),
            "n": n, "method": "t"}


def _wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion (respects [0,1])."""
    if n <= 0:
        return None
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
            / denom)
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize_proportion(counts):
    """Pooled 95% Wilson interval for a bounded probability metric.

    ``counts`` is a list of per-seed ``(k, n)`` success/trial counts (or
    None). A single ``(k, n)`` pair is also accepted. Pooling across
    seeds keeps the interval inside [0, 1], which a Student-t interval on
    per-seed means does not guarantee for small, extreme samples.
    """
    if isinstance(counts, tuple) and len(counts) == 2 and not isinstance(
            counts[0], (tuple, list)):
        counts = [counts]
    pairs = [tuple(c) for c in counts if c is not None and c[1] > 0]
    if not pairs:
        return {"mean": None, "sem": None, "ci": None, "n": 0,
                "n_events": 0, "method": "wilson"}
    k = sum(p[0] for p in pairs)
    n = sum(p[1] for p in pairs)
    return {
        "mean": k / n,
        "sem": None,
        "ci": _wilson_ci(k, n),
        "n": len(pairs),
        "n_events": n,
        "method": "wilson",
    }


def _rate_counts(recs, key):
    """(k, n) success/trial counts for a boolean per-episode field."""
    if not recs:
        return None
    return sum(1 for r in recs if r.get(key)), len(recs)


def _rate(recs, key):
    """Success / door-open fraction over recs (empty -> None)."""
    if not recs:
        return None
    return sum(1 for r in recs if r.get(key)) / len(recs)


def _mean(recs, key, only_success=True):
    vals = [r.get(key) for r in recs if r.get("success") or not only_success]
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else None


# ---------------------------------------------------------------------------
# Ground truth from a stream
# ---------------------------------------------------------------------------
def truth_rules(stream):
    """{(door, clause_frozenset)} for every distinct true rule in stream."""
    truths = set()
    for spec in stream:
        for door in ("D1", "D2"):
            rule = spec.config.rule_for(door)
            for clause in _rule_clauses(rule):
                truths.add((door, clause))
    return truths


def _antecedent_key(ante_list):
    return frozenset(tuple(a) for a in ante_list)


def match_hypotheses(mem, truths):
    """Classify the agent's hypotheses vs ground truth."""
    correct_any, verified_correct, verified_wrong = set(), set(), set()
    candidates_correct = set()
    contradicted_truth = set()
    for h in mem:
        ante = _antecedent_key(h["antecedent"])
        is_true = (h["door"], ante) in truths
        if is_true:
            correct_any.add((h["door"], ante))
            if h["status"] == "verified":
                verified_correct.add((h["door"], ante))
            elif h["status"] in ("hypothesis", "predicted"):
                candidates_correct.add((h["door"], ante))
        elif h["status"] == "verified":
            verified_wrong.add((h["door"], ante))
        if h["status"] == "contradicted":
            if is_true:
                contradicted_truth.add((h["door"], ante))
    return {
        "correct_any": correct_any,
        "verified_correct": verified_correct,
        "verified_wrong": verified_wrong,
        "candidates_correct": candidates_correct,
        "contradicted_truth": contradicted_truth,
    }


def discovery_metrics(mem, truths, generated=None):
    m = match_hypotheses(mem, truths)
    n_truths = len(truths) or 1
    n_verified = len(m["verified_correct"]) + len(m["verified_wrong"])
    return {
        "discovery_coverage": len(m["correct_any"]) / n_truths,
        "verified_correct": len(m["verified_correct"]),
        "verified_wrong": len(m["verified_wrong"]),
        "verified_precision": (len(m["verified_correct"]) / n_verified
                               if n_verified else None),
        "candidates_correct": len(m["candidates_correct"]),
        "contradicted_truth": len(m["contradicted_truth"]),
        "hypotheses_total": len(mem),
        "generated_lifetime": generated,
        # counts for pooled Wilson intervals
        "_counts": {
            "discovery_coverage": (len(m["correct_any"]), n_truths),
            "verified_precision": (len(m["verified_correct"]), n_verified),
        },
    }


# ---------------------------------------------------------------------------
# Per-protocol metric computation
# ---------------------------------------------------------------------------
def learning_curve(recs, truths):
    """Per-episode cumulative verified-correct discoveries."""
    curve = []
    seen = set()
    for r in recs:
        m = match_hypotheses(r.get("memory", []), truths)
        seen |= m["verified_correct"]
        curve.append(len(seen))
    return curve


def first_episode_with(recs, predicate):
    for i, r in enumerate(recs):
        if predicate(r):
            return i + 1
    return None


def train_metrics(recs, truths):
    half = len(recs) // 2 or 1
    first, second = recs[:half], recs[half:]
    curve = learning_curve(recs, truths)
    final = discovery_metrics(
        recs[-1].get("memory", []) if recs else [],
        truths, generated=recs[-1].get("diagnostics", {}).get(
            "memory", {}).get("generated") if recs else None)
    return {
        "success_rate": _rate(recs, "success"),
        "success_rate_first_half": _rate(first, "success"),
        "success_rate_second_half": _rate(second, "success"),
        "mean_steps_to_success_first_half": _mean(first, "steps"),
        "mean_steps_to_success_second_half": _mean(second, "steps"),
        "episodes_to_first_success": first_episode_with(
            recs, lambda r: r["success"]),
        "episodes_to_first_verified_correct": first_episode_with(
            recs, lambda r: match_hypotheses(
                r.get("memory", []), truths)["verified_correct"]),
        "discovery_curve": curve,
        "final_discovery": final,
        "_counts": {
            "success_rate": _rate_counts(recs, "success"),
            "success_rate_first_half": _rate_counts(first, "success"),
            "success_rate_second_half": _rate_counts(second, "success"),
            "discovery_coverage": final["_counts"]["discovery_coverage"],
            "verified_precision": final["_counts"]["verified_precision"],
        },
    }


def _trace_delta_sum(recs, key):
    return sum(r.get("trace_delta", {}).get(key, 0) for r in recs)


def eval_metrics(recs):
    steps_total = sum(r["steps"] for r in recs) if recs else 0
    # within-episode adaptation: episodes that produced at least one door
    # event and then reached the goal (discovered mechanism used in-episode)
    with_events = [r for r in recs if r.get("events")]
    fallback = _trace_delta_sum(recs, "random_fallback_steps")
    invalid = sum(r["invalid"] for r in recs)
    return {
        "success_rate": _rate(recs, "success"),
        "door_open_rate": _rate(recs, "door_open"),
        "within_episode_adaptation": (
            _rate(with_events, "success") if with_events else None),
        "mean_steps": _mean(recs, "steps", only_success=False),
        "mean_steps_to_success": _mean(recs, "steps"),
        "invalid_rate": (invalid / steps_total if steps_total else None),
        "fallback_rate": (fallback / steps_total if steps_total else None),
        "planner_ops_per_step": (_trace_delta_sum(recs, "planner_ops") /
                                 steps_total if steps_total else None),
        # NOTE: `plan_cache_hits` counts cache lookups that found an entry,
        # while `plans_used` counts executed plans — they are not a
        # hit/miss pair, so a "hit rate" would be misleading. Report both
        # per step instead, with accurate names.
        "plan_cache_hits_per_step": (
            _trace_delta_sum(recs, "plan_cache_hits") / steps_total
            if steps_total else None),
        "plans_used_per_step": (
            _trace_delta_sum(recs, "plans_used") / steps_total
            if steps_total else None),
        "_counts": {
            "success_rate": _rate_counts(recs, "success"),
            "door_open_rate": _rate_counts(recs, "door_open"),
            "within_episode_adaptation": (
                _rate_counts(with_events, "success")
                if with_events else None),
            "invalid_rate": (invalid, steps_total) if steps_total else None,
            "fallback_rate": (fallback, steps_total) if steps_total else None,
        },
    }


def zero_shot_coverage(pre_mem, eval_stream):
    """(covered, total) eval configs whose true door-rules are already
    verified in pre-evaluation memory (clean zero-shot knowledge measure)."""
    truths = truth_rules(eval_stream)
    m = match_hypotheses(pre_mem, truths)
    covered = set()
    total = len(set(config_signature(s.config) for s in eval_stream))
    for spec in eval_stream:
        sig = config_signature(spec.config)
        if sig in covered:
            continue
        for door in ("D1", "D2"):
            rule = spec.config.rule_for(door)
            for clause in _rule_clauses(rule):
                if (door, clause) in m["verified_correct"]:
                    covered.add(sig)
                    break
    return len(covered), total


def revisit_metrics(recs, occurrences):
    """Retention: success at later vs earlier occurrence of each config."""
    labels = []
    for sig, idxs in occurrences.items():
        if len(idxs) < 2:
            continue
        early, late = idxs[0], idxs[-1]
        labels.append({
            "config": sig,
            "early": bool(recs[early]["success"]),
            "late": bool(recs[late]["success"]),
        })
    keep = [l for l in labels if l["early"]]
    early_recs = [recs[l["early"]] for l in labels]
    late_recs = [recs[l["late"]] for l in labels]
    preserved = sum(1 for l in labels if l["early"] and l["late"])
    return {
        "configs": labels,
        "retention_preserved": (
            preserved / len(keep) if keep else None),
        "success_early": _rate(early_recs, "success"),
        "success_late": _rate(late_recs, "success"),
        "_counts": {
            "retention_preserved": (preserved, len(keep)) if keep else None,
            "success_early": _rate_counts(early_recs, "success"),
            "success_late": _rate_counts(late_recs, "success"),
        },
    }


def switch_metrics(recs, split_point, target_door="D1"):
    before, after = recs[:split_point], recs[split_point:]
    lag = None
    for i, r in enumerate(after):
        if r["success"]:
            lag = i + 1
            break
    # wrong-belief correction: pre-switch target-door hypotheses that end up
    # contradicted in the final memory.
    pre_hyps = []
    for r in before:
        for h in r.get("memory", []):
            if h["door"] == target_door and h["status"] != "contradicted":
                pre_hyps.append(h["id"])
    final_mem = recs[-1].get("memory", []) if recs else []
    contradicted_ids = {h["id"] for h in final_mem
                        if h["status"] == "contradicted"
                        and h["door"] == target_door}
    corrected = sum(1 for hid in set(pre_hyps) if hid in contradicted_ids)
    n_pre = len(set(pre_hyps))
    return {
        "success_before": _rate(before, "success"),
        "success_after": _rate(after, "success"),
        "recovery_lag": lag,
        "wrong_belief_correction": (
            corrected / n_pre if n_pre else None),
        "target_door": target_door,
        "_counts": {
            "success_before": _rate_counts(before, "success"),
            "success_after": _rate_counts(after, "success"),
            "wrong_belief_correction": (corrected, n_pre) if n_pre else None,
        },
    }


def restart_metrics(direct_recs, restored_recs):
    direct_success = _rate(direct_recs, "success")
    restored_success = _rate(restored_recs, "success")
    return {
        "success_direct": direct_success,
        "success_restored": restored_success,
        "parity": (restored_success / direct_success
                   if direct_success else None),
        "steps_direct": _mean(direct_recs, "steps", only_success=False),
        "steps_restored": _mean(restored_recs, "steps",
                                 only_success=False),
        "_counts": {
            "success_restored": _rate_counts(restored_recs, "success"),
        },
    }


# ---------------------------------------------------------------------------
# Scorecard assembly
# ---------------------------------------------------------------------------
def build_scorecard(seed_results, meta, extra_vectors=None,
                    train_stream_specs=None):
    """seed_results: {seed: {policy: {protocol: {"recs": [...]}}}}.

    ``extra_vectors`` supplies per-seed metric vectors that need context
    outside a single protocol's records (e.g. pre-eval memory for
    zero-shot coverage): {name: {policy: [per-seed values]}}.
    ``train_stream_specs`` is the training EpisodeSpec stream used for
    ground-truth rule extraction (kept out of the serialized output).
    """
    extra_vectors = extra_vectors or {}
    scorecard = {"meta": meta, "dimensions": {}, "per_seed": {}}
    truths = truth_rules(train_stream_specs) if train_stream_specs else set()

    protocols = meta["protocols"]

    # helpers to extract per-seed vectors
    def vectors(policy, protocol, metric_fn):
        out = []
        for seed, policies in seed_results.items():
            recs = policies.get(policy, {}).get(protocol, {}).get("recs")
            if recs is None:
                out.append(None)
                continue
            out.append(metric_fn(recs))
        return out

    def vectors_counts(policy, protocol, metric_fn):
        """Per-seed (k, n) counts for proportion metrics."""
        out = []
        for seed, policies in seed_results.items():
            recs = policies.get(policy, {}).get(protocol, {}).get("recs")
            if recs is None:
                out.append(None)
                continue
            out.append(metric_fn(recs))
        return out

    # ---- 1. causal discovery -------------------------------------------
    dim = {"title": "Causal discovery", "metrics": {}}
    for pol in ("metargon_trained", "metargon_muted"):
        per_seed = []
        for seed, policies in seed_results.items():
            recs = policies.get(pol, {}).get("train", {}).get("recs")
            per_seed.append(None if not recs else
                            discovery_metrics(recs[-1].get("memory", []),
                                              truths))
        for key in ("verified_correct", "verified_wrong",
                    "candidates_correct", "contradicted_truth",
                    "hypotheses_total"):
            dim["metrics"][f"{pol}.{key}"] = summarize(
                [d[key] if d else None for d in per_seed])
        for key in ("discovery_coverage", "verified_precision"):
            dim["metrics"][f"{pol}.{key}"] = summarize_proportion(
                [d["_counts"][key] if d else None for d in per_seed])
    scorecard["dimensions"]["causal_discovery"] = dim

    # ---- 2. learning efficiency ----------------------------------------
    dim = {"title": "Learning efficiency", "metrics": {}}
    for pol in ("metargon_trained", "metargon_muted"):
        tr = {
            "episodes_to_first_success": lambda r: train_metrics(r, truths)["episodes_to_first_success"],
            "episodes_to_first_verified_correct": lambda r: train_metrics(r, truths)["episodes_to_first_verified_correct"],
            "mean_steps_to_success_second_half": lambda r: train_metrics(r, truths)["mean_steps_to_success_second_half"],
        }
        for name, fn in tr.items():
            dim["metrics"][f"{pol}.{name}"] = summarize(
                vectors(pol, "train", fn))
        for name in ("success_rate", "success_rate_first_half",
                     "success_rate_second_half"):
            dim["metrics"][f"{pol}.{name}"] = summarize_proportion(
                vectors_counts(pol, "train",
                               lambda r, n=name: train_metrics(r, truths)["_counts"][n]))
    scorecard["dimensions"]["learning_efficiency"] = dim

    # ---- 3. knowledge retention ----------------------------------------
    dim = {"title": "Knowledge retention", "metrics": {}}
    for pol in ("metargon_trained", "metargon_muted"):
        for name in ("retention_preserved", "success_early", "success_late"):
            dim["metrics"][f"{pol}.{name}"] = summarize_proportion(
                vectors_counts(pol, "revisit",
                               lambda r, n=name: revisit_metrics(
                                   r, meta["revisit_occurrences"])["_counts"][n]))
        dim["metrics"][f"{pol}.restart_parity"] = summarize(
            vectors(pol, "restart",
                    lambda r: restart_metrics(
                        r.get("direct", []), r.get("restored", []))["parity"]))
        dim["metrics"][f"{pol}.restart_success_restored"] = summarize_proportion(
            vectors_counts(pol, "restart",
                           lambda r: restart_metrics(
                               r.get("direct", []),
                               r.get("restored", []))["_counts"]["success_restored"]))
    scorecard["dimensions"]["knowledge_retention"] = dim

    # ---- 4. generalization ----------------------------------------------
    dim = {"title": "Generalization", "metrics": {}}
    for pool in ("eval_val", "eval_test", "eval_ood"):
        for pol in ("metargon_trained", "metargon_fresh"):
            dim["metrics"][f"{pol}.{pool}.success_rate"] = summarize_proportion(
                vectors_counts(pol, pool,
                               lambda r: eval_metrics(r)["_counts"]["success_rate"]))
            dim["metrics"][f"{pol}.{pool}.mean_steps_to_success"] = summarize(
                vectors(pol, pool, lambda r: _mean(r, "steps")))
    # zero-shot knowledge transfer (pre-eval memory per seed) — counts
    for pool in ("eval_val", "eval_test", "eval_ood"):
        key = f"metargon_trained.{pool}.zero_shot_coverage"
        dim["metrics"][key] = summarize_proportion(
            extra_vectors.get(key))
    scorecard["dimensions"]["generalization"] = dim

    # ---- 5. adaptability ------------------------------------------------
    dim = {"title": "Adaptability", "metrics": {}}
    t_door = meta.get("switch_target_door", "D1")
    for pol in ("metargon_trained", "metargon_muted", "metargon_fresh"):
        for name in ("success_before", "success_after"):
            dim["metrics"][f"{pol}.{name}"] = summarize_proportion(
                vectors_counts(pol, "switch",
                               lambda r, n=name: switch_metrics(
                                   r, meta["switch_split"], t_door)["_counts"][n]))
        dim["metrics"][f"{pol}.recovery_lag"] = summarize(
            vectors(pol, "switch",
                    lambda r: switch_metrics(r, meta["switch_split"], t_door)["recovery_lag"]))
        if pol != "metargon_fresh":
            dim["metrics"][f"{pol}.wrong_belief_correction"] = summarize_proportion(
                vectors_counts(pol, "switch",
                               lambda r: switch_metrics(
                                   r, meta["switch_split"], t_door)["_counts"]["wrong_belief_correction"]))
    scorecard["dimensions"]["adaptability"] = dim

    # ---- 6. planning efficiency -----------------------------------------
    dim = {"title": "Planning efficiency", "metrics": {}}
    for pol in ("metargon_trained", "metargon_fresh"):
        suffix = "eval_test"
        for name in ("fallback_rate", "invalid_rate"):
            dim["metrics"][f"{pol}.{suffix}.{name}"] = summarize_proportion(
                vectors_counts(pol, suffix,
                               lambda r, n=name: eval_metrics(r)["_counts"][n]))
        for name in ("planner_ops_per_step", "plan_cache_hits_per_step",
                     "plans_used_per_step"):
            dim["metrics"][f"{pol}.{suffix}.{name}"] = summarize(
                vectors(pol, suffix, lambda r, n=name: eval_metrics(r)[n]))
    scorecard["dimensions"]["planning_efficiency"] = dim

    # ---- descriptive 0-1 indicators (NOT a composite IQ score) -----------
    scorecard["indicators"] = _indicators(scorecard)

    # baseline context table for eval pools
    scorecard["baselines"] = {}
    _add_baselines(scorecard, seed_results)

    # per-seed compact echoes
    scorecard["per_seed"] = _per_seed_echo(seed_results)

    return scorecard


def _mean_of(dim, key):
    return dim.get("metrics", {}).get(key, {}).get("mean")


def _indicators(sc):
    """Descriptive 0-1 indicators derived from raw metrics.

    Each indicator maps to exactly one raw metric; formulas are documented
    in the report. Nothing here is combined into a single score.
    """
    dims = sc["dimensions"]
    c = dims["causal_discovery"]
    e = dims["learning_efficiency"]
    r = dims["knowledge_retention"]
    g = dims["generalization"]
    a = dims["adaptability"]
    p = dims["planning_efficiency"]

    def clamp(x):
        return None if x is None else max(0.0, min(1.0, x))

    lag = _mean_of(a, "metargon_trained.recovery_lag")
    fresh_steps = _mean_of(g, "metargon_fresh.eval_test.mean_steps_to_success")
    trained_steps = _mean_of(g, "metargon_trained.eval_test.mean_steps_to_success")
    steps_improvement = (
        (1.0 - trained_steps / fresh_steps)
        if fresh_steps and trained_steps is not None else None)
    return {
        "causal_discovery": {
            "coverage": clamp(_mean_of(c, "metargon_trained.discovery_coverage")),
            "verified_precision": clamp(_mean_of(c, "metargon_trained.verified_precision")),
        },
        "learning_efficiency": {
            "success_rate_second_half": clamp(_mean_of(
                e, "metargon_trained.success_rate_second_half")),
            "first_verified_rule": clamp(1.0 / _mean_of(
                e, "metargon_trained.episodes_to_first_verified_correct")
                if _mean_of(e, "metargon_trained.episodes_to_first_verified_correct")
                else None),
        },
        "knowledge_retention": {
            "revisit_preserved": clamp(_mean_of(
                r, "metargon_trained.retention_preserved")),
            "restart_parity": clamp(_mean_of(
                r, "metargon_trained.restart_parity")),
        },
        "generalization": {
            "zero_shot_test_coverage": clamp(_mean_of(
                g, "metargon_trained.eval_test.zero_shot_coverage")),
            "test_success_rate": clamp(_mean_of(
                g, "metargon_trained.eval_test.success_rate")),
        },
        "adaptability": {
            "success_after_switch": clamp(_mean_of(
                a, "metargon_trained.success_after")),
            "recovery_speed": clamp(1.0 / (1.0 + lag) if lag is not None else None),
            "wrong_belief_correction": clamp(_mean_of(
                a, "metargon_trained.wrong_belief_correction")),
        },
        "planning_efficiency": {
            "fallback_reduction": clamp(1.0 - _mean_of(
                p, "metargon_trained.eval_test.fallback_rate")),
            "steps_to_success_improvement": clamp(steps_improvement),
        },
    }


def _add_baselines(sc, seed_results):
    def vectors(policy, protocol, metric_fn):
        out = []
        for seed, policies in seed_results.items():
            recs = policies.get(policy, {}).get(protocol, {}).get("recs")
            out.append(None if recs is None else metric_fn(recs))
        return out

    def vectors_counts(policy, protocol, metric_fn):
        out = []
        for seed, policies in seed_results.items():
            recs = policies.get(policy, {}).get(protocol, {}).get("recs")
            out.append(None if recs is None else metric_fn(recs))
        return out

    for pool in ("eval_val", "eval_test", "eval_ood"):
        row = {}
        for pol in ("metargon_trained", "metargon_fresh", "random",
                    "episodic", "qlearning", "oracle"):
            row[pol] = {
                "success_rate": summarize_proportion(
                    vectors_counts(pol, pool,
                                   lambda r: eval_metrics(r)["_counts"]["success_rate"])),
                "mean_steps_to_success": summarize(
                    vectors(pol, pool, lambda r: _mean(r, "steps"))),
                "within_episode_adaptation": summarize_proportion(
                    vectors_counts(pol, pool,
                                   lambda r: eval_metrics(r)["_counts"]["within_episode_adaptation"])),
            }
        sc["baselines"][pool] = row


def _per_seed_echo(seed_results):
    echo = {}
    for seed, policies in seed_results.items():
        echo[str(seed)] = {}
        for pol, prots in policies.items():
            echo[str(seed)][pol] = {}
            for proto, data in prots.items():
                recs = data.get("recs", [])
                if isinstance(recs, dict):
                    # composite protocols (e.g. restart) carry multiple arms
                    echo[str(seed)][pol][proto] = {
                        "n": len(recs.get("restored", [])),
                        "success": _rate(recs.get("restored", []),
                                         "success")}
                else:
                    echo[str(seed)][pol][proto] = {
                        "n": len(recs),
                        "success": _rate(recs, "success")}
    return echo
