# CORVID v0.5 — Sparse, Delayed Outcome Feedback

**Date:** 2026-11-06  
**Status:** Completed synthetic experiment, exploratory, no human-subject or physical experiments  
**Scientific claim:** Evidence of a limited performance difference under deliberately generated sensing conditions, **not** evidence of architectural novelty, biological fidelity, or real power savings.

## Primary question

Does a bounded-memory agent that jointly learns *where to observe* and *where to purchase scarce truth labels* improve decision accuracy beyond random observation and a strong, simple uncertainty heuristic when outcome labels are sparse and delayed?

## Hypothesis / failure criteria

We hoped joint allocation would win with the same sensing and verified-feedback budget, especially after an unannounced sensor-quality shift. The null result is that joint allocation fails to outperform contextwise uncertainty-based allocation (or wins only on a tuned generator), and/or does not generalize to new sensing patterns.

The experiment supports the null side: joint did **not** beat simple uncertainty in any of four environment families. Keep the failure visible.

## What policies may see

Before taking an action: only `AgentView(context, primary_sensor_bit)`. A paid optional observation reveals `secondary_bit` and `tertiary_bit`. The agent chooses whether to request a verified label *after* its action, before the simulator's target is used for labeling; the requested truth becomes available after exactly seven subsequent step boundaries. It cannot see hidden profile, actual error rate, correlation flag, or ground truth in other trials.

**Implementation caveat:** The Python evaluator constructs the hidden episodes and retains their truth to score outcomes; it also enqueues already-requested truth in its trusted scheduler. Correctness relies on respecting the policy/evaluator boundary; this is not a hardened runtime isolation layer.

## Budget and benchmark

- 30 paired seeds per environment, 800 trials each.
- Observation budget: `200/800` extra sensor queries, exactly `10/40` in each block.
- Label budget: `80/800` verified labels requested, exactly `4/40` in each block. `no_feedback` gets no labels by definition.
- Labels are delayed seven steps and can remain undelivered at trial end. Requested vs delivered counts are separate.
- The agent's memory stores a maximum of 32 *verified* error bits per context and sensor plus pairwise co-failures; some labels do not include extra readings, so corresponding histories are sparser.
- For `joint`, recency weighting decays older labels by `.93` per new label. `joint_no_decay` is an ablation that removes recency weighting. A contextwise online gain estimator updates only if both the baseline and queried decision can be scored from a **delivered** truth.
- The benchmark's experimental uncertainty-based method uses *learned primary sensor error* to prioritize observation and *prediction confidence* to prioritize label requests. This is a cheap baseline; it is not an exact Bayes-optimal agent.
- `novel` is a new, hand-constructed synthetic generator with stronger correlations and error inversion. It is an out-of-family stress test, **not** a rigorous independent real-world holdout.

## Results

All entries represent mean accuracy over 30 paired seeds. Metric is accuracy over 800 decisions; no difficult samples are excluded.

| Condition | No feedback | Random | Uncertainty | Joint | Joint no decay |
|---|---:|---:|---:|---:|---:|
| Stable | 78.74% | 79.22% | **81.15%** | 80.60% | 80.58% |
| Context-quality shift at midpoint | 78.94% | 78.78% | **79.97%** | 79.57% | 79.60% |
| IID quality, negative control | 78.47% | 78.18% | **78.49%** | 78.25% | 78.18% |
| Novel synthetic generator | 67.47% | 70.19% | **72.39%** | 71.77% | 71.83% |

Paired-bootstrap differences (30 seed-level differences, 2,000 bootstrap resamples, 95% percentile interval):

| Condition | Joint − random (pp) | Joint − uncertainty (pp) |
|---|---:|---:|
| Stable | +1.38 [+0.80, +2.00] | −0.56 [−1.09, −0.06] |
| Shift | +0.79 [+0.22, +1.54] | −0.40 [−0.83, +0.04] |
| IID | +0.08 [−0.29, +0.42] | −0.24 [−0.51, +0.03] |
| Novel | +1.58 [+0.88, +2.25] | −0.62 [−1.34, +0.10] |

These CIs quantify variability in the chosen synthetic seeds, not a generalization guarantee and not a preregistered confirmatory study. The first comparison shows the joint method below uncertainty on the stable generator; the remaining intervals overlap zero. Multiple comparisons are exploratory.

### Calibration and abstention limitations

Brier score (lower better) was best for `uncertainty` on all four families: 0.1431 stable, 0.1544 shift, 0.1674 IID and 0.1758 novel. We track "confident" predictions at posterior confidence ≥ 0.8, but confident coverage differs drastically among policies: do not compare accuracy of the confident subsets without controlling for coverage. The posterior assumes approximate sensor independence and is not assured to be calibrated.

### Negative outcomes / what we learned

1. **No clear joint-policy benefit over a strong cheap heuristic.** Adding a query-value learner and a second label-prioritization heuristic increased complexity without measurable superiority.
2. **Sparse labels are not free.** Labels requested late do not train the agent. In synthetic IID conditions, online estimation can be inferior to a fixed reasonably good prior.
3. **Memory decay was not a general improvement.** The no-decay variant often performed similarly or slightly better; a mechanism motivated only by forgetting lacks justification on these tasks.
4. **Identifiability limitation.** If all available sensors agree on a wrong answer, unlabeled observations cannot establish truth without external assumptions or intervention. Correlated errors are a fundamental obstacle.
5. **Not a biological model.** Crow-inspired selectivity is a design motivation, not anatomical reconstruction.

## Tests and trust boundaries

- 38 Python unit tests including preserved v0.2–0.4 regression tests: passed.
- New tests check unknown hidden labels, generated episode reproducibility, exact per-block budgets, labels not delivered early (including end-of-run pending labels), memory capacity, learned sensor inversion, zero-label no-learning behavior, and deterministic replay.
- `rust/src/v05.rs` is no-heap bounded-label logic and two Rust unit tests; **Rust toolchain unavailable here, no Rust build/test success claim**.
- No camera, microphone, Wi-Fi sensor, robot, mobile device or external AI API was involved.

## Planned go/no-go for v0.6

A new hypothesis worth testing: **selective causal interventions** may let an agent disambiguate correlated sensors more efficiently than repeated passive observation, even when labels are costly. This requires an intervention interface with safe environment transitions and visible outcomes, not a more elaborate confidence heuristic.

Pre-register: environments, held-out generators, label/sensing budgets, strong Bayesian/VOI and model-based baselines, accuracy and calibration, actual device latency/memory/energy, and at least one failure-oriented negative control. If it doesn't win against those baselines, do not claim a new architecture.

## Reproduction

```bash
python3 -m unittest discover -s python -v
python3 python/sparse_feedback.py --seeds 30 --trials 800 \
    --observation-fraction .25 --label-fraction .10 --delay 7 \
    --json results/sparse_feedback_30seeds.json
```

`results/sparse_feedback_30seeds.json` contains per-seed raw metrics and CIs; `results/summary_v05.csv` contains the table-friendly condition/policy summary.
