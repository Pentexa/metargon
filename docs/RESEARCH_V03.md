# CORVID v0.3 — Pre-registered-style experiment record

**Status:** completed synthetic experiment, **not** proof of new intelligence, biological equivalence, or lower energy usage.

## Research question

Can an agent **infer** which first-party observations are unreliable, without receiving per-trial quality labels, and reallocate a hard-limited number of second observations better than a random quota policy?

## What changed since v0.2

- No visible low/high quality signal. `AgentView` carries **only** `context` and the observed bit.
- Sensor error is unknown to the deployable policies; reliability is estimated online from disagreements between first and optional second readings.
- Calibrator uses a 16-sample sliding window, two pseudo-observations, and a fixed assumption that independent reference sensor error is 8%.
- Four contexts; primary sensor error depends on context, possibly changing at halfway. The decision mapping also reverses at halfway and is never revealed.
- Three environments: stationary error mapping, swapped context error mapping, and unpredictable per-trial errors (negative control).
- Same exact 25% query budget for all querying policies, distributed across 50-trial time blocks. All policies see the same seeded episodes and potential second readings. The `oracle_quality` arm is explicitly nondeployable.
- 30 independent paired seeds × 400 decisions × three environments; deterministic pair-bootstrap confidence intervals over **seed-level** differences.

## Policies

- `no_query`: makes no secondary observations; unknown sensor reliability stays at prior.
- `random_query`: spends matched quota randomly; calibrates reliability from its acquired readings.
- `learned_query`: uses **online-learned primary-sensor reliability** to weight its query probability. Its acquisition formula is **handcrafted**, not reinforcement learned: 0.1 + 4.5 × estimated_error² + 0.25 / sqrt(1 + sample_count). All other components and budget enforcement match random.
- `oracle_quality`: uses the actual primary error probability **for scheduling and inference**, so it is an optimistic nondeployable comparator. It does not see hidden cue or correct action.

## Scientific limitations

1. The 8% error rate and independence of the reference sensor are *given*. Correlated sensors can defeat the estimator.
2. The environment explicitly provides the context index; changing error rates are only predictable when context carries information.
3. The four contexts and hidden binary mode are tiny. This is a calibration and active-observation *toy problem*, not an open-world reasoning benchmark.
4. In `swap`, the rule and reliability change at the same time; the experiment does not isolate their individual effects.
5. Policy hyperparameters are chosen heuristically. We did not tune on a distinct train set nor run a held-out task family.
6. 30-seed bootstrap CIs only quantify seed variability in this simulation. They are not evidence of generalization to real devices.
7. There is no measurement of RSS, edge power, joules, thermal behavior or decision latency under target hardware.
8. The Rust kernel is an independently authored implementation of the calibration formula. **If Cargo is absent, its compilation is not verified.** Python tests cannot certify Rust behavior.
9. Our estimated error after a query is used immediately on the same reading. The estimator observes two readings from that trial but not the hidden cue or action outcome.
10. This is inspired by corvid selective attention and working memory, not a neural or anatomical crow-brain simulation.

## Interpretation requirements

- Success: learned > random on stationary reliability patterns at matched cost, and preferably adapt following a swap.
- Falsification/negative control: learned should **not** enjoy a systematic advantage where context has no predictive information about reliability (`iid`).
- A more serious result requires stronger policy baselines (e.g. UCB/Bayesian value of information), correlated sensors, variable observation costs, unseen contexts, and real energy measurements.
- Treat `oracle_quality` strictly as a reference ceiling, not as usable code for real agents.

## Exact reproduction

```sh
python3 -m unittest discover -s python -v
python3 python/learned_uncertainty.py --trials 400 --seeds 30 --query-fraction 0.25 --json results/learned_uncertainty_30seeds.json
cd rust && cargo test
```

See `results/learned_uncertainty_30seeds.json` for seed-level data, CI bounds and configuration. The JSON stores statistical observations, not physical measurements.

## Next go/no-go experiment (v0.4)

1. Make the reference sensor's reliability **unknown**, and sometimes correlated with the first sensor; check whether the learned confidence estimate is robust or detects invalid assumptions.
2. Replace the handcrafted query weight with a bounded online contextual bandit, and include UCB and value-of-information baselines at equal total *sensing + compute* cost.
3. Split train and test environments by distinct noise-generating processes and new context counts; do not tune on test set.
4. Add actual edge timing and power recording with device identification. Only claim efficiency if energy **per successful decision** improves at fixed accuracy or vice versa.

**Stop condition:** If CORVID cannot beat strong fair baselines or adaptation under held-out shifts, abandon the novelty narrative and revise the mechanism.
