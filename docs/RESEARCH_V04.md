# CORVID v0.4 — Conditional attention under correlated sensor failures

**Experiment type:** synthetic, supervised active perception; 30 independent paired seeds and 400 decisions/seed in each environment. **Not** evidence of a new brain architecture, general intelligence, real power savings, or superiority to state-of-the-art agents.

## Research question

Can a small bounded-memory agent infer the *incremental value* of another observation through delayed verified feedback, when neither sensor error probabilities nor a reference sensor's reliability are supplied, including cases in which two sensors fail together?

## What is new in this experiment relative to v0.3

- `AgentView` has only `(context, primary_bit)`; no quality flag, hidden profile, truth, or model error rate.
- Query reveals readings from **two** auxiliary sensors for one budget token. Two sensors may fail together. The estimator tracks each sensor's labeled error and a pair-cofailure statistic.
- After the agent has made its decision, the simulator discloses a **verified binary truth** as delayed feedback. This is an explicit strong assumption that removes the v0.3 need for a known 8% reference-sensor error. It is **not** a no-label or fully unsupervised algorithm.
- `bandit` is an online contextual observation-scheduling policy. For queried cases, it learns the observed gain/loss in decision correctness relative to the action that the unqueried sensor would have produced. It samples a small uncertainty term and uses a fixed probability-mapping formula. This is learned query value, not a learned neural controller.
- Policies have the same exact 10 optional observations per 40-trial block; `no_query` uses zero. Each policy sees identical pre-generated hidden episodes within a seed.
- Compare against `random`, primary-error-based `uncertainty` and UCB query-value allocation, not only a weak random baseline.

## Environments

- **stable:** four persistent hidden sensing profiles assigned to four visible context IDs.
- **shift:** hidden profile-to-context map rotates at midpoint without any visible shift notice.
- **iid:** each episode's profile is drawn afresh; context provides no predictive information. Negative control, where overfitting to contexts should not be rewarded.

Hidden profiles include: base noisy with good queries; very reliable base with two correlated bad auxiliary readings; primary+secondary jointly wrong with a reliable tertiary; and approximately independent, moderate noise.

## Frozen exploratory protocol

- `seeds=30`, `trials=400`, `query_fraction=0.25`; each querying policy uses exactly 100 queries per seed.
- Accuracy over all decisions is the primary metric.
- Secondary: post-midpoint 60 decision accuracy, accuracy on trials involving at least one co-failing pair, query helps/hurts, and a **synthetic work-unit count**.
- CIs use paired bootstrap over seed-level differences. These intervals represent synthetic simulation sampling uncertainty only; they do not quantify real-world generalization.
- Calibration: window of 32 verified feedback samples per context and sensor, Beta(1,3) prior. Optional sensors are only updated when queried.
- The query-benefit learner uses an exponentially weighted estimate with decay 0.95. This learner sees only rewards computed after action and only for queried episodes.

## Results (30 seeds × 400 trials)

| Condition | No query | Random | Fixed uncertainty | UCB value | CORVID bandit | Bandit − random (95% CI, percentage points) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stable | 77.65% | 80.35% | **81.80%** | 81.19% | 81.70% | +1.35 [+0.81,+1.86] |
| shift | 77.12% | 79.12% | 80.40% | 79.90% | **80.47%** | +1.36 [+0.88,+1.82] |
| iid negative control | 77.27% | **78.69%** | 78.38% | 78.62% | 78.17% | −0.53 [−1.09,+0.10] |

**Interpretation:** The learned-bandit query selection beats random under stable and changing context-reliability structure, but **does not show a meaningful improvement over the stronger fixed-uncertainty method**. It underperforms random in the iid control. The experimental core does **not** yet meet a compelling architectural novelty bar.

**Important comparability note:** v0.3 and v0.4 use different environments and feedback assumptions. Do not compare their raw accuracies as though v0.4 were an iteration of the identical benchmark.

## Threats to validity

1. Unlike most deployments, verified truth is guaranteed after every action. Without an independent outcome signal, one cannot generally identify jointly wrong sensors from pairwise agreement alone.
2. All observations are binary, contexts are explicitly provided, actions are one-bit classification, and the sensor count is three.
3. Synthetic profiles were deliberately designed with persistent context-reliability patterns and a midpoint shift. No physical camera/microphone or robot is tested.
4. The profile types, policy gain mapping, smoothing, decay, and confidence bonus are author-selected exploratory settings, **not** tuned on an independent training suite. Follow-up should freeze them and benchmark held-out profile families.
5. The fused signal treats sensor evidence as approximately independent except for a coarse downweighting of jointly failing auxiliaries; this is not a theoretically exact correlated Bayesian fusion method.
6. Other intelligent baselines could outperform our learner. A weak baseline comparison cannot establish novelty.
7. The `work_units` proxy counts selected agent operations, not wall time or energy. Rust module is added but not compiled in this environment (no `cargo` available).
8. The confidence intervals are over pre-generated synthetic seeds, and the experiment's examples are not a validated sample of the real world.

## Next go/no-go scientific test (v0.5)

We should **not** call this a new intelligence architecture yet. To advance:

- Train allocation on one set of sensing regimes and evaluate, without tuning, on genuinely **held-out sensor-generating mechanisms** including adversarial correlation patterns.
- Eliminate guaranteed per-step ground-truth feedback: introduce sparse, delayed, costly labels, and study *identifiability* / failure detection.
- Compare against a calibrated Bayes/VOI controller and contextual Thompson sampling; ablate memory decay, pair correlation penalty, query benefit adaptation separately.
- Measure real Edge hardware CPU latency, RSS, and energy per correct decision via an external power meter or platform-specific counters, reporting device name and measurement method.
- Seek a falsifiable mechanism that beats strong comparators across tasks, not merely a corvid-inspired label for known contextual-bandit ingredients.

## Reproduce

```bash
python3 -m unittest discover -s python -v
python3 python/adaptive_attention.py --seeds 30 --trials 400 --query-fraction .25 --json results/adaptive_attention_30seeds.json
cd rust && cargo test
```

The Python experiment and tests have been executed in the authoring environment. Rust compilation has **not** been verified there because Rust tooling was absent.
