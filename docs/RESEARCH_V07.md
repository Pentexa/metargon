# CORVID v0.7 research report — learning what an action *will show*

**Date:** 2026-11-06  
**Status:** Exploratory, synthetic, negative / mixed result. No claim of novel artificial intelligence or biological fidelity.

## Question

Can a tiny agent improve action selection by learning a forward model of what it will **observe** after each diagnostic action and imagining both possible responses *before* taking that action?

This advances the **hypothesis**, not necessarily performance. v0.6 learned retrospective action credits. v0.7 instead learns a bounded one-step empirical distribution `P(diagnostic_bit | context, first_bit, chosen_action)` from **unlabeled action outcomes**, and separately learns `P(truth | first_bit, action, outcome)` using **only delayed verified labels**. It estimates the expected correctness of each hypothetical action/outcome branch and chooses accordingly.

This is *not* a dynamics model of the general physical world, a neural crow-brain model, or a demonstration of causal understanding. No external device is actuated.

## Protocol

- Same generator as v0.6 (`python/causal_probe.py`), with five conditions: stable, regime shift, randomly selected hidden profile, new synthetic parameter set, and sham intervention with no useful causal information.
- 30 fixed paired seeds, 800 trials per seed, and per-40-trial fixed quotas of 10 optional diagnostics and 4 truth-label requests. Thus each policy gets 200 paid diagnostic opportunities and 80 requested labels per run.
- Requested truth labels are delivered **seven steps later**; some remain pending when a run ends. No unlabeled observation is treated as a verified truth.
- An action can be a passive sensor read or a reversible *synthetic* pulse followed by a before/after difference. The policy sees the context, first sensor bit, and chosen diagnostic's response; **not** the hidden truth, profile, simulator rates, or counterfactual result of the unchosen action.
- All policies share exactly the same pre-generated worlds and diagnostic/label quota scheduling for each seed; diagnostic **action types** can differ.
- Buffers are bounded: 64 entries per local substream for the online response and verified-truth models. The Rust kernel has fixed arrays, but is **not yet executed** here, and does not reproduce Python's global partial-pooling rule.
- Experimental selection uses a small `.010` notional pulse penalty when scoring an expected future outcome. This is **not an energy measurement or empirical intervention cost**; `synthetic_utility = accuracy - .010 × pulses / trials` is a sensitivity metric only.
- Experiments are exploratory. The approach and simulation are designed in the same project; the `novel` condition is an unseen rate configuration *within the same simulator*, **not an independently engineered holdout dataset**.

## Policies

| Name | Description |
|---|---|
| passive | Always use a passive reading when given a diagnostic slot |
| probe | Always perform a pulse in a diagnostic slot |
| random | Choose passive/pulse at random |
| bayes_voi | v0.6 approximate Bayesian value-of-information, with a knowingly imperfect conditional-independence assumption |
| corvid | v0.6 contextual action-credit controller |
| predictive | v0.7 response model + delayed verified truth model + one-step hypothetical diagnostic branches |
| uniform_outcomes | Ablation: same classification model, but fixes diagnostic outcome probabilities to 50/50, removing response prediction |
| no_outcome_label | Ablation: retains response model, but skips direct supervised conditional-truth correction |

The privileged simulator-profile oracle from v0.6 is deliberately **not included** in this v0.7 comparison: its inside knowledge is not deployable.

## Results

Each value is the mean of 30 paired seeds (800 trials/seed). All numbers are synthetic.

| Condition | Passive | Always pulse | Bayesian VOI | v0.6 CORVID | v0.7 predictive | v0.7 minus Bayesian VOI (95% paired bootstrap) |
|---|---:|---:|---:|---:|---:|---|
| Stable | 67.12% | **69.52%** | 68.91% | 68.48% | 68.48% | −0.43 pp [−0.81, −0.04] |
| Shift | 67.36% | **68.67%** | 68.08% | 68.07% | 67.99% | −0.08 pp [−0.35, +0.22] |
| IID | 67.75% | **68.28%** | 68.02% | 68.08% | 67.95% | −0.06 pp [−0.29, +0.17] |
| Novel | 66.62% | **68.38%** | 68.12% | 67.69% | 67.92% | −0.20 pp [−0.53, +0.15] |
| Sham | **67.12%** | 65.62% | 66.69% | 66.39% | 66.71% | +0.02 pp [−0.27, +0.30] |

**Negative primary finding:** the predictive controller is not consistently superior to a simpler Bayesian comparator or to always probing where probes are useful. In the stable condition, the interval of the paired difference excludes zero in favor of the Bayesian baseline. These intervals are descriptive only: multiple comparisons, small designed simulator, and no independent confirmatory study.

**Sham diagnostic behavior:** the new agent chose 58.6 pulses of 200 slots on average, compared to v0.6's 85.5; it reached 66.71% accuracy compared with v0.6's 66.39%. However, **always passive remains better at 67.12%**; the system still doesn't fully avoid an uninformative intervention.

**Forecast ablation:** compare `predictive` to `uniform_outcomes` to isolate learning `P(response | action, context, first)`. Difference in accuracy ranges from −0.12 to +0.29 percentage points across conditions, with no consistent positive effect. In the sham simulation the paired difference was +0.29 pp [about +0.05, +0.54], an exploratory within-simulator indication that modeling outcomes may matter for avoiding bad actions, **not** conclusive evidence of generalization. The v0.7 response forecast's action-outcome Brier score is ~0.215–0.233, but that score is computed on actions **the policy chose**, not a common counterfactual action dataset; do not compare it directly across policies as proof of superior forecasting.

## Limitations / failure modes

1. **One-step, not full physics:** our "world model" only predicts the binary response of a chosen diagnostic. It does not learn state transitions across time or long-horizon planning.
2. **Small labels:** 10% of decisions request truth, and many exact `(context, first, action, observed response)` categories see few labels. Shrinkage and hierarchical pooling may mix incompatible groups.
3. **Incoherent approximations:** the response probability and the truth posterior are estimated independently. Their implied marginals may not be Bayes-consistent. Future versions should estimate a coherent joint model and explicitly handle uncertainty over its parameters.
4. **No real power data:** synthetic pulse penalty cannot establish energy efficiency, CPU efficiency, or fitness for an Edge device.
5. **Not a truly independent test suite:** v0.6 and v0.7 share authored simulator assumptions. More seeds in the same simulator don't replace external validation.
6. **No unbiased estimate for unchosen actions:** the agent only observes the selected diagnostic. The response samples are adaptively selected, so naive response-frequency estimates can be biased on unseen contexts and trajectories.
7. **Biological inspiration only:** no claim that the algorithm models crow NCL, neural assemblies, memory anatomy or energy metabolism.

## Next falsifiable experiment (v0.8 proposal)

**Question:** Can a *coherent Bayesian latent transition model* learn when an intervention changes what happens next, rather than merely the appearance of one immediate diagnostic response?

Requirements before making a claim:

- A second, independently built stateful simulator with hidden transitions, not just new parameters inside the v0.6 generator.
- Explicit comparison with a Bayes-adaptive POMDP / information-gain baseline, a contextual bandit, always-probe, always-passive, and ablations.
- Track two different outcome scores: (a) task reward, (b) calibration on **randomized diagnostic evaluation slots** in which the response action is assigned externally to avoid selected-action evaluation bias.
- Define a reproducible stateful, low-dimensional transition system and test recovery after unannounced rule changes, not only recognition of noisy bits.
- Track CPU time and memory usage in Python and Rust; energy only when a physical test board is available.
- Stop early or report a negative result if the new mechanism fails to beat simpler controls on independent held-out domains.

## Related prior art

These ideas overlap established fields; calling them first-of-their-kind would be inaccurate:

- Zhang et al., *Active learning for optimal intervention design in causal models* (Nature Machine Intelligence, 2023): https://www.nature.com/articles/s42256-023-00719-0
- Zeng et al., *A Survey on Causal Reinforcement Learning* (2025): https://pubmed.ncbi.nlm.nih.gov/40030342/
- *Information gathering in POMDPs using active inference* (2025): https://research.tudelft.nl/en/publications/information-gathering-in-pomdps-using-active-inference/
- *Active inference and artificial reasoning* (2026): https://www.nature.com/articles/s41467-026-77209-5

## Reproduce

```bash
python3 -m unittest discover -s python -v
python3 python/predictive_world_model.py --seeds 30 --trials 800 --json results/predictive_world_model_30seeds.json
```

The JSON contains the exact protocol parameters, per-seed metrics for all eight policies, ablation comparisons, and descriptive paired bootstrap intervals. The CSV and plain-text summary are in `results/`.
