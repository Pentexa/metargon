# METARGON v0.8 — State, intervention and hidden-rule changes

**Date:** 6 November 2026. **Status:** Exploratory synthetic study. **Main finding:** A negative result for the more complex surprise-adaptive controller against a strong simple Bayesian comparison.

## Research question

Can an agent with a small context-dependent latent transition predictor adapt to hidden changes in which of two physical interventions opens a gate, under a strict action budget, without being given the simulator's hidden mode?

## Independently specified world

Five positions: start (0), key (1), switch (2), gate (3), goal (4). Public actions: move left/right, take/drop key, toggle switch, open gate. The key-only or switch-only requirement is hidden. In ordinary conditions, the matching mechanism succeeds with probability 0.93 and the nonmatching with 0.07. An action budget of 17 steps allows fallback; the tight-budget scenario allows 7 steps, so picking the wrong first mechanism has consequences.

The environment stores an independently generated uniform random draw for each mechanism *before* the policy acts. This provides matched potential outcomes for fair comparisons. The policy receives an immutable `Observation`, a context symbol and only the boolean gate result of its own attempts; it never receives the correct mechanism, latent probabilities, future changes or the evaluator's random numbers.

The public-effects BFS planner is written by hand and known to all agents. Only which gate transition works is predicted, so it is scientifically inaccurate to claim this is an autonomous general-purpose world model.

## Policies and baselines

- Always use key first; always use switch first; uniform random first attempt.
- Win-stay/lose-switch memory by context (simple, strong baseline).
- Fixed-hazard Bayesian model with probability updates from observed gate results.
- METARGON surprise-dependent forgetting, otherwise analogous Bayesian belief updates. Hyperparameters are fixed across conditions, not tuned against the test summaries.
- Oracle with hidden mechanism: a *privileged diagnostic comparator*, not a deployable competitor or guaranteed optimal policy.

All use the same environment, action budgets, movement planner, episode counts and paired random worlds. Their first mechanism differs; if budget permits, they try the other mechanism after failure. Real physical energy costs are not represented.

## Experiment

**30 paired random seeds × 240 episodes × eight conditions × seven policies** (403,200 executed episodes). Each policy starts without knowledge of the gate mode; policy memory persists across episodes of a given condition. Scenarios: stable context rules, one mid-run shift, repeated shifts, IID random modes, withheld novel context in the second half, noisy gate with 0.72/0.28 probabilities, tight 7-step budget with a mid-run shift, and sham gate openings independent of mode.

## Results

| Condition | METARGON success | Bayes success | Win-stay success | METARGON mean steps | Bayes mean steps |
|---|---:|---:|---:|---:|---:|
| stable | 93.65% | 93.65% | 93.65% | 6.51 | 6.51 |
| shift | 93.28% | 93.28% | 93.28% | 6.60 | 6.60 |
| rapid_shift | 93.56% | 93.56% | 93.56% | 7.29 | 7.29 |
| iid | 93.49% | 93.49% | 93.49% | 9.45 | 9.45 |
| novel | 93.94% | 93.94% | 93.94% | 6.50 | 6.50 |
| noisy_gate | 80.21% | 80.21% | 80.21% | 8.60 | 8.60 |
| tight_budget | 85.24% | 88.79% | 85.24% | 6.15 | 6.11 |
| sham | 80.40% | 80.40% | 80.40% | 8.92 | 8.92 |

**Primary discriminating result:** tight-budget METARGON success 85.24% vs Bayes 88.79%, difference -3.56 percentage points, paired-bootstrap 95% interval [-4.03, -3.03] pp across the 30 matched seeds. Bootstrap intervals describe this particular test generator and do not establish results for robotics in general.

The more elaborate surprise system **does not beat the simple Bayes baseline**. Under seven other conditions, the two models' *selected first mechanisms and overall success* matched across these samples. This means the current evaluation is insufficient to justify architectural complexity. A simple win-stay rule remains very competitive.

## Reasons for caution

1. The simple two-mode environment is so small that one observed gate success/failure often determines the optimal next action. It is not a rigorous test of rich causal models.
2. Even apparently high success (around 93% in generous-budget conditions) is **not** proof of smart selection: agents can try both mechanisms. Tight-budget success and first-choice accuracy are more meaningful.
3. The model is pre-programmed with the **class** of hidden rule and the nominal 0.93/0.07 probabilities; no general symbolic concept discovery occurs.
4. The new context is withheld only as an identifier in the **same** generator. It does not establish out-of-distribution transfer across independently designed worlds.
5. No real power, memory footprint of Python agent, hardware latency, robot stability or physical sensor reliability were measured. The Rust kernel has not been compiled here.
6. The oracle is privileged and can perform worse on individual stochastic samples; it is not a theoretical upper bound on every realized trajectory.
7. The experiment was developed and examined iteratively, so it is exploratory, not a held-out preregistered confirmation study.

## Follow-up experiments

- Replace the binary lock with 3–5 possible mechanisms and noisy partial state, making information-gathering choices genuinely necessary.
- Hide some public action effects and require learning a transition model from trajectories, rather than coding them into BFS.
- Use an independently written second simulator, with predetermined parameters and unseen evaluation seeds, to test transfer.
- Compare against contextual bandits, strong change-point methods, Bayesian POMDP solvers and simple finite-state strategies; perform ablations and fairness checks.
- Bring a verified model onto a microcontroller or Raspberry Pi class device and measure energy and memory per successful task.

## Reproduce

```bash
python3 python/metargon_demo.py
python3 -m unittest discover -s python -v
python3 python/metargon_world.py --seeds 30 --episodes 240 --json results/stateful_transfer_30seeds.json
```

The original historical experiments remain under their prior CORVID filenames to preserve the lineage; this project's name is **METARGON** from v0.8 onward.
