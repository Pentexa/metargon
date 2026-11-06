# CORVID v0.2 — Falsifiable research protocol

## Scientific goal

Can a bounded-memory agent use sensory uncertainty to decide **when** an extra observation is worth taking, improving decision accuracy with an equal observation budget? This is the first instrumented research question, **not** a claim that we reproduced a crow brain or invented a new AI architecture.

## Biological hypothesis vs engineering implementation

Biological inspiration: corvids can use working memory, selective attention and flexible decisions. This task is a tiny engineering analogue, not a biological model. The current algorithm is a Bayesian context belief tracker plus an observation budget policy; both are well-established concepts.

## Environment

For every trial the environment privately draws:
- A context (four values), a hidden binary cue, and an unobserved cue-to-action mapping.
- A *visible* initial noisy cue and a visible quality flag (low/high), with signal error probabilities 42% and 3% respectively.
- A second independently sampled cue (8% error), revealed **only if** the agent spends a query token.
- Distractor events. These do not carry useful information in this version.
- A binary outcome with 5% feedback noise.

The mapping reverses halfway through each 400-trial run. **The learner is not told when it changes.** The agent maintains a bounded belief per context; an additional observation may update the cue estimate. Query budgets are 100 per 400 trials, exactly identical for both query policies.

## Baselines (paired seeds)

- `no_query`: identical Bayesian action/outcome belief, no optional cue observation.
- `random_query`: identical Bayesian belief, query opportunities selected randomly online at exactly the 25% trial quota.
- `targeted_query`: identical Bayesian belief, query low-quality cues first subject to identical quota, falling back to the final trials if needed to spend the whole quota.

All agents use the same predetermined environment trials and potential second observations. Any advantage of `targeted_query` **is unsurprising**, because the visible quality flag reveals which first observations are unreliable. These baselines are meant to verify instrumentation and the possibility of a budgeted mechanism, not prove novel cognition.

## Success and failure criteria for next milestone

**Gate 1 — Reproducibility:** all Python tests pass, outputs reproducible by seed, equal query budgets and no hidden variables in model APIs.

**Gate 2 — Engine independence:** Rust core can run on CPU without allocating per decision and reproduces reference posterior computations. Rust source here is uncompiled until checked on a machine with Cargo.

**Gate 3 — Actual research:** remove the quality flag, use nonstationary sensory errors, introduce expensive *optional* sensing actions of varying usefulness, and learn a query policy from feedback rather than use hard-coded if-conditions. Compare against random, fixed allocation, UCB, entropy-based information gain and budgeted Bayesian baselines on held-out environments. Test multiple reward models and task families. If no statistically reliable benefit, reject the novelty hypothesis.

**Gate 4 — Edge evidence:** measure RSS, peak allocations, wall latency, and device-level joules, not just Python call counts; record device, CPU governor and power conditions.

## Threats to validity

- Quality labels are directly provided; selective allocation is an obvious rule.
- Environment is binary, with only four contexts, and not representative of real-world perception.
- Context transition hazard and sensor reliability are supplied as model assumptions; future tests must estimate or misspecify them.
- Group averages hide per-seed variance; inspect raw JSON.
- Runtime on a desktop Python interpreter is not edge-device energy.
- `decision_time_ms` measures Python overhead and is not a standardized performance claim.
- We do not claim general reasoning, open-ended problem solving, biological fidelity or novelty.

## Why Rust?

The small `rust/` crate provides an independent, bounded `Core<const CONTEXTS: usize>` with `#![forbid(unsafe_code)]`. It can be built and tested with `cargo test` and `cargo run --example basic`. It intentionally does not use a local LLM or need an external AI API for runtime inference.

## Next experiment to make this meaningful

Train a *learned* sensing scheduler on a family of novel sensor degradation processes, then test it on unseen kinds of drift and new tasks where the optimal sensing time cannot be known from a quality tag. Charge each optional perception operation in actual observed compute or energy. The research hypothesis is false if the learned scheduler cannot outperform matched simple heuristics under held-out conditions.
