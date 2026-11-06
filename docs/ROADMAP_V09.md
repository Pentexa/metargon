# METARGON v0.9 — Next scientific gate

**Do not claim novelty yet.** v0.8 found that surprise-gated forgetting trails a strong Bayesian baseline under strict budgets.

Hypothesis to test next: when both actuator dynamics and task goals are partially observed, a learned *factored* transition model may reduce necessary trials relative to recency rules at the same computation cost.

Minimum acceptance gates:

1. Independently authored simulator, not a variant of the v0.8 generator.
2. At least 3 hidden mechanisms, one correlated distractor and a noisy observation channel.
3. Ablate transition model, episodic memory and surprise component separately.
4. Pre-register parameters and a held-out seed list; publish negative results too.
5. Match action and observation costs, not only action counts; account for planner CPU time.
6. Compare Bayesian POMDP, contextual bandit, finite-state win-stay and random.
7. Later measure RAM, latency, and joules/goal on actual Edge hardware.

Passing Python tests means software correctness on tested cases, **not a new scientific result**.
