# CORVID v0.6 research report — active diagnostics, not passive sensing alone

Date: 2026-11-06. Status: **exploratory synthetic experiment, negative result for current learned policy**.

## Hypothesis

A resource-bounded controller might choose between different **types** of information-gathering actions, rather than simply choosing when to sample a sensor. A reversible probe perturbs a system, then records its response; passive sensing observes an unchanged system. In contexts where passive channels share failure modes, probing might reveal a hidden property more clearly. In contexts where the probe is uninformative or noisy, passive sensing may be better.

We test whether **learned action-specific benefit**, estimated only from sparse delayed verified labels, is preferable to strong simple strategies under a fixed number of diagnostic opportunities.

## Protocol and information barriers

- Binary hidden actuator state per trial. The first sensor bit and context identifier are visible. The hidden truth, simulator regime and diagnostic error probabilities are inaccessible to deployable policies.
- A paid diagnostic is either a passive sensor read or a simulated pulse followed by a before/after state-difference read.
- All strategies use identical pre-generated worlds per seed (paired comparison).
- Exactly 10 diagnostic slots per 40 steps; 200 per 800-decision run.
- Exactly 4 label requests per 40 steps; 80 total. Each requested truth is delivered after seven *subsequent* steps. Some outstanding labels remain undelivered at the end.
- Label requests are chosen before the hidden truth is scored. Non-queried truths never update online reliability or action credit.
- Histories are bounded to 64 labeled events per context and sensor channel. No unlabeled vote is treated as verified truth.
- 30 paired seeds, 800 steps each; conditions: stable, half-run shift, randomly changing profile, novel synthetic rates and sham probe.
- Randomness is fixed by explicit seeds. All algorithms have the same Python reference simulator and the same optional diagnostic *slot count*. **Different action types are not guaranteed equal real-world cost.**

## Comparators

1. `passive`: paid passive reading at every allocated slot.
2. `probe`: perform a pulse at every allocated slot.
3. `random`: choose passive/probe randomly.
4. `uncertainty`: simple threshold on labeled primary-channel error.
5. `plug_in_error`: choose the diagnostic with the smaller estimated labeled error rate, plus exploration.
6. `bayes_voi`: calculate expected 0–1 classification accuracy for each binary diagnostic using learned error rates and an independent-channel likelihood model; explore occasionally. This is **not** an exact Bayes-optimal causal intervention policy.
7. `corvid`: discounted contextual action credit from requested verified labels with an exploration bonus.
8. `oracle`: knows the hidden regime and uses a regime-specific preferred action. **Privileged and not guaranteed optimal.**

## Results

See `../results/summary_v06.txt` for human-readable numbers and `../results/causal_probe_30seeds.json` for all per-seed scores, selected action counts, budget use, Brier scores, and paired bootstrap differences.

| Condition | CORVID | Bayesian VOI | Always probe | Passive | CORVID - Bayesian VOI (95% paired bootstrap) |
|---|---:|---:|---:|---:|---:|
| Stable | 68.48% | 68.91% | 69.52% | 67.12% | −0.43 pp [−0.85, −0.02] |
| Shift | 68.07% | 68.08% | 68.67% | 67.36% | −0.00 pp [−0.28, +0.29] |
| IID | 68.08% | 68.02% | 68.28% | 67.75% | +0.06 pp [−0.24, +0.36] |
| Novel | 67.69% | 68.12% | 68.38% | 66.62% | −0.43 pp [−0.82, −0.05] |
| Sham | 66.39% | 66.69% | 65.62% | 67.12% | −0.30 pp [−0.60, −0.02] |

The `sham` control deliberately breaks the relationship between actuation and the hidden truth. Probing no longer helps. CORVID nevertheless uses the probe about 86 times in 200 diagnostic slots, showing **insufficient adaptation** to a useless intervention.

Important: Correlation and regime mechanisms are handcrafted, all data are synthetic and there is no external validation split. The confidence intervals show seed-to-seed variability conditional on this simulation, *not* broad applicability to other tasks.

## Interpretation

- **Active intervention can provide useful information:** always-probe beats passive in four of five handcrafted conditions, not in the sham condition. This does *not* demonstrate an intrinsically better intelligence architecture; we built informative and uninformative probes into the simulator.
- **The particular learned controller is insufficient:** selecting the most useful probe is harder than detecting uncertainty. Sparse labels (~80 of 800 decisions) leave few verified samples per action/context. Reward estimates are noisy, and correlated passive failures violate simplistic evidence fusion.
- **The privileged profile comparator is only a diagnostic reference:** it cannot be counted as a fair deployed policy and is not a mathematical accuracy upper bound.
- **Energy and latency were not measured:** all action costs are unit tokens. No conclusions about joules, cycles, battery life or biological neural efficiency follow.
- **Not a causal-discovery breakthrough:** the simulation explicitly encodes the intervention mechanism. We did not recover a novel causal structure from real data.

## Methodological shortcomings / next milestone

1. Replace the discounted contextual bandit with an explicit action-outcome model that predicts how diagnostic evidence changes under intervention. Compare against a proper finite POMDP controller (small enough for exact planning) and a well-tuned contextual bandit.
2. Pre-register a fresh family of synthetic environments and a separate benchmark implementation *before* tuning the new method. Otherwise, benchmark overfitting can masquerade as architecture progress.
3. Add an explicit multi-objective cost model for sensing, actuation, CPU, time and risk, and verify those costs on physical Edge hardware where safe.
4. Use a real reversible diagnostic example (e.g., a low-voltage test jig or a simulated controllable object with verified state transitions) before making claims about physical-world performance.
5. Explicitly study cases where no sequence of allowed probes identifies the hidden state; the agent should report non-identifiability rather than fabricate certainty.

## Relevant prior art

- Agrawal et al., 2019, **ABCD-Strategy: Budgeted Experimental Design for Targeted Causal Structure Discovery** — https://proceedings.mlr.press/v89/agrawal19b.html
- 2024, **Active sensing with predictive coding and uncertainty minimization** — https://doi.org/10.1016/j.patter.2024.100983
- Wang et al., ICML 2025, **Bayesian Active Learning for Bivariate Causal Discovery** — https://proceedings.mlr.press/v267/wang25bp.html

These works show that budgeted and causal experimental design already exist. CORVID has not demonstrated a novel algorithm beyond this literature.
