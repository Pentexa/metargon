# METARGON Cognitive Intelligence Benchmark v0.1

- Date: 2026-11-12
- Git head: `ccb1627f3f1fd504c62ee471c17d7df017fc415f`
- Seeds: [0, 1, 2, 3]
- Episode sizes: {'train': 24, 'eval': 8, 'revisit': 7, 'switch_half': 10}
- Frozen split counts: {'enumerated': 624, 'ood': 25, 'solvable': 245, 'test': 49, 'train': 147, 'val': 49}
- Distinct true (door, rule) pairs in training stream: 7

## Descriptive indicators (0–1; NOT a composite IQ)

| dimension | indicator | value |
|---|---|---|
| Causal discovery | coverage | 1.000 |
| Causal discovery | verified_precision | 0.200 |
| Learning efficiency | success_rate_second_half | 0.854 |
| Learning efficiency | first_verified_rule | 1.000 |
| Knowledge retention | revisit_preserved | 1.000 |
| Knowledge retention | restart_parity | 1.000 |
| Generalization | zero_shot_test_coverage | 0.469 |
| Generalization | test_success_rate | 0.812 |
| Adaptability | success_after_switch | 0.600 |
| Adaptability | recovery_speed | 0.500 |
| Adaptability | wrong_belief_correction | 0.526 |
| Planning efficiency | fallback_reduction | 0.832 |
| Planning efficiency | steps_to_success_improvement | 0.143 |

Formula notes:
- **causal_discovery**: coverage = fraction of true (door, rule) pairs found as hypotheses; precision = verified correct / verified.
- **learning_efficiency**: second-half success rate on the training stream; episodes-to-first-verified-rule inverted into 0-1.
- **knowledge_retention**: revisit = success preserved at late vs early occurrence; restart parity = restored/direct success.
- **generalization**: zero-shot coverage = true rules already verified in pre-eval memory; test success rate.
- **adaptability**: success after rule switch; recovery speed = 1/(1+lag); fraction of pre-switch wrong beliefs contradicted after.
- **planning_efficiency**: fallback reduction = 1 − fallback rate on test.

CI methods — **W** = pooled Wilson interval (bounded probability metrics); **t** = Student-t on per-seed means (unbounded metrics).

## 1. Causal discovery

| metric | mean ± 95% CI | CI |
|---|---|---|
| metargon_trained.verified_correct | 1.750  [1.056, 2.444] | t (4 seeds) |
| metargon_trained.verified_wrong | 7.000  [2.327, 11.673] | t (4 seeds) |
| metargon_trained.candidates_correct | 0.000  [0.000, 0.000] | t (4 seeds) |
| metargon_trained.contradicted_truth | 5.250  [4.556, 5.944] | t (4 seeds) |
| metargon_trained.hypotheses_total | 54.500  [34.338, 74.662] | t (4 seeds) |
| metargon_trained.discovery_coverage | 1.000  [0.879, 1.000] | W (28 events) |
| metargon_trained.verified_precision | 0.200  [0.100, 0.359] | W (35 events) |
| metargon_muted.verified_correct | 1.750  [1.056, 2.444] | t (4 seeds) |
| metargon_muted.verified_wrong | 7.000  [2.327, 11.673] | t (4 seeds) |
| metargon_muted.candidates_correct | 0.000  [0.000, 0.000] | t (4 seeds) |
| metargon_muted.contradicted_truth | 5.250  [4.556, 5.944] | t (4 seeds) |
| metargon_muted.hypotheses_total | 54.500  [34.338, 74.662] | t (4 seeds) |
| metargon_muted.discovery_coverage | 1.000  [0.879, 1.000] | W (28 events) |
| metargon_muted.verified_precision | 0.200  [0.100, 0.359] | W (35 events) |

## 2. Learning efficiency

| metric | mean ± 95% CI | CI |
|---|---|---|
| metargon_trained.episodes_to_first_success | 1.000  [1.000, 1.000] | t (4 seeds) |
| metargon_trained.episodes_to_first_verified_correct | 1.000  [1.000, 1.000] | t (4 seeds) |
| metargon_trained.mean_steps_to_success_second_half | 20.064  [10.668, 29.460] | t (4 seeds) |
| metargon_trained.success_rate | 0.698  [0.600, 0.781] | W (96 events) |
| metargon_trained.success_rate_first_half | 0.542  [0.403, 0.674] | W (48 events) |
| metargon_trained.success_rate_second_half | 0.854  [0.728, 0.928] | W (48 events) |
| metargon_muted.episodes_to_first_success | 1.000  [1.000, 1.000] | t (4 seeds) |
| metargon_muted.episodes_to_first_verified_correct | 1.000  [1.000, 1.000] | t (4 seeds) |
| metargon_muted.mean_steps_to_success_second_half | 20.064  [10.668, 29.460] | t (4 seeds) |
| metargon_muted.success_rate | 0.698  [0.600, 0.781] | W (96 events) |
| metargon_muted.success_rate_first_half | 0.542  [0.403, 0.674] | W (48 events) |
| metargon_muted.success_rate_second_half | 0.854  [0.728, 0.928] | W (48 events) |

## 3. Knowledge retention

| metric | mean ± 95% CI | CI |
|---|---|---|
| metargon_trained.retention_preserved | 1.000  [0.757, 1.000] | W (12 events) |
| metargon_trained.success_early | 1.000  [0.757, 1.000] | W (12 events) |
| metargon_trained.success_late | 1.000  [0.757, 1.000] | W (12 events) |
| metargon_trained.restart_parity | 1.000  [1.000, 1.000] | t (4 seeds) |
| metargon_trained.restart_success_restored | 0.812  [0.647, 0.911] | W (32 events) |
| metargon_muted.retention_preserved | 1.000  [0.757, 1.000] | W (12 events) |
| metargon_muted.success_early | 1.000  [0.757, 1.000] | W (12 events) |
| metargon_muted.success_late | 1.000  [0.757, 1.000] | W (12 events) |
| metargon_muted.restart_parity | 1.000  [1.000, 1.000] | t (4 seeds) |
| metargon_muted.restart_success_restored | 0.719  [0.546, 0.844] | W (32 events) |

## 4. Generalization

| metric | mean ± 95% CI | CI |
|---|---|---|
| metargon_trained.eval_val.success_rate | 0.781  [0.612, 0.890] | W (32 events) |
| metargon_trained.eval_val.mean_steps_to_success | 23.375  [15.807, 30.943] | t (4 seeds) |
| metargon_fresh.eval_val.success_rate | 0.500  [0.336, 0.664] | W (32 events) |
| metargon_fresh.eval_val.mean_steps_to_success | 10.250  [9.849, 10.651] | t (4 seeds) |
| metargon_trained.eval_test.success_rate | 0.812  [0.647, 0.911] | W (32 events) |
| metargon_trained.eval_test.mean_steps_to_success | 18.333  [12.389, 24.277] | t (4 seeds) |
| metargon_fresh.eval_test.success_rate | 0.750  [0.579, 0.867] | W (32 events) |
| metargon_fresh.eval_test.mean_steps_to_success | 21.381  [10.796, 31.966] | t (4 seeds) |
| metargon_trained.eval_ood.success_rate | 0.156  [0.069, 0.318] | W (32 events) |
| metargon_trained.eval_ood.mean_steps_to_success | 12.500  [1.742, 23.258] | t (2 seeds) |
| metargon_fresh.eval_ood.success_rate | 0.312  [0.180, 0.486] | W (32 events) |
| metargon_fresh.eval_ood.mean_steps_to_success | 15.600  [11.297, 19.903] | t (2 seeds) |
| metargon_trained.eval_val.zero_shot_coverage | 0.438  [0.282, 0.607] | W (32 events) |
| metargon_trained.eval_test.zero_shot_coverage | 0.469  [0.309, 0.636] | W (32 events) |
| metargon_trained.eval_ood.zero_shot_coverage | 0.562  [0.393, 0.718] | W (32 events) |

## 5. Adaptability

| metric | mean ± 95% CI | CI |
|---|---|---|
| metargon_trained.success_before | 1.000  [0.912, 1.000] | W (40 events) |
| metargon_trained.success_after | 0.600  [0.446, 0.737] | W (40 events) |
| metargon_trained.recovery_lag | 1.000  [1.000, 1.000] | t (4 seeds) |
| metargon_trained.wrong_belief_correction | 0.526  [0.317, 0.727] | W (19 events) |
| metargon_muted.success_before | 1.000  [0.912, 1.000] | W (40 events) |
| metargon_muted.success_after | 0.625  [0.470, 0.758] | W (40 events) |
| metargon_muted.recovery_lag | 1.500  [0.699, 2.301] | t (4 seeds) |
| metargon_muted.wrong_belief_correction | 1.000  [0.940, 1.000] | W (60 events) |
| metargon_fresh.success_before | 1.000  [0.912, 1.000] | W (40 events) |
| metargon_fresh.success_after | 0.450  [0.307, 0.602] | W (40 events) |
| metargon_fresh.recovery_lag | 2.500  [0.096, 4.904] | t (4 seeds) |

## 6. Planning efficiency

| metric | mean ± 95% CI | CI |
|---|---|---|
| metargon_trained.eval_test.fallback_rate | 0.168  [0.146, 0.193] | W (964 events) |
| metargon_trained.eval_test.invalid_rate | 0.000  [0.000, 0.004] | W (964 events) |
| metargon_trained.eval_test.planner_ops_per_step | 177.821  [107.685, 247.956] | t (4 seeds) |
| metargon_trained.eval_test.plan_cache_hits_per_step | 2.906  [2.807, 3.006] | t (4 seeds) |
| metargon_trained.eval_test.plans_used_per_step | 0.548  [0.355, 0.741] | t (4 seeds) |
| metargon_fresh.eval_test.fallback_rate | 0.246  [0.222, 0.271] | W (1196 events) |
| metargon_fresh.eval_test.invalid_rate | 0.000  [0.000, 0.003] | W (1196 events) |
| metargon_fresh.eval_test.planner_ops_per_step | 289.386  [217.109, 361.662] | t (4 seeds) |
| metargon_fresh.eval_test.plan_cache_hits_per_step | 2.420  [1.904, 2.936] | t (4 seeds) |
| metargon_fresh.eval_test.plans_used_per_step | 0.496  [0.417, 0.575] | t (4 seeds) |

## Baseline context (evaluation pools)

### eval_val

| policy | success_rate | success CI | mean_steps_to_success | within_episode_adaptation |
|---|---|---|---|---|
| metargon_trained | 0.781  [0.612, 0.890] | W (32 events) | 23.375  [15.807, 30.943] | 0.806  [0.637, 0.908] |
| metargon_fresh | 0.500  [0.336, 0.664] | W (32 events) | 10.250  [9.849, 10.651] | 0.516  [0.348, 0.680] |
| random | 0.031  [0.006, 0.157] | W (32 events) | 28.000 | 0.034  [0.006, 0.172] |
| episodic | 0.031  [0.006, 0.157] | W (32 events) | 28.000 | 0.034  [0.006, 0.172] |
| qlearning | 0.406  [0.255, 0.577] | W (32 events) | 37.042  [34.902, 39.182] | 0.448  [0.284, 0.625] |
| oracle | 1.000  [0.893, 1.000] | W (32 events) | 12.875  [12.875, 12.875] | 1.000  [0.893, 1.000] |

### eval_test

| policy | success_rate | success CI | mean_steps_to_success | within_episode_adaptation |
|---|---|---|---|---|
| metargon_trained | 0.812  [0.647, 0.911] | W (32 events) | 18.333  [12.389, 24.277] | 0.839  [0.674, 0.929] |
| metargon_fresh | 0.750  [0.579, 0.867] | W (32 events) | 21.381  [10.796, 31.966] | 0.750  [0.579, 0.867] |
| random | 0.000  [0.000, 0.107] | W (32 events) | — | 0.000  [0.000, 0.110] |
| episodic | 0.000  [0.000, 0.107] | W (32 events) | — | 0.000  [0.000, 0.110] |
| qlearning | 0.250  [0.133, 0.421] | W (32 events) | 34.167  [21.813, 46.521] | 0.286  [0.153, 0.471] |
| oracle | 1.000  [0.893, 1.000] | W (32 events) | 14.250  [13.449, 15.051] | 1.000  [0.893, 1.000] |

### eval_ood

| policy | success_rate | success CI | mean_steps_to_success | within_episode_adaptation |
|---|---|---|---|---|
| metargon_trained | 0.156  [0.069, 0.318] | W (32 events) | 12.500  [1.742, 23.258] | 0.156  [0.069, 0.318] |
| metargon_fresh | 0.312  [0.180, 0.486] | W (32 events) | 15.600  [11.297, 19.903] | 0.312  [0.180, 0.486] |
| random | 0.031  [0.006, 0.157] | W (32 events) | 40.000 | 0.033  [0.006, 0.167] |
| episodic | 0.031  [0.006, 0.157] | W (32 events) | 40.000 | 0.033  [0.006, 0.167] |
| qlearning | 0.094  [0.032, 0.242] | W (32 events) | 60.750  [42.462, 79.038] | 0.111  [0.039, 0.281] |
| oracle | 1.000  [0.893, 1.000] | W (32 events) | 15.969  [14.045, 17.892] | 1.000  [0.893, 1.000] |

## Methodology

- **Subject**: the unmodified Academy student (`MetargonAcademyAgent`, default configuration).
- **Splits**: frozen train/val/test/OOD pools snapshotted from the Academy's canonical split function and verified (disjointness, solvability, equality with the canonical computation).
- **Matched streams**: every policy runs the identical deterministic episode stream per seed (same configs, seeds, noise, budgets).
- **Protocols**: training stream; zero-shot evaluation on val/test/OOD; continual revisit (A,B,C,A,B,C,A); rule-switch adaptability (10×A then 10×B); save/restore restart.
- **Controls**: fresh student; memory-muted student (identical training, declarative memory wiped before evaluation); random / episodic / Q-learning baselines; privileged oracle (diagnostic only).
- **Leakage prevention**: students receive public observations only; hidden dynamics are used solely by the oracle; a test audits this.
- **Statistics**: bounded probability metrics (rates, coverage, precision, correction fractions) use a pooled 95% Wilson interval, which stays inside [0, 1]; unbounded metrics (steps, episodes-to-event, planner counters) use a Student-t interval on per-seed means.
- **Planning counters**: the earlier `plan_cache_hit_rate` (cache hits / plans used) was removed because the two counters are not a hit/miss pair; it is replaced by `plan_cache_hits_per_step` and `plans_used_per_step`, reported separately with accurate names.

## Limitations

- Results describe this student on this task distribution (Academy gridworld, 80-step budget, no observation noise); they are not a general statement about the agent.
- The memory-muted control still allows within-episode learning; it removes only cross-episode declarative knowledge.
- Muted discovery metrics are measured on the training stream before wiping (identical by construction to the trained arm); the muted arm's *behavioral* effect appears in the evaluation protocols.
- Zero-shot coverage is a knowledge-content measure (pre-eval memory vs test truths), distinct from task success, which also involves planning and exploration.
- OOD results have wide confidence intervals; the lower trained vs fresh OOD success is a suggestive negative-transfer observation, not a definitive claim.
- Confidence intervals quantify seed variability, not environmental breadth; more families and noise regimes are future work.
