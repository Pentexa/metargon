# METARGON Cognitive Intelligence Benchmark v0.1

- Date: 2026-11-12
- Git head: `6531b6b48de307c6f83828785dcf4be4038e3496`
- Seeds: [0, 1, 2, 3]
- Episode sizes: {'train': 24, 'eval': 8, 'revisit': 7, 'switch_half': 10}
- Frozen split counts: {'enumerated': 624, 'ood': 25, 'solvable': 245, 'test': 49, 'train': 147, 'val': 49}
- Distinct true (door, rule) pairs in training stream: 7

## Descriptive indicators (0–1; NOT a composite IQ)

| dimension | indicator | value |
|---|---|---|
| Causal discovery | coverage | 1.000 |
| Causal discovery | verified_precision | 0.229 |
| Learning efficiency | success_rate_second_half | 0.854 |
| Learning efficiency | first_verified_rule | 1.000 |
| Knowledge retention | revisit_preserved | 1.000 |
| Knowledge retention | restart_parity | 1.000 |
| Generalization | zero_shot_test_coverage | 0.469 |
| Generalization | test_success_rate | 0.812 |
| Adaptability | success_after_switch | 0.600 |
| Adaptability | recovery_speed | 0.500 |
| Adaptability | wrong_belief_correction | 0.389 |
| Planning efficiency | fallback_reduction | 0.831 |
| Planning efficiency | steps_to_success_improvement | 0.143 |

Formula notes:
- **causal_discovery**: coverage = fraction of true (door, rule) pairs found as hypotheses; precision = verified correct / verified.
- **learning_efficiency**: second-half success rate on the training stream; episodes-to-first-verified-rule inverted into 0-1.
- **knowledge_retention**: revisit = success preserved at late vs early occurrence; restart parity = restored/direct success.
- **generalization**: zero-shot coverage = true rules already verified in pre-eval memory; test success rate.
- **adaptability**: success after rule switch; recovery speed = 1/(1+lag); fraction of pre-switch wrong beliefs contradicted after.
- **planning_efficiency**: fallback reduction = 1 − fallback rate on test.

## 1. Causal discovery

| metric | mean ± 95% CI | n |
|---|---|---|
| metargon_trained.discovery_coverage | 1.000  [1.000, 1.000] | 4 |
| metargon_trained.verified_correct | 1.750  [1.056, 2.444] | 4 |
| metargon_trained.verified_wrong | 7.000  [2.327, 11.673] | 4 |
| metargon_trained.verified_precision | 0.229  [0.051, 0.407] | 4 |
| metargon_trained.candidates_correct | 0.000  [0.000, 0.000] | 4 |
| metargon_trained.contradicted_truth | 5.250  [4.556, 5.944] | 4 |
| metargon_trained.hypotheses_total | 54.500  [34.338, 74.662] | 4 |
| metargon_muted.discovery_coverage | 1.000  [1.000, 1.000] | 4 |
| metargon_muted.verified_correct | 1.750  [1.056, 2.444] | 4 |
| metargon_muted.verified_wrong | 7.000  [2.327, 11.673] | 4 |
| metargon_muted.verified_precision | 0.229  [0.051, 0.407] | 4 |
| metargon_muted.candidates_correct | 0.000  [0.000, 0.000] | 4 |
| metargon_muted.contradicted_truth | 5.250  [4.556, 5.944] | 4 |
| metargon_muted.hypotheses_total | 54.500  [34.338, 74.662] | 4 |

## 2. Learning efficiency

| metric | mean ± 95% CI | n |
|---|---|---|
| metargon_trained.success_rate | 0.698  [0.501, 0.895] | 4 |
| metargon_trained.success_rate_first_half | 0.542  [0.320, 0.763] | 4 |
| metargon_trained.success_rate_second_half | 0.854  [0.657, 1.052] | 4 |
| metargon_trained.episodes_to_first_success | 1.000  [1.000, 1.000] | 4 |
| metargon_trained.episodes_to_first_verified_correct | 1.000  [1.000, 1.000] | 4 |
| metargon_trained.mean_steps_to_success_second_half | 20.064  [10.668, 29.460] | 4 |
| metargon_muted.success_rate | 0.698  [0.501, 0.895] | 4 |
| metargon_muted.success_rate_first_half | 0.542  [0.320, 0.763] | 4 |
| metargon_muted.success_rate_second_half | 0.854  [0.657, 1.052] | 4 |
| metargon_muted.episodes_to_first_success | 1.000  [1.000, 1.000] | 4 |
| metargon_muted.episodes_to_first_verified_correct | 1.000  [1.000, 1.000] | 4 |
| metargon_muted.mean_steps_to_success_second_half | 20.064  [10.668, 29.460] | 4 |

## 3. Knowledge retention

| metric | mean ± 95% CI | n |
|---|---|---|
| metargon_trained.retention_preserved | 1.000  [1.000, 1.000] | 4 |
| metargon_trained.success_early | 1.000  [1.000, 1.000] | 4 |
| metargon_trained.success_late | 1.000  [1.000, 1.000] | 4 |
| metargon_trained.restart_parity | 1.000  [1.000, 1.000] | 4 |
| metargon_trained.restart_success_restored | 0.812  [0.712, 0.913] | 4 |
| metargon_muted.retention_preserved | 1.000  [1.000, 1.000] | 4 |
| metargon_muted.success_early | 1.000  [1.000, 1.000] | 4 |
| metargon_muted.success_late | 1.000  [1.000, 1.000] | 4 |
| metargon_muted.restart_parity | 1.000  [1.000, 1.000] | 4 |
| metargon_muted.restart_success_restored | 0.719  [0.500, 0.937] | 4 |

## 4. Generalization

| metric | mean ± 95% CI | n |
|---|---|---|
| metargon_trained.eval_val.success_rate | 0.781  [0.695, 0.868] | 4 |
| metargon_trained.eval_val.mean_steps_to_success | 23.375  [15.807, 30.943] | 4 |
| metargon_fresh.eval_val.success_rate | 0.500  [0.500, 0.500] | 4 |
| metargon_fresh.eval_val.mean_steps_to_success | 10.250  [9.849, 10.651] | 4 |
| metargon_trained.eval_test.success_rate | 0.812  [0.712, 0.913] | 4 |
| metargon_trained.eval_test.mean_steps_to_success | 18.333  [12.389, 24.277] | 4 |
| metargon_fresh.eval_test.success_rate | 0.750  [0.403, 1.097] | 4 |
| metargon_fresh.eval_test.mean_steps_to_success | 21.381  [10.796, 31.966] | 4 |
| metargon_trained.eval_ood.success_rate | 0.156  [-0.104, 0.416] | 4 |
| metargon_trained.eval_ood.mean_steps_to_success | 12.500  [1.742, 23.258] | 2 |
| metargon_fresh.eval_ood.success_rate | 0.312  [-0.188, 0.813] | 4 |
| metargon_fresh.eval_ood.mean_steps_to_success | 15.600  [11.297, 19.903] | 2 |
| metargon_trained.eval_val.zero_shot_coverage | 0.438  [0.264, 0.611] | 4 |
| metargon_trained.eval_test.zero_shot_coverage | 0.469  [0.382, 0.555] | 4 |
| metargon_trained.eval_ood.zero_shot_coverage | 0.562  [0.149, 0.976] | 4 |

## 5. Adaptability

| metric | mean ± 95% CI | n |
|---|---|---|
| metargon_trained.success_before | 1.000  [1.000, 1.000] | 4 |
| metargon_trained.success_after | 0.600  [0.373, 0.827] | 4 |
| metargon_trained.recovery_lag | 1.000  [1.000, 1.000] | 4 |
| metargon_trained.wrong_belief_correction | 0.389  [-0.283, 1.061] | 4 |
| metargon_muted.success_before | 1.000  [1.000, 1.000] | 4 |
| metargon_muted.success_after | 0.625  [0.450, 0.800] | 4 |
| metargon_muted.recovery_lag | 1.500  [0.699, 2.301] | 4 |
| metargon_muted.wrong_belief_correction | 1.000  [1.000, 1.000] | 4 |
| metargon_fresh.success_before | 1.000  [1.000, 1.000] | 4 |
| metargon_fresh.success_after | 0.450  [0.210, 0.690] | 4 |
| metargon_fresh.recovery_lag | 2.500  [0.096, 4.904] | 4 |

## 6. Planning efficiency

| metric | mean ± 95% CI | n |
|---|---|---|
| metargon_trained.eval_test.fallback_rate | 0.169  [0.013, 0.326] | 4 |
| metargon_trained.eval_test.invalid_rate | 0.000  [0.000, 0.000] | 4 |
| metargon_trained.eval_test.plan_cache_hit_rate | 5.505  [4.052, 6.959] | 4 |
| metargon_trained.eval_test.planner_ops_per_step | 177.821  [107.685, 247.956] | 4 |
| metargon_fresh.eval_test.fallback_rate | 0.229  [0.091, 0.368] | 4 |
| metargon_fresh.eval_test.invalid_rate | 0.000  [0.000, 0.000] | 4 |
| metargon_fresh.eval_test.plan_cache_hit_rate | 4.951  [3.406, 6.495] | 4 |
| metargon_fresh.eval_test.planner_ops_per_step | 289.386  [217.109, 361.662] | 4 |

## Baseline context (evaluation pools)

### eval_val

| policy | success_rate | mean_steps_to_success | within_episode_adaptation |
|---|---|---|---|
| metargon_trained | 0.781  [0.695, 0.868] | 23.375  [15.807, 30.943] | 0.808  [0.714, 0.902] |
| metargon_fresh | 0.500  [0.500, 0.500] | 10.250  [9.849, 10.651] | 0.518  [0.468, 0.567] |
| random | 0.031  [-0.055, 0.118] | 28.000 | 0.036  [-0.063, 0.135] |
| episodic | 0.031  [-0.055, 0.118] | 28.000 | 0.036  [-0.063, 0.135] |
| qlearning | 0.406  [0.320, 0.493] | 37.042  [34.902, 39.182] | 0.446  [0.397, 0.496] |
| oracle | 1.000  [1.000, 1.000] | 12.875  [12.875, 12.875] | 1.000  [1.000, 1.000] |

### eval_test

| policy | success_rate | mean_steps_to_success | within_episode_adaptation |
|---|---|---|---|
| metargon_trained | 0.812  [0.712, 0.913] | 18.333  [12.389, 24.277] | 0.839  [0.756, 0.923] |
| metargon_fresh | 0.750  [0.403, 1.097] | 21.381  [10.796, 31.966] | 0.750  [0.403, 1.097] |
| random | 0.000  [0.000, 0.000] | — | 0.000  [0.000, 0.000] |
| episodic | 0.000  [0.000, 0.000] | — | 0.000  [0.000, 0.000] |
| qlearning | 0.250  [0.108, 0.392] | 34.167  [21.813, 46.521] | 0.286  [0.124, 0.448] |
| oracle | 1.000  [1.000, 1.000] | 14.250  [13.449, 15.051] | 1.000  [1.000, 1.000] |

### eval_ood

| policy | success_rate | mean_steps_to_success | within_episode_adaptation |
|---|---|---|---|
| metargon_trained | 0.156  [-0.104, 0.416] | 12.500  [1.742, 23.258] | 0.156  [-0.104, 0.416] |
| metargon_fresh | 0.312  [-0.188, 0.813] | 15.600  [11.297, 19.903] | 0.312  [-0.188, 0.813] |
| random | 0.031  [-0.055, 0.118] | 40.000 | 0.031  [-0.055, 0.118] |
| episodic | 0.031  [-0.055, 0.118] | 40.000 | 0.031  [-0.055, 0.118] |
| qlearning | 0.094  [-0.072, 0.260] | 60.750  [42.462, 79.038] | 0.098  [-0.070, 0.267] |
| oracle | 1.000  [1.000, 1.000] | 15.969  [14.045, 17.892] | 1.000  [1.000, 1.000] |

## Methodology

- **Subject**: the unmodified Academy student (`MetargonAcademyAgent`, default configuration).
- **Splits**: frozen train/val/test/OOD pools snapshotted from the Academy's canonical split function and verified (disjointness, solvability, equality with the canonical computation).
- **Matched streams**: every policy runs the identical deterministic episode stream per seed (same configs, seeds, noise, budgets).
- **Protocols**: training stream; zero-shot evaluation on val/test/OOD; continual revisit (A,B,C,A,B,C,A); rule-switch adaptability (10×A then 10×B); save/restore restart.
- **Controls**: fresh student; memory-muted student (identical training, declarative memory wiped before evaluation); random / episodic / Q-learning baselines; privileged oracle (diagnostic only).
- **Leakage prevention**: students receive public observations only; hidden dynamics are used solely by the oracle; a test audits this.
- **Statistics**: per-seed metrics summarized as mean ± 95% CI (Student-t).

## Limitations

- Results describe this student on this task distribution (Academy gridworld, 80-step budget, no observation noise); they are not a general statement about the agent.
- The memory-muted control still allows within-episode learning; it removes only cross-episode declarative knowledge.
- Muted discovery metrics are measured on the training stream before wiping (identical by construction to the trained arm); the muted arm's *behavioral* effect appears in the evaluation protocols.
- Zero-shot coverage is a knowledge-content measure (pre-eval memory vs test truths), distinct from task success, which also involves planning and exploration.
- OOD results have wide confidence intervals; the lower trained vs fresh OOD success is a suggestive negative-transfer observation, not a definitive claim.
- Confidence intervals quantify seed variability, not environmental breadth; more families and noise regimes are future work.
