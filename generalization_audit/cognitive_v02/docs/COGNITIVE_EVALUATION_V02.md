# METARGON Cognitive Evaluation v0.2 — independent audit

- Date: 2026-11-12
- Git head: `0d1a71c70b4d2f18ac02c2b9b92894c09a804e29`
- Seeds: [0, 1, 2, 3]
- Sizes: {'train': 12, 'eval': 6, 'ab_train': 8, 'ab_eval': 4}
- Phases: ['standard', 'ab']
- Frozen split counts: {'enumerated': 624, 'ood': 25, 'solvable': 245, 'test': 49, 'train': 147, 'val': 49}

## 0. Questions asked

- Do cognitive improvements reflect genuine learning, memory transfer and adaptation, or more search effort?
- Does a trained agent outperform a fresh agent with identical action and computation budgets?
- Do apparent gains disappear when learned memory is removed (muted) or its associations are randomized (shuffled)?
- Separating within-episode discovery from cross-episode knowledge transfer.
- Catastrophic forgetting after a conflicting A/B/A task sequence.
- Distinguishing improved planning from improved causal knowledge.

## 1. Training phase (knowledge + within-episode discovery)

| arm | success | within-episode discovery | coverage | precision | verified-correct | verified-wrong |
|---|---|---|---|---|---|---|
| trained | 0.542  [0.403, 0.674] | 0.167  [0.087, 0.296] | 0.714  [0.529, 0.847] | 0.129  [0.051, 0.289] | 1.000  [-0.133, 2.133] | 6.750  [2.605, 10.895] |
| epoch_reset | 0.583  [0.443, 0.712] | 0.167  [0.087, 0.296] | 0.750  [0.566, 0.873] | 0.259  [0.161, 0.389] | 3.500  [2.699, 4.301] | 10.000  [4.111, 15.889] |
| explore | 0.562  [0.423, 0.693] | 0.229  [0.133, 0.365] | 0.821  [0.644, 0.921] | 0.071  [0.025, 0.190] | 0.750  [-0.579, 2.079] | 9.750  [6.672, 12.828] |
| explore_epoch_reset | 0.521  [0.383, 0.655] | 0.312  [0.199, 0.453] | 0.964  [0.823, 0.994] | 0.292  [0.182, 0.432] | 3.500  [2.112, 4.888] | 8.500  [4.336, 12.664] |
| fresh | — | — | 0.000  [0.000, 0.121] | — | 0.000  [0.000, 0.000] | 0.000  [0.000, 0.000] |
| muted | 0.542  [0.403, 0.674] | 0.167  [0.087, 0.296] | 0.714  [0.529, 0.847] | 0.129  [0.051, 0.289] | 1.000  [-0.133, 2.133] | 6.750  [2.605, 10.895] |
| shuffled | 0.542  [0.403, 0.674] | 0.167  [0.087, 0.296] | 0.714  [0.529, 0.847] | 0.129  [0.051, 0.289] | 1.000  [-0.133, 2.133] | 6.750  [2.605, 10.895] |

## 2. Eval pools (clean / noisy / OOD)

### val_clean

| arm | success | zero-shot coverage | within-episode discovery | steps→success | planner ops/step | fallback |
|---|---|---|---|---|---|---|
| trained | 0.458  [0.279, 0.649] | 0.130  [0.045, 0.321] | 0.000  [0.000, 0.138] | 19.117  [-5.527, 43.760] | 100.864  [36.028, 165.700] | 0.064  [0.052, 0.079] |
| epoch_reset | 0.583  [0.388, 0.755] | 0.565  [0.368, 0.744] | 0.042  [0.007, 0.202] | 18.238  [7.472, 29.003] | 257.898  [163.695, 352.102] | 0.382  [0.354, 0.412] |
| explore | 0.708  [0.508, 0.851] | 0.130  [0.045, 0.321] | 0.125  [0.043, 0.310] | 25.371  [10.696, 40.046] | 86.928  [52.562, 121.293] | 0.284  [0.256, 0.313] |
| explore_epoch_reset | 0.875  [0.690, 0.957] | 0.478  [0.292, 0.670] | 0.000  [0.000, 0.138] | 25.667  [14.723, 36.610] | 117.522  [92.608, 142.436] | 0.064  [0.049, 0.083] |
| fresh | 0.458  [0.279, 0.649] | 0.000  [0.000, 0.143] | 0.292  [0.149, 0.492] | 10.000  [10.000, 10.000] | 134.693  [77.118, 192.269] | 0.003  [0.001, 0.008] |
| muted | 0.542  [0.351, 0.721] | 0.000  [0.000, 0.143] | 0.333  [0.180, 0.533] | 10.438  [9.722, 11.153] | 68.163  [43.223, 93.103] | 0.003  [0.001, 0.009] |
| shuffled | 0.458  [0.279, 0.649] | 0.174  [0.070, 0.371] | 0.167  [0.067, 0.359] | 16.483  [-0.303, 33.270] | 65.948  [-24.835, 156.730] | 0.066  [0.053, 0.081] |

### test_clean

| arm | success | zero-shot coverage | within-episode discovery | steps→success | planner ops/step | fallback |
|---|---|---|---|---|---|---|
| trained | 0.750  [0.551, 0.880] | 0.160  [0.064, 0.347] | 0.208  [0.092, 0.405] | 20.271  [11.286, 29.256] | 209.134  [128.221, 290.046] | 0.278  [0.249, 0.309] |
| epoch_reset | 0.875  [0.690, 0.957] | 0.480  [0.300, 0.665] | 0.167  [0.067, 0.359] | 19.658  [14.267, 25.050] | 233.635  [153.985, 313.285] | 0.241  [0.210, 0.275] |
| explore | 0.667  [0.467, 0.820] | 0.120  [0.042, 0.300] | 0.125  [0.043, 0.310] | 16.667  [13.580, 19.753] | 74.234  [-18.626, 167.093] | 0.207  [0.182, 0.234] |
| explore_epoch_reset | 0.833  [0.641, 0.933] | 0.440  [0.267, 0.629] | 0.042  [0.007, 0.202] | 28.375  [16.843, 39.907] | 100.049  [79.759, 120.338] | 0.053  [0.040, 0.070] |
| fresh | 0.792  [0.595, 0.908] | 0.000  [0.000, 0.133] | 0.292  [0.149, 0.492] | 22.642  [9.735, 35.549] | 312.796  [290.901, 334.691] | 0.316  [0.286, 0.348] |
| muted | 0.833  [0.641, 0.933] | 0.000  [0.000, 0.133] | 0.292  [0.149, 0.492] | 20.767  [17.431, 24.103] | 206.793  [149.777, 263.809] | 0.185  [0.158, 0.214] |
| shuffled | 0.833  [0.641, 0.933] | 0.200  [0.089, 0.391] | 0.250  [0.120, 0.449] | 20.583  [14.329, 26.837] | 236.758  [158.603, 314.912] | 0.249  [0.219, 0.281] |

### test_noisy

| arm | success | zero-shot coverage | within-episode discovery | steps→success | planner ops/step | fallback |
|---|---|---|---|---|---|---|
| trained | 0.833  [0.641, 0.933] | 0.160  [0.064, 0.347] | 0.292  [0.149, 0.492] | 29.875  [19.174, 40.576] | 445.950  [223.608, 668.292] | 0.064  [0.050, 0.082] |
| epoch_reset | 0.833  [0.641, 0.933] | 0.480  [0.300, 0.665] | 0.375  [0.212, 0.573] | 29.250  [15.009, 43.491] | 376.702  [296.437, 456.968] | 0.088  [0.071, 0.108] |
| explore | 0.625  [0.427, 0.788] | 0.120  [0.042, 0.300] | 0.333  [0.180, 0.533] | 20.783  [14.018, 27.549] | 85.252  [-18.546, 189.049] | 0.174  [0.152, 0.198] |
| explore_epoch_reset | 0.500  [0.314, 0.686] | 0.440  [0.267, 0.629] | 0.250  [0.120, 0.449] | 30.292  [13.616, 46.967] | 129.026  [73.303, 184.749] | 0.028  [0.020, 0.038] |
| fresh | 0.875  [0.690, 0.957] | 0.000  [0.000, 0.133] | 0.375  [0.212, 0.573] | 25.217  [16.361, 34.072] | 484.200  [201.707, 766.693] | 0.062  [0.047, 0.082] |
| muted | 0.875  [0.690, 0.957] | 0.000  [0.000, 0.133] | 0.417  [0.245, 0.612] | 26.225  [14.505, 37.945] | 390.478  [156.965, 623.990] | 0.091  [0.073, 0.113] |
| shuffled | 0.792  [0.595, 0.908] | 0.200  [0.089, 0.391] | 0.542  [0.351, 0.721] | 22.438  [14.002, 30.873] | 392.328  [50.609, 734.047] | 0.123  [0.103, 0.147] |

### ood

| arm | success | zero-shot coverage | within-episode discovery | steps→success | planner ops/step | fallback |
|---|---|---|---|---|---|---|
| trained | 0.375  [0.212, 0.573] | 0.167  [0.067, 0.359] | 0.083  [0.023, 0.258] | 13.833  [10.046, 17.620] | 113.229  [31.263, 195.194] | 0.000  [0.000, 0.003] |
| epoch_reset | 0.125  [0.043, 0.310] | 0.417  [0.245, 0.612] | 0.000  [0.000, 0.138] | 10.000  [10.000, 10.000] | 139.518  [115.873, 163.163] | 0.000  [0.000, 0.002] |
| explore | 0.292  [0.149, 0.492] | 0.125  [0.043, 0.310] | 0.083  [0.023, 0.258] | 39.917  [-3.648, 83.482] | 269.475  [236.649, 302.302] | 0.347  [0.324, 0.371] |
| explore_epoch_reset | 0.292  [0.149, 0.492] | 0.417  [0.245, 0.612] | 0.000  [0.000, 0.138] | 35.000  [-2.019, 72.019] | 372.472  [321.760, 423.184] | 0.098  [0.084, 0.114] |
| fresh | 0.250  [0.120, 0.449] | 0.000  [0.000, 0.138] | 0.167  [0.067, 0.359] | 14.167  [9.146, 19.187] | 70.902  [56.418, 85.385] | 0.000  [0.000, 0.003] |
| muted | 0.250  [0.120, 0.449] | 0.000  [0.000, 0.138] | 0.167  [0.067, 0.359] | 13.500  [5.611, 21.389] | 107.470  [54.085, 160.854] | 0.000  [0.000, 0.003] |
| shuffled | 0.292  [0.149, 0.492] | 0.125  [0.043, 0.310] | 0.125  [0.043, 0.310] | 14.250  [9.916, 18.584] | 148.914  [66.494, 231.335] | 0.000  [0.000, 0.003] |

## 3. Paired comparisons (95% bootstrap CI)

| comparison | Δ | 95% CI | n |
|---|---|---|---|
| trained-fresh.val_clean.success_rate | 0.000 | [-0.20833333333333331, 0.16666666666666669] | 4 |
| muted-trained.val_clean.success_rate | 0.083 | [-0.08333333333333338, 0.25] | 4 |
| shuffled-trained.val_clean.success_rate | 0.000 | [-0.12499999999999997, 0.125] | 4 |
| epoch_reset-trained.val_clean.success_rate | 0.125 | [0.0, 0.25] | 4 |
| explore-trained.val_clean.success_rate | 0.250 | [0.041666666666666685, 0.5416666666666666] | 4 |
| explore+epoch-trained.val_clean.success_rate | 0.417 | [0.24999999999999997, 0.5833333333333334] | 4 |
| explore+epoch-epoch.val_clean.success_rate | 0.292 | [0.16666666666666666, 0.4166666666666667] | 4 |
| trained-fresh.test_clean.success_rate | -0.042 | [-0.25000000000000006, 0.20833333333333331] | 4 |
| muted-trained.test_clean.success_rate | 0.083 | [-0.08333333333333331, 0.25000000000000006] | 4 |
| shuffled-trained.test_clean.success_rate | 0.083 | [0.0, 0.16666666666666674] | 4 |
| epoch_reset-trained.test_clean.success_rate | 0.125 | [0.041666666666666685, 0.16666666666666674] | 4 |
| explore-trained.test_clean.success_rate | -0.083 | [-0.16666666666666663, 0.0] | 4 |
| explore+epoch-trained.test_clean.success_rate | 0.083 | [0.0, 0.25] | 4 |
| explore+epoch-epoch.test_clean.success_rate | -0.042 | [-0.16666666666666674, 0.08333333333333331] | 4 |
| trained-fresh.test_noisy.success_rate | -0.042 | [-0.16666666666666669, 0.0833333333333334] | 4 |
| muted-trained.test_noisy.success_rate | 0.042 | [-0.0833333333333334, 0.16666666666666669] | 4 |
| shuffled-trained.test_noisy.success_rate | -0.042 | [-0.16666666666666669, 0.08333333333333331] | 4 |
| epoch_reset-trained.test_noisy.success_rate | -0.000 | [-0.12500000000000006, 0.12499999999999997] | 4 |
| explore-trained.test_noisy.success_rate | -0.208 | [-0.2916666666666667, -0.16666666666666663] | 4 |
| explore+epoch-trained.test_noisy.success_rate | -0.333 | [-0.5, -0.125] | 4 |
| explore+epoch-epoch.test_noisy.success_rate | -0.333 | [-0.5, -0.125] | 4 |
| trained-fresh.ood.success_rate | 0.125 | [-0.08333333333333334, 0.375] | 4 |
| muted-trained.ood.success_rate | -0.125 | [-0.375, 0.08333333333333334] | 4 |
| shuffled-trained.ood.success_rate | -0.083 | [-0.16666666666666669, 0.0] | 4 |
| epoch_reset-trained.ood.success_rate | -0.250 | [-0.4166666666666667, -0.08333333333333333] | 4 |
| explore-trained.ood.success_rate | -0.083 | [-0.33333333333333337, 0.16666666666666669] | 4 |
| explore+epoch-trained.ood.success_rate | -0.083 | [-0.25, 0.0] | 4 |
| explore+epoch-epoch.ood.success_rate | 0.167 | [0.041666666666666664, 0.2916666666666667] | 4 |
| ab.forgetting.trained.a2-a1 | 0.250 | [0.25, 0.25] | 4 |

## 4. A/B/A repeated learning (conflicting sequence)

| arm | A1 | B (conflict) | A2 | A-rule retained final |
|---|---|---|---|---|
| trained | 0.750  [0.505, 0.898] | 0.750  [0.505, 0.898] | 1.000  [0.806, 1.000] | 0.000  [0.000, 0.324] |
| epoch_reset | 1.000  [0.806, 1.000] | 0.500  [0.280, 0.720] | 1.000  [0.806, 1.000] | 1.000  [0.676, 1.000] |
| explore | 0.938  [0.717, 0.989] | 0.375  [0.185, 0.614] | 0.812  [0.570, 0.934] | 0.000  [0.000, 0.324] |
| explore_epoch_reset | 0.750  [0.505, 0.898] | 0.125  [0.035, 0.360] | 0.750  [0.505, 0.898] | 1.000  [0.676, 1.000] |
| fresh | — | — | — | — |
| muted | — | — | — | — |
| shuffled | — | — | — | — |

## 5. Floors (sanity context)

| floor | success |
|---|---|
| random val_clean | 0.042  [-0.074, 0.157] |
| random test_clean | 0.000  [0.000, 0.000] |
| random test_noisy | 0.000  [0.000, 0.000] |
| random ood | 0.042  [-0.074, 0.157] |
| oracle val_clean | 1.000  [1.000, 1.000] |
| oracle test_clean | 1.000  [1.000, 1.000] |
| oracle test_noisy | 1.000  [1.000, 1.000] |
| oracle ood | 1.000  [1.000, 1.000] |

## 6. Methodology

- Frozen train/val/test/OOD splits (snapshotted and verified); matched deterministic episode streams per seed for every arm.
- Arms: trained (default agent), epoch_reset (v0.7 episode-scoped evidence + per-episode probe reset), explore (Cognitive Exploration v0.1 on), explore+epoch_reset, fresh, muted (declarative memory wiped), shuffled (associations randomized).
- Bounded probability metrics use pooled 95% Wilson intervals; unbounded metrics use mean ± 95% t intervals; paired comparisons use bootstrap 95% CIs.
- Zero-shot coverage is measured from pre-evaluation memory against the true rules of each evaluation pool.
- Within-episode discovery marks episodes where a new verified-correct hypothesis was created inside the episode.
- A/B/A: train rule A, eval A', train conflicting rule B, eval B', retrain A, eval A'' — A'' vs A' quantifies forgetting; B' quantifies recovery after the mechanism change.

## 7. Limitations

- Budgets and seeds are finite; results describe this student on the Academy gridworld distribution (80-step cap).
- Muted/shuffled controls still allow within-episode learning during evaluation; they remove only cross-episode declarative knowledge.
- Zero-shot coverage is a knowledge-content measure, distinct from task success (which also involves planning/exploration).
- The shuffle control preserves counts/support levels and only randomizes associations; it does not remove all structure.
