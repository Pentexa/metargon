# METARGON Cognitive Integration Evaluation v0.1

- Date: 2026-11-13
- Git head: `5349bb8f59c9795350f8b04aa403b01e8b2584f9`
- Seeds (confirmatory): [101, 103, 107, 109]
- Sizes: {'train': 12, 'eval': 6, 'ab_train': 8, 'ab_eval': 4}
- Frozen split counts: {'enumerated': 624, 'ood': 25, 'solvable': 245, 'test': 49, 'train': 147, 'val': 49}
- Preregistration: docs/PREREGISTRATION.md (frozen before confirmatory execution)

## 1. Arms overview (success on primary pool, val_clean)

| arm | success | zero-shot coverage | within-episode discovery | planner ops/step | fallback |
|---|---|---|---|---|---|
| fresh | 0.542  [0.351, 0.721] | 0.000  [0.000, 0.133] | 0.375  [0.212, 0.573] | 204.021  [138.016, 270.025] | 0.105  [0.088, 0.125] |
| trained | 0.583  [0.388, 0.755] | 0.080  [0.022, 0.250] | 0.042  [0.007, 0.202] | 127.507  [-31.445, 286.460] | 0.145  [0.125, 0.167] |
| muted | 0.542  [0.351, 0.721] | 0.000  [0.000, 0.133] | 0.292  [0.149, 0.492] | 133.179  [62.631, 203.728] | 0.093  [0.077, 0.112] |
| shuffled | 0.542  [0.351, 0.721] | 0.160  [0.064, 0.347] | 0.083  [0.023, 0.258] | 142.032  [20.399, 263.665] | 0.154  [0.134, 0.176] |
| epoch_reset | 0.583  [0.388, 0.755] | 0.440  [0.267, 0.629] | 0.000  [0.000, 0.138] | 160.002  [92.049, 227.955] | 0.270  [0.244, 0.298] |
| muted_epoch_reset | 0.625  [0.427, 0.788] | 0.000  [0.000, 0.133] | 0.208  [0.092, 0.405] | 267.970  [164.531, 371.408] | 0.190  [0.167, 0.216] |
| explore | 0.583  [0.388, 0.755] | 0.080  [0.022, 0.250] | 0.000  [0.000, 0.138] | 70.690  [15.800, 125.579] | 0.294  [0.269, 0.321] |
| explore_epoch_reset | 0.667  [0.467, 0.820] | 0.400  [0.234, 0.593] | 0.042  [0.007, 0.202] | 81.718  [37.286, 126.150] | 0.089  [0.073, 0.109] |
| muted_explore_epoch_reset | 0.583  [0.388, 0.755] | 0.000  [0.000, 0.133] | 0.333  [0.180, 0.533] | 92.046  [39.853, 144.240] | 0.093  [0.077, 0.111] |
| shuffled_explore_epoch_reset | 0.708  [0.508, 0.851] | 0.320  [0.172, 0.516] | 0.167  [0.067, 0.359] | 114.443  [54.533, 174.353] | 0.098  [0.080, 0.119] |

## 2. All pools (success; memory controls)

### val_clean

| arm | success | zero-shot coverage | steps→success | planner ops/step | fallback |
|---|---|---|---|---|---|
| fresh | 0.542  [0.351, 0.721] | 0.000  [0.000, 0.133] | 12.312  [6.780, 17.845] | 204.021  [138.016, 270.025] | 0.105  [0.088, 0.125] |
| trained | 0.583  [0.388, 0.755] | 0.080  [0.022, 0.250] | 19.800  [10.810, 28.790] | 127.507  [-31.445, 286.460] | 0.145  [0.125, 0.167] |
| muted | 0.542  [0.351, 0.721] | 0.000  [0.000, 0.133] | 12.400  [5.738, 19.062] | 133.179  [62.631, 203.728] | 0.093  [0.077, 0.112] |
| shuffled | 0.542  [0.351, 0.721] | 0.160  [0.064, 0.347] | 16.100  [10.772, 21.428] | 142.032  [20.399, 263.665] | 0.154  [0.134, 0.176] |
| epoch_reset | 0.583  [0.388, 0.755] | 0.440  [0.267, 0.629] | 16.450  [5.842, 27.058] | 160.002  [92.049, 227.955] | 0.270  [0.244, 0.298] |
| muted_epoch_reset | 0.625  [0.427, 0.788] | 0.000  [0.000, 0.133] | 16.267  [7.759, 24.775] | 267.970  [164.531, 371.408] | 0.190  [0.167, 0.216] |
| explore | 0.583  [0.388, 0.755] | 0.080  [0.022, 0.250] | 26.200  [10.626, 41.774] | 70.690  [15.800, 125.579] | 0.294  [0.269, 0.321] |
| explore_epoch_reset | 0.667  [0.467, 0.820] | 0.400  [0.234, 0.593] | 20.317  [6.160, 34.474] | 81.718  [37.286, 126.150] | 0.089  [0.073, 0.109] |
| muted_explore_epoch_reset | 0.583  [0.388, 0.755] | 0.000  [0.000, 0.133] | 23.104  [-0.022, 46.230] | 92.046  [39.853, 144.240] | 0.093  [0.077, 0.111] |
| shuffled_explore_epoch_reset | 0.708  [0.508, 0.851] | 0.320  [0.172, 0.516] | 20.871  [8.842, 32.900] | 114.443  [54.533, 174.353] | 0.098  [0.080, 0.119] |

### test_clean

| arm | success | zero-shot coverage | steps→success | planner ops/step | fallback |
|---|---|---|---|---|---|
| fresh | 0.625  [0.427, 0.788] | 0.000  [0.000, 0.129] | 17.396  [11.041, 23.750] | 295.063  [91.507, 498.620] | 0.312  [0.284, 0.342] |
| trained | 0.667  [0.467, 0.820] | 0.154  [0.061, 0.335] | 19.300  [15.779, 22.821] | 234.419  [30.576, 438.262] | 0.240  [0.214, 0.268] |
| muted | 0.667  [0.467, 0.820] | 0.000  [0.000, 0.129] | 17.521  [10.929, 24.113] | 202.293  [-21.074, 425.659] | 0.292  [0.263, 0.322] |
| shuffled | 0.583  [0.388, 0.755] | 0.192  [0.085, 0.379] | 21.100  [17.727, 24.473] | 149.818  [25.797, 273.840] | 0.166  [0.146, 0.190] |
| epoch_reset | 0.625  [0.427, 0.788] | 0.462  [0.288, 0.645] | 19.562  [14.670, 24.455] | 142.236  [57.355, 227.117] | 0.313  [0.285, 0.342] |
| muted_epoch_reset | 0.583  [0.388, 0.755] | 0.000  [0.000, 0.129] | 17.896  [9.856, 25.936] | 244.941  [158.534, 331.348] | 0.198  [0.176, 0.223] |
| explore | 0.583  [0.388, 0.755] | 0.154  [0.061, 0.335] | 23.375  [7.503, 39.247] | 59.383  [26.571, 92.196] | 0.181  [0.160, 0.204] |
| explore_epoch_reset | 0.667  [0.467, 0.820] | 0.462  [0.288, 0.645] | 24.900  [13.934, 35.866] | 94.579  [41.646, 147.513] | 0.101  [0.084, 0.121] |
| muted_explore_epoch_reset | 0.667  [0.467, 0.820] | 0.000  [0.000, 0.129] | 26.488  [13.501, 39.474] | 121.890  [85.629, 158.151] | 0.075  [0.061, 0.092] |
| shuffled_explore_epoch_reset | 0.625  [0.427, 0.788] | 0.308  [0.165, 0.500] | 20.850  [6.793, 34.907] | 118.674  [60.045, 177.304] | 0.077  [0.063, 0.095] |

### test_noisy

| arm | success | zero-shot coverage | steps→success | planner ops/step | fallback |
|---|---|---|---|---|---|
| fresh | 0.708  [0.508, 0.851] | 0.000  [0.000, 0.129] | 25.542  [17.976, 33.107] | 350.804  [259.794, 441.813] | 0.184  [0.161, 0.210] |
| trained | 0.667  [0.467, 0.820] | 0.154  [0.061, 0.335] | 30.033  [22.430, 37.636] | 371.260  [121.184, 621.336] | 0.129  [0.110, 0.150] |
| muted | 0.708  [0.508, 0.851] | 0.000  [0.000, 0.129] | 28.017  [14.033, 42.000] | 290.325  [179.239, 401.411] | 0.132  [0.112, 0.154] |
| shuffled | 0.667  [0.467, 0.820] | 0.192  [0.085, 0.379] | 32.104  [23.571, 40.637] | 321.266  [187.504, 455.029] | 0.091  [0.076, 0.109] |
| epoch_reset | 0.708  [0.508, 0.851] | 0.462  [0.288, 0.645] | 28.942  [11.021, 46.862] | 279.976  [97.956, 461.996] | 0.120  [0.102, 0.142] |
| muted_epoch_reset | 0.625  [0.427, 0.788] | 0.000  [0.000, 0.129] | 28.688  [11.224, 46.151] | 345.942  [145.542, 546.341] | 0.178  [0.156, 0.201] |
| explore | 0.667  [0.467, 0.820] | 0.154  [0.061, 0.335] | 26.008  [16.560, 35.456] | 73.564  [37.893, 109.234] | 0.171  [0.150, 0.195] |
| explore_epoch_reset | 0.583  [0.388, 0.755] | 0.462  [0.288, 0.645] | 31.954  [4.572, 59.337] | 87.434  [44.151, 130.717] | 0.009  [0.005, 0.016] |
| muted_explore_epoch_reset | 0.542  [0.351, 0.721] | 0.000  [0.000, 0.129] | 28.900  [20.614, 37.186] | 103.898  [60.547, 147.249] | 0.027  [0.019, 0.037] |
| shuffled_explore_epoch_reset | 0.542  [0.351, 0.721] | 0.308  [0.165, 0.500] | 30.775  [27.961, 33.589] | 108.764  [74.011, 143.517] | 0.007  [0.004, 0.013] |

### ood

| arm | success | zero-shot coverage | steps→success | planner ops/step | fallback |
|---|---|---|---|---|---|
| fresh | 0.292  [0.149, 0.492] | 0.000  [0.000, 0.155] | 15.556  [14.281, 16.830] | 80.322  [70.757, 89.886] | 0.000  [0.000, 0.003] |
| trained | 0.208  [0.092, 0.405] | 0.429  [0.245, 0.635] | 13.778  [7.644, 19.912] | 97.558  [56.750, 138.366] | 0.000  [0.000, 0.002] |
| muted | 0.292  [0.149, 0.492] | 0.000  [0.000, 0.155] | 15.556  [14.281, 16.830] | 126.057  [105.649, 146.466] | 0.000  [0.000, 0.003] |
| shuffled | 0.292  [0.149, 0.492] | 0.238  [0.106, 0.451] | 18.800  [-1.766, 39.366] | 153.488  [59.045, 247.930] | 0.077  [0.065, 0.092] |
| epoch_reset | 0.208  [0.092, 0.405] | 0.429  [0.245, 0.635] | 13.778  [7.644, 19.912] | 251.626  [183.587, 319.664] | 0.000  [0.000, 0.002] |
| muted_epoch_reset | 0.208  [0.092, 0.405] | 0.000  [0.000, 0.155] | 13.778  [7.644, 19.912] | 223.496  [160.250, 286.741] | 0.000  [0.000, 0.002] |
| explore | 0.208  [0.092, 0.405] | 0.333  [0.172, 0.546] | 43.000  [-3.011, 89.011] | 140.648  [-12.195, 293.491] | 0.355  [0.333, 0.378] |
| explore_epoch_reset | 0.458  [0.279, 0.649] | 0.524  [0.324, 0.717] | 41.062  [7.126, 74.999] | 394.892  [324.021, 465.763] | 0.081  [0.068, 0.097] |
| muted_explore_epoch_reset | 0.500  [0.314, 0.686] | 0.000  [0.000, 0.155] | 34.050  [30.516, 37.584] | 336.980  [273.817, 400.144] | 0.111  [0.095, 0.129] |
| shuffled_explore_epoch_reset | 0.542  [0.351, 0.721] | 0.381  [0.208, 0.591] | 31.792  [23.121, 40.463] | 444.735  [296.459, 593.012] | 0.115  [0.098, 0.133] |

## 3. Preregistered comparisons (paired bootstrap 95% CI)

| comparison | Δ | 95% CI | n |
|---|---|---|---|
| integ-vs-epoch.val_clean.success_rate | 0.083 | [-0.083, 0.167] | 4 |
| integ-vs-epoch.val_clean.zero_shot_coverage | -0.042 | [-0.125, 0.000] | 4 |
| integ-vs-explore.val_clean.success_rate | 0.083 | [0.000, 0.167] | 4 |
| integ-vs-explore.val_clean.zero_shot_coverage | 0.327 | [0.071, 0.583] | 4 |
| integ-vs-trained.val_clean.success_rate | 0.083 | [-0.083, 0.250] | 4 |
| integ-vs-trained.val_clean.zero_shot_coverage | 0.327 | [0.071, 0.583] | 4 |
| epoch-vs-trained.val_clean.success_rate | -0.000 | [-0.125, 0.125] | 4 |
| epoch-vs-trained.val_clean.zero_shot_coverage | 0.369 | [0.155, 0.583] | 4 |
| explore-vs-trained.val_clean.success_rate | -0.000 | [-0.125, 0.125] | 4 |
| explore-vs-trained.val_clean.zero_shot_coverage | 0.000 | [0.000, 0.000] | 4 |
| muted-integ-vs-muted-epoch.val_clean.success_rate | -0.042 | [-0.250, 0.125] | 4 |
| muted-integ-vs-muted-epoch.val_clean.zero_shot_coverage | 0.000 | [0.000, 0.000] | 4 |
| muted-integ-vs-integ.val_clean.success_rate | -0.083 | [-0.167, 0.000] | 4 |
| muted-integ-vs-integ.val_clean.zero_shot_coverage | -0.411 | [-0.750, -0.071] | 4 |
| shuffled-integ-vs-integ.val_clean.success_rate | 0.042 | [0.000, 0.125] | 4 |
| shuffled-integ-vs-integ.val_clean.zero_shot_coverage | -0.077 | [-0.167, 0.083] | 4 |
| integ-vs-epoch.test_clean.success_rate | 0.042 | [-0.083, 0.167] | 4 |
| integ-vs-epoch.test_clean.zero_shot_coverage | 0.000 | [0.000, 0.000] | 4 |
| integ-vs-explore.test_clean.success_rate | 0.083 | [0.000, 0.167] | 4 |
| integ-vs-explore.test_clean.zero_shot_coverage | 0.310 | [0.083, 0.536] | 4 |
| integ-vs-trained.test_clean.success_rate | 0.000 | [-0.125, 0.125] | 4 |
| integ-vs-trained.test_clean.zero_shot_coverage | 0.310 | [0.083, 0.536] | 4 |
| epoch-vs-trained.test_clean.success_rate | -0.042 | [-0.167, 0.083] | 4 |
| epoch-vs-trained.test_clean.zero_shot_coverage | 0.310 | [0.083, 0.536] | 4 |
| explore-vs-trained.test_clean.success_rate | -0.083 | [-0.167, 0.000] | 4 |
| explore-vs-trained.test_clean.zero_shot_coverage | 0.000 | [0.000, 0.000] | 4 |
| muted-integ-vs-muted-epoch.test_clean.success_rate | 0.083 | [-0.167, 0.458] | 4 |
| muted-integ-vs-muted-epoch.test_clean.zero_shot_coverage | 0.000 | [0.000, 0.000] | 4 |
| muted-integ-vs-integ.test_clean.success_rate | 0.000 | [-0.125, 0.125] | 4 |
| muted-integ-vs-integ.test_clean.zero_shot_coverage | -0.458 | [-0.762, -0.155] | 4 |
| shuffled-integ-vs-integ.test_clean.success_rate | -0.042 | [-0.125, 0.000] | 4 |
| shuffled-integ-vs-integ.test_clean.zero_shot_coverage | -0.161 | [-0.286, -0.042] | 4 |
| integ-vs-epoch.test_noisy.success_rate | -0.125 | [-0.375, 0.000] | 4 |
| integ-vs-epoch.test_noisy.zero_shot_coverage | 0.000 | [0.000, 0.000] | 4 |
| integ-vs-explore.test_noisy.success_rate | -0.083 | [-0.458, 0.167] | 4 |
| integ-vs-explore.test_noisy.zero_shot_coverage | 0.310 | [0.083, 0.536] | 4 |
| integ-vs-trained.test_noisy.success_rate | -0.083 | [-0.375, 0.125] | 4 |
| integ-vs-trained.test_noisy.zero_shot_coverage | 0.310 | [0.083, 0.536] | 4 |
| epoch-vs-trained.test_noisy.success_rate | 0.042 | [0.000, 0.125] | 4 |
| epoch-vs-trained.test_noisy.zero_shot_coverage | 0.310 | [0.083, 0.536] | 4 |
| explore-vs-trained.test_noisy.success_rate | -0.000 | [-0.125, 0.125] | 4 |
| explore-vs-trained.test_noisy.zero_shot_coverage | 0.000 | [0.000, 0.000] | 4 |
| muted-integ-vs-muted-epoch.test_noisy.success_rate | -0.083 | [-0.167, 0.083] | 4 |
| muted-integ-vs-muted-epoch.test_noisy.zero_shot_coverage | 0.000 | [0.000, 0.000] | 4 |
| muted-integ-vs-integ.test_noisy.success_rate | -0.042 | [-0.292, 0.333] | 4 |
| muted-integ-vs-integ.test_noisy.zero_shot_coverage | -0.458 | [-0.762, -0.155] | 4 |
| shuffled-integ-vs-integ.test_noisy.success_rate | -0.042 | [-0.333, 0.292] | 4 |
| shuffled-integ-vs-integ.test_noisy.zero_shot_coverage | -0.161 | [-0.286, -0.042] | 4 |
| integ-vs-epoch.ood.success_rate | 0.250 | [0.167, 0.333] | 4 |
| integ-vs-epoch.ood.zero_shot_coverage | 0.100 | [0.000, 0.300] | 4 |
| integ-vs-explore.ood.success_rate | 0.250 | [0.167, 0.417] | 4 |
| integ-vs-explore.ood.zero_shot_coverage | 0.146 | [-0.125, 0.417] | 4 |
| integ-vs-trained.ood.success_rate | 0.250 | [0.167, 0.333] | 4 |
| integ-vs-trained.ood.zero_shot_coverage | 0.062 | [-0.146, 0.250] | 4 |
| epoch-vs-trained.ood.success_rate | 0.000 | [0.000, 0.000] | 4 |
| epoch-vs-trained.ood.zero_shot_coverage | -0.038 | [-0.325, 0.250] | 4 |
| explore-vs-trained.ood.success_rate | -0.000 | [-0.208, 0.167] | 4 |
| explore-vs-trained.ood.zero_shot_coverage | -0.083 | [-0.250, 0.000] | 4 |
| muted-integ-vs-muted-epoch.ood.success_rate | 0.292 | [0.208, 0.333] | 4 |
| muted-integ-vs-muted-epoch.ood.zero_shot_coverage | 0.000 | [0.000, 0.000] | 4 |
| muted-integ-vs-integ.ood.success_rate | 0.042 | [0.000, 0.125] | 4 |
| muted-integ-vs-integ.ood.zero_shot_coverage | -0.483 | [-0.667, -0.167] | 4 |
| shuffled-integ-vs-integ.ood.success_rate | 0.083 | [0.000, 0.250] | 4 |
| shuffled-integ-vs-integ.ood.zero_shot_coverage | -0.125 | [-0.250, 0.000] | 4 |

## 4. Interaction and attribution analysis

Interaction term per pool: (explore_epoch_reset − epoch_reset) − (explore − trained) on success rate.

| pool | interaction Δ | 95% CI | n |
|---|---|---|---|
| val_clean.success_rate | 0.083 | [-0.000, 0.167] | 4 |
| test_clean.success_rate | 0.125 | [0.042, 0.167] | 4 |
| test_noisy.success_rate | -0.125 | [-0.500, 0.125] | 4 |
| ood.success_rate | 0.250 | [0.167, 0.417] | 4 |

Attribution deltas (memory-wiped variants):

| comparison | Δ | 95% CI | n |
|---|---|---|---|
| muted_explore_epoch_reset-muted_epoch_reset.val_clean.success_rate | -0.042 | [-0.250, 0.125] | 4 |
| muted_explore_epoch_reset-muted_epoch_reset.val_clean.zero_shot_coverage | 0.000 | [0.000, 0.000] | 4 |
| muted_explore_epoch_reset-explore_epoch_reset.val_clean.success_rate | -0.083 | [-0.167, 0.000] | 4 |
| muted_explore_epoch_reset-explore_epoch_reset.val_clean.zero_shot_coverage | -0.411 | [-0.750, -0.071] | 4 |
| shuffled_explore_epoch_reset-explore_epoch_reset.val_clean.success_rate | 0.042 | [0.000, 0.125] | 4 |
| shuffled_explore_epoch_reset-explore_epoch_reset.val_clean.zero_shot_coverage | -0.077 | [-0.167, 0.083] | 4 |
| muted_explore_epoch_reset-muted_epoch_reset.test_clean.success_rate | 0.083 | [-0.167, 0.458] | 4 |
| muted_explore_epoch_reset-muted_epoch_reset.test_clean.zero_shot_coverage | 0.000 | [0.000, 0.000] | 4 |
| muted_explore_epoch_reset-explore_epoch_reset.test_clean.success_rate | 0.000 | [-0.125, 0.125] | 4 |
| muted_explore_epoch_reset-explore_epoch_reset.test_clean.zero_shot_coverage | -0.458 | [-0.762, -0.155] | 4 |
| shuffled_explore_epoch_reset-explore_epoch_reset.test_clean.success_rate | -0.042 | [-0.125, 0.000] | 4 |
| shuffled_explore_epoch_reset-explore_epoch_reset.test_clean.zero_shot_coverage | -0.161 | [-0.286, -0.042] | 4 |
| muted_explore_epoch_reset-muted_epoch_reset.test_noisy.success_rate | -0.083 | [-0.167, 0.083] | 4 |
| muted_explore_epoch_reset-muted_epoch_reset.test_noisy.zero_shot_coverage | 0.000 | [0.000, 0.000] | 4 |
| muted_explore_epoch_reset-explore_epoch_reset.test_noisy.success_rate | -0.042 | [-0.292, 0.333] | 4 |
| muted_explore_epoch_reset-explore_epoch_reset.test_noisy.zero_shot_coverage | -0.458 | [-0.762, -0.155] | 4 |
| shuffled_explore_epoch_reset-explore_epoch_reset.test_noisy.success_rate | -0.042 | [-0.333, 0.292] | 4 |
| shuffled_explore_epoch_reset-explore_epoch_reset.test_noisy.zero_shot_coverage | -0.161 | [-0.286, -0.042] | 4 |
| muted_explore_epoch_reset-muted_epoch_reset.ood.success_rate | 0.292 | [0.208, 0.333] | 4 |
| muted_explore_epoch_reset-muted_epoch_reset.ood.zero_shot_coverage | 0.000 | [0.000, 0.000] | 4 |
| muted_explore_epoch_reset-explore_epoch_reset.ood.success_rate | 0.042 | [0.000, 0.125] | 4 |
| muted_explore_epoch_reset-explore_epoch_reset.ood.zero_shot_coverage | -0.483 | [-0.667, -0.167] | 4 |
| shuffled_explore_epoch_reset-explore_epoch_reset.ood.success_rate | 0.083 | [0.000, 0.250] | 4 |
| shuffled_explore_epoch_reset-explore_epoch_reset.ood.zero_shot_coverage | -0.125 | [-0.250, 0.000] | 4 |

## 5. A/B/A retention (interference)

| arm | A1 | B (conflict) | A2 | A-rule retained final |
|---|---|---|---|---|
| fresh | — | — | — | — |
| trained | 0.750  [0.505, 0.898] | 0.938  [0.717, 0.989] | 0.750  [0.505, 0.898] | 0.375  [0.137, 0.694] |
| muted | — | — | — | — |
| shuffled | — | — | — | — |
| epoch_reset | 0.562  [0.332, 0.769] | 0.875  [0.640, 0.965] | 0.688  [0.444, 0.858] | 0.875  [0.529, 0.978] |
| muted_epoch_reset | — | — | — | — |
| explore | 0.438  [0.231, 0.668] | 0.750  [0.505, 0.898] | 0.625  [0.386, 0.815] | 0.250  [0.071, 0.591] |
| explore_epoch_reset | 0.438  [0.231, 0.668] | 0.688  [0.444, 0.858] | 0.562  [0.332, 0.769] | 0.750  [0.409, 0.929] |
| muted_explore_epoch_reset | — | — | — | — |
| shuffled_explore_epoch_reset | — | — | — | — |

## 6. Floors (sanity)

| floor | success |
|---|---|
| random val_clean | 0.125  [-0.096, 0.346] |
| random test_clean | 0.083  [-0.148, 0.315] |
| random test_noisy | 0.083  [-0.148, 0.315] |
| random ood | 0.000  [0.000, 0.000] |
| oracle val_clean | 1.000  [1.000, 1.000] |
| oracle test_clean | 1.000  [1.000, 1.000] |
| oracle test_noisy | 1.000  [1.000, 1.000] |
| oracle ood | 1.000  [1.000, 1.000] |

## 7. Methodology

- Reuses the unmodified Cognitive Evaluation v0.2 framework (merged PR #52); additional arms registered via its public candidate interface.
- Matched deterministic streams per seed across all arms; equal action/computation budgets.
- Preregistered confirmatory seeds 101/103/107/109; smoke uses seed 0 only.
- Bounded probabilities: pooled 95% Wilson intervals; unbounded scalars: mean ± 95% t; comparisons: paired bootstrap deltas.

## 8. Limitations

- Small fixed seed set; finite budgets; Academy gridworld distribution only.
- Muted/shuffled controls still allow within-episode learning; they remove only cross-episode declarative knowledge.
- Interaction terms depend on the specific arm baselines and are exploratory beyond the preregistered primary pool.
