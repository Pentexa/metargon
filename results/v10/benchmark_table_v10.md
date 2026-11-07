# METARGON v0.10 — benchmark table

Environment: Mechanism Lab (`python/lab_env.py`). Paired pre-generated worlds; same seeds per policy; oracle is privileged (labelled, excluded from deployable rankings).

Run: 8 seeds × 100 episodes × 7 conditions.

Uncertainty: paired bootstrap 95% intervals (1500 resamples, seed 4301).

Policies: `metargon09` = v0.9 agent; `metargon10` = v0.10 full; `-rb` = robust belief ablated; `-cd` = probe cooldowns ablated; `-x` = information-gathering fallback ablated; `bayes` = model-based particle-filter planner (given the declared family F).


## stable

| policy | success | steps | invalid | loops | stall | events | firstEv | cpu s | peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| metargon09 | 35.12% | 61.12 | 0.00 | 169.9 | 0.0 | 91.0 | 7.85 | 30.64 | — |
| metargon10 | 80.75% | 38.19 | 0.00 | 130.1 | 0.0 | 200.9 | 5.26 | 45.54 | — |
| metargon10-rb | 80.75% | 38.19 | 0.00 | 130.1 | 0.0 | 200.9 | 5.26 | 48.98 | — |
| metargon10-cd | 80.88% | 38.08 | 0.00 | 116.9 | 0.0 | 200.5 | 5.16 | 78.50 | — |
| metargon10-x | 82.12% | 36.13 | 0.00 | 156.5 | 0.0 | 227.6 | 5.16 | 48.35 | — |
| bayes | 77.50% | 45.63 | 0.00 | 178.2 | 0.0 | 221.8 | 8.44 | 32.33 | — |
| oracle | 100.00% | 17.75 | 0.00 | 100.0 | 0.0 | 200.0 | 4.75 | 0.35 | — |

## shift

| policy | success | steps | invalid | loops | stall | events | firstEv | cpu s | peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| metargon09 | 46.38% | 55.22 | 0.00 | 162.6 | 0.0 | 110.2 | 6.92 | 24.56 | — |
| metargon10 | 89.25% | 32.88 | 0.00 | 141.9 | 0.0 | 250.0 | 5.47 | 37.82 | — |
| metargon10-rb | 89.25% | 32.88 | 0.00 | 141.9 | 0.0 | 250.0 | 5.47 | 42.30 | — |
| metargon10-cd | 90.00% | 32.34 | 0.00 | 132.9 | 0.0 | 208.8 | 5.31 | 45.84 | — |
| metargon10-x | 82.25% | 34.78 | 0.00 | 160.1 | 0.0 | 621.0 | 5.37 | 49.11 | — |
| bayes | 80.62% | 41.41 | 0.00 | 153.2 | 0.0 | 206.8 | 8.66 | 32.60 | — |
| oracle | 100.00% | 20.62 | 0.00 | 106.2 | 0.0 | 212.5 | 6.25 | 0.49 | — |

## rapid_shift

| policy | success | steps | invalid | loops | stall | events | firstEv | cpu s | peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| metargon09 | 40.75% | 59.21 | 0.00 | 152.2 | 0.0 | 96.0 | 4.32 | 35.71 | — |
| metargon10 | 86.88% | 36.42 | 0.00 | 150.4 | 0.0 | 345.4 | 5.90 | 71.95 | — |
| metargon10-rb | 86.88% | 36.42 | 0.00 | 150.4 | 0.0 | 345.4 | 5.90 | 67.26 | — |
| metargon10-cd | 90.00% | 34.99 | 0.00 | 138.1 | 0.0 | 286.8 | 5.88 | 56.46 | — |
| metargon10-x | 86.12% | 34.57 | 0.00 | 168.5 | 0.0 | 279.9 | 6.04 | 51.71 | — |
| bayes | 76.88% | 44.66 | 0.00 | 156.1 | 0.0 | 206.6 | 9.93 | 38.85 | — |
| oracle | 91.25% | 26.55 | 0.00 | 126.8 | 0.0 | 193.8 | 6.59 | 0.76 | — |

## iid

| policy | success | steps | invalid | loops | stall | events | firstEv | cpu s | peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| metargon09 | 53.62% | 53.80 | 0.00 | 169.9 | 0.0 | 154.4 | 8.39 | 83.91 | — |
| metargon10 | 74.50% | 42.66 | 0.00 | 175.2 | 0.0 | 295.6 | 5.89 | 112.22 | — |
| metargon10-rb | 74.50% | 42.66 | 0.00 | 175.2 | 0.0 | 295.6 | 5.89 | 120.69 | — |
| metargon10-cd | 73.75% | 42.41 | 0.00 | 174.9 | 0.0 | 332.0 | 5.66 | 119.90 | — |
| metargon10-x | 77.75% | 40.28 | 0.00 | 231.9 | 0.0 | 323.5 | 6.50 | 78.63 | — |
| bayes | 79.00% | 43.04 | 0.00 | 139.1 | 0.0 | 210.5 | 9.81 | 38.97 | — |
| oracle | 78.88% | 39.98 | 0.00 | 173.6 | 0.0 | 195.1 | 7.23 | 1.37 | — |

## novel

| policy | success | steps | invalid | loops | stall | events | firstEv | cpu s | peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| metargon09 | 31.25% | 62.47 | 0.00 | 157.6 | 0.0 | 88.1 | 7.91 | 38.70 | — |
| metargon10 | 89.75% | 31.66 | 0.00 | 126.1 | 0.0 | 201.6 | 6.13 | 46.29 | — |
| metargon10-rb | 89.75% | 31.66 | 0.00 | 126.1 | 0.0 | 201.6 | 6.13 | 51.43 | — |
| metargon10-cd | 77.50% | 38.80 | 0.00 | 129.9 | 0.0 | 219.0 | 5.53 | 71.74 | — |
| metargon10-x | 86.62% | 32.32 | 0.00 | 147.6 | 0.0 | 218.5 | 6.05 | 54.26 | — |
| bayes | 78.50% | 43.69 | 0.00 | 155.5 | 0.0 | 201.8 | 9.68 | 41.62 | — |
| oracle | 93.75% | 25.25 | 0.00 | 113.1 | 0.0 | 187.5 | 7.94 | 0.75 | — |

## noisy_obs

| policy | success | steps | invalid | loops | stall | events | firstEv | cpu s | peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| metargon09 | 33.75% | 68.63 | 31.88 | 213.8 | 0.0 | 196.8 | 11.72 | 245.87 | — |
| metargon10 | 49.38% | 62.01 | 35.88 | 140.5 | 0.0 | 264.1 | 8.52 | 200.41 | — |
| metargon10-rb | 51.50% | 61.89 | 28.12 | 149.6 | 0.0 | 265.4 | 8.98 | 190.31 | — |
| metargon10-cd | 50.88% | 61.47 | 33.75 | 143.9 | 0.0 | 262.8 | 8.66 | 193.19 | — |
| metargon10-x | 58.00% | 56.09 | 20.00 | 217.6 | 0.0 | 242.1 | 8.20 | 144.76 | — |
| bayes | 73.12% | 48.10 | 54.75 | 178.9 | 0.0 | 232.8 | 8.99 | 40.73 | — |
| oracle | 98.38% | 21.38 | 3.88 | 108.6 | 0.0 | 210.5 | 5.34 | 2.13 | — |

## tight_budget

| policy | success | steps | invalid | loops | stall | events | firstEv | cpu s | peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| metargon09 | 45.00% | 35.36 | 0.00 | 138.4 | 0.0 | 112.6 | 6.82 | 25.96 | — |
| metargon10 | 83.62% | 28.69 | 0.00 | 146.1 | 0.0 | 223.2 | 5.33 | 28.17 | — |
| metargon10-rb | 83.62% | 28.69 | 0.00 | 146.1 | 0.0 | 223.2 | 5.33 | 26.73 | — |
| metargon10-cd | 83.88% | 28.63 | 0.00 | 132.9 | 0.0 | 208.8 | 5.31 | 25.37 | — |
| metargon10-x | 86.75% | 27.70 | 0.00 | 162.8 | 0.0 | 271.5 | 5.25 | 24.99 | — |
| bayes | 64.88% | 32.05 | 0.00 | 135.0 | 0.0 | 181.5 | 7.75 | 22.62 | — |
| oracle | 100.00% | 20.62 | 0.00 | 106.2 | 0.0 | 212.5 | 6.25 | 0.42 | — |

## Paired comparisons (success-rate deltas, pp)

| condition | comparison | Δ pp | 95% CI pp |
|---|---:|---:|---:|
| stable | v10 vs v09 | +45.62 | [0.88, 85.50] |
| stable | v10 vs bayes | +3.25 | [-21.00, 23.00] |
| stable | v10 vs -rb (no robust belief) | +0.00 | [0.00, 0.00] |
| stable | v10 vs -cd (no cooldowns) | -0.13 | [-0.38, 0.00] |
| stable | v10 vs -x (no info fallback) | -1.37 | [-23.75, 15.75] |
| shift | v10 vs v09 | +42.88 | [16.00, 66.62] |
| shift | v10 vs bayes | +8.62 | [-4.38, 20.88] |
| shift | v10 vs -rb (no robust belief) | +0.00 | [0.00, 0.00] |
| shift | v10 vs -cd (no cooldowns) | -0.75 | [-2.00, 0.25] |
| shift | v10 vs -x (no info fallback) | +7.00 | [-9.63, 23.38] |
| rapid_shift | v10 vs v09 | +46.12 | [12.37, 75.37] |
| rapid_shift | v10 vs bayes | +10.00 | [-4.63, 23.00] |
| rapid_shift | v10 vs -rb (no robust belief) | +0.00 | [0.00, 0.00] |
| rapid_shift | v10 vs -cd (no cooldowns) | -3.13 | [-7.63, 0.00] |
| rapid_shift | v10 vs -x (no info fallback) | +0.75 | [-10.50, 10.00] |
| iid | v10 vs v09 | +20.88 | [12.25, 30.12] |
| iid | v10 vs bayes | -4.50 | [-12.12, 3.12] |
| iid | v10 vs -rb (no robust belief) | +0.00 | [0.00, 0.00] |
| iid | v10 vs -cd (no cooldowns) | +0.75 | [-2.13, 4.00] |
| iid | v10 vs -x (no info fallback) | -3.25 | [-7.75, 1.63] |
| novel | v10 vs v09 | +58.50 | [24.37, 90.00] |
| novel | v10 vs bayes | +11.25 | [-1.62, 23.38] |
| novel | v10 vs -rb (no robust belief) | +0.00 | [0.00, 0.00] |
| novel | v10 vs -cd (no cooldowns) | +12.25 | [0.13, 30.12] |
| novel | v10 vs -x (no info fallback) | +3.12 | [-9.50, 13.12] |
| noisy_obs | v10 vs v09 | +15.62 | [8.38, 23.38] |
| noisy_obs | v10 vs bayes | -23.75 | [-33.62, -14.50] |
| noisy_obs | v10 vs -rb (no robust belief) | -2.12 | [-7.25, 3.25] |
| noisy_obs | v10 vs -cd (no cooldowns) | -1.50 | [-3.25, -0.13] |
| noisy_obs | v10 vs -x (no info fallback) | -8.62 | [-14.50, -3.62] |
| tight_budget | v10 vs v09 | +38.62 | [6.25, 67.62] |
| tight_budget | v10 vs bayes | +18.75 | [1.88, 33.50] |
| tight_budget | v10 vs -rb (no robust belief) | +0.00 | [0.00, 0.00] |
| tight_budget | v10 vs -cd (no cooldowns) | -0.25 | [-0.88, 0.25] |
| tight_budget | v10 vs -x (no info fallback) | -3.13 | [-20.13, 11.50] |
