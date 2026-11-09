# METARGON v0.12 — f family

Run: 2 seeds × 20 episodes × 2 conditions.

Uncertainty: paired bootstrap 95% intervals (1500 resamples, seed 4301).


## stable

| policy | success | exact D1 | exact D2 | steps | invalid | cpu s |
|---|---:|---:|---:|---:|---:|---:|
| metargon10 | 72.50% | 92.5% | 70.0% | 34.52 | 0.00 | 175.25 |
| metargon11 | 72.50% | 85.0% | 47.5% | 34.33 | 0.00 | 113.39 |
| metargon12 | 72.50% | 90.0% | 67.5% | 34.33 | 0.00 | 115.75 |
| metargon12-nodc | 72.50% | 85.0% | 47.5% | 34.33 | 0.00 | 94.99 |
| metargon12-norc | 72.50% | 90.0% | 67.5% | 34.33 | 0.00 | 83.16 |
| metargon12-noad | 72.50% | 90.0% | 67.5% | 34.33 | 0.00 | 62.92 |
| bayes | 82.50% | 0.0% | 0.0% | 42.78 | 0.00 | 27.32 |
| oracle | 100.00% | 0.0% | 0.0% | 16.50 | 0.00 | 0.37 |

### Paired comparisons (success-rate deltas, pp)

| comparison | Δ pp | 95% CI pp |
|---|---:|---:|
| comparison:metargon12_minus_metargon10 | +0.00 | [0.00, 0.00] |
| comparison:metargon12_minus_metargon11 | +0.00 | [0.00, 0.00] |
| comparison:metargon12_minus_bayes | -10.00 | [-25.00, 5.00] |
| comparison:metargon12_minus_metargon12-nodc | +0.00 | [0.00, 0.00] |
| comparison:metargon12_minus_metargon12-norc | +0.00 | [0.00, 0.00] |
| comparison:metargon12_minus_metargon12-noad | +0.00 | [0.00, 0.00] |

## noisy_obs

| policy | success | exact D1 | exact D2 | steps | invalid | cpu s |
|---|---:|---:|---:|---:|---:|---:|
| metargon10 | 47.50% | 95.0% | 50.0% | 55.70 | 3.50 | 224.48 |
| metargon11 | 62.50% | 35.0% | 45.0% | 42.50 | 8.00 | 123.45 |
| metargon12 | 70.00% | 85.0% | 85.0% | 41.67 | 12.00 | 103.23 |
| metargon12-nodc | 72.50% | 37.5% | 50.0% | 39.97 | 11.50 | 87.41 |
| metargon12-norc | 70.00% | 70.0% | 57.5% | 38.58 | 14.50 | 93.86 |
| metargon12-noad | 70.00% | 85.0% | 85.0% | 41.67 | 12.00 | 76.66 |
| bayes | 72.50% | 0.0% | 0.0% | 44.10 | 9.50 | 31.42 |
| oracle | 97.50% | 0.0% | 0.0% | 19.95 | 1.00 | 1.48 |

### Paired comparisons (success-rate deltas, pp)

| comparison | Δ pp | 95% CI pp |
|---|---:|---:|
| comparison:metargon12_minus_metargon10 | +22.50 | [5.00, 40.00] |
| comparison:metargon12_minus_metargon11 | +7.50 | [-5.00, 20.00] |
| comparison:metargon12_minus_bayes | -2.50 | [-10.00, 5.00] |
| comparison:metargon12_minus_metargon12-nodc | -2.50 | [-5.00, 0.00] |
| comparison:metargon12_minus_metargon12-norc | +0.00 | [-5.00, 5.00] |
| comparison:metargon12_minus_metargon12-noad | +0.00 | [0.00, 0.00] |
