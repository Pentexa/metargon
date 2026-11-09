# METARGON v0.12 — f family

Run: 4 seeds × 40 episodes × 1 conditions.

Uncertainty: paired bootstrap 95% intervals (1500 resamples, seed 4301).


## noisy_obs

| policy | success | exact D1 | exact D2 | steps | invalid | cpu s |
|---|---:|---:|---:|---:|---:|---:|
| metargon10 | 44.38% | 97.5% | 59.4% | 62.46 | 9.75 | 60.42 |
| metargon11 | 80.62% | 63.7% | 31.9% | 36.48 | 10.00 | 30.81 |
| metargon12 | 72.50% | 87.5% | 76.9% | 42.49 | 16.00 | 23.75 |
| metargon12-nodc | 80.00% | 62.5% | 43.8% | 39.79 | 13.00 | 18.98 |
| metargon12-norc | 81.25% | 80.6% | 75.6% | 33.41 | 12.75 | 16.48 |
| metargon12-noad | 72.50% | 87.5% | 76.9% | 42.49 | 16.00 | 22.22 |
| bayes | 76.25% | 0.0% | 0.0% | 46.01 | 21.00 | 10.06 |
| oracle | 98.12% | 0.0% | 0.0% | 22.48 | 3.00 | 0.59 |

### Paired comparisons (success-rate deltas, pp)

| comparison | Δ pp | 95% CI pp |
|---|---:|---:|
| comparison:metargon12_minus_metargon10 | +28.12 | [15.00, 45.63] |
| comparison:metargon12_minus_metargon11 | -8.12 | [-24.37, 3.75] |
| comparison:metargon12_minus_bayes | -3.75 | [-20.00, 13.75] |
| comparison:metargon12_minus_metargon12-nodc | -7.50 | [-18.75, 0.00] |
| comparison:metargon12_minus_metargon12-norc | -8.75 | [-22.50, 1.25] |
| comparison:metargon12_minus_metargon12-noad | +0.00 | [0.00, 0.00] |
