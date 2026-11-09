# METARGON v0.12 — f family

Run: 4 seeds × 40 episodes × 5 conditions.

Uncertainty: paired bootstrap 95% intervals (1500 resamples, seed 4301).


## stable

| policy | success | exact D1 | exact D2 | steps | invalid | cpu s |
|---|---:|---:|---:|---:|---:|---:|
| metargon10 | 85.62% | 88.1% | 83.8% | 34.03 | 0.00 | 110.79 |
| metargon11 | 85.62% | 96.2% | 65.6% | 27.77 | 0.00 | 107.95 |
| metargon12 | 85.62% | 97.5% | 84.4% | 27.87 | 0.00 | 109.97 |
| metargon12-nodc | 85.62% | 96.2% | 66.9% | 27.87 | 0.00 | 150.26 |
| metargon12-norc | 85.62% | 97.5% | 84.4% | 27.87 | 0.00 | 108.59 |
| metargon12-noad | 85.62% | 97.5% | 84.4% | 27.87 | 0.00 | 80.17 |
| bayes | 78.75% | 0.0% | 0.0% | 45.49 | 0.00 | 36.77 |
| oracle | 100.00% | 0.0% | 0.0% | 18.75 | 0.00 | 0.51 |

### Paired comparisons (success-rate deltas, pp)

| comparison | Δ pp | 95% CI pp |
|---|---:|---:|
| comparison:metargon12_minus_metargon10 | +0.00 | [0.00, 0.00] |
| comparison:metargon12_minus_metargon11 | +0.00 | [0.00, 0.00] |
| comparison:metargon12_minus_bayes | +6.87 | [-10.63, 21.25] |
| comparison:metargon12_minus_metargon12-nodc | +0.00 | [0.00, 0.00] |
| comparison:metargon12_minus_metargon12-norc | +0.00 | [0.00, 0.00] |
| comparison:metargon12_minus_metargon12-noad | +0.00 | [0.00, 0.00] |

## shift

| policy | success | exact D1 | exact D2 | steps | invalid | cpu s |
|---|---:|---:|---:|---:|---:|---:|
| metargon10 | 86.25% | 86.2% | 83.8% | 36.48 | 0.00 | 114.91 |
| metargon11 | 90.62% | 90.6% | 65.0% | 25.43 | 0.00 | 70.63 |
| metargon12 | 91.25% | 95.0% | 88.8% | 25.21 | 0.00 | 67.50 |
| metargon12-nodc | 90.62% | 90.0% | 66.2% | 25.60 | 0.00 | 74.14 |
| metargon12-norc | 91.25% | 95.0% | 88.8% | 25.21 | 0.00 | 101.96 |
| metargon12-noad | 91.25% | 95.0% | 88.8% | 25.21 | 0.00 | 91.89 |
| bayes | 80.62% | 0.0% | 0.0% | 42.63 | 0.00 | 52.18 |
| oracle | 100.00% | 0.0% | 0.0% | 22.38 | 0.00 | 0.72 |

### Paired comparisons (success-rate deltas, pp)

| comparison | Δ pp | 95% CI pp |
|---|---:|---:|
| comparison:metargon12_minus_metargon10 | +5.00 | [0.00, 15.00] |
| comparison:metargon12_minus_metargon11 | +0.62 | [0.00, 1.87] |
| comparison:metargon12_minus_bayes | +10.62 | [-1.88, 20.00] |
| comparison:metargon12_minus_metargon12-nodc | +0.62 | [0.00, 1.87] |
| comparison:metargon12_minus_metargon12-norc | +0.00 | [0.00, 0.00] |
| comparison:metargon12_minus_metargon12-noad | +0.00 | [0.00, 0.00] |

## noisy_obs

| policy | success | exact D1 | exact D2 | steps | invalid | cpu s |
|---|---:|---:|---:|---:|---:|---:|
| metargon10 | 44.38% | 97.5% | 59.4% | 62.46 | 9.75 | 278.40 |
| metargon11 | 80.62% | 63.7% | 31.9% | 36.48 | 10.00 | 191.44 |
| metargon12 | 72.50% | 87.5% | 76.9% | 42.49 | 16.00 | 203.16 |
| metargon12-nodc | 80.00% | 62.5% | 43.8% | 39.79 | 13.00 | 159.25 |
| metargon12-norc | 81.25% | 80.6% | 75.6% | 33.41 | 12.75 | 152.91 |
| metargon12-noad | 72.50% | 87.5% | 76.9% | 42.49 | 16.00 | 177.23 |
| bayes | 76.25% | 0.0% | 0.0% | 46.01 | 21.00 | 54.56 |
| oracle | 98.12% | 0.0% | 0.0% | 22.48 | 3.00 | 2.99 |

### Paired comparisons (success-rate deltas, pp)

| comparison | Δ pp | 95% CI pp |
|---|---:|---:|
| comparison:metargon12_minus_metargon10 | +28.12 | [15.00, 45.63] |
| comparison:metargon12_minus_metargon11 | -8.12 | [-24.37, 3.75] |
| comparison:metargon12_minus_bayes | -3.75 | [-20.00, 13.75] |
| comparison:metargon12_minus_metargon12-nodc | -7.50 | [-18.75, 0.00] |
| comparison:metargon12_minus_metargon12-norc | -8.75 | [-22.50, 1.25] |
| comparison:metargon12_minus_metargon12-noad | +0.00 | [0.00, 0.00] |

## novel

| policy | success | exact D1 | exact D2 | steps | invalid | cpu s |
|---|---:|---:|---:|---:|---:|---:|
| metargon10 | 91.25% | 86.9% | 84.4% | 28.23 | 0.00 | 81.90 |
| metargon11 | 91.88% | 92.5% | 66.9% | 23.52 | 0.00 | 68.31 |
| metargon12 | 91.25% | 96.2% | 88.1% | 23.98 | 0.00 | 100.43 |
| metargon12-nodc | 91.88% | 92.5% | 68.1% | 23.62 | 0.00 | 84.41 |
| metargon12-norc | 91.25% | 96.2% | 88.1% | 23.98 | 0.00 | 76.93 |
| metargon12-noad | 91.25% | 96.2% | 88.1% | 23.98 | 0.00 | 73.29 |
| bayes | 79.38% | 0.0% | 0.0% | 42.52 | 0.00 | 57.98 |
| oracle | 100.00% | 0.0% | 0.0% | 21.88 | 0.00 | 1.34 |

### Paired comparisons (success-rate deltas, pp)

| comparison | Δ pp | 95% CI pp |
|---|---:|---:|
| comparison:metargon12_minus_metargon10 | +0.00 | [0.00, 0.00] |
| comparison:metargon12_minus_metargon11 | -0.63 | [-1.88, 0.00] |
| comparison:metargon12_minus_bayes | +11.88 | [-1.25, 25.00] |
| comparison:metargon12_minus_metargon12-nodc | -0.63 | [-1.88, 0.00] |
| comparison:metargon12_minus_metargon12-norc | +0.00 | [0.00, 0.00] |
| comparison:metargon12_minus_metargon12-noad | +0.00 | [0.00, 0.00] |

## tight_budget

| policy | success | exact D1 | exact D2 | steps | invalid | cpu s |
|---|---:|---:|---:|---:|---:|---:|
| metargon10 | 73.12% | 85.0% | 80.6% | 31.13 | 0.00 | 98.77 |
| metargon11 | 90.00% | 87.5% | 64.4% | 21.73 | 0.00 | 50.07 |
| metargon12 | 90.62% | 93.1% | 86.9% | 21.58 | 0.00 | 49.86 |
| metargon12-nodc | 90.00% | 87.5% | 64.4% | 21.73 | 0.00 | 45.72 |
| metargon12-norc | 90.62% | 93.1% | 86.9% | 21.58 | 0.00 | 43.89 |
| metargon12-noad | 90.62% | 93.1% | 86.9% | 21.58 | 0.00 | 39.41 |
| bayes | 65.62% | 0.0% | 0.0% | 32.66 | 0.00 | 43.01 |
| oracle | 100.00% | 0.0% | 0.0% | 22.38 | 0.00 | 0.86 |

### Paired comparisons (success-rate deltas, pp)

| comparison | Δ pp | 95% CI pp |
|---|---:|---:|
| comparison:metargon12_minus_metargon10 | +17.50 | [0.00, 35.62] |
| comparison:metargon12_minus_metargon11 | +0.63 | [0.00, 1.88] |
| comparison:metargon12_minus_bayes | +25.00 | [9.38, 37.50] |
| comparison:metargon12_minus_metargon12-nodc | +0.63 | [0.00, 1.88] |
| comparison:metargon12_minus_metargon12-norc | +0.00 | [0.00, 0.00] |
| comparison:metargon12_minus_metargon12-noad | +0.00 | [0.00, 0.00] |
