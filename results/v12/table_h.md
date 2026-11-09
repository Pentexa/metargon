# METARGON v0.12 — h family

Run: 4 seeds × 40 episodes × 2 conditions.

Uncertainty: paired bootstrap 95% intervals (1500 resamples, seed 4301).


## stable

| policy | success | exact D1 | exact D2 | steps | invalid | cpu s |
|---|---:|---:|---:|---:|---:|---:|
| metargon10 | 83.75% | 88.8% | 63.1% | 36.61 | 0.00 | 127.14 |
| metargon11 | 73.12% | 92.5% | 41.2% | 34.09 | 0.00 | 148.14 |
| metargon12 | 73.12% | 95.6% | 70.0% | 34.09 | 0.00 | 158.47 |
| metargon12-nodc | 73.12% | 92.5% | 41.2% | 34.09 | 0.00 | 140.74 |
| metargon12-norc | 73.12% | 95.6% | 70.0% | 34.09 | 0.00 | 96.39 |
| metargon12-noad | 73.12% | 95.6% | 70.0% | 34.09 | 0.00 | 57.23 |
| bayes | 71.88% | 0.0% | 0.0% | 46.00 | 0.00 | 34.37 |
| oracle | 100.00% | 0.0% | 0.0% | 16.50 | 0.00 | 0.32 |

### Paired comparisons (success-rate deltas, pp)

| comparison | Δ pp | 95% CI pp |
|---|---:|---:|
| comparison:metargon12_minus_metargon10 | -10.63 | [-30.00, 0.00] |
| comparison:metargon12_minus_metargon11 | +0.00 | [0.00, 0.00] |
| comparison:metargon12_minus_bayes | +1.25 | [-22.50, 23.75] |
| comparison:metargon12_minus_metargon12-nodc | +0.00 | [0.00, 0.00] |
| comparison:metargon12_minus_metargon12-norc | +0.00 | [0.00, 0.00] |
| comparison:metargon12_minus_metargon12-noad | +0.00 | [0.00, 0.00] |

## novel

| policy | success | exact D1 | exact D2 | steps | invalid | cpu s |
|---|---:|---:|---:|---:|---:|---:|
| metargon10 | 83.75% | 82.5% | 61.9% | 31.44 | 0.00 | 110.42 |
| metargon11 | 79.38% | 92.5% | 43.8% | 30.05 | 0.00 | 111.21 |
| metargon12 | 79.38% | 95.6% | 76.9% | 30.05 | 0.00 | 113.09 |
| metargon12-nodc | 79.38% | 92.5% | 43.8% | 30.05 | 0.00 | 120.35 |
| metargon12-norc | 79.38% | 95.6% | 76.9% | 30.05 | 0.00 | 130.02 |
| metargon12-noad | 79.38% | 95.6% | 76.9% | 30.05 | 0.00 | 122.30 |
| bayes | 72.50% | 0.0% | 0.0% | 45.78 | 0.00 | 42.21 |
| oracle | 100.00% | 0.0% | 0.0% | 18.00 | 0.00 | 0.43 |

### Paired comparisons (success-rate deltas, pp)

| comparison | Δ pp | 95% CI pp |
|---|---:|---:|
| comparison:metargon12_minus_metargon10 | -4.38 | [-11.25, 0.00] |
| comparison:metargon12_minus_metargon11 | +0.00 | [0.00, 0.00] |
| comparison:metargon12_minus_bayes | +6.88 | [-7.50, 21.25] |
| comparison:metargon12_minus_metargon12-nodc | +0.00 | [0.00, 0.00] |
| comparison:metargon12_minus_metargon12-norc | +0.00 | [0.00, 0.00] |
| comparison:metargon12_minus_metargon12-noad | +0.00 | [0.00, 0.00] |
