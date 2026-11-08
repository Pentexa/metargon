# METARGON v0.11 — benchmark table

Run: 8 seeds × 100 episodes × 7 conditions (family f).

Uncertainty: paired bootstrap 95% intervals (1500 resamples).


## stable

| policy | success | steps | invalid | events | firstEv | cpu s | peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| metargon10 | 80.62% | 37.15 | 0.00 | 174.4 | 5.26 | 56.89 | — |
| metargon11 | 74.62% | 33.75 | 0.00 | 164.0 | 5.26 | 68.75 | — |
| metargon11-cf | 74.50% | 35.14 | 0.00 | 163.2 | 5.32 | 64.93 | — |
| metargon11-ct | 74.62% | 33.75 | 0.00 | 164.0 | 5.26 | 60.35 | — |
| metargon11-rp | 74.62% | 33.75 | 0.00 | 164.0 | 5.26 | 59.72 | — |
| metargon11-rw | 74.62% | 33.75 | 0.00 | 164.0 | 5.26 | 58.41 | — |
| bayes | 77.50% | 45.63 | 0.00 | 221.8 | 8.44 | 26.61 | — |
| oracle | 100.00% | 17.75 | 0.00 | 200.0 | 4.75 | 0.30 | — |

Learning diagnostics (mean per seed-run; offline oracle false-verification not available in summary):

| policy | verified | contradicted | replay applied | confirmed events | dropped candidates |
|---|---:|---:|---:|---:|---:|
| metargon10 | 12.125 | 11.75 | 2777.25 | None | None |
| metargon11 | 11.0 | 10.125 | 1293.25 | 163.0 | 0.0 |
| metargon11-cf | 8.5 | 12.625 | 1295.25 | 0.0 | 0.0 |
| metargon11-ct | 11.0 | 10.125 | 1293.25 | 163.0 | 0.0 |
| metargon11-rp | 9.375 | 10.125 | 0.0 | 163.0 | 0.0 |
| metargon11-rw | 11.0 | 10.125 | 1293.25 | 163.0 | 0.0 |

### Paired comparisons (success-rate deltas, pp)

| condition | comparison | Δ pp | 95% CI pp |
|---|---:|---:|---:|
| stable | v11 vs v10 | -6.00 | [-18.00, 0.00] |
| stable | v11 vs bayes | -2.88 | [-27.63, 20.62] |
| stable | v11 vs -cf (no confirmed events) | +0.13 | [-3.87, 4.62] |
| stable | v11 vs -ct (no certainty) | +0.00 | [0.00, 0.00] |
| stable | v11 vs -rp (replay off) | +0.00 | [0.00, 0.00] |
| stable | v11 vs -rw (replay weighted) | +0.00 | [0.00, 0.00] |

## shift

| policy | success | steps | invalid | events | firstEv | cpu s | peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| metargon10 | 86.88% | 33.56 | 0.00 | 188.6 | 5.35 | 41.07 | — |
| metargon11 | 86.25% | 26.84 | 0.00 | 181.0 | 5.40 | 36.45 | — |
| metargon11-cf | 83.50% | 29.39 | 0.00 | 175.0 | 5.44 | 45.67 | — |
| metargon11-ct | 86.25% | 26.84 | 0.00 | 181.0 | 5.40 | 36.01 | — |
| metargon11-rp | 86.62% | 26.91 | 0.00 | 181.8 | 5.39 | 34.40 | — |
| metargon11-rw | 86.25% | 26.84 | 0.00 | 181.0 | 5.40 | 35.92 | — |
| bayes | 80.62% | 41.41 | 0.00 | 206.8 | 8.66 | 22.43 | — |
| oracle | 100.00% | 20.62 | 0.00 | 212.5 | 6.25 | 0.30 | — |

Learning diagnostics (mean per seed-run; offline oracle false-verification not available in summary):

| policy | verified | contradicted | replay applied | confirmed events | dropped candidates |
|---|---:|---:|---:|---:|---:|
| metargon10 | 18.375 | 12.125 | 3005.25 | None | None |
| metargon11 | 16.125 | 18.0 | 1429.25 | 180.0 | 0.0 |
| metargon11-cf | 14.5 | 15.5 | 1389.25 | 0.0 | 0.0 |
| metargon11-ct | 16.125 | 18.0 | 1429.25 | 180.0 | 0.0 |
| metargon11-rp | 16.625 | 18.625 | 0.0 | 180.75 | 0.0 |
| metargon11-rw | 16.125 | 18.0 | 1429.25 | 180.0 | 0.0 |

### Paired comparisons (success-rate deltas, pp)

| condition | comparison | Δ pp | 95% CI pp |
|---|---:|---:|---:|
| shift | v11 vs v10 | -0.63 | [-9.00, 7.12] |
| shift | v11 vs bayes | +5.62 | [-7.88, 20.00] |
| shift | v11 vs -cf (no confirmed events) | +2.75 | [-1.63, 9.12] |
| shift | v11 vs -ct (no certainty) | +0.00 | [0.00, 0.00] |
| shift | v11 vs -rp (replay off) | -0.38 | [-1.00, 0.00] |
| shift | v11 vs -rw (replay weighted) | +0.00 | [0.00, 0.00] |

## rapid_shift

| policy | success | steps | invalid | events | firstEv | cpu s | peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| metargon10 | 83.62% | 37.02 | 0.00 | 252.8 | 5.90 | 38.51 | — |
| metargon11 | 86.50% | 27.78 | 0.00 | 186.1 | 5.77 | 38.38 | — |
| metargon11-cf | 85.12% | 28.88 | 0.00 | 181.1 | 5.73 | 44.11 | — |
| metargon11-ct | 86.50% | 27.78 | 0.00 | 186.1 | 5.77 | 38.43 | — |
| metargon11-rp | 87.88% | 26.74 | 0.00 | 184.2 | 5.78 | 33.93 | — |
| metargon11-rw | 86.50% | 27.78 | 0.00 | 186.1 | 5.77 | 38.50 | — |
| bayes | 76.88% | 44.66 | 0.00 | 206.6 | 9.93 | 23.06 | — |
| oracle | 91.25% | 26.55 | 0.00 | 193.8 | 6.59 | 0.47 | — |

Learning diagnostics (mean per seed-run; offline oracle false-verification not available in summary):

| policy | verified | contradicted | replay applied | confirmed events | dropped candidates |
|---|---:|---:|---:|---:|---:|
| metargon10 | 13.125 | 15.125 | 4031.25 | None | None |
| metargon11 | 11.125 | 19.875 | 1469.25 | 185.0 | 0.0 |
| metargon11-cf | 10.0 | 16.25 | 1438.25 | 0.0 | 0.0 |
| metargon11-ct | 11.125 | 19.875 | 1469.25 | 185.0 | 0.0 |
| metargon11-rp | 9.25 | 15.875 | 0.0 | 183.25 | 0.0 |
| metargon11-rw | 11.125 | 19.875 | 1469.25 | 185.0 | 0.0 |

### Paired comparisons (success-rate deltas, pp)

| condition | comparison | Δ pp | 95% CI pp |
|---|---:|---:|---:|
| rapid_shift | v11 vs v10 | +2.88 | [-4.50, 13.50] |
| rapid_shift | v11 vs bayes | +9.62 | [-2.38, 22.25] |
| rapid_shift | v11 vs -cf (no confirmed events) | +1.37 | [-1.75, 6.13] |
| rapid_shift | v11 vs -ct (no certainty) | +0.00 | [0.00, 0.00] |
| rapid_shift | v11 vs -rp (replay off) | -1.37 | [-3.37, 0.00] |
| rapid_shift | v11 vs -rw (replay weighted) | +0.00 | [0.00, 0.00] |

## iid

| policy | success | steps | invalid | events | firstEv | cpu s | peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| metargon10 | 67.75% | 46.27 | 0.00 | 411.0 | 6.65 | 93.08 | — |
| metargon11 | 68.25% | 41.55 | 0.00 | 326.8 | 6.55 | 94.52 | — |
| metargon11-cf | 72.50% | 38.03 | 0.00 | 236.9 | 6.15 | 81.31 | — |
| metargon11-ct | 68.25% | 41.55 | 0.00 | 326.8 | 6.55 | 92.04 | — |
| metargon11-rp | 76.50% | 36.29 | 0.00 | 305.1 | 5.99 | 65.42 | — |
| metargon11-rw | 68.25% | 41.55 | 0.00 | 326.8 | 6.55 | 96.91 | — |
| bayes | 79.00% | 43.04 | 0.00 | 210.5 | 9.81 | 23.43 | — |
| oracle | 78.88% | 39.98 | 0.00 | 195.1 | 7.23 | 0.90 | — |

Learning diagnostics (mean per seed-run; offline oracle false-verification not available in summary):

| policy | verified | contradicted | replay applied | confirmed events | dropped candidates |
|---|---:|---:|---:|---:|---:|
| metargon10 | 15.875 | 38.375 | 6562.625 | None | None |
| metargon11 | 7.0 | 29.125 | 1648.25 | 207.375 | 0.0 |
| metargon11-cf | 3.75 | 40.875 | 1884.25 | 0.0 | 0.0 |
| metargon11-ct | 7.0 | 29.125 | 1648.25 | 207.375 | 0.0 |
| metargon11-rp | 1.0 | 43.5 | 0.0 | 221.75 | 0.0 |
| metargon11-rw | 7.0 | 29.125 | 1648.25 | 207.375 | 0.0 |

### Paired comparisons (success-rate deltas, pp)

| condition | comparison | Δ pp | 95% CI pp |
|---|---:|---:|---:|
| iid | v11 vs v10 | +0.50 | [-7.00, 5.75] |
| iid | v11 vs bayes | -10.75 | [-19.75, -0.75] |
| iid | v11 vs -cf (no confirmed events) | -4.25 | [-11.38, 2.00] |
| iid | v11 vs -ct (no certainty) | +0.00 | [0.00, 0.00] |
| iid | v11 vs -rp (replay off) | -8.25 | [-15.50, -0.88] |
| iid | v11 vs -rw (replay weighted) | +0.00 | [0.00, 0.00] |

## novel

| policy | success | steps | invalid | events | firstEv | cpu s | peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| metargon10 | 89.75% | 31.65 | 0.00 | 186.4 | 6.71 | 25.73 | — |
| metargon11 | 86.75% | 27.31 | 0.00 | 182.0 | 6.26 | 34.21 | — |
| metargon11-cf | 83.88% | 29.42 | 0.00 | 175.8 | 6.05 | 46.14 | — |
| metargon11-ct | 86.75% | 27.31 | 0.00 | 182.0 | 6.26 | 39.72 | — |
| metargon11-rp | 86.62% | 26.52 | 0.00 | 181.8 | 6.05 | 39.55 | — |
| metargon11-rw | 86.75% | 27.31 | 0.00 | 182.0 | 6.26 | 41.27 | — |
| bayes | 78.50% | 43.69 | 0.00 | 201.8 | 9.68 | 23.62 | — |
| oracle | 93.75% | 25.25 | 0.00 | 187.5 | 7.94 | 0.43 | — |

Learning diagnostics (mean per seed-run; offline oracle false-verification not available in summary):

| policy | verified | contradicted | replay applied | confirmed events | dropped candidates |
|---|---:|---:|---:|---:|---:|
| metargon10 | 12.875 | 18.125 | 2969.25 | None | None |
| metargon11 | 12.625 | 20.5 | 1437.25 | 181.0 | 0.0 |
| metargon11-cf | 10.125 | 21.625 | 1395.25 | 0.0 | 0.0 |
| metargon11-ct | 12.625 | 20.5 | 1437.25 | 181.0 | 0.0 |
| metargon11-rp | 12.75 | 18.625 | 0.0 | 180.75 | 0.0 |
| metargon11-rw | 12.625 | 20.5 | 1437.25 | 181.0 | 0.0 |

### Paired comparisons (success-rate deltas, pp)

| condition | comparison | Δ pp | 95% CI pp |
|---|---:|---:|---:|
| novel | v11 vs v10 | -3.00 | [-9.38, 0.38] |
| novel | v11 vs bayes | +8.25 | [-4.75, 21.88] |
| novel | v11 vs -cf (no confirmed events) | +2.88 | [-1.75, 9.75] |
| novel | v11 vs -ct (no certainty) | +0.00 | [0.00, 0.00] |
| novel | v11 vs -rp (replay off) | +0.13 | [0.00, 0.38] |
| novel | v11 vs -rw (replay weighted) | +0.00 | [0.00, 0.00] |

## noisy_obs

| policy | success | steps | invalid | events | firstEv | cpu s | peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| metargon10 | 45.12% | 63.19 | 23.00 | 477.9 | 10.80 | 151.10 | — |
| metargon11 | 84.88% | 33.38 | 22.00 | 219.4 | 6.66 | 56.61 | — |
| metargon11-cf | 73.12% | 41.03 | 24.00 | 203.1 | 7.05 | 103.45 | — |
| metargon11-ct | 86.62% | 31.92 | 27.62 | 202.6 | 6.95 | 54.62 | — |
| metargon11-rp | 80.50% | 33.47 | 41.38 | 181.1 | 6.37 | 63.43 | — |
| metargon11-rw | 79.62% | 36.62 | 31.50 | 220.6 | 6.54 | 62.49 | — |
| bayes | 73.12% | 48.10 | 54.75 | 232.8 | 8.99 | 28.83 | — |
| oracle | 98.38% | 21.38 | 3.88 | 210.5 | 5.34 | 1.51 | — |

Learning diagnostics (mean per seed-run; offline oracle false-verification not available in summary):

| policy | verified | contradicted | replay applied | confirmed events | dropped candidates |
|---|---:|---:|---:|---:|---:|
| metargon10 | 47.5 | 9.75 | 34991.875 | None | None |
| metargon11 | 9.0 | 23.75 | 2561.5 | 325.125 | 2.375 |
| metargon11-cf | 10.625 | 10.5 | 5141.625 | 0.0 | 5.75 |
| metargon11-ct | 8.75 | 24.375 | 2324.5 | 292.125 | 3.125 |
| metargon11-rp | 6.0 | 23.0 | 0.0 | 280.375 | 4.5 |
| metargon11-rw | 10.125 | 21.125 | 2627.5 | 330.0 | 4.375 |

### Paired comparisons (success-rate deltas, pp)

| condition | comparison | Δ pp | 95% CI pp |
|---|---:|---:|---:|
| noisy_obs | v11 vs v10 | +39.75 | [25.12, 56.38] |
| noisy_obs | v11 vs bayes | +11.75 | [-3.75, 25.12] |
| noisy_obs | v11 vs -cf (no confirmed events) | +11.75 | [1.00, 23.75] |
| noisy_obs | v11 vs -ct (no certainty) | -1.75 | [-15.38, 8.25] |
| noisy_obs | v11 vs -rp (replay off) | +4.37 | [-3.75, 14.50] |
| noisy_obs | v11 vs -rw (replay weighted) | +5.25 | [-1.38, 14.62] |

## tight_budget

| policy | success | steps | invalid | events | firstEv | cpu s | peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| metargon10 | 82.62% | 28.40 | 0.00 | 232.8 | 5.02 | 25.28 | — |
| metargon11 | 85.88% | 21.88 | 0.00 | 178.9 | 5.30 | 23.50 | — |
| metargon11-cf | 84.75% | 22.90 | 0.00 | 174.4 | 5.09 | 30.23 | — |
| metargon11-ct | 85.88% | 21.88 | 0.00 | 178.9 | 5.30 | 21.45 | — |
| metargon11-rp | 86.50% | 22.03 | 0.00 | 180.1 | 5.31 | 20.48 | — |
| metargon11-rw | 85.88% | 21.88 | 0.00 | 178.9 | 5.30 | 18.68 | — |
| bayes | 64.88% | 32.05 | 0.00 | 181.5 | 7.75 | 15.46 | — |
| oracle | 100.00% | 20.62 | 0.00 | 212.5 | 6.25 | 0.28 | — |

Learning diagnostics (mean per seed-run; offline oracle false-verification not available in summary):

| policy | verified | contradicted | replay applied | confirmed events | dropped candidates |
|---|---:|---:|---:|---:|---:|
| metargon10 | 16.75 | 12.25 | 3711.25 | None | None |
| metargon11 | 16.125 | 13.0 | 1416.0 | 178.375 | 0.0 |
| metargon11-cf | 15.25 | 16.5 | 1384.25 | 0.0 | 0.0 |
| metargon11-ct | 16.125 | 13.0 | 1416.0 | 178.375 | 0.0 |
| metargon11-rp | 16.625 | 13.875 | 0.0 | 179.625 | 0.0 |
| metargon11-rw | 16.125 | 13.0 | 1416.0 | 178.375 | 0.0 |

### Paired comparisons (success-rate deltas, pp)

| condition | comparison | Δ pp | 95% CI pp |
|---|---:|---:|---:|
| tight_budget | v11 vs v10 | +3.25 | [-10.38, 19.75] |
| tight_budget | v11 vs bayes | +21.00 | [9.62, 32.50] |
| tight_budget | v11 vs -cf (no confirmed events) | +1.12 | [-9.38, 12.12] |
| tight_budget | v11 vs -ct (no certainty) | +0.00 | [0.00, 0.00] |
| tight_budget | v11 vs -rp (replay off) | -0.63 | [-1.38, 0.00] |
| tight_budget | v11 vs -rw (replay weighted) | +0.00 | [0.00, 0.00] |
