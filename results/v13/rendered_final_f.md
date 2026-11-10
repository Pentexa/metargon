# METARGON v0.13 — family f
seeds 4 × 40 episodes; conditions ['stable', 'shift', 'noisy_obs', 'novel', 'tight_budget']; policies ['metargon10', 'metargon11', 'metargon12', 'metargon13', 'bayes', 'oracle']

## stable
| policy | success_rate | exact_discovery_D1 | exact_discovery_D2 | covered_discovery_D1 | covered_discovery_D2 | mean_steps | invalid_actions | cpu_s | temporal_probes | temporal_generated | temporal_verified | temporal_evictions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| metargon10 | 85.6% | 88.1% | 83.8% | 88.1% | 83.8% | 34.03 | 0.00 | 16.5 | — | — | — | — |
| metargon11 | 85.6% | 96.2% | 65.6% | 96.2% | 65.6% | 27.77 | 0.00 | 14.2 | — | — | — | — |
| metargon12 | 85.6% | 97.5% | 84.4% | 97.5% | 84.4% | 27.87 | 0.00 | 15.0 | 0.00 | — | — | — |
| metargon13 | 85.6% | 97.5% | 84.4% | 97.5% | 84.4% | 27.87 | 0.00 | 14.6 | 0.00 | — | — | — |
| bayes | 78.8% | 0.0% | 0.0% | 0.0% | 0.0% | 45.49 | 0.00 | 7.7 | — | — | — | — |
| oracle | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 18.75 | 0.00 | 0.1 | — | — | — | — |

### per-seed exact D1 (frozen criteria: seed sensitivity)
| seed | metargon10 | metargon11 | metargon12 | metargon13 |
|---|---|---|---|---|
| 0 | 95.0% | 87.5% | 92.5% | 92.5% |
| 1 | 97.5% | 97.5% | 97.5% | 97.5% |
| 2 | 80.0% | 100.0% | 100.0% | 100.0% |
| 3 | 80.0% | 100.0% | 100.0% | 100.0% |
- metargon10: exact D1 nonzero seeds 4/4, population stddev 0.082
- metargon11: exact D1 nonzero seeds 4/4, population stddev 0.052
- metargon12: exact D1 nonzero seeds 4/4, population stddev 0.031
- metargon13: exact D1 nonzero seeds 4/4, population stddev 0.031
- comparison:metargon13_minus_metargon12: success delta +0.00 pp CI (0.0, 0.0)
- comparison:metargon13_minus_metargon10: success delta +0.00 pp CI (0.0, 0.0)
- comparison:metargon13_minus_metargon11: success delta +0.00 pp CI (0.0, 0.0)
- comparison:metargon12_minus_metargon10: success delta +0.00 pp CI (0.0, 0.0)
- comparison:metargon12_minus_metargon11: success delta +0.00 pp CI (0.0, 0.0)

## shift
| policy | success_rate | exact_discovery_D1 | exact_discovery_D2 | covered_discovery_D1 | covered_discovery_D2 | mean_steps | invalid_actions | cpu_s | temporal_probes | temporal_generated | temporal_verified | temporal_evictions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| metargon10 | 86.2% | 86.2% | 83.8% | 86.2% | 83.8% | 36.48 | 0.00 | 16.8 | — | — | — | — |
| metargon11 | 90.6% | 90.6% | 65.0% | 90.6% | 65.0% | 25.43 | 0.00 | 9.9 | — | — | — | — |
| metargon12 | 91.2% | 95.0% | 88.8% | 95.6% | 88.8% | 25.21 | 0.00 | 9.0 | 0.00 | — | — | — |
| metargon13 | 91.2% | 95.0% | 88.8% | 95.6% | 88.8% | 25.21 | 0.00 | 9.8 | 0.00 | — | — | — |
| bayes | 80.6% | 0.0% | 0.0% | 0.0% | 0.0% | 42.63 | 0.00 | 7.4 | — | — | — | — |
| oracle | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 22.38 | 0.00 | 0.1 | — | — | — | — |

### per-seed exact D1 (frozen criteria: seed sensitivity)
| seed | metargon10 | metargon11 | metargon12 | metargon13 |
|---|---|---|---|---|
| 0 | 87.5% | 65.0% | 85.0% | 85.0% |
| 1 | 97.5% | 97.5% | 97.5% | 97.5% |
| 2 | 80.0% | 100.0% | 97.5% | 97.5% |
| 3 | 80.0% | 100.0% | 100.0% | 100.0% |
- metargon10: exact D1 nonzero seeds 4/4, population stddev 0.072
- metargon11: exact D1 nonzero seeds 4/4, population stddev 0.148
- metargon12: exact D1 nonzero seeds 4/4, population stddev 0.059
- metargon13: exact D1 nonzero seeds 4/4, population stddev 0.059
- comparison:metargon13_minus_metargon12: success delta +0.00 pp CI (0.0, 0.0)
- comparison:metargon13_minus_metargon10: success delta +5.00 pp CI (0.0, 15.0)
- comparison:metargon13_minus_metargon11: success delta +0.62 pp CI (0.0, 1.9)
- comparison:metargon12_minus_metargon10: success delta +5.00 pp CI (0.0, 15.0)
- comparison:metargon12_minus_metargon11: success delta +0.62 pp CI (0.0, 1.9)

## noisy_obs
| policy | success_rate | exact_discovery_D1 | exact_discovery_D2 | covered_discovery_D1 | covered_discovery_D2 | mean_steps | invalid_actions | cpu_s | temporal_probes | temporal_generated | temporal_verified | temporal_evictions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| metargon10 | 44.4% | 97.5% | 59.4% | 97.5% | 59.4% | 62.46 | 9.75 | 35.8 | — | — | — | — |
| metargon11 | 80.6% | 63.7% | 31.9% | 70.0% | 41.2% | 36.48 | 10.00 | 24.5 | — | — | — | — |
| metargon12 | 81.2% | 80.6% | 75.6% | 81.2% | 75.6% | 33.41 | 12.75 | 18.0 | 0.00 | — | — | — |
| metargon13 | 81.2% | 80.6% | 75.6% | 81.2% | 75.6% | 33.41 | 12.75 | 17.7 | 0.00 | — | — | — |
| bayes | 76.2% | 0.0% | 0.0% | 0.0% | 0.0% | 46.01 | 21.00 | 8.8 | — | — | — | — |
| oracle | 98.1% | 0.0% | 0.0% | 0.0% | 0.0% | 22.48 | 3.00 | 0.5 | — | — | — | — |

### per-seed exact D1 (frozen criteria: seed sensitivity)
| seed | metargon10 | metargon11 | metargon12 | metargon13 |
|---|---|---|---|---|
| 0 | 92.5% | 20.0% | 40.0% | 40.0% |
| 1 | 100.0% | 42.5% | 90.0% | 90.0% |
| 2 | 97.5% | 92.5% | 92.5% | 92.5% |
| 3 | 100.0% | 100.0% | 100.0% | 100.0% |
- metargon10: exact D1 nonzero seeds 4/4, population stddev 0.031
- metargon11: exact D1 nonzero seeds 4/4, population stddev 0.336
- metargon12: exact D1 nonzero seeds 4/4, population stddev 0.237
- metargon13: exact D1 nonzero seeds 4/4, population stddev 0.237
- comparison:metargon13_minus_metargon12: success delta +0.00 pp CI (0.0, 0.0)
- comparison:metargon13_minus_metargon10: success delta +36.88 pp CI (18.8, 68.8)
- comparison:metargon13_minus_metargon11: success delta +0.63 pp CI (-1.9, 3.7)
- comparison:metargon12_minus_metargon10: success delta +36.88 pp CI (18.8, 68.8)
- comparison:metargon12_minus_metargon11: success delta +0.63 pp CI (-1.9, 3.7)

## novel
| policy | success_rate | exact_discovery_D1 | exact_discovery_D2 | covered_discovery_D1 | covered_discovery_D2 | mean_steps | invalid_actions | cpu_s | temporal_probes | temporal_generated | temporal_verified | temporal_evictions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| metargon10 | 91.2% | 86.9% | 84.4% | 86.9% | 84.4% | 28.23 | 0.00 | 10.1 | — | — | — | — |
| metargon11 | 91.9% | 92.5% | 66.9% | 92.5% | 66.9% | 23.52 | 0.00 | 8.3 | — | — | — | — |
| metargon12 | 91.2% | 96.2% | 88.1% | 96.2% | 88.1% | 23.98 | 0.00 | 7.7 | 0.00 | — | — | — |
| metargon13 | 91.2% | 96.2% | 88.1% | 96.2% | 88.1% | 23.98 | 0.00 | 6.6 | 0.00 | — | — | — |
| bayes | 79.4% | 0.0% | 0.0% | 0.0% | 0.0% | 42.52 | 0.00 | 6.3 | — | — | — | — |
| oracle | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 21.88 | 0.00 | 0.1 | — | — | — | — |

### per-seed exact D1 (frozen criteria: seed sensitivity)
| seed | metargon10 | metargon11 | metargon12 | metargon13 |
|---|---|---|---|---|
| 0 | 90.0% | 72.5% | 87.5% | 87.5% |
| 1 | 97.5% | 97.5% | 97.5% | 97.5% |
| 2 | 80.0% | 100.0% | 100.0% | 100.0% |
| 3 | 80.0% | 100.0% | 100.0% | 100.0% |
- metargon10: exact D1 nonzero seeds 4/4, population stddev 0.074
- metargon11: exact D1 nonzero seeds 4/4, population stddev 0.116
- metargon12: exact D1 nonzero seeds 4/4, population stddev 0.052
- metargon13: exact D1 nonzero seeds 4/4, population stddev 0.052
- comparison:metargon13_minus_metargon12: success delta +0.00 pp CI (0.0, 0.0)
- comparison:metargon13_minus_metargon10: success delta +0.00 pp CI (0.0, 0.0)
- comparison:metargon13_minus_metargon11: success delta -0.63 pp CI (-1.9, 0.0)
- comparison:metargon12_minus_metargon10: success delta +0.00 pp CI (0.0, 0.0)
- comparison:metargon12_minus_metargon11: success delta -0.63 pp CI (-1.9, 0.0)

## tight_budget
| policy | success_rate | exact_discovery_D1 | exact_discovery_D2 | covered_discovery_D1 | covered_discovery_D2 | mean_steps | invalid_actions | cpu_s | temporal_probes | temporal_generated | temporal_verified | temporal_evictions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| metargon10 | 73.1% | 85.0% | 80.6% | 85.0% | 80.6% | 31.13 | 0.00 | 9.8 | — | — | — | — |
| metargon11 | 90.0% | 87.5% | 64.4% | 88.1% | 64.4% | 21.73 | 0.00 | 5.7 | — | — | — | — |
| metargon12 | 90.6% | 93.1% | 86.9% | 93.8% | 86.9% | 21.58 | 0.00 | 4.5 | 0.00 | — | — | — |
| metargon13 | 90.6% | 93.1% | 86.9% | 93.8% | 86.9% | 21.58 | 0.00 | 4.7 | 0.00 | — | — | — |
| bayes | 65.6% | 0.0% | 0.0% | 0.0% | 0.0% | 32.66 | 0.00 | 4.9 | — | — | — | — |
| oracle | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 22.38 | 0.00 | 0.1 | — | — | — | — |

### per-seed exact D1 (frozen criteria: seed sensitivity)
| seed | metargon10 | metargon11 | metargon12 | metargon13 |
|---|---|---|---|---|
| 0 | 82.5% | 60.0% | 82.5% | 82.5% |
| 1 | 97.5% | 97.5% | 97.5% | 97.5% |
| 2 | 80.0% | 95.0% | 95.0% | 95.0% |
| 3 | 80.0% | 97.5% | 97.5% | 97.5% |
- metargon10: exact D1 nonzero seeds 4/4, population stddev 0.073
- metargon11: exact D1 nonzero seeds 4/4, population stddev 0.159
- metargon12: exact D1 nonzero seeds 4/4, population stddev 0.062
- metargon13: exact D1 nonzero seeds 4/4, population stddev 0.062
- comparison:metargon13_minus_metargon12: success delta +0.00 pp CI (0.0, 0.0)
- comparison:metargon13_minus_metargon10: success delta +17.50 pp CI (0.0, 35.6)
- comparison:metargon13_minus_metargon11: success delta +0.63 pp CI (0.0, 1.9)
- comparison:metargon12_minus_metargon10: success delta +17.50 pp CI (0.0, 35.6)
- comparison:metargon12_minus_metargon11: success delta +0.63 pp CI (0.0, 1.9)
