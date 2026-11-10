# METARGON v0.13 — family h
seeds 4 × 40 episodes; conditions ['stable', 'novel']; policies ['metargon10', 'metargon11', 'metargon12', 'metargon13', 'bayes', 'oracle']

## stable
| policy | success_rate | exact_discovery_D1 | exact_discovery_D2 | covered_discovery_D1 | covered_discovery_D2 | mean_steps | invalid_actions | cpu_s | temporal_probes | temporal_generated | temporal_verified | temporal_evictions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| metargon10 | 83.8% | 88.8% | 63.1% | 88.8% | 63.1% | 36.61 | 0.00 | 16.6 | — | — | — | — |
| metargon11 | 73.1% | 92.5% | 41.2% | 92.5% | 41.2% | 34.09 | 0.00 | 25.7 | — | — | — | — |
| metargon12 | 73.1% | 95.6% | 70.0% | 95.6% | 70.0% | 34.09 | 0.00 | 26.0 | 0.00 | — | — | — |
| metargon13 | 73.1% | 95.6% | 70.0% | 95.6% | 70.0% | 34.09 | 0.00 | 25.8 | 0.00 | — | — | — |
| bayes | 71.9% | 0.0% | 0.0% | 0.0% | 0.0% | 46.00 | 0.00 | 8.2 | — | — | — | — |
| oracle | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 16.50 | 0.00 | 0.1 | — | — | — | — |

### per-seed exact D1 (frozen criteria: seed sensitivity)
| seed | metargon10 | metargon11 | metargon12 | metargon13 |
|---|---|---|---|---|
| 0 | 82.5% | 95.0% | 97.5% | 97.5% |
| 1 | 82.5% | 95.0% | 97.5% | 97.5% |
| 2 | 95.0% | 87.5% | 92.5% | 92.5% |
| 3 | 95.0% | 92.5% | 95.0% | 95.0% |
- metargon10: exact D1 nonzero seeds 4/4, population stddev 0.062
- metargon11: exact D1 nonzero seeds 4/4, population stddev 0.031
- metargon12: exact D1 nonzero seeds 4/4, population stddev 0.021
- metargon13: exact D1 nonzero seeds 4/4, population stddev 0.021
- comparison:metargon13_minus_metargon12: success delta +0.00 pp CI (0.0, 0.0)
- comparison:metargon13_minus_metargon10: success delta -10.63 pp CI (-30.0, 0.0)
- comparison:metargon13_minus_metargon11: success delta +0.00 pp CI (0.0, 0.0)
- comparison:metargon12_minus_metargon10: success delta -10.63 pp CI (-30.0, 0.0)
- comparison:metargon12_minus_metargon11: success delta +0.00 pp CI (0.0, 0.0)

## novel
| policy | success_rate | exact_discovery_D1 | exact_discovery_D2 | covered_discovery_D1 | covered_discovery_D2 | mean_steps | invalid_actions | cpu_s | temporal_probes | temporal_generated | temporal_verified | temporal_evictions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| metargon10 | 83.8% | 82.5% | 61.9% | 82.5% | 61.9% | 31.44 | 0.00 | 15.6 | — | — | — | — |
| metargon11 | 79.4% | 92.5% | 43.8% | 92.5% | 43.8% | 30.05 | 0.00 | 16.4 | — | — | — | — |
| metargon12 | 79.4% | 95.6% | 76.9% | 95.6% | 76.9% | 30.05 | 0.00 | 15.1 | 0.00 | — | — | — |
| metargon13 | 79.4% | 95.6% | 76.9% | 95.6% | 76.9% | 30.05 | 0.00 | 14.7 | 0.00 | — | — | — |
| bayes | 72.5% | 0.0% | 0.0% | 0.0% | 0.0% | 45.78 | 0.00 | 5.9 | — | — | — | — |
| oracle | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 18.00 | 0.00 | 0.1 | — | — | — | — |

### per-seed exact D1 (frozen criteria: seed sensitivity)
| seed | metargon10 | metargon11 | metargon12 | metargon13 |
|---|---|---|---|---|
| 0 | 82.5% | 95.0% | 97.5% | 97.5% |
| 1 | 57.5% | 95.0% | 97.5% | 97.5% |
| 2 | 95.0% | 87.5% | 92.5% | 92.5% |
| 3 | 95.0% | 92.5% | 95.0% | 95.0% |
- metargon10: exact D1 nonzero seeds 4/4, population stddev 0.153
- metargon11: exact D1 nonzero seeds 4/4, population stddev 0.031
- metargon12: exact D1 nonzero seeds 4/4, population stddev 0.021
- metargon13: exact D1 nonzero seeds 4/4, population stddev 0.021
- comparison:metargon13_minus_metargon12: success delta +0.00 pp CI (0.0, 0.0)
- comparison:metargon13_minus_metargon10: success delta -4.38 pp CI (-11.3, 0.0)
- comparison:metargon13_minus_metargon11: success delta +0.00 pp CI (0.0, 0.0)
- comparison:metargon12_minus_metargon10: success delta -4.38 pp CI (-11.3, 0.0)
- comparison:metargon12_minus_metargon11: success delta +0.00 pp CI (0.0, 0.0)
