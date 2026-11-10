# METARGON v0.13 — family l
seeds 4 × 40 episodes; conditions ['stable', 'novel']; policies ['metargon12', 'metargon13', 'metargon13-gen', 'metargon13-notp', 'metargon12-notp', 'oracle', 'random', 'metargon12-notp', 'qlearning', 'episodic']

## stable
| policy | success_rate | exact_discovery_D1 | exact_discovery_D2 | covered_discovery_D1 | covered_discovery_D2 | mean_steps | invalid_actions | cpu_s | temporal_probes | temporal_generated | temporal_verified | temporal_evictions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| metargon12 | 27.5% | 16.9% | 0.0% | 16.9% | 0.0% | 68.01 | 0.00 | 47.2 | 44.75 | — | — | — |
| metargon13 | 29.4% | 39.4% | 20.0% | 39.4% | 20.0% | 65.26 | 0.00 | 44.7 | 57.00 | 18.25 | 3.75 | 0.00 |
| metargon13-gen | 27.5% | 16.9% | 0.0% | 16.9% | 0.0% | 68.01 | 0.00 | 52.0 | 44.75 | 8.75 | 0.75 | 0.00 |
| metargon13-notp | 18.1% | 0.0% | 0.0% | 0.0% | 0.0% | 75.31 | 0.00 | 61.5 | 0.00 | — | — | — |
| metargon12-notp | 18.1% | 0.0% | 0.0% | 0.0% | 0.0% | 75.31 | 0.00 | 67.3 | 0.00 | — | — | — |
| oracle | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 24.25 | 0.00 | 0.2 | — | — | — | — |
| random | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 80.00 | 0.00 | 0.5 | — | — | — | — |
| metargon12-notp | 18.1% | 0.0% | 0.0% | 0.0% | 0.0% | 75.31 | 0.00 | 67.3 | 0.00 | — | — | — |
| qlearning | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 80.00 | 0.00 | 0.6 | — | — | — | — |
| episodic | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 80.00 | 0.00 | 0.6 | — | — | — | — |

### per-seed exact D1 (frozen criteria: seed sensitivity)
| seed | metargon12 | metargon13 | metargon13-gen | metargon13-notp | metargon12-notp | metargon12-notp |
|---|---|---|---|---|---|---|
| 0 | 0.0% | 77.5% | 0.0% | 0.0% | 0.0% | 0.0% |
| 1 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 2 | 67.5% | 80.0% | 67.5% | 0.0% | 0.0% | 0.0% |
| 3 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
- metargon12: exact D1 nonzero seeds 1/4, population stddev 0.292
- metargon13: exact D1 nonzero seeds 2/4, population stddev 0.394
- metargon13-gen: exact D1 nonzero seeds 1/4, population stddev 0.292
- metargon13-notp: exact D1 nonzero seeds 0/4, population stddev 0.000
- metargon12-notp: exact D1 nonzero seeds 0/4, population stddev 0.000
- metargon12-notp: exact D1 nonzero seeds 0/4, population stddev 0.000
- comparison:metargon13_minus_metargon12: success delta +1.87 pp CI (0.0, 5.6)
- comparison:metargon13_minus_metargon13-notp: success delta +11.25 pp CI (-9.4, 45.6)
- comparison:metargon13_minus_metargon13-gen: success delta +1.87 pp CI (0.0, 5.6)
- comparison:temporal_minus_notp_exact_D1: delta +39.38 pp CI (0.0, 78.8)
- comparison:v12_temporal_minus_notp_exact_D1: delta +16.88 pp CI (0.0, 50.6)

## novel
| policy | success_rate | exact_discovery_D1 | exact_discovery_D2 | covered_discovery_D1 | covered_discovery_D2 | mean_steps | invalid_actions | cpu_s | temporal_probes | temporal_generated | temporal_verified | temporal_evictions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| metargon12 | 29.4% | 23.1% | 6.2% | 23.1% | 6.2% | 67.21 | 0.00 | 57.4 | 42.50 | — | — | — |
| metargon13 | 27.5% | 33.1% | 13.8% | 33.1% | 13.8% | 67.27 | 0.00 | 60.2 | 50.50 | 19.25 | 3.25 | 0.00 |
| metargon13-gen | 29.4% | 23.1% | 6.2% | 23.1% | 6.2% | 67.21 | 0.00 | 59.3 | 42.50 | 9.00 | 0.75 | 0.00 |
| metargon13-notp | 18.8% | 0.0% | 0.0% | 0.0% | 0.0% | 74.01 | 0.00 | 61.0 | 0.00 | — | — | — |
| metargon12-notp | 18.8% | 0.0% | 0.0% | 0.0% | 0.0% | 74.01 | 0.00 | 44.2 | 0.00 | — | — | — |
| oracle | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 23.75 | 0.00 | 0.2 | — | — | — | — |
| random | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 80.00 | 0.00 | 0.4 | — | — | — | — |
| metargon12-notp | 18.8% | 0.0% | 0.0% | 0.0% | 0.0% | 74.01 | 0.00 | 44.2 | 0.00 | — | — | — |
| qlearning | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 80.00 | 0.00 | 0.4 | — | — | — | — |
| episodic | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 80.00 | 0.00 | 0.5 | — | — | — | — |

### per-seed exact D1 (frozen criteria: seed sensitivity)
| seed | metargon12 | metargon13 | metargon13-gen | metargon13-notp | metargon12-notp | metargon12-notp |
|---|---|---|---|---|---|---|
| 0 | 0.0% | 27.5% | 0.0% | 0.0% | 0.0% | 0.0% |
| 1 | 25.0% | 25.0% | 25.0% | 0.0% | 0.0% | 0.0% |
| 2 | 67.5% | 80.0% | 67.5% | 0.0% | 0.0% | 0.0% |
| 3 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
- metargon12: exact D1 nonzero seeds 2/4, population stddev 0.276
- metargon13: exact D1 nonzero seeds 3/4, population stddev 0.291
- metargon13-gen: exact D1 nonzero seeds 2/4, population stddev 0.276
- metargon13-notp: exact D1 nonzero seeds 0/4, population stddev 0.000
- metargon12-notp: exact D1 nonzero seeds 0/4, population stddev 0.000
- metargon12-notp: exact D1 nonzero seeds 0/4, population stddev 0.000
- comparison:metargon13_minus_metargon12: success delta -1.88 pp CI (-5.6, 0.0)
- comparison:metargon13_minus_metargon13-notp: success delta +8.75 pp CI (-3.8, 21.3)
- comparison:metargon13_minus_metargon13-gen: success delta -1.88 pp CI (-5.6, 0.0)
- comparison:temporal_minus_notp_exact_D1: delta +33.13 pp CI (6.9, 66.3)
- comparison:v12_temporal_minus_notp_exact_D1: delta +23.12 pp CI (0.0, 50.6)
