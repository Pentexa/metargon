# METARGON v0.12.1 — Regression Recovery (findings so far)

**Branch:** `research/v121-regression-recovery`
**Base:** origin/main (7b706be, v12 merged)

## Diagnosed (4×40 paired, matched seeds)

### F-noisy (v12 72.5% vs v11 80.6%)
| variant | success | D1 exact | D2 exact |
|---|---:|---:|---:|
| v10 | 44.38 | 97.5 | 59.4 |
| v11 | 80.62 | 63.7 | 31.9 |
| v12 (dual) | 72.50 | 87.5 | 76.9 |
| v12 -nodc (clean context) | 80.00 | 62.5 | 43.8 |
| v12 -norc (no recent credit) | 81.25 | 80.6 | 75.6 |
| v12 -noad (no adaptive confirm) | 72.50 | 87.5 | 76.9 |
| v12 -dst (dual_stable) | 73.12 | 79.4 | — |

→ Dual-context costs −7.5pp on noisy but restores D2 (76.9 vs 43.8).
→ Recent-credit costs −8.75pp on noisy AND keeps D2 (75.6). Best single fix
  for F-noisy: recent-credit off (81.25 ≈ v11).
→ dual_stable only +0.6pp over dual; not the answer.

### H-stable (v12 73.1% vs v10 83.8%)
| variant | success | D2 exact |
|---|---:|---:|
| v10 | 83.75 | 63.1 |
| v11 | 73.12 | 41.2 |
| v11 -cf (no confirmed events) | 79.38 | 20.6 |
| v11 -ct (no certainty) | 73.12 | 41.2 |
| v11 -rp (no replay) | 73.12 | 41.2 |
| v12 (dual) | 73.12 | 70.0 |
| v12 -dst | 73.12 | 70.0 |

→ H loss is NOT from v12 mechanisms (all v12 ablations identical). The
  loss resides in the shared v11 confirmed-event gating (−6.3pp).
→ Confirmed-off recovers success but kills D2 discovery (20.6).
→ v12 dual restores D2 (70.0) but keeps success at 73.1.

### R-stable (v12 60.6% vs v11 71.9%)
| variant | success |
|---|---:|
| v10 | 61.25 |
| v11 | 71.88 |
| v12 | 60.62 |
| v12 -norc | 60.62 |
| v12 -noad | 60.62 |
| v12 -dst | 60.00 |
| v12 -nodc | (pending) |

→ Recent-credit NOT the cause (report hypothesis wrong). Dual_stable no
  help. -nodc measurement pending.

### L (temporal preservation check)
- v12 original (dual + temporal): exact D1 16.9%(stable)/23.1%(novel).
- Adaptive gating prototype broke temporal exact D1 (0%) and was rejected
  (default reverted to off).
- eval2_l (original v12 on L) pending — must confirm preservation.

## Prototype tested and REJECTED (honest negative)
- Adaptive confirmed-event gating by anomaly rate: F-noisy 53.1% (disaster:
  noise pollution during the un-gated warm-up window), H D2 exact 44.4%,
  L exact D1 0%. Reverted.

## Candidate fixes (evidence-supported)
1. Recent-credit OFF – single best F-noisy fix (+8.75pp, D2 kept).
   Check L preservation before adopting.
2. H: confirmed-events gating communicates with D2 discovery — need a
   mechanism that keeps D2 learning without the gating cost on success.

## Status
- [x] F/H/R ablation panels (v121 diag + eval + eval2)
- [x] Adaptive gating prototype measured & rejected
- [ ] R -nodc measurement (running)
- [ ] L preservation (running)
- [ ] Fix implementation
- [ ] Re-measure + full tests
- [ ] Commit / push / PR
