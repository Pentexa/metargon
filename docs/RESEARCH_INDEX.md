# METARGON — Research Index

This index catalogues every research report in the repository. It is the authoritative map from version → report → results → artifacts.

> **Reading guide.** Percentages across versions are **not directly comparable** — each study uses its own environment and protocol (matched seeds and bootstrap CIs apply *within* a study). "Bayes" always refers to the **privileged** family-knowing Bayesian particle-filter planner; "Oracle" is a privileged perfect-knowledge diagnostic **excluded from all rankings**. Family-agnostic student agents are the subject of study.

## Merged research (on `main`)

### CORVID v0.1–v0.7 (historical)

Under its original name CORVID, the project built a sequence of bounded-memory experiments. Preserved verbatim for provenance.

| Version | Research question | Report |
|---|---|---|
| v0.1 | Active perception — when is an extra observation worth its cost? | [`docs/RESEARCH_V03.md`](RESEARCH_V03.md) (v0.3–v0.7 live in the same folder) |
| v0.2 | Selective attention and bounded working memory | `python/adaptive_attention.py` |
| v0.3 | Active causal probing under sparse, delayed feedback | `docs/RESEARCH_V03.md` |
| v0.4 | Learned sensor reliability without quality labels | `docs/RESEARCH_V04.md` |
| v0.5 | Sparse and delayed feedback with verified outcomes | `docs/RESEARCH_V05.md` |
| v0.6 | Bounded verified-feedback memory for diagnostics | `docs/RESEARCH_V06.md` |
| v0.7 | Lightweight predictive world models | `docs/RESEARCH_V07.md` |

### METARGON v0.8 — Learning a Changing World

The rename to METARGON. An independently specified stateful five-position workshop (key, switch, gate with a hidden mechanism that can change mid-run, noisy outcomes).

- **Headline (negative):** the more complex surprise-adaptive controller **underperformed a strong fixed-hazard Bayesian baseline by 3.56 percentage points** under a tight action budget (85.24% vs 88.79%).
- Full preservation: `docs/RESEARCH_V08.md`, `results/stateful_transfer_30seeds.json`.
- Table (30 matched seeds × 240 episodes, tight budget): privileged comparator 92.88%, fixed-hazard Bayes **88.79%**, METARGON surprise-adaptive 85.24%, win-stay/lose-switch 85.24%, random first mechanism 49.24%.

### METARGON v0.9 — Compositional World Models & OOD Transfer

Second independent environment: the **Mechanism Lab** (7×5 grid, two objects, two plates, two switches, two doors, noise). The agent composes family-agnostic hypotheses about object–action–outcome relationships.

| Condition (8 paired seeds × 100 episodes) | METARGON | Bayes | Random | Oracle |
|---|---:|---:|---:|---:|
| stable | 35.1% | 77.5% | 0.3% | 100.0% |
| shift | 46.4% | 80.6% | 0.6% | 100.0% |
| rapid_shift | 40.8% | 76.9% | 0.5% | 91.3% |
| iid | 53.6% | 79.0% | 0.4% | 78.9% |
| novel (held-out) | 31.3% | 78.5% | 0.6% | 93.8% |
| noisy_obs | 33.8% | 73.1% | 0.3% | 98.4% |
| tight_budget | 45.0% | 64.9% | 0.0% | 100.0% |

- **Positive:** METARGON clearly beats model-free baselines (random/episodic/Q-learning ~0–1%) and transfers mechanism knowledge to held-out configurations.
- **Negative:** it does not outperform the family-knowing Bayesian planner. This is the headline negative result: hypothesis composition alone did not close the gap.
- Files: `docs/RESEARCH_V09.md`, `results/v09/benchmark_table.md`, `docs/V09_BASELINE_AUDIT.md`.

### METARGON v0.10 — Robust Perception, Evidence Decay, Structured-OOD Transfer

Diagnosed v0.9's pathologies (discovery freeze, contradictory protected atoms, raw-noise belief corruption, candidate truncation, no mechanism-change memory) and fixed them with family-agnostic mechanisms: robust reconciled belief, probe cooldowns, information-gathering fallback, evidence decay + prioritized replay, complete candidate generation.

| Condition (8 seeds × 100 episodes) | v0.9 | v0.10 | Bayes | Oracle |
|---|---:|---:|---:|---:|
| stable | 35.12% | **80.75%** | 77.50% | 100% |
| shift | 46.38% | **89.25%** | 80.62% | 100% |
| rapid_shift | 40.75% | **86.88%** | 76.88% | 91.25% |
| iid | 53.62% | 74.50% | **79.00%** | 78.88% |
| novel | 31.25% | **89.75%** | 78.50% | 93.75% |
| noisy_obs | 33.75% | 49.37% | **73.12%** | 98.38% |
| tight_budget | 45.00% | **83.62%** | 64.88% | 100% |

- **Positive:** closes the gap to Bayes on stable/shift/rapid_shift/novel/tight_budget; on structurally novel family H (Bayes's model class misspecified) v0.10 sustains ~80% while Bayes drops to ~67% — reported as a **directional** transfer result (CIs wide at 8 seeds).
- **Negative:** `noisy_obs` still trails Bayes (−23.75pp, CI excludes 0); `iid` trails slightly (−4.5pp, CI includes 0). Prioritized replay is functionally validated but empirically neutral-to-negative (−4.7pp under noise).
- Ablations: robust belief (−rb) mainly reduces invalid actions and false-verified hypotheses, not raw success on clean conditions; cooldowns (−cd) cost 12.25pp on novel; fallback (−x) costs 8.6pp under noise.
- Files: `docs/RESEARCH_V10.md`, `results/v10/benchmark_table_v10.md`, `results/v10/`.

### METARGON v0.11 — Noise-Aware Learning (latest merged)

v0.11 targets the remaining `noisy_obs` gap: **confirmed events** (learning stream), **certainty-weighted evidence**, and **replay filtering/weighting**.

| Condition (8 seeds × 100 episodes, F) | v0.10 | v0.11 | Bayes | v11 − v10 pp [95% CI] | v11 − Bayes pp [95% CI] |
|---|---:|---:|---:|---:|---:|
| stable | 80.62 | 74.62 | 77.50 | −6.00 [−18.0, 0.0] | −2.88 [−27.6, 20.6] |
| shift | 86.88 | 86.25 | 80.62 | −0.63 [−9.0, 7.1] | +5.62 [−7.9, 20.0] |
| rapid_shift | 83.62 | 86.50 | 76.88 | +2.88 [−4.5, 13.5] | +9.62 [−2.4, 22.3] |
| iid | 67.75 | 68.25 | 79.00 | +0.50 [−7.0, 5.8] | −10.75 [−19.8, −0.8] |
| novel | 89.75 | 86.75 | 78.50 | −3.00 [−9.4, 0.4] | +8.25 [−4.8, 21.9] |
| noisy_obs | 45.12 | **84.88** | 73.12 | **+39.75 [25.1, 56.4]** | **+11.75 [−3.8, 25.1]** |
| tight_budget | 82.62 | 85.88 | 64.88 | +3.25 [−10.4, 19.8] | +21.00 [9.6, 32.5] |

Family H (8 seeds × 100 episodes): H-stable 79.88 → 68.00 (Bayes 66.88; Δ −11.88 [−28.9, −0.4]); H-novel 79.88 → 74.25 (Bayes 68.12; Δ −5.62 [−13.3, −0.4]).

Ablations (F, noisy_obs): `-cf` (no confirmed events) 73.12 → confirmed events are the main driver (+11.75pp claim); `-ct` (no certainty weighting) 86.62 → net-neutral-to-negative (−1.75pp, CI includes 0); `-rp` (replay off) 80.50 → filtered replay helps; `-rw` (weighted replay) 79.62 → weighted < filtered.

- **Positive:** noisy success more than doubled (45.1% → 84.9%), now exceeding Bayes under noise; the agent meets or exceeds Bayes on 9 of 11 condition×family cells; CPU/memory are lower than v0.10 on noisy (56.6s vs 151.1s per seed-run; peak memory 0.13 vs 0.23 MB stable sample).
- **Negative (measured regressions):** clean-condition `stable` −6.0pp and H-stable −11.9pp vs v0.10; certainty weighting is not a measured win; invalid-action rate did not improve (22 vs 23 per seed-run); `iid` remains below Bayes.
- Files: `docs/RESEARCH_V11.md`, `results/v11/benchmark_table_v11.md`, `results/v11/diagnostics_noise.json`, `docs/WORKLOG_V11.md`.

### Academy v0.1 (development layer)

A separate reproducible continual-learning laboratory (student agents, procedural tasks, frozen splits, checkpointing). Results are **agent-reported**; they were protocol-audited but not independently re-run.

- Held-out test (6 seeds × 40 episodes): oracle 100.0%, **METARGON 50.0%**, Bayes 27.5%, Q-learning 23.3%, episodic 3.7%, random 3.3%. Validation pool: METARGON 66.7% vs Bayes 50.4%. OOD: METARGON 50.0% vs Bayes 20.8%.
- Training (480 episodes/condition, matched exposure): METARGON 0.263, Bayes 0.327, Q-learning 0.106, random/episodic ~0.037.
- **Neutral result:** replay on/off are identical on all pools (0.500/0.667/0.500).
- Files: `docs/ACADEMY_RESEARCH_V01.md`, `docs/ACADEMY_ARCHITECTURE.md`, `results/academy_v01/`.

### EVREN integration + Curriculum v0.2

EVREN is an optional LLM curriculum teacher; proposals are validated tasks, never answers.

- Integration report: `docs/EVREN_INTEGRATION_REPORT.md` (validation chain, quota guard, offline fallback generator, security).
- **Curriculum v0.2 experiment** (pinned offline run, 160 episodes/condition, matched budgets): adaptive learning-progress selection demonstrably ran and gave the best training-phase success (0.444 vs 0.419 for uniform), but produced **no measurable transfer benefit** — LP and teacher-source effects are exactly +0.0000 (zero-width CI). Retention is flat (+0.000–0.125). The only non-uniform classic result is `ascending` (+0.042, one config).
- **Root cause:** the student's hypothesis-memory saturation (66–73 contradicted hypotheses of 128 slots, ~0 verified) is the binding constraint, not the curriculum source. The archived v0.1 `0.000` result was an evaluation artifact (one config per seed; fresh agents also fail those configs).
- Files: `docs/EVREN_CURRICULUM_V02.md`, `results/academy_v01/evren/curriculum_v02_summary.json`.

### Academy Memory v0.3 — Saturation and Negative Transfer

Audited the hypothesis memory and tested minimal ablatable improvements (all off by default) under matched budgets.

| Variant | Train | Classic (16 cfg) | OOD (12 cfg) |
|---|---:|---:|---:|
| **fresh (control)** | — | **0.695** | **0.417** |
| original | 0.362 | 0.562 | 0.333 |
| recycle | 0.362 | 0.562 | 0.333 |
| ratio (r=2) | 0.394 | 0.344 | 0.417 |
| window (w=24) | 0.381 | 0.312 | 0.306 |
| probe | 0.506 | 0.562 | 0.333 |
| protect_local | 0.369 | 0.562 | 0.333 |
| probe_local | 0.469 | 0.562 | 0.333 |
| combined | 0.188 | 0.375 | 0.417 |

- **Negative transfer quantified:** an untrained fresh student scores 0.695 on the classic grid vs 0.562 for every trained variant — training itself costs ~0.13 on unseen configs. Any improvement here is a *recovery* of the fresh baseline.
- **Positive:** adaptive probing (`probe`, `probe_local`) raises training-phase success (+0.144/+0.106) while preserving held-out transfer bit-for-bit (+0.0000).
- **Negative:** ratio gating and evidence windowing retain more verified hypotheses but degrade classic transfer numerically (0.344/0.312 vs 0.562; CIs include 0); the combined variant is strictly worse (train −0.175); no variant lifts held-out transfer above original.
- Files: `docs/MEMORY_V03.md`, `results/academy_v01/memory_v03/`.

### Independent generalization audit

A fully standalone protocol audit plus a new benchmark on two structurally novel families: **R** (sustained-hold gates, `HOLD_K=3`) and **L** (ordered-sequence gates).

| Family R (2 seeds × 15 episodes) | metargon10 | random/episodic | qlearning | bayes | oracle |
|---|---:|---:|---:|---:|---:|
| stable | 50.0% | 3.3% | 0.0% | **83.3%** | 100% |
| shift | 63.3% | 0.0% | 0.0% | **90.0%** | 100% |
| novel | 76.7% | 3.3% | 0.0% | **90.0%** | 100% |
| cross_family | 70.0% | 0.0% | 0.0% | **86.7%** | — |

| Family L (3 seeds × 20 episodes) | metargon10 | random/episodic | qlearning | bayes | oracle |
|---|---:|---:|---:|---:|---:|
| stable | 25.0% | 0.0% | 0.0% | **53.3%** | 100% |
| shift | 23.3% | 0.0% | 0.0% | **56.7%** | 100% |
| novel | 23.3% | 0.0% | 0.0% | **46.7%** | 100% |
| cross_family | 58.3% | 0.0% | 0.0% | **60.0%** | — |

- **Positive:** v0.10 transfers its machinery to the sustained-hold axis (50–93% exact D1 discovery on R, far above model-free baselines); compute matching is a measurable fairness lever; model-free baselines are at chance on both families.
- **Negative:** on R the student still does not beat the family-F Bayes planner on any condition (−13 to −33pp). On L the student's unordered conjunctive hypothesis language **cannot express** the ordered rule — exact D1 discovery is 0.0% on every condition; success is purely behavioral. Cap-120 compute costs the student 20–50pp on R.
- Protocol-audit findings (leakage, baseline fairness, artifacts) are recorded in the report, including the Academy `EpisodeSpec` interface caveat and the v0.2 doc/artifact discrepancy.
- Files: [`generalization_audit/docs/AUDIT_REPORT.md`](../generalization_audit/docs/AUDIT_REPORT.md), [`generalization_audit/results/rendered_tables.md`](../generalization_audit/results/rendered_tables.md), `generalization_audit/results/summary_r.json`, `summary_l.json`.

## Unmerged development work (do not cite as completed)

The following are active branches/worktrees owned by other working sessions. Their features are **not** part of this documentation and must not be presented as completed research:

- `research/v12-general-causality` — METARGON v0.12 (in development).
- `academy/memory-v04` — Academy Memory v0.4 (in development).
- Follow-up independent audit work (Independent Audit v0.2, in development).

## Cross-cutting limitations

1. All environments are synthetic 7×5 grid worlds; no real-world generalization is claimed or measured.
2. Study sizes are small (8 paired seeds for research benchmarks; 2–3 seeds for audit R/L); bootstrap CIs reflect that.
3. The Bayesian comparator is deliberately privileged (receives the declared family); it is a strong upper reference, not a fair competitor.
4. No hardware energy/latency measurements exist; operation counts are not equated with energy.
5. Academy summary metrics reproduce under fixed `PYTHONHASHSEED`; internal hypothesis bookkeeping may vary across hash seeds.
