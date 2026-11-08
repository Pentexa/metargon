# Nightly Notes — 2026-11-07/08 (memory-v03)

Session paused by user at ~02:00. Everything below is the exact
continuation point for the morning.

## Branch / worktree
- Branch: `academy/memory-v03` (base origin/main @ 570b826)
- Worktree: `C:\Users\USER\AppData\Local\Temp\opencode\metargon-memory-v03`
  (separate worktree; the main workspace belongs to another session and was
  never touched)

## Committed already (295125c)
- `academy/agents/memory.py` — v0.3 flags (off by default):
  recycle_contradicted, contradiction_ratio, evidence_window
- `academy/agents/metargon_agent.py` — probe_adaptive,
  probe_episode_local, protect_episode_local
- `academy/tools/memory_diagnosis.py`, `academy/tools/memory_experiment.py`
- `academy/tests/test_memory_v03.py` (14 tests, green)
- `docs/MEMORY_V03.md` — draft report (numbers to be updated)

## Results so far (matched budget 160 eps, seed 20261107, HASHSEED=0)
Recorded in results/academy_v01/memory_v03/memory_v03_runs.jsonl (append-only):
- original:  train=0.375  classic=0.667  ood=0.375
- recycle:   train=0.375  classic=0.667  ood=0.375
- window:    (recorded; numbers in jsonl)
Pending/status:
- ratio crashed with rc=4294967295 (external termination, no Python
  traceback) — RETRY tomorrow; if it keeps crashing, rerun the full set
  at episodes=120 + 12/8 grid for consistency.
- probe, protect_local, probe_local, combined: not yet run (driver was
  executing them sequentially; some may have completed since this note
  was written).

## Morning checklist
1. Check `memory_v03_chunks.log` + `memory_v03_runs.jsonl`; confirm all
   variants recorded; retry ratio (and any missing) via
   `python x/run_chunks.py` (skips already-recorded variants).
   NOTE: as of 02:30 the driver was still on `window` (no further
   progress logged); if it hasn't finished by morning, kill it and
   rerun `python x/run_chunks.py` (resumes from jsonl).
2. Merge: `python academy/tools/memory_merge.py` → writes
   memory_v03_summary.json (comparisons + paired CIs).
3. Replace the placeholder numbers in `docs/MEMORY_V03.md` with the
   merged table; keep honest negative results (ratio/window/combined
   degrade transfer; probe_local is the only training-positive variant).
4. Run full suite: `python -m unittest discover -s academy/tests`
   (long; expect ~8 min).
   DONE (02:20): full suite green — **187 tests OK** (incl. the 14
   new test_memory_v03.py).
5. Final commit, push, and open a PR:
   `academy/memory-v03` → `main`.

## Environment notes
- Always run experiments with `PYTHONHASHSEED=0` (recorded in outputs).
- The environment has terminated long-running subprocesses twice at the
  same point (after ~2 variants) — chunked runner + JSONL append is the
  mitigation; do not go back to one monolithic run.
- Do NOT touch `python/` or `rust/` research dirs (another session owns
  them), and do not commit anything outside this worktree's own files.
