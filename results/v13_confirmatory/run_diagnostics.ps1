# Mission C diagnostics on confirmatory L seeds (8 seeds x 40 episodes,
# instrumented v13 and v12), as preregistered in
# docs/RESEARCH_V13_CONFIRMATORY.md PART III.
# Usage (in repo root): powershell -File results/v13_confirmatory/run_diagnostics.ps1

$env:PYTHONHASHSEED = "0"

# stable, v13 and v12
python python/diagnose_v13.py --family l --conditions stable --seeds 8 `
    --episodes 40 --policy v13 --seed-base 21800 `
    --json results/v13_confirmatory/diag_l_stable_8x40.json
python python/diagnose_v13.py --family l --conditions stable --seeds 8 `
    --episodes 40 --policy v12 --seed-base 21800 `
    --json results/v13_confirmatory/diag_l_stable_v12_8x40.json

# novel, v13 and v12
python python/diagnose_v13.py --family l --conditions novel --seeds 8 `
    --episodes 40 --policy v13 --seed-base 21800 `
    --json results/v13_confirmatory/diag_l_novel_8x40.json
python python/diagnose_v13.py --family l --conditions novel --seeds 8 `
    --episodes 40 --policy v12 --seed-base 21800 `
    --json results/v13_confirmatory/diag_l_novel_v12_8x40.json

Write-Host "ALL DIAGNOSTICS DONE"
