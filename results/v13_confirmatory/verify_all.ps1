# Verify all confirmatory manifests with the Quality v0.5 verifier.
# Usage (in repo root): powershell -File results/v13_confirmatory/verify_all.ps1

$manifs = @(
    "v13_confirmatory_l",
    "v13_confirmatory_f",
    "v13_confirmatory_h",
    "v13_confirmatory_r",
    "v13_confirmatory_r_nodc",
    "v13_confirmatory_diag_l_stable",
    "v13_confirmatory_diag_l_stable_v12",
    "v13_confirmatory_diag_l_novel",
    "v13_confirmatory_diag_l_novel_v12",
    "v13_confirmatory_diag_h_stable",
    "v13_confirmatory_diag_r_stable"
)

foreach ($m in $manifs) {
    $mf = "generalization_audit/manifests/$m.manifest.json"
    if (-not (Test-Path $mf)) {
        Write-Host "MISSING $mf"
        continue
    }
    python -m generalization_audit.tools.verify_experiment `
        --manifest $mf --json -o "results/v13_confirmatory/verify_$m.json" | Out-Null
    Write-Host "$m exit=$LASTEXITCODE"
    python -c @"
import json
d = json.load(open('results/v13_confirmatory/verify_$m.json', encoding='utf-8'))
s = d['summary']
print('   overall', d['overall'], 'PASS', s['PASS'], 'FAIL', s['FAIL'], 'INC', s['INCONCLUSIVE'])
for c in d['checks']:
    if c['status'] == 'FAIL':
        print('   FAIL', c['check'], '-', c['message'])
"@
}
