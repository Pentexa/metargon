import json, sys, glob
from pathlib import Path
sys.path.insert(0, '.'); sys.path.insert(0, 'python'); sys.path.insert(0, 'generalization_audit')
from generalization_audit.verification import load_manifest, VerifierContext, run_checks, FAIL, INCONCLUSIVE, PASS
from generalization_audit.verification.artifact_gate import check_frozen_target_artifacts

root = Path('.').resolve()
report = {
    "audit": "post-merge-v06",
    "date": "2026-11-10",
    "starting_commit": "cb56c0bf514e7b5b5e08fcd33c94c2565e21efee",
    "manifest_verification": {},
    "artifact_gate": {},
    "research_v13_recomputation": None,
    "academy_v06_recomputation": None,
}

manifests = sorted(glob.glob('generalization_audit/manifests/v13_*.manifest.json')) + \
    sorted(glob.glob('generalization_audit/manifests/memory_v06*.manifest.json'))
per = {}
for mf in manifests:
    m = load_manifest(mf)
    results = run_checks(VerifierContext(m, Path(mf)))
    fails = [r for r in results if r.status == FAIL]
    incs = [r for r in results if r.status == INCONCLUSIVE]
    passn = sum(1 for r in results if r.status == PASS)
    per[mf] = {
        "PASS": passn, "FAIL": len(fails), "INCONCLUSIVE": len(incs),
        "overall": "FAIL" if fails else ("INCONCLUSIVE" if incs else "PASS"),
        "fail_checks": [r.name for r in fails],
        "inconclusive_checks": [(r.name, r.message) for r in incs],
    }
report["manifest_verification"]["sum"] = {
    "PASS": sum(v["PASS"] for v in per.values()),
    "FAIL": sum(v["FAIL"] for v in per.values()),
    "INCONCLUSIVE": sum(v["INCONCLUSIVE"] for v in per.values()),
}
report["manifest_verification"]["per_manifest"] = per

for r in check_frozen_target_artifacts(root):
    report["artifact_gate"][r.target] = {
        "overall": r.overall,
        "checks": [{"name": c.name, "status": c.status, "message": c.message}
                   for c in r.checks],
    }

for key, path in (("research_v13_recomputation", "results/verification_v13/post_merge_b_recompute.json"),
                  ("academy_v06_recomputation", "results/verification_v13/post_merge_c_recompute.json")):
    p = Path(path)
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        report[key] = {
            "checks": len(d["checks"]),
            "FAIL": sum(1 for c in d["checks"] if c["status"] == "FAIL"),
            "PASS": sum(1 for c in d["checks"] if c["status"] == "PASS"),
        }

out = root / "results" / "verification_v13" / "post_merge_audit_v06.json"
out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(f"wrote {out}")
print(json.dumps({"manifest_sum": report["manifest_verification"]["sum"],
                  "gate": {k: v["overall"] for k, v in report["artifact_gate"].items()},
                  "b": report["research_v13_recomputation"],
                  "c": report["academy_v06_recomputation"]}, indent=2))
