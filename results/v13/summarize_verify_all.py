import json
from pathlib import Path

total_pass = total_fail = total_incon = 0
for p in sorted(Path("results/verification_v13").glob("verify_v13_*.json")):
    d = json.loads(p.read_text(encoding="utf-8"))
    s = d["summary"]
    fails = [c for c in d["checks"] if c["status"] == "FAIL"]
    incon = [c for c in d["checks"] if c["status"] == "INCONCLUSIVE"]
    total_pass += s["PASS"]; total_fail += s["FAIL"]; total_incon += s["INCONCLUSIVE"]
    print(f"{p.name}: overall={d['overall']} PASS={s['PASS']} "
          f"FAIL={s['FAIL']} INCONCLUSIVE={s['INCONCLUSIVE']}")
    for c in incon:
        print(f"    inconclusive: {c['check']} — {c['message']}")
    for c in fails:
        print(f"    FAIL: {c['check']} — {c['message']}")
print(f"\nTOTAL PASS={total_pass} FAIL={total_fail} INCONCLUSIVE={total_incon}")
