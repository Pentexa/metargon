import json
import sys
from pathlib import Path

for p in sorted(Path("results/v13").glob("verify_v13_*.json")):
    d = json.loads(p.read_text(encoding="utf-8"))
    s = d["summary"]
    fails = [c for c in d["checks"] if c["status"] == "FAIL"]
    incon = [c for c in d["checks"] if c["status"] == "INCONCLUSIVE"]
    print(f"{p.name}: overall={d['overall']} PASS={s['PASS']} "
          f"FAIL={s['FAIL']} INCONCLUSIVE={s['INCONCLUSIVE']}")
    for c in incon:
        print(f"    inconclusive: {c['check']} — {c['message']}")
    for c in fails:
        print(f"    FAIL: {c['check']} — {c['message']}")
