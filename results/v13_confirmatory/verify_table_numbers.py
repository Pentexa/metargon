import json
import statistics
import sys

d = json.load(open(sys.argv[1], encoding="utf-8"))
for key, t in sorted(d["tables"].items()):
    su = [v for v in t["success_rate"] if v is not None]
    d1 = [v for v in t["exact_discovery_D1"] if v is not None]
    d2 = [v for v in t["exact_discovery_D2"] if v is not None]
    print(f"{key}: success={statistics.fmean(su)*100:.2f} "
          f"D1={statistics.fmean(d1)*100:.2f} D2={statistics.fmean(d2)*100:.2f}")
