import json
import sys

path = sys.argv[1]
d = json.load(open(path, encoding="utf-8"))
for run in d["runs"]:
    clean = sum(e["gate_events"]["clean"] for e in run["episodes"])
    dual = sum(e["gate_events"]["dual"] for e in run["episodes"])
    succ = sum(e["success"] for e in run["episodes"])
    print(f"seed {run['seed']}: clean_events {clean} dual_events {dual} "
          f"success {succ}/{len(run['episodes'])}")
