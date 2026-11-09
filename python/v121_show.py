#!/usr/bin/env python3
"""Print a compact one-line summary per policy for a v121 eval JSON."""
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
d = json.loads(path.read_text(encoding="utf-8"))
policies = d["metadata"]["policies"]
for cond, row in d["summary"].items():
    print(f"[{cond}]")
    for p in policies:
        x = row.get(p)
        if not x:
            continue
        ex1 = x.get("exact_discovery_D1")
        ex2 = x.get("exact_discovery_D2")
        m = (f"{p:>18} succ {x['success_rate']*100:6.2f}% "
             f"exD1 {ex1*100 if ex1 is not None else float('nan'):5.1f}% "
             f"exD2 {ex2*100 if ex2 is not None else float('nan'):5.1f}% "
             f"steps {x['mean_steps']:5.2f} "
             f"inv {x['invalid_actions']:5.2f} "
             f"cpu {x['cpu_s']:7.2f}s")
        print(m)
