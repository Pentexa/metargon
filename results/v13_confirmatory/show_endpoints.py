import json
import sys

d = json.load(open(sys.argv[1], encoding="utf-8"))
for e in d["endpoints"]:
    print(f"{e['label']}: delta {e['delta']*100:+.2f} pp "
          f"CI [{e['ci'][0]*100:.2f}, {e['ci'][1]*100:.2f}] "
          f"nz={e['n_positive_delta_seeds']}/{e['n_seeds']}")
