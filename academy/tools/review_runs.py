"""Review currently recorded JSONL rows."""
import json

path = "results/academy_v01/memory_v03/memory_v03_runs.jsonl"
for line in open(path, encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    p = r["per"]
    print(f"{r['variant']:10s} train={p['train_success']:.3f} "
          f"classic={p['classic_test']['config_mean_success']:.3f} "
          f"ood={p['ood']['config_mean_success']:.3f} "
          f"mem={p['memory']['status_counts']}")
