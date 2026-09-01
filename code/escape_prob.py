"""Characterise the escape probability vs degree (DECISIONS.md D16). n=1000, 10 seeds."""
import json, os
import gen_graphs, pignn
from runner import ROOT
OUT = os.path.join(ROOT, "results", "escape.jsonl")
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        try:
            r = json.loads(l); done.add((r["d"], r["seed"]))
        except Exception: pass
for d in [4, 6, 7, 8, 9]:
    for seed in range(10):
        if (d, seed) in done: continue
        import random as _r; _r.seed(seed)
        import igraph as ig; ig.set_random_number_generator(_r)
        binp, npzp = gen_graphs.generate(1000, d, seed)
        r = pignn.run(npzp, "cpu", seed=seed, log_every=0)
        rec = dict(d=d, n=1000, seed=seed, raw_size=r["raw_size"],
                   escaped=bool(r["raw_size"] > 0), epochs=r["epochs_run"],
                   final_loss=r["final_loss"])
        with open(OUT, "a") as f: f.write(json.dumps(rec)+"\n")
        print(f"  d={d:2d} s={seed}: raw={r['raw_size']:>4d} "
              f"{'ESCAPED' if rec['escaped'] else 'collapsed'} "
              f"loss={r['final_loss']:>9.2f}", flush=True)
