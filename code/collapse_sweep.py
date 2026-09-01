"""Locate the degree at which the published PI-GNN objective collapses to the origin.
Cheap (n=1000) but decisive: reports raw_size BEFORE post-processing, which is the
quantity that reveals the collapse."""
import json, os, sys, time, math
import numpy as np, gen_graphs, pignn
from runner import ROOT, REPAIR
import subprocess

OUT = os.path.join(ROOT, "results", "collapse.jsonl")
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        try:
            r = json.loads(l); done.add((r["d"], r["seed"]))
        except Exception: pass

for d in [3, 5, 7, 10, 12, 14, 16, 18, 20, 25]:
    for seed in [0, 1, 2]:
        if (d, seed) in done: 
            print(f"  .. skip d={d} s={seed}"); continue
        import random as _r; _r.seed(seed)
        import igraph as ig; ig.set_random_number_generator(_r)
        binp, npzp = gen_graphs.generate(1000, d, seed)
        r = pignn.run(npzp, "cpu", seed=seed, log_every=0)
        rec = dict(d=d, n=1000, seed=seed, raw_size=r["raw_size"],
                   raw_density=r["raw_size"]/1000.0, epochs=r["epochs_run"],
                   final_loss=r["final_loss"], best_loss=r["best_loss"],
                   t_train_s=r["t_train_s"])
        with open(OUT, "a") as f: f.write(json.dumps(rec)+"\n")
        print(f"  d={d:2d} s={seed}: raw_size={r['raw_size']:>4d} "
              f"final_loss={r['final_loss']:>10.2f} epochs={r['epochs_run']}", flush=True)
