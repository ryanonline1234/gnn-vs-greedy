"""Test the defence the Replies ACTUALLY published (D36), which the pre-registered grid
missed: GraphSAGE instead of GCN, and penalty P=10 instead of P=2.

Reply to Angelini (arXiv:2302.03602): "by simply setting P = 10 we find even further
consistent improvements. Specifically, for d = 3 at P = 10 GraphSAGE achieves approximation
ratios of AR ~ 0.947, on par with the DGA-based results reported by Angelini and
Ricci-Tersenghi."  That is a falsifiable claim; this script tests it on our harness.
"""
import json, os, subprocess, time
import numpy as np, gen_graphs, pignn
from runner import ROOT, REPAIR, RHO_UB

OUT = os.path.join(ROOT, "results", "reply_defence.jsonl")
tmp = os.path.join(ROOT, "data", "tmp"); os.makedirs(tmp, exist_ok=True)
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        r = json.loads(l); done.add((r["method"], r["d"], r["seed"]))

ARMS = [("gcn_P2",   dict(arch="gcn",  penalty=2.0)),   # as published (control)
        ("gcn_P10",  dict(arch="gcn",  penalty=10.0)),  # tweak 2 alone
        ("sage_P2",  dict(arch="sage", penalty=2.0)),   # tweak 1 alone
        ("sage_P10", dict(arch="sage", penalty=10.0))]  # the Reply's configuration

for tag, hp in ARMS:
    for d in (3, 5, 20):
        for seed in (0, 1, 2):
            if (tag, d, seed) in done:
                print(f"  .. skip {tag} d={d} s={seed}", flush=True); continue
            import random as _r; _r.seed(seed)
            import igraph as ig; ig.set_random_number_generator(_r)
            binp, npzp = gen_graphs.generate(1000, d, seed)
            bp = os.path.join(tmp, f"rd_{tag}_{d}_{seed}.bin")
            t0 = time.time()
            r = pignn.run(npzp, "cpu", seed=seed, log_every=0, dump=bp, **hp)
            j = json.loads(subprocess.run([REPAIR, binp, bp], capture_output=True,
                                          text=True).stdout)
            os.remove(bp)
            rec = dict(method=tag, arch=hp["arch"], penalty=hp["penalty"], d=d, n=1000,
                       seed=seed, raw_size=r["raw_size"], size=j["repaired_size"],
                       density=j["density"], epochs_run=r["epochs_run"],
                       final_loss=r["final_loss"], t_train_s=r["t_train_s"],
                       wall_s=time.time()-t0)
            rec["ar"] = rec["density"]/RHO_UB[d] if d in RHO_UB else None
            with open(OUT, "a") as f: f.write(json.dumps(rec)+"\n")
            ars = f"AR={rec['ar']:.4f}" if rec["ar"] else f"rho={rec['density']:.5f}"
            print(f"  {tag:<9} d={d:<2} s={seed}: raw={rec['raw_size']:>5} {ars}", flush=True)
print("REPLY_DEFENCE COMPLETE", flush=True)
