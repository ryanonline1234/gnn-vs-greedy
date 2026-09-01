"""POST-HOC escape attempts at d=20 (DECISIONS.md D13 objections b,c).
These are NOT part of the pre-registered arm and are labelled separately.

Arms, all at n=1000, seeds 0-2:
  posthoc_lr1e-3 / lr1e-2   - larger steps may escape the origin's basin
  posthoc_dim256            - a bigger embedding breaks symmetry harder
  posthoc_patience5000      - rules out "stopped too early"
  posthoc_combined          - all three together, the strongest charitable config
  modified_linear           - MODIFIES the published objective: diagonal -sum p_i
                              instead of -sum p_i^2, which removes the trivial
                              critical point at the origin. Reported as a
                              modification, never as PI-GNN.
"""
import json, os, subprocess, time
import numpy as np, gen_graphs, pignn
from runner import ROOT, REPAIR, RHO_UB

OUT = os.path.join(ROOT, "results", "posthoc.jsonl")
ARMS = {
  "posthoc_lr1e-3":      dict(lr=1e-3),
  "posthoc_lr1e-2":      dict(lr=1e-2),
  "posthoc_dim256":      dict(dim_embedding=256),
  "posthoc_patience5000":dict(patience=5000),
  "posthoc_combined":    dict(lr=1e-3, dim_embedding=256, patience=5000),
  "modified_linear":     dict(diag="linear"),
  "modified_linear_lr1e-3": dict(diag="linear", lr=1e-3),
}
DS = [3, 5, 20]
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        try:
            r = json.loads(l); done.add((r["method"], r["d"], r["seed"]))
        except Exception: pass

tmpdir = os.path.join(ROOT, "data", "tmp"); os.makedirs(tmpdir, exist_ok=True)
for d in DS:
    for tag, hp in ARMS.items():
        for seed in [0, 1, 2]:
            if (tag, d, seed) in done:
                print(f"  .. skip {tag} d={d} s={seed}", flush=True); continue
            import random as _r; _r.seed(seed)
            import igraph as ig; ig.set_random_number_generator(_r)
            binp, npzp = gen_graphs.generate(1000, d, seed)
            t0 = time.time()
            bp = os.path.join(tmpdir, f"ph_{tag}_{d}_{seed}.bin")
            r = pignn.run(npzp, "cpu", seed=seed, dump=bp, log_every=0, **hp)
            j = json.loads(subprocess.run([REPAIR, binp, bp], capture_output=True,
                                          text=True).stdout)
            os.remove(bp)
            rec = dict(method=tag, d=d, n=1000, seed=seed, size=j["repaired_size"],
                       density=j["density"], raw_size=j["raw_size"],
                       t_total_s=r["t_build_s"]+r["t_train_s"]+j["t_post_s"],
                       t_train_s=r["t_train_s"], epochs_run=r["epochs_run"],
                       final_loss=r["final_loss"], diag=r["diag"], lr=r["lr"],
                       dim_embedding=r["dim_embedding"], patience=r["patience"],
                       violations=j["violations_after"], wall_s=time.time()-t0)
            rec["ar"] = rec["density"]/RHO_UB[d] if d in RHO_UB else None
            with open(OUT, "a") as f: f.write(json.dumps(rec)+"\n")
            ars = f" AR={rec['ar']:.4f}" if rec["ar"] else ""
            print(f"  {tag:<24} d={d:2d} s={seed}: raw={j['raw_size']:>5d} "
                  f"repaired={j['repaired_size']:>5d} rho={rec['density']:.4f}{ars} "
                  f"({r['epochs_run']} ep)", flush=True)
