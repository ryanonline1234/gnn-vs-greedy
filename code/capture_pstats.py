"""Direct measurement of the final p-vector's uniformity for the modified_linear arm at
d=20 — the discriminating check the mean-field claim needs (review round, mechanism
finding 1): loss-matching alone cannot distinguish a uniform configuration from one with
graph-correlated scatter, because the loss is second-order blind to zero-mean
perturbations around the stationary point. So measure p itself.
"""
import json, os
import numpy as np
import pignn, gen_graphs

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results", "pstats.jsonl")
n, d, P = 1000, 20, 2.0
pstar = 1.0 / (P * d)
Lstar = -n / (2 * P * d)

captured = {}
_orig = pignn.loss_fn
def hook(p, eu, ev, penalty=2.0, diag="quadratic"):
    captured["p"] = p.detach()
    return _orig(p, eu, ev, penalty, diag)
pignn.loss_fn = hook

for seed in (0, 1, 2):
    import random as _r; _r.seed(seed)
    import igraph as ig; ig.set_random_number_generator(_r)
    binp, npzp = gen_graphs.generate(n, d, seed)
    r = pignn.run(npzp, "cpu", seed=seed, log_every=0, diag="linear")
    p = captured["p"].numpy()
    rec = dict(method="pstats_modified_linear", d=d, n=n, seed=seed,
               pstar=pstar, Lstar=Lstar, final_loss=r["final_loss"],
               p_mean=float(p.mean()), p_std=float(p.std()),
               p_max=float(p.max()), p_min=float(p.min()),
               std_over_pstar=float(p.std() / pstar),
               mean_over_pstar=float(p.mean() / pstar),
               frac_above_threshold=float((p >= 0.5).mean()),
               epochs_run=r["epochs_run"])
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"seed {seed}: loss={r['final_loss']:.4f} (L*={Lstar}) mean(p)={p.mean():.5f} "
          f"(p*={pstar}) std/p*={p.std()/pstar:.3f} max={p.max():.4f}", flush=True)
