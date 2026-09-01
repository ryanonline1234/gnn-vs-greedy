"""Stage A of the pre-registered tuning budget (DECISIONS.md D11)."""
import argparse, json, os, subprocess, itertools, time
import numpy as np, gen_graphs, pignn
from runner import run_gnn, load_done, emit, ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--d", type=int, nargs="+", default=[3, 5, 20])
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--device", default="cpu")
a = ap.parse_args()

GRID = list(itertools.product([1e-4, 3e-4, 1e-3, 1e-2], [2.0, 3.0]))
tmpdir = os.path.join(ROOT, "data", "tmp"); os.makedirs(tmpdir, exist_ok=True)
done = load_done(a.out)

for d in a.d:
    for (lr, pen) in GRID:
        tag = f"tune_lr{lr:g}_pen{pen:g}"
        for seed in a.seeds:
            if (tag, d, a.n, seed) in done:
                print(f"  .. skip {tag} d={d} s={seed}", flush=True); continue
            import random as _r; _r.seed(seed)
            import igraph as ig; ig.set_random_number_generator(_r)
            binp, npzp = gen_graphs.generate(a.n, d, seed)
            t0 = time.time()
            rec = run_gnn(npzp, binp, d, a.n, seed, a.device, tag, tmpdir,
                          epochs=100_000, dim_embedding=None, lr=lr, penalty=pen,
                          log_every=0)
            rec["wall_s"] = time.time() - t0; rec["lr"] = lr; rec["penalty"] = pen
            emit(a.out, rec)
