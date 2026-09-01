"""Strengthen the d=10..15 escape probability to 10 seeds.

Motivation (D32): Krutsky, Sir, Kungurtsev & Korpas, ECAI 2025 (arXiv:2507.13703) report a
density-driven phase transition in PI-GNN training dynamics located BETWEEN d=10 and d=20,
with non-collapsed behaviour for d <= 10. This study measures collapse from d >= 8 at n=1000
in the published configuration, with only 3 seeds at d >= 10. That is a direct quantitative
disagreement with a peer-reviewed result at d=10, and 3 seeds cannot carry it. Same published
config, same generator, 10 seeds.
"""
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
for d in [10, 11, 12, 13, 15]:
    for seed in range(10):
        if (d, seed) in done:
            print(f"  .. skip d={d} s={seed}", flush=True); continue
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
print("D10PLUS COMPLETE", flush=True)
