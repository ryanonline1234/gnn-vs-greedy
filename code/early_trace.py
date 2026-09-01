"""Discriminating experiment for the n=1e6 projection (peer-review concern 4c).

The budgeted n=1e6 run's raw thresholded size decays geometrically (137,531 -> 220 over 36
epochs, halving every ~3) with the loss still POSITIVE (+81,089), i.e. best_bits is still the
all-zeros bitstring under the reference's own bookkeeping. Two readings:
  (a) this is the normal early transient at every n (p starts near 0.5, descends, symmetry
      breaks later) -> the projection is sound;
  (b) escape probability is n-dependent as well as d-dependent -> the 7.31-day figure is a
      projection of the time to produce the EMPTY SET.
These are distinguishable: trace the same quantities over early epochs at n=1e3 and n=1e4,
d=3, where escape is KNOWN to occur, and compare the shape.
"""
import json, os, time
import numpy as np, torch
import gen_graphs, pignn
from pignn import PIGNN, build_tensors, loss_fn
from runner import ROOT

OUT = os.path.join(ROOT, "results", "early_trace.jsonl")

def trace(n, d, seed, max_epochs, every):
    import random as _r; _r.seed(seed)
    import igraph as ig; ig.set_random_number_generator(_r)
    binp, npzp = gen_graphs.generate(n, d, seed)
    z = np.load(npzp); edges = z["edges"]
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cpu")
    import math
    dim = int(math.sqrt(n)); hid = max(1, dim // 2)
    src, dst, norm, eu, ev = build_tensors(edges, n, dev)
    net = PIGNN(n, dim, hid, 0.0).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    pts = []
    for ep in range(max_epochs + 1):
        p = net(src, dst, norm)
        loss = loss_fn(p, eu, ev, 2.0)
        if ep % every == 0:
            raw = int((p.detach() >= 0.5).sum())
            pts.append(dict(epoch=ep, raw=raw, loss=float(loss.item()),
                            frac=raw / n))
        opt.zero_grad(); loss.backward(); opt.step()
    rec = dict(method="early_trace", n=n, d=d, seed=seed, dim_embedding=dim,
               max_epochs=max_epochs, points=pts)
    with open(OUT, "a") as f: f.write(json.dumps(rec) + "\n")
    print(f"\n n={n:,} d={d} seed={seed}  (dim={dim})")
    print(f"   {'epoch':>6} {'raw':>9} {'frac':>8} {'ratio':>7} {'loss':>12}")
    prev = None
    for x in pts:
        ratio = f"{x['raw']/prev:.2f}" if prev else "  -  "
        print(f"   {x['epoch']:>6} {x['raw']:>9} {x['frac']:>8.4f} {ratio:>7} {x['loss']:>12.1f}")
        prev = x["raw"] if x["raw"] else None
    return pts

if __name__ == "__main__":
    for (n, mx, ev) in [(1000, 60, 3), (10000, 60, 3)]:
        trace(n, 3, 0, mx, ev)
    print("\nEARLY_TRACE COMPLETE", flush=True)
