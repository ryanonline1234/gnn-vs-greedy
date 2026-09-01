"""When does the raw thresholded set stop shrinking and start growing? (peer-review 4c)
If escape at n=1e4 happens only after thousands of epochs, then 36 epochs at n=1e6 is far
too early to read as collapse, and the projection survives. If escape is early, the n=1e6
trajectory is anomalous."""
import json, os, math
import numpy as np, torch
import gen_graphs
from pignn import PIGNN, build_tensors, loss_fn
from runner import ROOT
OUT = os.path.join(ROOT, "results", "early_trace.jsonl")

def run(n, d, seed, max_epochs, every):
    import random as _r; _r.seed(seed)
    import igraph as ig; ig.set_random_number_generator(_r)
    binp, npzp = gen_graphs.generate(n, d, seed)
    z = np.load(npzp); edges = z["edges"]
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cpu"); dim = int(math.sqrt(n)); hid = max(1, dim // 2)
    src, dst, norm, eu, ev = build_tensors(edges, n, dev)
    net = PIGNN(n, dim, hid, 0.0).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    pts = []; best_raw = None; best_ep = None
    for ep in range(max_epochs + 1):
        p = net(src, dst, norm)
        loss = loss_fn(p, eu, ev, 2.0)
        if ep % every == 0:
            raw = int((p.detach() >= 0.5).sum())
            pts.append((ep, raw, float(loss.item())))
            if best_raw is None or raw < best_raw:
                best_raw, best_ep = raw, ep
        opt.zero_grad(); loss.backward(); opt.step()
    rec = dict(method="escape_epoch", n=n, d=d, seed=seed, max_epochs=max_epochs,
               min_raw=best_raw, min_raw_epoch=best_ep,
               points=[dict(epoch=e, raw=r, loss=l) for e, r, l in pts])
    with open(OUT, "a") as f: f.write(json.dumps(rec) + "\n")
    print(f"n={n:,} d={d}: minimum raw={best_raw} at epoch {best_ep} "
          f"(of {max_epochs}); final raw={pts[-1][1]}")
    print("  trajectory (every %d):" % every)
    for e, r, l in pts:
        mark = "  <-- MIN" if e == best_ep else ""
        print(f"    ep {e:>6}  raw {r:>7}  loss {l:>11.1f}{mark}")
    return best_ep, best_raw

if __name__ == "__main__":
    run(10000, 3, 0, 3000, 150)
    print("ESCAPE_EPOCH COMPLETE", flush=True)
