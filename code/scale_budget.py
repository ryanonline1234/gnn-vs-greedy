"""Claim B, measured rather than projected: run PI-GNN at n=1e6 in the published
configuration under a fixed WALL-CLOCK budget, and report the best valid independent set
it has produced when the budget expires. DGA's time on the same host is the reference.

This is a real measurement of 'what does the method deliver at a million variables',
as opposed to an extrapolation. A truncated run is reported as truncated.
"""
import argparse, json, math, os, subprocess, time
import numpy as np, torch
import gen_graphs
from pignn import PIGNN, build_tensors, loss_fn
from runner import ROOT, GREEDY, REPAIR, RHO_UB

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1_000_000)
ap.add_argument("--d", type=int, default=3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--budget-s", type=float, default=2700.0)
ap.add_argument("--checkpoint-every", type=int, default=250)
ap.add_argument("--out", default=os.path.join(ROOT, "results", "scale.jsonl"))
a = ap.parse_args()

import random as _r; _r.seed(a.seed)
import igraph as ig; ig.set_random_number_generator(_r)
binp, npzp = gen_graphs.generate(a.n, a.d, a.seed)
z = np.load(npzp); edges = z["edges"]; n = a.n

dim = int(math.sqrt(n)); hid = max(1, dim // 2)
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = torch.device("cpu")
print(f"n={n:,} d={a.d} dim_embedding={dim} hidden={hid} budget={a.budget_s:.0f}s", flush=True)

t_build0 = time.time()
src, dst, norm, eu, ev = build_tensors(edges, n, dev)
net = PIGNN(n, dim, hid, 0.0).to(dev)
opt = torch.optim.Adam(net.parameters(), lr=1e-4)
t_build = time.time() - t_build0
print(f"build {t_build:.1f}s", flush=True)

tmpdir = os.path.join(ROOT, "data", "tmp"); os.makedirs(tmpdir, exist_ok=True)
bp = os.path.join(tmpdir, f"scale_{n}_{a.d}.bin")

def score(p):
    bits = (p.detach() >= 0.5).to(torch.int8).cpu().numpy().astype(np.int8)
    with open(bp, "wb") as f:
        f.write(np.array([n], dtype=np.int32).tobytes()); f.write(bits.tobytes())
    return json.loads(subprocess.run([REPAIR, binp, bp], capture_output=True, text=True).stdout)

t0 = time.time(); ep = 0; ckpts = []
prev_loss, count, patience, tol = 1.0, 0, 100, 1e-4
stop_reason = "budget"
while True:
    el = time.time() - t0
    if el > a.budget_s:
        break
    p = net(src, dst, norm)
    loss = loss_fn(p, eu, ev, 2.0)
    loss_ = loss.item()
    if (abs(loss_ - prev_loss) <= tol) or ((loss_ - prev_loss) > 0):
        count += 1
    else:
        count = 0
    if count >= patience:
        stop_reason = "converged (published early-stopping rule)"
        break
    prev_loss = loss_
    opt.zero_grad(); loss.backward(); opt.step()
    ep += 1
    if ep % a.checkpoint_every == 0:
        j = score(p)
        ckpts.append(dict(epoch=ep, t=el, loss=loss_, repaired=j["repaired_size"],
                          raw=j["raw_size"], density=j["density"]))
        print(f"  ep {ep:>6d} t={el:7.1f}s loss={loss_:>12.1f} raw={j['raw_size']:>7d} "
              f"repaired={j['repaired_size']:>7d} rho={j['density']:.5f}", flush=True)

t_train = time.time() - t0
j = score(p)
if os.path.exists(bp): os.remove(bp)
rec = dict(method="pignn_pub_budgeted", n=n, d=a.d, seed=a.seed, dim_embedding=dim,
           hidden_dim=hid, epochs_run=ep, t_build_s=t_build, t_train_s=t_train,
           t_total_s=t_build + t_train + j["t_post_s"], budget_s=a.budget_s,
           stop_reason=stop_reason, size=j["repaired_size"], raw_size=j["raw_size"],
           density=j["density"], violations=j["violations_after"],
           truncated=(stop_reason == "budget"), checkpoints=ckpts)
rec["ar"] = rec["density"] / RHO_UB[a.d] if a.d in RHO_UB else None
with open(a.out, "a") as f: f.write(json.dumps(rec) + "\n")
print(f"\nFINAL after {ep} epochs / {t_train:.0f}s ({stop_reason}): "
      f"repaired={j['repaired_size']} rho={j['density']:.5f} "
      + (f"AR={rec['ar']:.4f}" if rec["ar"] else ""), flush=True)
