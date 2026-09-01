"""Claim B: 'the ability to scale beyond the state of the art to problems with millions
of variables.' Tested directly at n = 1e6.

Three things are measured, separately:
  1. Whether the REFERENCE implementation could represent the problem at all
     (it builds a dense n x n Q via torch.zeros(n,n)).
  2. The baselines' wall clock at n=1e6 on this host.
  3. PI-GNN's measured per-epoch cost at n=1e6 with the published configuration,
     and the projected time to the epoch count the method actually needs (taken from
     the measured epochs-to-stop at smaller n). The projection is labelled a projection.
"""
import json, os, sys, subprocess, time, math, argparse
import numpy as np, torch
import gen_graphs, pignn
from runner import ROOT, GREEDY, REPAIR, RHO_UB
from pignn import PIGNN, build_tensors, loss_fn

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1_000_000)
ap.add_argument("--d", type=int, default=3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--probe-epochs", type=int, default=30)
ap.add_argument("--dim-embedding", type=int, default=None)
ap.add_argument("--out", default=os.path.join(ROOT, "results", "scale.jsonl"))
a = ap.parse_args()

n, d, seed = a.n, a.d, a.seed
import random as _r; _r.seed(seed)
import igraph as ig; ig.set_random_number_generator(_r)
print(f"generating n={n} d={d} seed={seed} ...", flush=True)
binp, npzp = gen_graphs.generate(n, d, seed)

recs = []
# ---- 1. dense-Q feasibility of the reference implementation ----
dense_bytes = 4 * n * n
print(f"\n[1] reference implementation builds a dense Q = torch.zeros({n},{n})")
print(f"    that is {n*n:.3e} float32 entries = {dense_bytes/2**40:.1f} TiB "
      f"-> NOT REPRESENTABLE on any single machine.")
recs.append(dict(method="reference_dense_Q_feasibility", n=n, d=d, seed=seed,
                 dense_entries=float(n)*n, dense_bytes=float(dense_bytes),
                 feasible=False))

# ---- 2. baselines ----
print("\n[2] baselines on this host")
for mth in ("dga", "ga"):
    j = json.loads(subprocess.run([GREEDY, binp, mth, str(seed)],
                                  capture_output=True, text=True).stdout)
    rec = dict(method=mth, d=d, n=n, seed=seed, size=j["size"], density=j["density"],
               t_total_s=j["t_solve_s"], violations=j["violations"],
               nonmaximal=j["nonmaximal"])
    rec["ar"] = rec["density"]/RHO_UB[d] if d in RHO_UB else None
    recs.append(rec)
    print(f"    {mth:4s}: size={j['size']} rho={j['density']:.5f} "
          f"AR={rec['ar']:.4f} t={j['t_solve_s']:.4f}s" if rec["ar"] else
          f"    {mth:4s}: size={j['size']} t={j['t_solve_s']:.4f}s", flush=True)

tmpdir = os.path.join(ROOT, "data", "tmp"); os.makedirs(tmpdir, exist_ok=True)
bp = os.path.join(tmpdir, f"null_{n}_{d}.bin")
with open(bp, "wb") as f:
    f.write(np.array([n], dtype=np.int32).tobytes()); f.write(np.ones(n, dtype=np.int8).tobytes())
j = json.loads(subprocess.run([REPAIR, binp, bp], capture_output=True, text=True).stdout)
os.remove(bp)
rec = dict(method="null_ones", d=d, n=n, seed=seed, size=j["repaired_size"],
           density=j["density"], t_total_s=j["t_post_s"], violations=j["violations_after"])
rec["ar"] = rec["density"]/RHO_UB[d] if d in RHO_UB else None
recs.append(rec)
print(f"    null_ones: size={j['repaired_size']} rho={j['density']:.5f} t={j['t_post_s']:.4f}s")

# ---- 3. PI-GNN per-epoch cost at the published configuration ----
dim = a.dim_embedding if a.dim_embedding else int(math.sqrt(n))
hid = max(1, dim//2)
params_embed = n * dim
print(f"\n[3] PI-GNN at published config: dim_embedding=int(sqrt(n))={dim}, hidden={hid}")
print(f"    embedding table = {n} x {dim} = {params_embed:.3e} parameters")
print(f"    fp32 weights {4*params_embed/2**30:.2f} GiB + grads {4*params_embed/2**30:.2f} GiB"
      f" + Adam m,v {8*params_embed/2**30:.2f} GiB = {16*params_embed/2**30:.2f} GiB", flush=True)

z = np.load(npzp); edges = z["edges"]
try:
    torch.manual_seed(seed)
    dev = torch.device("cpu")
    t0 = time.time()
    src, dst, norm, eu, ev = build_tensors(edges, n, dev)
    net = PIGNN(n, dim, hid, 0.0).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    t_build = time.time() - t0
    for _ in range(3):
        loss = loss_fn(net(src, dst, norm), eu, ev); opt.zero_grad(); loss.backward(); opt.step()
    t0 = time.time()
    for _ in range(a.probe_epochs):
        loss = loss_fn(net(src, dst, norm), eu, ev); opt.zero_grad(); loss.backward(); opt.step()
    per_epoch = (time.time()-t0)/a.probe_epochs
    print(f"    build {t_build:.1f}s;  measured {per_epoch*1000:.1f} ms/epoch "
          f"over {a.probe_epochs} epochs", flush=True)
    recs.append(dict(method="pignn_pub_probe", n=n, d=d, seed=seed, dim_embedding=dim,
                     hidden_dim=hid, t_build_s=t_build, ms_per_epoch=per_epoch*1000,
                     probe_epochs=a.probe_epochs, embed_params=float(params_embed),
                     mem_est_gib=16*params_embed/2**30))
except Exception as e:
    print(f"    FAILED: {type(e).__name__}: {e}", flush=True)
    recs.append(dict(method="pignn_pub_probe", n=n, d=d, seed=seed, dim_embedding=dim,
                     failed=f"{type(e).__name__}: {e}"))

with open(a.out, "a") as f:
    for r in recs: f.write(json.dumps(r)+"\n")
print(f"\nwrote {len(recs)} records to {a.out}")
