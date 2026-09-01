"""Generate and cache random d-regular graphs.

Binary format consumed by greedy.c:  int32 n, int32 m, then 2*m int32 (u,v) pairs.
Also caches an .npz with the edge array for the torch side.
"""
import sys, os, time, argparse, random
import numpy as np
import igraph as ig

CACHE = os.path.join(os.path.dirname(__file__), "..", "data")

def path_for(n, d, seed, ext):
    return os.path.join(CACHE, f"rrg_n{n}_d{d}_s{seed}.{ext}")

def generate(n, d, seed):
    """Uniform random d-regular graph on n nodes."""
    binp, npzp = path_for(n, d, seed, "bin"), path_for(n, d, seed, "npz")
    if os.path.exists(binp) and os.path.exists(npzp):
        return binp, npzp
    t0 = time.time()
    # igraph K_Regular: uniform sampling of simple d-regular graphs
    g = ig.Graph.K_Regular(n, d)
    e = np.array(g.get_edgelist(), dtype=np.int32)
    assert e.shape[0] == n * d // 2, (e.shape, n * d // 2)
    # sanity: simple, d-regular
    degs = np.bincount(e.ravel(), minlength=n)
    assert degs.min() == d and degs.max() == d, (degs.min(), degs.max())
    os.makedirs(CACHE, exist_ok=True)
    with open(binp, "wb") as f:
        f.write(np.array([n, e.shape[0]], dtype=np.int32).tobytes())
        f.write(e.tobytes())
    np.savez_compressed(npzp, edges=e, n=n, d=d, seed=seed)
    print(f"  gen n={n} d={d} s={seed} m={e.shape[0]} in {time.time()-t0:.1f}s", flush=True)
    return binp, npzp

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", required=True)
    ap.add_argument("--d", type=int, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    a = ap.parse_args()
    for d in a.d:
        for n in a.n:
            for s in a.seeds:
                random.seed(s); ig.set_random_number_generator(random)
                generate(n, d, s)
