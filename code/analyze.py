"""Aggregate JSONL results into per-(method,d,n) summaries with mean +/- s.e.m."""
import json, sys, os, glob
import numpy as np
from collections import defaultdict

RHO_UB = {3: 0.45537, 5: 0.38443}
AR_1RSB = {3: 0.990, 5: 0.987}
ORDER = ["dga", "ga", "null_ones", "null_rand", "pignn_pub", "pignn_tuned"]
LABEL = {"dga": "DGA (greedy)", "ga": "GA (random greedy)",
         "null_ones": "null: all-ones", "null_rand": "null: coin-flip",
         "pignn_pub": "PI-GNN (as published)", "pignn_tuned": "PI-GNN (tuned)"}

def load(paths):
    recs = []
    for p in paths:
        if not os.path.exists(p): continue
        for line in open(p):
            line = line.strip()
            if line:
                try: recs.append(json.loads(line))
                except Exception: pass
    return recs

def agg(recs):
    g = defaultdict(list)
    for r in recs:
        g[(r["method"], r["d"], r["n"])].append(r)
    out = {}
    for k, v in g.items():
        dens = np.array([x["density"] for x in v])
        t = np.array([x["t_total_s"] for x in v])
        sz = np.array([x["size"] for x in v])
        out[k] = dict(
            trials=len(v), density=dens.mean(),
            density_sem=dens.std(ddof=1)/np.sqrt(len(v)) if len(v) > 1 else 0.0,
            size=sz.mean(), t=t.mean(),
            t_sem=t.std(ddof=1)/np.sqrt(len(v)) if len(v) > 1 else 0.0,
            t_min=t.min(), t_max=t.max(),
            ar=(dens.mean()/RHO_UB[k[1]]) if k[1] in RHO_UB else None,
            viol=sum(x.get("violations", 0) for x in v),
        )
    return out

def main(paths):
    recs = load(paths)
    if not recs:
        print("no results yet"); return
    a = agg(recs)
    ds = sorted({k[1] for k in a}); ns = sorted({k[2] for k in a})
    for d in ds:
        ub = RHO_UB.get(d)
        print(f"\n{'='*104}\nd = {d}" + (f"   rho_UB = {ub}   AR_1RSB = {AR_1RSB[d]}" if ub else "   (no published rho_UB adopted)"))
        print(f"{'='*104}")
        print(f"{'n':>8} {'method':<24} {'trials':>6} {'density':>18} {'AR':>8} "
              f"{'time (s)':>14} {'viol':>5}")
        print("-"*104)
        for n in ns:
            any_row = False
            for m in ORDER:
                k = (m, d, n)
                if k not in a: continue
                r = a[k]; any_row = True
                arS = f"{r['ar']:.4f}" if r["ar"] else "  --  "
                print(f"{n:>8} {LABEL.get(m,m):<24} {r['trials']:>6} "
                      f"{r['density']:.5f} +/- {r['density_sem']:.5f} {arS:>8} "
                      f"{r['t']:>10.4f}+/-{r['t_sem']:<7.4f} {r['viol']:>5}")
            if any_row: print()
        # head-to-head
        for n in ns:
            kg, kd = ("pignn_pub", d, n), ("dga", d, n)
            kn = ("null_ones", d, n)
            if kg in a and kd in a:
                g, dd = a[kg], a[kd]
                nl = a[kn]["density"] if kn in a else None
                gap = 100*(dd["density"] - g["density"])/g["density"]
                spd = g["t"]/dd["t"] if dd["t"] > 0 else float("inf")
                s = (f"  n={n:>7}: DGA beats PI-GNN by {gap:+.2f}% in size, "
                     f"and is {spd:,.0f}x faster")
                if nl is not None:
                    lift = 100*(g["density"] - nl)/nl
                    s += f";  PI-GNN vs null floor: {lift:+.2f}%"
                print(s)
    print()

if __name__ == "__main__":
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(
        os.path.dirname(__file__), "..", "results", "*.jsonl")))
    main(paths)
