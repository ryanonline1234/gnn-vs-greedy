import json, os, glob, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RHO_UB = {3: 0.45537, 5: 0.38443}
AR_1RSB = {3: 0.990, 5: 0.987}
STYLE = {
    "dga":       dict(c="#1b7837", m="o", l="DGA (degree-based greedy)"),
    "pignn_pub": dict(c="#c51b7d", m="s", l="PI-GNN (as published)"),
    "pignn_tuned": dict(c="#7b3294", m="D", l="PI-GNN (tuned)"),
    "null_ones": dict(c="#888888", m="^", l="null control: all-ones + same post-processing"),
    "ga":        dict(c="#2166ac", m="v", l="GA (random greedy)"),
}

def load():
    recs = []
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "*.jsonl"))):
        for line in open(p):
            line = line.strip()
            if line:
                try: recs.append(json.loads(line))
                except Exception: pass
    return recs

def agg(recs):
    g = defaultdict(list)
    for r in recs: g[(r["method"], r["d"], r["n"])].append(r)
    return {k: dict(
        dens=np.mean([x["density"] for x in v]),
        sem=(np.std([x["density"] for x in v], ddof=1)/np.sqrt(len(v))) if len(v)>1 else 0.0,
        t=np.mean([x["t_total_s"] for x in v]), k=len(v)) for k, v in g.items()}

def main():
    a = agg(load())
    ds = sorted({k[1] for k in a})
    fig, axes = plt.subplots(2, len(ds), figsize=(5.2*len(ds), 8.6))
    if len(ds) == 1: axes = axes.reshape(2, 1)
    for j, d in enumerate(ds):
        ns = sorted({k[2] for k in a if k[1] == d})
        ax = axes[0][j]
        for mth, st in STYLE.items():
            xs = [n for n in ns if (mth, d, n) in a]
            if not xs: continue
            ys = [a[(mth, d, n)]["dens"]/RHO_UB[d] if d in RHO_UB else a[(mth,d,n)]["dens"] for n in xs]
            es = [a[(mth, d, n)]["sem"]/RHO_UB[d] if d in RHO_UB else a[(mth,d,n)]["sem"] for n in xs]
            ax.errorbar(xs, ys, yerr=es, color=st["c"], marker=st["m"], ms=6,
                        lw=1.8, capsize=3, label=st["l"])
        if d in AR_1RSB:
            ax.axhline(AR_1RSB[d], ls="--", c="k", lw=1.2)
            ax.text(ns[0], AR_1RSB[d]+0.003, "1RSB optimum", fontsize=8)
        ax.set_xscale("log"); ax.set_xlabel("n"); ax.grid(alpha=.25)
        ax.set_ylabel("approximation ratio  $\\rho/\\rho_{UB}$" if d in RHO_UB else "IS density $\\rho$")
        ax.set_title(f"d = {d}   " + ("quality (higher is better)" if j==0 else "quality"))
        if j == 0: ax.legend(fontsize=7.5, loc="lower left")

        ax = axes[1][j]
        for mth, st in STYLE.items():
            if mth in ("null_ones", "ga"): continue
            xs = [n for n in ns if (mth, d, n) in a]
            if not xs: continue
            ys = [a[(mth, d, n)]["t"] for n in xs]
            ax.plot(xs, ys, color=st["c"], marker=st["m"], ms=6, lw=1.8, label=st["l"])
        ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=.25)
        ax.set_xlabel("n"); ax.set_ylabel("wall clock (s), this host")
        ax.set_title(f"d = {d}   time (lower is better)")
        if j == 0: ax.legend(fontsize=7.5, loc="upper left")
    fig.suptitle("PI-GNN vs degree-based greedy for Maximum Independent Set on random d-regular graphs\n"
                 "same host (Apple M1 Pro), same instances, same seeds", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(ROOT, "results", "figure1.png")
    fig.savefig(out, dpi=160)
    print("wrote", out)

if __name__ == "__main__":
    main()
