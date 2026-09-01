import json, os, glob
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from collections import defaultdict
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

rs = []
for f in ("collapse.jsonl", "escape.jsonl"):
    p = os.path.join(ROOT, "results", f)
    if os.path.exists(p):
        rs += [json.loads(l) for l in open(p) if l.strip()]
g = defaultdict(list)
for r in rs: g[r["d"]].append(r)
ds = sorted(g)
esc = [sum(1 for x in g[d] if x["raw_size"] > 0)/len(g[d]) for d in ds]
ntr = [len(g[d]) for d in ds]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))
ax1.plot(ds, esc, "o-", color="#B02A3A", lw=2, ms=7)
for d, e, k in zip(ds, esc, ntr):
    ax1.annotate(f"{int(e*k)}/{k}", (d, e), textcoords="offset points",
                 xytext=(0, 9), ha="center", fontsize=7.5)
ax1.axvspan(16, max(ds)+1, color="#888", alpha=.10)
ax1.text(16.4, .55, "clustering transition\n(d > 16), where the critics\nsaid the problem gets hard",
         fontsize=7.5, va="center")
ax1.set_xlabel("degree d of the random regular graph")
ax1.set_ylabel("fraction of runs producing a non-empty set")
ax1.set_title("PI-GNN, as published, stops producing any output at d $\\approx$ 7")
ax1.set_ylim(-.05, 1.12); ax1.grid(alpha=.25); ax1.set_xlim(2, max(ds)+1)

lo = [x for x in rs if x["raw_size"] == 0]; hi = [x for x in rs if x["raw_size"] > 0]
ax2.scatter([x["d"] for x in hi], [x["final_loss"] for x in hi], c="#1b7837",
            s=34, label="escaped the origin", zorder=3)
ax2.scatter([x["d"] for x in lo], [x["final_loss"] for x in lo], c="#B02A3A",
            marker="x", s=42, label="collapsed to p = 0", zorder=3)
ax2.axhline(0, color="k", lw=1.1, ls="--")
ax2.text(max(ds)-1, 6, "L = 0: the all-zeros critical point", fontsize=7.5, ha="right")
ax2.set_yscale("symlog", linthresh=1)
ax2.set_xlabel("degree d"); ax2.set_ylabel("final training loss")
ax2.set_title("The two outcomes are trivially separable")
ax2.legend(fontsize=8); ax2.grid(alpha=.25)
fig.suptitle("The published objective $L(p)=p^{\\mathsf{T}}Qp$ is a pure quadratic form, so $\\nabla L(0)=0$:\n"
             "the empty set is a critical point at every degree, and its basin grows with $d$", fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.90])
out = os.path.join(ROOT, "results", "figure2_collapse.png")
fig.savefig(out, dpi=160); print("wrote", out)
