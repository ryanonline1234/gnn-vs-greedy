"""review/DATA-APPENDIX.md — every measured configuration, complete, for citation.
Data only; no interpretation. Regenerate after any new run."""
import json, os
from collections import defaultdict
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
R = os.path.join(ROOT, "results")
RHO_UB = {3: 0.45537, 5: 0.38443}
def L(n):
    p = os.path.join(R, n)
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []

out = ["# DATA APPENDIX", "",
       "Every measured configuration. Mean ± s.e.m. over seeds where more than one seed was run.",
       "Host: Apple M1 Pro, 10 core, 32 GB. Greedy baselines in C (`-O3 -march=native`); "
       "PI-GNN in PyTorch 2.13.0 on CPU.", "",
       "Approximation ratio AR = density / ρ_UB(d), with ρ_UB(3)=0.45537, ρ_UB(5)=0.38443 "
       "(McKay 1987). No ρ_UB is adopted for d=20.", ""]

def block(title, recs, keyf, cols):
    g = defaultdict(list)
    for r in recs: g[keyf(r)].append(r)
    out.append(f"## {title}"); out.append("")
    out.append("| " + " | ".join(c[0] for c in cols) + " |")
    out.append("|" + "|".join("---" for _ in cols) + "|")
    for k in sorted(g):
        v = g[k]
        out.append("| " + " | ".join(c[1](k, v) for c in cols) + " |")
    out.append("")

def ms(v, f):
    x = np.array([r[f] for r in v])
    if len(x) == 1: return f"{x[0]:.5f}"
    return f"{x.mean():.5f} ± {x.std(ddof=1)/np.sqrt(len(x)):.5f}"

main = L("phase1.jsonl") + L("phase2.jsonl")
block("Main sweep — quality and time", main,
      lambda r: (r["d"], r["n"], r["method"]),
      [("d", lambda k, v: str(k[0])),
       ("n", lambda k, v: f"{k[1]:,}"),
       ("method", lambda k, v: k[2]),
       ("trials", lambda k, v: str(len(v))),
       ("density", lambda k, v: ms(v, "density")),
       ("AR", lambda k, v: f"{np.mean([r['density'] for r in v])/RHO_UB[k[0]]:.4f}" if k[0] in RHO_UB else "—"),
       ("wall clock (s)", lambda k, v: f"{np.mean([r['t_total_s'] for r in v]):,.6g}"),
       ("violations", lambda k, v: str(sum(r.get("violations", 0) for r in v)))])

block("Pre-registered tuning grid (Stage A, n=1000, seeds 0–2)", L("tuning.jsonl"),
      lambda r: (r["d"], r["lr"], r["penalty"]),
      [("d", lambda k, v: str(k[0])),
       ("lr", lambda k, v: f"{k[1]:g}"),
       ("penalty", lambda k, v: f"{k[2]:g}"),
       ("non-empty", lambda k, v: f"{sum(1 for r in v if r['raw_size']>0)}/{len(v)}"),
       ("density", lambda k, v: ms(v, "density")),
       ("AR", lambda k, v: f"{np.mean([r['density'] for r in v])/RHO_UB[k[0]]:.4f}" if k[0] in RHO_UB else "—")])

block("Post-hoc escape arms (n=1000, seeds 0–2)", L("posthoc.jsonl"),
      lambda r: (r["d"], r["method"]),
      [("d", lambda k, v: str(k[0])),
       ("arm", lambda k, v: k[1]),
       ("non-empty", lambda k, v: f"{sum(1 for r in v if r['raw_size']>0)}/{len(v)}"),
       ("mean raw size", lambda k, v: f"{np.mean([r['raw_size'] for r in v]):,.1f}"),
       ("density", lambda k, v: ms(v, "density")),
       ("mean final loss", lambda k, v: f"{np.mean([r['final_loss'] for r in v]):,.3f}")])

block("Escape probability vs degree (n=1000, published config; overlapping (d, seed) cells de-duplicated per D28)",
      list({(r["d"], r["seed"]): r for r in L("collapse.jsonl") + L("escape.jsonl")}.values()),
      lambda r: r["d"],
      [("d", lambda k, v: str(k)),
       ("runs", lambda k, v: str(len(v))),
       ("non-empty", lambda k, v: str(sum(1 for r in v if r["raw_size"] > 0))),
       ("p(non-empty)", lambda k, v: f"{sum(1 for r in v if r['raw_size']>0)/len(v):.2f}"),
       ("mean epochs", lambda k, v: f"{np.mean([r['epochs'] if 'epochs' in r else r.get('epochs_run',0) for r in v]):,.0f}")])

block("Exact MIS via HiGHS ILP", L("exact.jsonl"), lambda r: (r["d"], r["n"]),
      [("d", lambda k, v: str(k[0])),
       ("n", lambda k, v: str(k[1])),
       ("instances", lambda k, v: str(len(v))),
       ("density", lambda k, v: ms(v, "density")),
       ("/ρ_UB", lambda k, v: f"{np.mean([r['density'] for r in v])/RHO_UB[k[0]]:.4f}" if k[0] in RHO_UB else "—"),
       ("all proven optimal", lambda k, v: "yes" if all(r["status"] == "Optimal" for r in v) else "**no — time limit**")])

sc = L("scale.jsonl")
out += ["## n = 10^6 (scale test)", "", "All n=10^6 rows are a single instance/seed (seed 0).", "", "| d | method | size | density | AR | wall clock | note |",
        "|---|---|---|---|---|---|---|"]
for r in sc:
    if r["method"] == "reference_dense_Q_feasibility":
        out.append(f"| {r['d']} | reference dense Q | — | — | — | — | "
                   f"{r['dense_entries']:.2e} entries = {r['dense_bytes']/2**40:,.1f} TiB — **not representable** |")
    elif r["method"] == "pignn_pub_probe":
        out.append(f"| {r['d']} | PI-GNN probe | — | — | — | {r['ms_per_epoch']:,.1f} ms/epoch | "
                   f"{r['probe_epochs']} epochs; {r['embed_params']:.1e} embedding params, {r['mem_est_gib']:,.1f} GiB |")
    elif r["method"] == "pignn_pub_budgeted":
        out.append(f"| {r['d']} | PI-GNN budgeted | {r['size']:,} | {r['density']:.5f} | "
                   f"{r['ar']:.4f} | {r['t_train_s']:,.0f} s | {r['epochs_run']} epochs; "
                   f"stop={r['stop_reason']}; truncated={r['truncated']} |")
    else:
        ar = f"{r['ar']:.4f}" if r.get("ar") else "—"
        out.append(f"| {r['d']} | {r['method']} | {r['size']:,} | {r['density']:.5f} | {ar} | "
                   f"{r['t_total_s']:.4f} s | |")
out.append("")
ps = L("pstats.jsonl")
if ps:
    out += ["## Final p-vector statistics, modified linear diagonal, d=20 (D27)", "",
            "| seed | final loss | L* | mean p | p* | std(p)/p* | max p | frac >= 0.5 |",
            "|---|---|---|---|---|---|---|---|"]
    for r in ps:
        out.append(f"| {r['seed']} | {r['final_loss']:.4f} | {r['Lstar']} | {r['p_mean']:.5f} | "
                   f"{r['pstar']} | {r['std_over_pstar']:.3f} | {r['p_max']:.4f} | {r['frac_above_threshold']:.0f} |")
    out.append("")
p = os.path.join(ROOT, "review", "DATA-APPENDIX.md")
open(p, "w").write("\n".join(out))
print(f"wrote {p} ({len('\n'.join(out)):,} bytes)")
