"""review/DATA-APPENDIX.md — every measured configuration in results/, for citation.
Data only; no interpretation. Regenerate after any new run.
The header counts the records in every results/*.jsonl at build time and flags any file
that no section below tabulates, so the "every measured configuration" claim stays checkable."""
import glob, json, math, os
from collections import defaultdict
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
R = os.path.join(ROOT, "results")
RHO_UB = {3: 0.45537, 5: 0.38443}
def L(n):
    p = os.path.join(R, n)
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []

def wilson(k, n, z=1.96):
    """Wilson score interval (same formula as build_report.py) — 0/n needs an upper bound (D35)."""
    if n == 0: return (0.0, 1.0)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0.0, c-h), min(1.0, c+h))

# Files tabulated by a section below. Anything else in results/ is flagged in the header.
COVERED = {"phase1.jsonl", "phase2.jsonl", "tuning.jsonl", "posthoc.jsonl", "reply_defence.jsonl",
           "collapse.jsonl", "escape.jsonl", "exact.jsonl", "scale.jsonl", "pstats.jsonl",
           "early_trace.jsonl"}
_files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(R, "*.jsonl")))
_counts = {f: len(L(f)) for f in _files}
_uncovered = [f for f in _files if f not in COVERED]
_scope = ("Every measured configuration in `results/`" if not _uncovered else
          "Measured configurations from " + ", ".join(f"`{f}`" for f in _files if f in COVERED)
          + " (NOT tabulated here: " + ", ".join(f"`{f}`" for f in _uncovered) + ")")
out = ["# DATA APPENDIX", "",
       f"{_scope}: {sum(_counts.values())} measurement records across {len(_files)} files ("
       + " · ".join(f"{f.removesuffix('.jsonl')} {c}" for f, c in _counts.items()) + "). "
       "Some cells were executed by more than one driver (e.g. n=1000, d ∈ {3, 5, 20}, seeds 0–2), "
       "so records are not distinct runs; each table states its own de-duplication rule.",
       "Mean ± s.e.m. over seeds where more than one seed was run.",
       "Host: Apple M1 Pro, 10 core, 32 GB. Greedy baselines in C (`-O3 -march=native`); "
       "PI-GNN in PyTorch 2.13.0 on CPU.", "",
       "Approximation ratio AR = density / ρ_UB(d), with ρ_UB(3)=0.45537, ρ_UB(5)=0.38443 "
       "(McKay 1987). No ρ_UB is adopted for d=20.", "",
       "`pignn_pub` = PI-GNN at the released implementation's configuration: embedding size "
       "d0 = int(√n) at every n, as in the authors' released example notebook. The paper's text sets "
       "d0 = int(√n) only for n ≥ 10^5 and d0 = int(∛n) below that, so at n = 1000 and n = 10^4 "
       "these runs use d0 = 31 / 100 where the text gives 10 / 21; at n ≥ 10^5 the two agree (D38).", ""]

def block(title, recs, keyf, cols, note=None):
    g = defaultdict(list)
    for r in recs: g[keyf(r)].append(r)
    out.append(f"## {title}"); out.append("")
    if note: out.extend([note, ""])
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

# The Reply's published defence (D36-D37), with DGA on the SAME seeds (0-2) for comparison (D38).
_ARM_ORDER = {"gcn_P2": 0, "gcn_P10": 1, "sage_P2": 2, "sage_P10": 3}
_rd = L("reply_defence.jsonl")
_rd_d = sorted({r["d"] for r in _rd}); _rd_seeds = sorted({r["seed"] for r in _rd})
_dga_same = [r for r in L("phase1.jsonl") if r["method"] == "dga" and r["n"] == 1000
             and r["d"] in _rd_d and r["seed"] in _rd_seeds]
if _rd:
    block("The Reply's published defence — {GCN, GraphSAGE} × P ∈ {2, 10} (n=1000, seeds "
          f"{_rd_seeds[0]}–{_rd_seeds[-1]}; D37)",
          _rd + _dga_same,
          lambda r: (r["d"], _ARM_ORDER.get(r["method"], 9), r["method"]),
          [("d", lambda k, v: str(k[0])),
           ("arm", lambda k, v: k[2]),
           ("runs", lambda k, v: str(len(v))),
           ("non-empty", lambda k, v: f"{sum(1 for r in v if r['raw_size']>0)}/{len(v)}" if "raw_size" in v[0] else "—"),
           ("mean raw size", lambda k, v: f"{np.mean([r['raw_size'] for r in v]):,.1f}" if "raw_size" in v[0] else "—"),
           ("density", lambda k, v: ms(v, "density")),
           ("AR", lambda k, v: f"{np.mean([r['density'] for r in v])/RHO_UB[k[0]]:.4f}" if k[0] in RHO_UB else "—"),
           ("mean final loss", lambda k, v: f"{np.mean([r['final_loss'] for r in v]):,.3f}" if "final_loss" in v[0] else "—")],
          note="The GraphSAGE layer is this study's mean-aggregator implementation; the Reply gives no "
               "architecture detail and no reference code exists (D37). `dga` rows are phase-1 DGA on the "
               "same graphs and seeds, for a like-for-like comparison. non-empty = the network's raw "
               "output (raw_size > 0), before post-processing.")

# Escape table: collapse + escape de-duplicated by (d, seed) per D28; phase-1 pignn_pub n=1000
# runs are pooled only where (d, seed) is not already present (listed first, so collapse/escape
# records win on overlap) — D38. "non-empty" is the network's RAW output (raw_size > 0).
_p1pub = [r for r in L("phase1.jsonl") if r["method"] == "pignn_pub" and r["n"] == 1000]
_esc = list({(r["d"], r["seed"]): r for r in _p1pub + L("collapse.jsonl") + L("escape.jsonl")}.values())
_eg = defaultdict(list)
for r in _esc: _eg[r["d"]].append(r)
def _wfmt(k, n):
    lo, hi = wilson(k, n); return f"[{lo:.2f}, {hi:.2f}]"
out += ["## Escape probability vs degree (n=1000, released implementation's configuration, d0 = int(√n) = 31)", "",
        "Sources: collapse.jsonl + escape.jsonl, overlapping (d, seed) cells de-duplicated per D28; phase-1 "
        "`pignn_pub` n=1000 runs pooled only where (d, seed) is not already present (D38). "
        "non-empty = the network's raw output (raw_size > 0), before post-processing. "
        "The paper's text setting d0 = int(∛n) = 10 is tabulated separately below (D39).", "",
        "| d | runs | non-empty | p(non-empty) | Wilson 95% CI | mean epochs |",
        "|---|---|---|---|---|---|"]
for d in sorted(_eg):
    v = _eg[d]; k = sum(1 for r in v if r["raw_size"] > 0)
    ep = np.mean([r['epochs'] if 'epochs' in r else r.get('epochs_run', 0) for r in v])
    out.append(f"| {d} | {len(v)} | {k} | {k/len(v):.2f} | {_wfmt(k, len(v))} | {ep:,.0f} |")
_hi = [r for r in _esc if r["d"] >= 8]
if _hi:
    _k = sum(1 for r in _hi if r["raw_size"] > 0); _lo, _up = wilson(_k, len(_hi))
    out.append(f"| pooled d ≥ 8 | {len(_hi)} | {_k} | {_k/len(_hi):.2f} | [{_lo:.3f}, {_up:.3f}] | — |")
out += ["", f"Total: {len(_esc)} runs in this table.", ""]

# D39 (pre-registered): the same ceiling test at the paper's text embedding width
_cb = L("cbrt_d0.jsonl")
if _cb:
    _cg = defaultdict(list)
    for r in _cb: _cg[r["d"]].append(r)
    out += ["## Escape at the paper's embedding width (n=1000, d0 = int(∛n) = 10, hidden 5; pre-registered, D39)", "",
            "Same instances and seeds as the released-configuration runs; patience 100. Source: "
            "results/cbrt_d0.jsonl. non-empty = raw_size > 0.", "",
            "| d | runs | non-empty | Wilson 95% CI | mean AR / density | mean epochs |",
            "|---|---|---|---|---|---|"]
    for d in sorted(_cg):
        v = _cg[d]; k = sum(1 for r in v if r["raw_size"] > 0)
        dens = np.mean([r["density"] for r in v])
        val = f"AR {dens/RHO_UB[d]:.4f}" if d in RHO_UB else f"density {dens:.4f}"
        out.append(f"| {d} | {len(v)} | {k} | {_wfmt(k, len(v))} | {val} | {np.mean([r['epochs_run'] for r in v]):,.0f} |")
    _h = [r for r in _cb if r["d"] >= 8]; _hk = sum(1 for r in _h if r["raw_size"] > 0)
    _l, _u = wilson(_hk, len(_h))
    out += [f"| pooled d ≥ 8 | {len(_h)} | {_hk} | [{_l:.3f}, {_u:.3f}] | — | — |", ""]

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
et = L("early_trace.jsonl")
_bud = [r for r in L("scale.jsonl") if r["method"] == "pignn_pub_budgeted" and r.get("checkpoints")]
if et or _bud:
    out += ["## Early-epoch raw-output traces (D34) — instrumented diagnostic re-runs", "",
            "Instrumented re-runs (code/early_trace.py, code/find_escape_epoch.py) that record the raw "
            "thresholded set size during training; they are diagnostics, not additional sweep cells. "
            "The n = 10^6 row is the checkpoint trace of the budgeted scale run above (scale.jsonl), "
            "not a separate record.", "",
            "| source | n | d | seed | d0 | epochs traced | raw at first traced epoch | min raw in window (epoch) | "
            "raw at last traced epoch | loss at last traced epoch |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    rows = [(f"early_trace.jsonl ({r['method']})", r, r["points"]) for r in et]
    rows += [("scale.jsonl (budgeted checkpoints)", r, r["checkpoints"]) for r in _bud]
    for src, r, pts in rows:
        first, last = pts[0], pts[-1]
        mn = min(pts, key=lambda p: p["raw"])
        mn_note = " — end of window, still falling" if mn is last and len(pts) > 1 and pts[-2]["raw"] > last["raw"] else ""
        d0 = r.get("dim_embedding", int(math.sqrt(r["n"])))
        out.append(f"| {src} | {r['n']:,} | {r['d']} | {r['seed']} | {d0} | "
                   f"{last['epoch']} (max {r.get('max_epochs', r.get('epochs_run'))}) | "
                   f"{first['raw']:,} (epoch {first['epoch']}) | {mn['raw']:,} (epoch {mn['epoch']}{mn_note}) | "
                   f"{last['raw']:,} | {last['loss']:,.1f} |")
    out += ["", "Full traces (epoch: raw):", ""]
    for src, r, pts in rows:
        out.append(f"- n={r['n']:,}, {src}: " + ", ".join(f"{p['epoch']}: {p['raw']:,}" for p in pts))
    out.append("")
p = os.path.join(ROOT, "review", "DATA-APPENDIX.md")
open(p, "w").write("\n".join(out))
print(f"wrote {p} ({len('\n'.join(out)):,} bytes)")
