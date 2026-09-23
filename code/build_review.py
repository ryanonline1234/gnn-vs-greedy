"""Generate review/REVIEW.md — an orientation + verification package for an independent
reviewing agent. Every headline number is pulled from results/, never transcribed."""
import json, os, glob, math
from collections import defaultdict
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
R = os.path.join(ROOT, "results")
RHO_UB = {3: 0.45537, 5: 0.38443}

def loadf(n):
    p = os.path.join(R, n)
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []

main = loadf("phase1.jsonl") + loadf("phase2.jsonl")
tune, post = loadf("tuning.jsonl"), loadf("posthoc.jsonl")
_coll_raw = loadf("collapse.jsonl") + loadf("escape.jsonl")
_coll_by = {(r["d"], r["seed"]): r for r in _coll_raw}  # d=7 overlap de-dup (D28)
# Pool phase-1 released-configuration n=1000 runs ONLY where (d, seed) is not already present
# (D38: phase-1 seeds 0-2 at d=3/5/20 are the same cells as collapse.jsonl seeds 0-2).
for _r in loadf("phase1.jsonl"):
    if _r["method"] == "pignn_pub" and _r["n"] == 1000 and (_r["d"], _r["seed"]) not in _coll_by:
        _coll_by[(_r["d"], _r["seed"])] = _r
coll = list(_coll_by.values())
pstats = loadf("pstats.jsonl")
exact, scale = loadf("exact.jsonl"), loadf("scale.jsonl")
rdef = loadf("reply_defence.jsonl")
# F1/D38: the public count is raw measurement records, computed here, never transcribed.
_files = sorted(glob.glob(os.path.join(R, "*.jsonl")))
N_RECORDS = sum(len(loadf(os.path.basename(f))) for f in _files)
N_FILES = len(_files)

def wilson(k, n, z=1.96):
    if n == 0: return 0.0, 1.0
    p = k/n; den = 1 + z*z/n
    c = (p + z*z/(2*n))/den; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return max(0.0, c-h), min(1.0, c+h)

A = defaultdict(list)
for r in main: A[(r["method"], r["d"], r["n"])].append(r)
def st(m, d, n, f="density"):
    v = A.get((m, d, n))
    if not v: return None
    x = np.array([r[f] for r in v]); return x.mean()
def AR(m, d, n):
    s = st(m, d, n); return None if s is None or d not in RHO_UB else s/RHO_UB[d]
def st_seeds(m, d, n, seeds, f="density"):
    v = [r[f] for r in A.get((m, d, n), []) if r["seed"] in seeds]
    return float(np.mean(v)) if v else None

def tbl():
    L = ["| d | n | method | density | AR | wall clock (s) | trials |",
         "|---|---|---|---|---|---|---|"]
    for d in sorted({k[1] for k in A}):
        for n in sorted({k[2] for k in A if k[1] == d}):
            for m, lab in [("dga","DGA"),("pignn_pub","PI-GNN"),("null_ones","null all-ones"),("ga","GA")]:
                s = st(m,d,n)
                if s is None: continue
                a = AR(m,d,n); t = st(m,d,n,"t_total_s"); k = len(A[(m,d,n)])
                L.append(f"| {d} | {n:,} | {lab} | {s:.5f} | {a:.4f} | {t:,.4g} | {k} |"
                         if a else
                         f"| {d} | {n:,} | {lab} | {s:.5f} | — | {t:,.4g} | {k} |")
    return "\n".join(L)

# escape probability
esc = defaultdict(list)
for r in coll: esc[r["d"]].append(r)
ESC = {d:(sum(1 for x in v if x.get("raw_size", 0)>0), len(v)) for d,v in esc.items()}
def _ci(e, k):
    lo, hi = wilson(e, k); return f"[{lo:.2f}, {hi:.2f}]"
esc_tbl = "\n".join(["| d | raw non-empty / runs | p | Wilson 95% CI |","|---|---|---|---|"] +
    [f"| {d} | {e}/{k} | {e/k:.2f} | {_ci(e, k)} |" for d,(e,k) in sorted(ESC.items())])
ESC_RUNS = sum(k for _, k in ESC.values())
HI_E = sum(e for d, (e, k) in ESC.items() if d >= 8); HI_K = sum(k for d, (e, k) in ESC.items() if d >= 8)
HI_LO, HI_HI = wilson(HI_E, HI_K)

# tuning
tn20 = sum(1 for r in tune if r["d"]==20); tz20 = sum(1 for r in tune if r["d"]==20 and r["raw_size"]>0)
pn20 = sum(1 for r in post if r["d"]==20); pz20 = sum(1 for r in post if r["d"]==20 and r["raw_size"]>0)
def bestof(recs, d):
    g = defaultdict(list)
    for r in recs:
        if r["d"]==d: g[r["method"]].append(r["density"])
    if not g: return None, None
    m = max(g, key=lambda k: np.mean(g[k])); return m, np.mean(g[m])

# mean field — both linear-diagonal arms (D27: the sister arm is shown, not omitted)
mf = []
for arm in ("modified_linear", "modified_linear_lr1e-3"):
    for d in (3,5,20):
        v=[r for r in post if r["d"]==d and r["method"]==arm]
        if v:
            n,P=1000,2.0; Ls=-n/(2.0*P*d); ob=np.mean([x["final_loss"] for x in v])
            mf.append(f"| {arm} | {d} | {1/(P*d):.4f} | {Ls:.3f} | {ob:.3f} | "
                      f"{'symmetry-broken' if ob < Ls-0.02*abs(Ls) else '**stuck at mean field**'} |")
mf_tbl = "\n".join(["| arm | d | uniform p* | closed-form L* | observed L | outcome |","|---|---|---|---|---|---|"]+mf)
_ml20 = [r["final_loss"] for r in post if r["d"]==20 and r["method"]=="modified_linear"]
ML20 = float(np.mean(_ml20)); LS20 = -1000/(2.0*2.0*20)
ML20_BELOW = sum(1 for x in _ml20 if x < LS20)
_ps = [r for r in pstats if r["method"]=="pstats_modified_linear"]
PS_MEAN = (min(r["p_mean"] for r in _ps), max(r["p_mean"] for r in _ps))
PS_SD = (min(r["std_over_pstar"] for r in _ps), max(r["std_over_pstar"] for r in _ps))
PS_MAX = max(r["p_max"] for r in _ps); PSTAR = _ps[0]["pstar"]

# F6: raw vs post-processed decomposition of PI-GNN's output (published objective)
def _decomp(d, n):
    v = A.get(("pignn_pub", d, n), [])
    if not v: return None
    raw = float(np.mean([r["raw_size"] for r in v])); fin = float(np.mean([r["size"] for r in v]))
    pct = 100*(fin-raw)/fin
    return d, pct, (f"d={d}, n={n:,}: raw {raw:,.0f} → final {fin:,.0f} "
                    f"({'100' if pct > 99.95 else f'{pct:.1f}'}%)")
_dc = [x for x in (_decomp(3,1000), _decomp(3,100000), _decomp(5,1000), _decomp(20,1000)) if x]
DECOMP = "; ".join(x[2] for x in _dc)
_works = [x[1] for x in _dc if x[0] in (3, 5)]
DECOMP_LO, DECOMP_HI = min(_works), max(_works)

# F4 / D37: the Reply's published defence (GraphSAGE + P=10), as the Reply describes it
RD_ARMS = [("gcn_P2", "GCN, P=2"), ("gcn_P10", "GCN, P=10"), ("sage_P2", "GraphSAGE, P=2"),
           ("sage_P10", "GraphSAGE, P=10 (the Reply's configuration)")]
_rg = defaultdict(list)
for r in rdef: _rg[(r["method"], r["d"])].append(r)
RD_SEEDS = sorted({r["seed"] for r in rdef})
def rd_ar(arm, d):
    v = _rg.get((arm, d)); return float(np.mean([r["density"] for r in v]))/RHO_UB[d] if v else None
def rd_ne(arm, d):
    v = _rg.get((arm, d), []); return sum(1 for r in v if r["raw_size"] > 0), len(v)
DGA_S = {d: st_seeds("dga", d, 1000, RD_SEEDS) for d in (3, 5, 20)}
rd_rows = []
for arm, lab in RD_ARMS:
    c5 = f"{rd_ar(arm,5):.4f}"
    e5, k5 = rd_ne(arm, 5)
    if e5 < k5: c5 += f" (raw non-empty only {e5}/{k5})"
    e20, k20 = rd_ne(arm, 20)
    rd_rows.append(f"| {lab} | {rd_ar(arm,3):.4f} | {c5} | {e20}/{k20} |")
rd_rows.append(f"| **DGA, same seeds {RD_SEEDS[0]}–{RD_SEEDS[-1]}** | **{DGA_S[3]/RHO_UB[3]:.4f}** | "
               f"**{DGA_S[5]/RHO_UB[5]:.4f}** | density {DGA_S[20]:.4f} |")
rd_tbl = "\n".join(["| arm | d=3 AR | d=5 AR | d=20 raw non-empty |", "|---|---|---|---|"] + rd_rows)
RD20_E = sum(rd_ne(a, 20)[0] for a, _ in RD_ARMS); RD20_K = sum(rd_ne(a, 20)[1] for a, _ in RD_ARMS)
RD20_POST = sorted({round(float(np.mean([r["density"] for r in _rg[(a, 20)]])), 4) for a, _ in RD_ARMS})
SAGE10_3 = rd_ar("sage_P10", 3); REPLY_AR = 0.947   # the Reply's stated figure (arXiv:2302.03602)
# The prose below states these as facts; refuse to write it if the data stop supporting them.
assert RD20_E == 0, "defence prose says d=20 raw output is empty in every run"
assert all(rd_ar(a, d) < DGA_S[d]/RHO_UB[d] for a, _ in RD_ARMS for d in (3, 5)), "prose says no arm reaches DGA"
assert SAGE10_3 < REPLY_AR - 0.01, "prose says the Reply's GraphSAGE AR is not reproduced"
assert HI_E == 0, "ceiling prose says no non-empty raw output at d >= 8"

# scaling
pts={}
for r in main:
    if r["method"]=="pignn_pub" and r["d"]==3: pts.setdefault(r["n"],[]).append(r["t_train_s"]/r["epochs_run"]*1000)
for r in scale:
    if r["method"]=="pignn_pub_probe" and r["d"]==3: pts[r["n"]]=r["ms_per_epoch"]
pts={n:(float(np.mean(v)) if isinstance(v,list) else v) for n,v in pts.items()}
sc_rows=[]; prev=None
for n in sorted(pts):
    v=pts[n]
    if prev:
        ex=math.log10(v/prev[1])/math.log10(n/prev[0])
        sc_rows.append(f"| {n:,} | {v:,.2f} | {v/prev[1]:.1f}× | {ex:.2f} |")
    else: sc_rows.append(f"| {n:,} | {v:,.2f} | — | — |")
    prev=(n,v)
sc_tbl="\n".join(["| n | ms/epoch | ×/decade | local exponent |","|---|---|---|---|"]+sc_rows)

bud = [r for r in scale if r["method"]=="pignn_pub_budgeted"]
dga1e6 = next((r for r in scale if r["method"]=="dga" and r["d"]==3), None)
nul1e6 = next((r for r in scale if r["method"]=="null_ones" and r["d"]==3), None)
probe = next((r for r in scale if r["method"]=="pignn_pub_probe" and r["d"]==3), None)

# epochs-to-convergence per n (PI-GNN, d=3); projection uses the two largest measured sizes (D33)
EP = {n: float(np.mean([r["epochs_run"] for r in A[("pignn_pub", 3, n)]]))
      for n in sorted({k[2] for k in A if k[0]=="pignn_pub" and k[1]==3})}
EP_BIG = [EP[n] for n in sorted(EP)[-2:]]
_costs = [probe["ms_per_epoch"]/1000] + ([bud[0]["t_train_s"]/bud[0]["epochs_run"]] if bud else [])
C_LO, C_HI = min(_costs), max(_costs)
T_LO, T_HI = min(EP_BIG)*C_LO, max(EP_BIG)*C_HI
X_LO, X_HI = T_LO/dga1e6["t_total_s"], T_HI/dga1e6["t_total_s"]
def _ep(n): return f"{EP[n]:,.0f}"
EP_TXT = " / ".join(_ep(n) for n in sorted(EP)) + " at n=" + "/".join(f"1e{int(round(math.log10(n)))}" for n in sorted(EP))

bud_txt = "_The budgeted n=1e6 run had not finished when this was generated._"
if bud:
    b = bud[0]
    bud_txt = (f"Measured, not projected: after **{b['t_train_s']:,.0f} s** of training "
        f"({b['epochs_run']} epochs of the ~{EP[max(EP)]:,.0f} needed, i.e. "
        f"{100*b['epochs_run']/EP[max(EP)]:.2f}% of the schedule), PI-GNN's output repairs to "
        f"**{b['density']:.5f}** (AR {b['ar']:.4f}). On the same instance DGA reached "
        f"**{dga1e6['density']:.5f}** (AR {dga1e6['ar']:.4f}) in **{dga1e6['t_total_s']:.4f} s**, "
        f"and the all-ones null control reached **{nul1e6['density']:.5f}**. "
        f"Stop reason: `{b['stop_reason']}`. Truncated: `{b['truncated']}`. It stopped on its "
        f"wall-clock budget, so it is a lower bound, not evidence of non-convergence; its "
        f"raw-output decay over those epochs is consistent with the normal early transient (D34).")

_ks = {len(A[(m, 3, 10000)]) for m in ("dga", "pignn_pub")}
REP_SEEDS = f"{_ks.pop()} seeds each" if len(_ks) == 1 else "DGA {0} / PI-GNN {1} seeds".format(
    len(A[("dga", 3, 10000)]), len(A[("pignn_pub", 3, 10000)]))
TUNE_SEEDS = sorted({r["seed"] for r in tune})
DGA_T3 = st_seeds("dga", 3, 1000, TUNE_SEEDS)/RHO_UB[3]
_pb_name = bestof(post, 3)[0]
POST_BEST_LAB = (f"(`{_pb_name}`, a modified objective — not a tuned arm)" if "modified" in _pb_name
                 else f"(`{_pb_name}`)")
RD20_POST_TXT = ", ".join(f"{x:.4f}" for x in RD20_POST)
# Mirrors README.md's re-measure lines: these drivers take no --out (they append to a fixed file
# in results/ and skip cells already there), so the committed file is moved aside first.
REMEASURE = """# collapse_sweep.py, escape_prob.py, escape_d10plus.py and reply_defence.py take no --out: each
# appends to a fixed file in results/ and skips cells already there, so move that file aside first.
# mv -n never overwrites an earlier move-aside; the published files are tracked by git, so
# `git checkout -- results/` restores them.
mkdir -p results.published
mv -n results/collapse.jsonl results/escape.jsonl results.published/
python code/collapse_sweep.py && python code/escape_prob.py && python code/escape_d10plus.py   # escape table: 121 of its 127 runs (the other 6 are phase-1 runs, re-measured by runner.py above)
mv -n results/reply_defence.jsonl results.published/ && python code/reply_defence.py             # the Reply's defence, GraphSAGE + P=10 (D37)
# Open (D38), not yet run: the ceiling under the paper's d0 = int(∛n) at n = 1000
python code/runner.py --out results/cbrt_d0.jsonl --methods pignn_pub --tag pignn_cbrt --dim-embedding 10 --n 1000 --d 3 5 8 12 20 --seeds 0 1 2 3 4"""

DOC = f"""# REVIEW PACKAGE — adjudicating PI-GNN vs greedy for Maximum Independent Set

**For an independent reviewing agent.** Everything below is measured on one machine
(Apple M1 Pro, 10 core, 32 GB). {N_RECORDS:,} measurement records in `results/*.jsonl`
({N_FILES} files). The measurements are the author's, produced by running the code on his
machine; the code and this file's prose are AI-written (generated by `code/build_review.py`).

Your job is not to agree. It is to find the load-bearing claim that is wrong. §6 lists where
I think the weight sits.

---

## 1. What is on trial

Schuetz, Brubaker & Katzgraber, *Nat Mach Intell* **4**, 367 (2022), abstract, verbatim:

> "the graph neural network optimizer performs on par or outperforms existing solvers, with the
> ability to scale beyond the state of the art to problems with millions of variables."

Angelini & Ricci-Tersenghi, arXiv:2206.13211 **listing abstract**, verbatim (published as
*Nat Mach Intell* **5**, 29 (2023); note the paper's own abstract words this differently —
"much better quality than the GNN in a much shorter time", with the body claiming a speedup
"larger than a factor 10⁴"):

> "…a simple greedy algorithm, running in almost linear time, can find solutions for the MIS
> problem of much better quality than the GNN. The greedy algorithm is faster by a factor of
> $10^4$ with respect to the GNN for problems with a million variables."

The measured object is **the validity of the published sentence against an exact objective**,
not model performance. The sentence decomposes into two claims scored separately:
- **A** — parity/superiority in solution quality.
- **B** — scaling to ~10^6 variables.

**Prior art (D32) — read this study against it.** Böther et al., ICLR 2022 (arXiv:2201.10494)
ran the null-control idea (the GNN replaceable by random values) for a different MIS method.
Ichikawa, NeurIPS 2024 (arXiv:2309.16965) names both failure modes seen here (trapping in the
relaxed optimisation; reliance on post-learning rounding). Krutsky et al., ECAI 2025
(arXiv:2507.13703) report the density-driven transition in PI-GNN training. Their MIS tables
(Tables 2, 3 and 11) put the baseline PI-GNN at 0.00 at d = 10, consistent with the
{ESC[10][0]}/{ESC[10][1]} raw non-empty measured here; their non-collapsed d = 10 result is for MaxCut.
D35's claimed disagreement compared across problems and is **withdrawn (D38)**. What this study
adds on that axis is resolution: its degree grid places the MIS escape transition at d ≈ 6–7,
between their grid points d = 5 and d = 10. What is
left as new: the closed-form mean-field optimum checked against measurement, the null control
applied to PI-GNN with a measured floor, and the location of the transition.

## 2. Verdicts and where the evidence is

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| A | on par or outperforms existing solvers | **Refuted** | `results/phase1.jsonl`, `phase2.jsonl` — §3 |
| B | scales to millions of variables | **Upheld only after a reformulation the released implementation does not contain** | `results/scale.jsonl` — §5 |
| — | undisclosed operating ceiling in d | **Fails for d ≥ 8** (raw output non-empty in {HI_E}/{HI_K} runs pooled over d ≥ 8, n=1000, in the released implementation's configuration d0 = int(√n); untested under the paper's d0 = int(∛n), D38) | `results/collapse.jsonl`, `escape.jsonl`, `phase1.jsonl` — §4 |
| — | the Reply's published defence (GraphSAGE + P=10, arXiv:2302.03602); our earlier "Answered" verdict is **retracted** (D36) | **Tested as described (D37): raw output empty at d=20 in {RD20_K-RD20_E}/{RD20_K} runs across all four arms; no arm reaches DGA at d=3 or d=5; the Reply's GraphSAGE AR ≈{REPLY_AR} at d=3 not reproduced (ours {SAGE10_3:.4f}) — a failure to reproduce, not a refutation** | `results/reply_defence.jsonl`; pre-registered grid `tuning.jsonl`, `posthoc.jsonl` — §4 |

## 3. Claim A — the head-to-head

{tbl()}

"PI-GNN" is our port run in the released implementation's configuration: embedding width
d0 = int(√n) at every n. The paper's text uses int(√n) only for n ≥ 10⁵ and int(∛n) below it,
so at n = 1,000 and 10,000 this study ran d0 = 31 / 100 where the paper's text gives 10 / 21;
at n ≥ 10⁵ the two agree. Early-stopping patience is also the implementation's 100, where the
paper's text gives 10³ (D38).

**Replication check.** At d=3, n=10,000 ({REP_SEEDS}), DGA lands at AR {AR('dga',3,10000):.4f} against the critique's
published ≈0.95, and PI-GNN at {AR('pignn_pub',3,10000):.4f} against their ≈0.92. Both sides of
their Figure 1 reproduce on independently written code. This is the single strongest reason to
believe the harness is sound.

**The null control (absent from all five documents in the exchange; the idea itself is prior
art for a different MIS method — Böther et al., ICLR 2022, D32).** An all-ones bitstring,
pushed through the *identical* repair-and-maximalize pipeline the GNN's output goes through,
reaches AR {AR('null_ones',3,10000):.4f} at d=3, n=10,000, against the trained network's
{AR('pignn_pub',3,10000):.4f}. At d=20 the control **beats** the network
({st('null_ones',20,10000):.5f} vs {st('pignn_pub',20,10000):.5f}). The bitstring carries no information about
the instance, but the pipeline applied to it does: on the all-ones input `repair.c` reduces to
**iterated maximum-degree vertex deletion**, a classical MIS heuristic (D33). So the primary
evidence is the raw-vs-post-processed decomposition of PI-GNN's own output (share of the final
set added by post-processing in brackets): {DECOMP}. Where the network works (d = 3, 5)
post-processing adds {DECOMP_LO:.1f}–{DECOMP_HI:.1f}%; at d = 20 the entire score is post-processing.

## 4. Mechanism, ceiling, and the Reply's defence

### Escape probability vs degree (n=1000, released implementation's configuration)

{esc_tbl}

**Pooled d ≥ 8: {HI_E}/{HI_K}, Wilson 95% CI [{HI_LO:.3f}, {HI_HI:.3f}]** — no single degree row can carry the
plateau claim; the pooled one can. {ESC_RUNS} runs in this table: `collapse.jsonl` + `escape.jsonl`
de-duplicated by (d, seed) (D28), plus phase-1 released-configuration n=1000 runs only where
(d, seed) is not already present (D38). "Non-empty" means the network's **raw** output, before
post-processing. Open (D38): whether this ceiling holds under the paper's text rule
d0 = int(∛n) at n = 1000 is untested.

### Why it fails — failure to symmetry-break

For a uniform assignment on a d-regular graph, the relaxed objective has a closed-form optimum.
Under a **modified, post-hoc objective** — the diagonal changed to linear, so the origin is *not*
a critical point — it is p* = 1/(Pd), L* = −n/(2Pd). The table checks that closed form against
measurement (D17 derived it after the arm had run):

{mf_tbl}

At d=20 the measured loss is {ML20:.3f} (3-seed mean of `modified_linear`) vs L* = {LS20:.3f} — but the
loss alone cannot certify uniformity (it is second-order blind to zero-mean scatter, and {ML20_BELOW} of
{len(_ml20)} seeds land *below* L*, which no uniform configuration can do). The direct measurement
(`results/pstats.jsonl`, `code/capture_pstats.py`): mean p = {PS_MEAN[0]:.4f}–{PS_MEAN[1]:.4f} vs p* = {PSTAR},
std(p)/p* = {100*PS_SD[0]:.0f}–{100*PS_SD[1]:.0f}%, max p ≤ {PS_MAX:.3f} (≈{PS_MAX/PSTAR:.0f}p*) — every output a factor
~{0.5/PS_MAX:.0f} below the 0.5 threshold. **The network stays near the uniform mean-field saddle
(near, not exactly uniform — D27): graph-correlated fluctuations form but are never amplified
into symmetry breaking.** Under the **published** quadratic diagonal the uniform optimum is
p* = 0, L* = 0, which is why being trapped there shows up as an empty raw output. What this
analysis does NOT explain: the location of the escape threshold at d ≈ 6–7 — the uniform point
is a strict saddle at every degree, so the threshold is measured, not derived.

Verify with: `python code/verify_meanfield.py` (reads the committed `results/posthoc.jsonl`). `code/capture_pstats.py` re-trains and APPENDS to `results/pstats.jsonl` with no skip check, so move that file aside before running it.

### The pre-registered tuning grid

Pre-registered budget (fixed in `DECISIONS.md` D11 **before** any sweep ran): 4 learning rates ×
2 penalties × 3 seeds × 3 degrees, GCN only. Then 7 clearly-labelled post-hoc arms.
At d=20: **{tz20}/{tn20}** pre-registered runs and **{pz20}/{pn20}** post-hoc runs produced a
non-empty raw output. At d=3 the best pre-registered arm reaches AR {bestof(tune,3)[1]/RHO_UB[3]:.4f}, and the
best post-hoc arm {POST_BEST_LAB} {bestof(post,3)[1]/RHO_UB[3]:.4f}, vs DGA's {DGA_T3:.4f}
on the same seeds ({TUNE_SEEDS[0]}–{TUNE_SEEDS[-1]}). These remain valid measurements, but the grid was built against an assumed
"untuned" defence, not the one the Reply published (next subsection).

### The Reply's published defence (D36–D37)

The Reply to Angelini & Ricci-Tersenghi (arXiv:2302.03602) proposes two specific changes —
GraphSAGE instead of GCN, and penalty P = 10 instead of P = 2 — and states that "for d = 3 at
P = 10 GraphSAGE achieves approximation ratios of AR ~ 0.947". The pre-registered grid
(GCN only, P ∈ {{2, 3}}) did not test that, so our earlier verdict that the objection was
answered on its own terms is **retracted** (D36). D37 ran the defence as the Reply describes it.
The Reply gives no architecture detail and no code, so the GraphSAGE layer is our own
mean-aggregator implementation (`code/pignn.py`), which `code/test_port.py` does not cover.
n=1000, seeds {RD_SEEDS[0]}–{RD_SEEDS[-1]}:

{rd_tbl}

AR is the seed-mean post-processed density / ρ_UB. At d=20 there is no AR bound, so the column
counts runs whose **raw** output is non-empty; post-processed density there is
{RD20_POST_TXT} for every arm, i.e. first-fit on an empty set.
- **d=20: raw output empty in {RD20_K-RD20_E}/{RD20_K} runs across all four arms**, the Reply's own configuration
  included. The defence was run at d = 3, 5 and 20 only; this says nothing new about other d ≥ 8.
- **No arm reaches DGA at d=3 or d=5.** P=10 lifts GCN slightly at d=3
  ({rd_ar('gcn_P2',3):.4f} → {rd_ar('gcn_P10',3):.4f}) and makes it markedly worse at d=5
  ({rd_ar('gcn_P2',5):.4f} → {rd_ar('gcn_P10',5):.4f}), which the Reply does not mention.
- **Not reproduced:** our GraphSAGE + P=10 reaches {SAGE10_3:.4f} at d=3, against the Reply's ≈{REPLY_AR}
  and below our own GCN. The likeliest reading is that our GraphSAGE is not theirs. This is a
  failure to reproduce, not a refutation; resolving it needs their implementation.

Two further points from the Reply. It dismisses d > 16 on relevance, not capability ("While
this is certainly an interesting academic exercise, we are not convinced about its practical
usefulness") — a different objection from the one the operating-ceiling result answers. And it
reports an updated post-processing whose scaling improves from ~n^2.0 to ~n^1.0 (total runtime
~n^1.7 → ~n^0.8); this study timed the original released implementation, not that update (§5).

## 5. Claim B — scale

### Measured per-epoch cost, d=3

{sc_tbl}

The released implementation's rule `dim_embedding = int(sqrt(n))`, `hidden = dim/2` (which the
paper's text also uses for n ≥ 10⁵; below that it gives int(∛n)) makes the first GCN layer a
dense `(n × dim) @ (dim × hidden)` matmul, so at n ≥ 10⁵ FLOPs grow as `n·dim·hidden = n²/2`
**independently of graph sparsity**. Measured time does not track that cleanly — it grew more slowly than the
FLOP count below n=1e5, and ~3× faster than it in the last decade (memory-bound at 14.90 GiB).
See D22/D23.

### The released example implementation's hard wall
`utils.py` builds a dense `torch.zeros(n, n)`. At n=1e6 that is 1.0e12 float32 entries =
**3.6 TiB**. The full paper never describes the QUBO construction used at n=1e6 (the word
"dense" does not appear in it), so this is a property of the released example implementation,
not of a published method description. Our sparse reformulation (D4) is mathematically
identical in the forward math (checked by `code/test_port.py`) and is what makes any n=1e6 test possible — a deliberate improvement in the *claim's*
favour.

### Budgeted n=1e6 measurement
{bud_txt}

### Projection (labelled as such)
Epochs-to-convergence is stable across three decades ({EP_TXT}), so the epoch count transfers.
Two per-epoch costs were measured at n=1e6 and disagree (probe {probe['ms_per_epoch']/1000:.2f} s; budgeted run
{bud[0]['t_train_s']/bud[0]['epochs_run']:.2f} s), so the projection is a **range** (D33): {min(EP_BIG):,.0f}–{max(EP_BIG):,.0f} epochs (the two largest measured
sizes) × {C_LO:.2f}–{C_HI:.2f} s = **{T_LO:,.0f}–{T_HI:,.0f} s ≈ {T_LO/86400:.1f}–{T_HI/86400:.1f} days**, against DGA's
**{dga1e6['t_total_s']:.4f} s** on the same instance → **roughly {X_LO/1e6:.1f}–{X_HI/1e6:.1f} million×**. The critique claimed
10^4; the same-host figure is ~{X_LO/1e4:.0f}–{X_HI/1e4:.0f}× larger, because their 10^4 compared their own laptop's greedy
against GPU timings read off the paper's Figure 5 (they say so explicitly). This times the
original released post-processing, not the Reply's updated one (§4). D12's rule (run each size on
whichever device is faster) was never executed at n=1e6 — an unexecuted protocol step (D33).

## 6. Where to attack this — ranked by how much weight it carries

1. **`code/test_port.py` is the keystone — and its scope is stated precisely.** It checks the
   port's *forward math* only against the reference implementation's math, re-derived inline:
   sparse loss ≡ dense `pᵀQp` to machine precision (true asymmetric Q, `−p²` diagonal, no
   self-loops), GCN layer to float32 epsilon; an adversarial audit found it kills the natural
   mutations of that math except a row-normalisation mutant (see the next sentence). It needs nothing in `refs/`. Its normalisation check is vacuous
   on d-regular graphs (D26), and the GraphSAGE layer is untested (no reference exists). The
   training loop, early stopping and bitstring selection sit OUTSIDE the test — their fidelity
   rests on line-by-line replication of the reference's bookkeeping, 6/6 fresh reruns matching the
   recorded results (training is not bit-deterministic; D38), and the DGL 0.5.x source confirming the transcribed layer
   semantics (D26). One disclosed reporting deviation favours the network: the port keeps
   max(best, final) bitstring; the reference keeps best only. If the forward-math test is
   vacuous, the study falls.
2. **Our GraphSAGE may not be theirs — the weakest point.** The Reply reports AR ≈{REPLY_AR} at d=3
   with GraphSAGE + P=10; ours reaches {SAGE10_3:.4f}, below our own GCN. The Reply gives no
   architecture detail or code, so this is a failure to reproduce, not a refutation (D37). If
   our SAGE layer is wrong, every GraphSAGE row in §4 is void; the GCN + P=10 rows still stand.
3. **The ceiling was measured in the released implementation's configuration only.** At n=1000
   that is d0 = 31, where the paper's text gives int(∛n) = 10. Whether the d ≥ 8 ceiling holds
   under ∛n is untested (D38).
4. **The mean-field result is the most falsifiable claim.** L*(uniform) = −n/(2Pd) is a closed
   form, checked against measurement — D17 derived it after the arm had run, not before.
   Re-derive it by hand and check it against `results/posthoc.jsonl`. If the algebra is wrong,
   the mechanism section is wrong.
5. **The null control could be argued to be too generous to itself.** It benefits from a
   maximalization pass, and on all-ones input the pipeline is iterated maximum-degree deletion,
   which reads the graph (D33). Counter-argument in D5: DGA's output is maximal by construction,
   so comparing a non-maximal set to a maximal one would understate PI-GNN. The decomposition in
   §3 is the primary evidence. Decide if you buy it.
6. **The Krutsky comparison was wrong (withdrawn, D38).** D35 claimed a disagreement at d=10,
   but their non-collapsed d = 10 result is MaxCut; their MIS tables agree with our
   {ESC[10][0]}/{ESC[10][1]}. Check that what remains — the transition at d ≈ 6–7, from
   {ESC[6][0]}/{ESC[6][1]} at d=6 and {ESC[7][0]}/{ESC[7][1]} at d=7 — is not over-read from 10 seeds per degree.
7. **Single-seed cells.** n=1e5 PI-GNN and everything at n=1e6 are one seed. Variance at smaller
   n is small (see the trials column) but this is a real limitation.
8. **The n=1e6 numbers are a probe plus a truncated run, not a completed run.** The
   {T_LO/86400:.1f}–{T_HI/86400:.1f}-day figure is a projection and is labelled as one everywhere. If you think the
   epoch count does not transfer to n=1e6, that is the place to say so. The timing covers the
   original released post-processing only, and D12's device rule was never run at n=1e6.
9. **Scope cuts.** PI-GNN at d=5 and d=20, n=1e5 were deliberately not run (D20), and d=5 at
   n=1e6 was killed (only its baselines were captured). Stated, not hidden.
10. **MaxCut is untested.** Boettcher's concurrent critique targets MaxCut; this study is MIS only.
   No claim is made about it.
11. **This does not show GNNs cannot do combinatorial optimisation.** It shows *this* relaxation
   with *this* projection does not. Watch for any sentence that overreaches past that.

## 7. Reproduce

```bash
# Python 3.12
pip install -r requirements.txt
cc -O3 -march=native -o code/greedy code/greedy.c     # drop -march=native if your compiler rejects it
cc -O3 -march=native -o code/repair code/repair.c
python code/test_port.py            # run first — forward-math port-equivalence check; needs nothing else
python code/analyze.py              # summary tables from the committed results
python code/build_report.py && python code/make_site.py      # regenerate report.html and site/
python code/fetch_refs.py           # optional: the papers + the reference implementation into refs/
# To re-measure rather than re-read, write to a NEW file — drivers skip cells already in --out:
python code/runner.py --out results/rerun_phase1.jsonl --d 3 5 20 --n 1000 10000 --seeds 0 1 2 3 4
{REMEASURE}
```

## 8. File map

| Path | What |
|---|---|
| `DECISIONS.md` | D1–D38, append-only. Superseded entries are corrected by later ones, not edited: D15→D17→D27 (mechanism), D4→D26 (port), D18→D28→D35→D38 (escape table), D20→D28 (ratio), D22→D23 (scaling), D24→D25 (Claim B measured), D27→D29 (below-L* count), D6/D11→D36 (tuning premise retracted), D31→D38 (run-count convention retired). |
| `STATE.md` | current position, in-flight work, next action |
| `code/test_port.py` | port-equivalence test of the forward math (sparse loss, GCN layer); needs nothing in `refs/` |
| `code/pignn.py` | the PI-GNN port (GCN, plus the D37 GraphSAGE layer) |
| `code/greedy.c`, `code/repair.c` | baselines and post-processing, C |
| `results/*.jsonl` | all raw measurements, one record per run |
| `refs/` | **not committed** (gitignored; the papers are copyrighted). `python code/fetch_refs.py` retrieves the papers and the authors' reference implementation (Apache-2.0) into it. |
| `notebook/AI-USE-LOG.md` | Regeneron STS Appendix 4 compliance record |
| `report.html` | the findings page (AI-written prose — see the log's boundary note) |
"""
os.makedirs(os.path.join(ROOT, "review"), exist_ok=True)
out = os.path.join(ROOT, "review", "REVIEW.md")
open(out, "w").write(DOC)
print(f"wrote {out} ({len(DOC):,} bytes)")
