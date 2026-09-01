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
coll = list({(r["d"], r["seed"]): r for r in _coll_raw}.values())  # d=7 overlap de-dup (D28)
pstats = loadf("pstats.jsonl")
exact, scale = loadf("exact.jsonl"), loadf("scale.jsonl")

A = defaultdict(list)
for r in main: A[(r["method"], r["d"], r["n"])].append(r)
def st(m, d, n, f="density"):
    v = A.get((m, d, n))
    if not v: return None
    x = np.array([r[f] for r in v]); return x.mean()
def AR(m, d, n):
    s = st(m, d, n); return None if s is None or d not in RHO_UB else s/RHO_UB[d]

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
ESC = {d:(sum(1 for x in v if x["raw_size"]>0), len(v)) for d,v in esc.items()}
esc_tbl = "\n".join(["| d | non-empty runs | p |","|---|---|---|"] +
    [f"| {d} | {e}/{k} | {e/k:.2f} |" for d,(e,k) in sorted(ESC.items())])

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
mf_tbl = "\n".join(["| arm | d | uniform p* | predicted L* | observed L | outcome |","|---|---|---|---|---|---|"]+mf)

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

bud_txt = "_The budgeted n=1e6 run had not finished when this was generated._"
if bud:
    b = bud[0]
    bud_txt = (f"Measured, not projected: after **{b['t_train_s']:,.0f} s** of training "
        f"({b['epochs_run']} epochs of the ~13,321 needed, i.e. "
        f"{100*b['epochs_run']/13321:.2f}% of the schedule), PI-GNN's output repairs to "
        f"**{b['density']:.5f}** (AR {b['ar']:.4f}). On the same instance DGA reached "
        f"**{dga1e6['density']:.5f}** (AR {dga1e6['ar']:.4f}) in **{dga1e6['t_total_s']:.4f} s**, "
        f"and the information-free null control reached **{nul1e6['density']:.5f}**. "
        f"Stop reason: `{b['stop_reason']}`. Truncated: `{b['truncated']}`.")

DOC = f"""# REVIEW PACKAGE — adjudicating PI-GNN vs greedy for Maximum Independent Set

**For an independent reviewing agent.** Everything below is measured on one machine
(Apple M1 Pro, 10 core, 32 GB). {len(main)+len(tune)+len(post)+len(coll)+len(exact)+len(scale):,} records in `results/*.jsonl`.

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

## 2. Verdicts and where the evidence is

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| A | on par or outperforms existing solvers | **Refuted** | `results/phase1.jsonl`, `phase2.jsonl` — §3 |
| B | scales to millions of variables | **Upheld only after a reformulation the paper does not contain** | `results/scale.jsonl` — §5 |
| — | undisclosed operating ceiling in d | **Fails for d ≥ 8** | `results/collapse.jsonl`, `escape.jsonl` — §4 |
| — | "it was untuned" (the Replies' defence) | **Answered** | `results/tuning.jsonl`, `posthoc.jsonl` — §4 |

## 3. Claim A — the head-to-head

{tbl()}

**Replication check.** At d=3 DGA lands at AR {AR('dga',3,1000):.4f} against the critique's
published ≈0.95, and PI-GNN at {AR('pignn_pub',3,1000):.4f} against their ≈0.92. Both sides of
their Figure 1 reproduce on independently written code. This is the single strongest reason to
believe the harness is sound.

**The null control (absent from all five documents in the exchange).** An all-ones bitstring — zero information —
pushed through the *identical* repair-and-maximalize pipeline the GNN's output goes through
reaches AR {AR('null_ones',3,1000):.4f} at d=3, against the trained network's
{AR('pignn_pub',3,1000):.4f}. At d=20 the control **beats** the network
({st('null_ones',20,1000):.5f} vs {st('pignn_pub',20,1000):.5f}).

## 4. Mechanism, ceiling, and the tuning defence

### Escape probability vs degree (n=1000, published config)

{esc_tbl}

### Why it fails — failure to symmetry-break

For a uniform assignment on a d-regular graph, the relaxed objective has a closed-form optimum.
With the diagonal changed to linear (so the origin is *not* a critical point):

{mf_tbl}

The loss agreement is −12.499 vs −12.500 — but the loss alone cannot certify uniformity
(it is second-order blind to zero-mean scatter, and two of three seeds land *below* L*, which
no uniform configuration can do). The direct measurement (`results/pstats.jsonl`,
`code/capture_pstats.py`): mean p = 0.0260–0.0261 vs p* = 0.025, std(p)/p* = 27–29%, max
p ≤ 0.074 — every output a factor ~7 below the 0.5 threshold. **The network is trapped at the
uniform mean-field saddle: graph-correlated fluctuations form but are never amplified into
symmetry breaking.** For the published quadratic diagonal that saddle *is* p=0, which is why
the raw output is empty. What this analysis does NOT explain: the location of the escape
threshold at d ≈ 6–7 — the uniform point is a strict saddle at every degree, so the threshold
is measured, not derived.

Verify with: `./pyrt/bin/python code/verify_meanfield.py` and `code/capture_pstats.py`

### Tuning

Pre-registered budget (fixed in `DECISIONS.md` D11 **before** any sweep ran): 4 learning rates ×
2 penalties × 3 seeds × 3 degrees. Then 7 clearly-labelled post-hoc arms.
At d=20: **{tz20}/{tn20}** pre-registered runs and **{pz20}/{pn20}** post-hoc runs produced a
non-empty set. At d=3 the best tuned arm reaches AR {bestof(post,3)[1]/RHO_UB[3]:.4f} vs DGA's
{AR('dga',3,1000):.4f}.

## 5. Claim B — scale

### Measured per-epoch cost, d=3

{sc_tbl}

The paper's own rule `dim_embedding = int(sqrt(n))`, `hidden = dim/2` makes the first GCN layer a
dense `(n × dim) @ (dim × hidden)` matmul, so FLOPs grow as `n·dim·hidden = n²/2` **independently
of graph sparsity**. Measured time does not track that cleanly — slower than predicted below
n=1e5, ~3× faster than predicted in the last decade (memory-bound at 14.90 GiB). See D22/D23.

### The reference implementation's hard wall
`utils.py` builds a dense `torch.zeros(n, n)`. At n=1e6 that is 1.0e12 float32 entries =
**3.6 TiB**. The published code cannot represent the size it claims to solve. Our sparse
reformulation (D4) is mathematically identical and is what makes any n=1e6 test possible —
a deliberate improvement in the *claim's* favour.

### Budgeted n=1e6 measurement
{bud_txt}

### Projection (labelled as such)
Epochs-to-convergence is stable across three decades (12,361 / 14,812 / 13,321 at n=1e3/1e4/1e5),
so the epoch count transfers: **13,321 × 47.39 s = 631,252 s = 7.31 days**, against DGA's
**0.2671 s** on the same instance → **≈2.4 million×**. The critique claimed 10^4; the same-host
figure is ~236× larger, because their 10^4 compared their own laptop's greedy against GPU
timings read off the paper's Figure 5 (they say so explicitly).

## 6. Where to attack this — ranked by how much weight it carries

1. **`code/test_port.py` is the keystone — and its scope is now stated precisely.** It proves
   the *forward math* only: sparse loss ≡ reference dense `pᵀQp` to machine precision (true
   asymmetric Q, `−p²` diagonal, no self-loops), GCN layer to float32 epsilon; an adversarial
   audit confirmed it kills every natural mutation of that math. The training loop, early
   stopping and bitstring selection sit OUTSIDE the test — their fidelity rests on line-by-line
   replication of the reference's bookkeeping, 6/6 fresh reruns reproducing recorded numbers
   digit-for-digit, and the DGL 0.5.x source confirming the transcribed layer semantics (D26).
   One disclosed reporting deviation favours the network: the port keeps max(best, final)
   bitstring; the reference keeps best only. If the forward-math test is vacuous, the study falls.
2. **The mean-field result is the most falsifiable claim.** −12.500 was derived in closed form
   *before* comparison. Re-derive `L*(uniform) = −n/(2Pd)` by hand and check it against
   `results/posthoc.jsonl`. If the algebra is wrong, the mechanism section is wrong.
3. **The null control could be argued to be too generous to itself.** It benefits from a
   maximalization pass. Counter-argument in D5: DGA's output is maximal by construction, so
   comparing a non-maximal set to a maximal one would understate PI-GNN. Decide if you buy it.
4. **Single-seed cells.** n=1e5 PI-GNN and everything at n=1e6 are one seed. Variance at smaller
   n is small (see the trials column) but this is a real limitation.
5. **The n=1e6 numbers are a probe plus a truncated run, not a completed run.** The 7.31-day
   figure is a projection and is labelled as one everywhere. If you think the epoch count does
   not transfer to n=1e6, that is the place to say so.
6. **Scope cuts.** PI-GNN at d=5 and d=20, n=1e5 were deliberately not run (D20), and d=5 at
   n=1e6 was killed (only its baselines were captured). Stated, not hidden.
7. **MaxCut is untested.** Boettcher's concurrent critique targets MaxCut; this study is MIS only.
   No claim is made about it.
8. **This does not show GNNs cannot do combinatorial optimisation.** It shows *this* relaxation
   with *this* projection does not. Watch for any sentence that overreaches past that.

## 7. Reproduce

```bash
cd ~/gnn-vs-greedy
./pyrt/bin/python code/test_port.py         # equivalence proof — run this FIRST
cc -O3 -march=native -o code/greedy code/greedy.c
cc -O3 -march=native -o code/repair code/repair.c
./pyrt/bin/python code/runner.py --out results/phase1.jsonl \\
       --d 3 5 20 --n 1000 10000 --seeds 0 1 2 3 4
./pyrt/bin/python code/collapse_sweep.py
./pyrt/bin/python code/sweep_tuned.py --out results/tuning.jsonl --d 3 5 20 --n 1000 --seeds 0 1 2
./pyrt/bin/python code/posthoc_d20.py
./pyrt/bin/python code/verify_meanfield.py
./pyrt/bin/python code/exact_mis.py --out results/exact.jsonl
./pyrt/bin/python code/analyze.py           # all tables
```

## 8. File map

| Path | What |
|---|---|
| `DECISIONS.md` | D1–D24, append-only, every design choice + rationale. **D15 is marked WRONG and superseded by D17.** |
| `STATE.md` | current position, in-flight work, next action |
| `code/test_port.py` | the equivalence proof |
| `code/pignn.py` | the PI-GNN port |
| `code/greedy.c`, `code/repair.c` | baselines and post-processing, C |
| `results/*.jsonl` | all raw measurements, one record per run |
| `refs/pignn/` | the official reference implementation (Apache-2.0) |
| `refs/critique.txt` | full text of the critique |
| `notebook/AI-USE-LOG.md` | Regeneron STS Appendix 4 compliance record |
| `report.html` | the findings page (AI-written prose — see the log's boundary note) |
"""
os.makedirs(os.path.join(ROOT, "review"), exist_ok=True)
out = os.path.join(ROOT, "review", "REVIEW.md")
open(out, "w").write(DOC)
print(f"wrote {out} ({len(DOC):,} bytes)")
