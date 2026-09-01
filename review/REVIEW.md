# REVIEW PACKAGE — adjudicating PI-GNN vs greedy for Maximum Independent Set

**For an independent reviewing agent.** Everything below is measured on one machine
(Apple M1 Pro, 10 core, 32 GB). 483 records in `results/*.jsonl`.

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

| d | n | method | density | AR | wall clock (s) | trials |
|---|---|---|---|---|---|---|
| 3 | 1,000 | DGA | 0.43260 | 0.9500 | 4.98e-05 | 5 |
| 3 | 1,000 | PI-GNN | 0.41820 | 0.9184 | 13.74 | 5 |
| 3 | 1,000 | null all-ones | 0.39660 | 0.8709 | 0.0001028 | 5 |
| 3 | 1,000 | GA | 0.37880 | 0.8319 | 5.34e-05 | 5 |
| 3 | 10,000 | DGA | 0.43270 | 0.9502 | 0.0005792 | 5 |
| 3 | 10,000 | PI-GNN | 0.41612 | 0.9138 | 118.3 | 5 |
| 3 | 10,000 | null all-ones | 0.39386 | 0.8649 | 0.0004734 | 5 |
| 3 | 10,000 | GA | 0.37442 | 0.8222 | 0.0005266 | 5 |
| 3 | 100,000 | DGA | 0.43254 | 0.9499 | 0.00664 | 3 |
| 3 | 100,000 | PI-GNN | 0.41516 | 0.9117 | 2,176 | 1 |
| 3 | 100,000 | null all-ones | 0.39588 | 0.8694 | 0.004563 | 3 |
| 3 | 100,000 | GA | 0.37547 | 0.8245 | 0.006074 | 3 |
| 5 | 1,000 | DGA | 0.35300 | 0.9182 | 7.74e-05 | 5 |
| 5 | 1,000 | PI-GNN | 0.34000 | 0.8844 | 20.71 | 5 |
| 5 | 1,000 | null all-ones | 0.32800 | 0.8532 | 0.00013 | 5 |
| 5 | 1,000 | GA | 0.29900 | 0.7778 | 7.26e-05 | 5 |
| 5 | 10,000 | DGA | 0.35672 | 0.9279 | 0.0007978 | 5 |
| 5 | 10,000 | PI-GNN | 0.34144 | 0.8882 | 191.2 | 5 |
| 5 | 10,000 | null all-ones | 0.32670 | 0.8498 | 0.0007183 | 5 |
| 5 | 10,000 | GA | 0.30190 | 0.7853 | 0.0007272 | 5 |
| 5 | 100,000 | DGA | 0.35665 | 0.9277 | 0.009042 | 3 |
| 5 | 100,000 | null all-ones | 0.32689 | 0.8503 | 0.007353 | 3 |
| 5 | 100,000 | GA | 0.30152 | 0.7843 | 0.008772 | 3 |
| 20 | 1,000 | DGA | 0.17000 | — | 0.0001952 | 5 |
| 20 | 1,000 | PI-GNN | 0.13540 | — | 20.65 | 5 |
| 20 | 1,000 | null all-ones | 0.15720 | — | 0.000268 | 5 |
| 20 | 1,000 | GA | 0.13820 | — | 0.0001858 | 5 |
| 20 | 10,000 | DGA | 0.17324 | — | 0.002103 | 5 |
| 20 | 10,000 | PI-GNN | 0.13956 | — | 272.7 | 5 |
| 20 | 10,000 | null all-ones | 0.15880 | — | 0.00234 | 5 |
| 20 | 10,000 | GA | 0.13956 | — | 0.00202 | 5 |
| 20 | 100,000 | DGA | 0.17371 | — | 0.02987 | 3 |
| 20 | 100,000 | null all-ones | 0.15873 | — | 0.03015 | 3 |
| 20 | 100,000 | GA | 0.13948 | — | 0.02888 | 3 |

**Replication check.** At d=3 DGA lands at AR 0.9500 against the critique's
published ≈0.95, and PI-GNN at 0.9184 against their ≈0.92. Both sides of
their Figure 1 reproduce on independently written code. This is the single strongest reason to
believe the harness is sound.

**The null control (absent from all five documents in the exchange).** An all-ones bitstring — zero information —
pushed through the *identical* repair-and-maximalize pipeline the GNN's output goes through
reaches AR 0.8709 at d=3, against the trained network's
0.9184. At d=20 the control **beats** the network
(0.15720 vs 0.13540).

## 4. Mechanism, ceiling, and the tuning defence

### Escape probability vs degree (n=1000, published config)

| d | non-empty runs | p |
|---|---|---|
| 3 | 3/3 | 1.00 |
| 4 | 10/10 | 1.00 |
| 5 | 3/3 | 1.00 |
| 6 | 9/10 | 0.90 |
| 7 | 2/10 | 0.20 |
| 8 | 0/10 | 0.00 |
| 9 | 0/10 | 0.00 |
| 10 | 0/10 | 0.00 |
| 11 | 0/10 | 0.00 |
| 12 | 0/10 | 0.00 |
| 13 | 0/10 | 0.00 |
| 14 | 0/3 | 0.00 |
| 15 | 0/10 | 0.00 |
| 16 | 0/3 | 0.00 |
| 18 | 0/3 | 0.00 |
| 20 | 0/3 | 0.00 |
| 25 | 0/3 | 0.00 |

### Why it fails — failure to symmetry-break

For a uniform assignment on a d-regular graph, the relaxed objective has a closed-form optimum.
With the diagonal changed to linear (so the origin is *not* a critical point):

| arm | d | uniform p* | predicted L* | observed L | outcome |
|---|---|---|---|---|---|
| modified_linear | 3 | 0.1667 | -83.333 | -423.254 | symmetry-broken |
| modified_linear | 5 | 0.1000 | -50.000 | -341.256 | symmetry-broken |
| modified_linear | 20 | 0.0250 | -12.500 | -12.499 | **stuck at mean field** |
| modified_linear_lr1e-3 | 3 | 0.1667 | -83.333 | -421.949 | symmetry-broken |
| modified_linear_lr1e-3 | 5 | 0.1000 | -50.000 | -342.936 | symmetry-broken |
| modified_linear_lr1e-3 | 20 | 0.0250 | -12.500 | -12.519 | **stuck at mean field** |

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
At d=20: **0/24** pre-registered runs and **0/21** post-hoc runs produced a
non-empty set. At d=3 the best tuned arm reaches AR 0.9296 vs DGA's
0.9500.

## 5. Claim B — scale

### Measured per-epoch cost, d=3

| n | ms/epoch | ×/decade | local exponent |
|---|---|---|---|
| 1,000 | 1.10 | — | — |
| 10,000 | 7.98 | 7.3× | 0.86 |
| 100,000 | 163.22 | 20.4× | 1.31 |
| 1,000,000 | 47,387.71 | 290.3× | 2.46 |

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
Measured, not projected: after **1,503 s** of training (36 epochs of the ~13,321 needed, i.e. 0.27% of the schedule), PI-GNN's output repairs to **0.37470** (AR 0.8229). On the same instance DGA reached **0.43295** (AR 0.9508) in **0.2671 s**, and the information-free null control reached **0.39527**. Stop reason: `budget`. Truncated: `True`.

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
./pyrt/bin/python code/runner.py --out results/phase1.jsonl \
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
