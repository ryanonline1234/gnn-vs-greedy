# DECISIONS — adjudicating Schuetz et al. (NMI 2022) vs Angelini & Ricci-Tersenghi (NMI 2023)

Append-only. Newest at bottom.

## D1 — What is being measured
The measured object is **the validity of a published claim against an exact objective**,
not model performance. The claim under test (Schuetz, Brubaker & Katzgraber,
Nat Mach Intell 4, 367 (2022), abstract, verbatim):

> "the graph neural network optimizer performs on par or outperforms existing solvers,
> with the ability to scale beyond the state of the art to problems with millions of variables."

This decomposes into two independently falsifiable claims:
- **A (quality/parity):** PI-GNN is on par with or better than existing solvers for MIS.
- **B (scale):** PI-GNN scales to ~1e6 variables.
A and B can have different verdicts. B being true does not rescue A, because the
baseline also scales. They are scored separately and never averaged.

## D2 — Ground truth (the answer key)
From Angelini & Ricci-Tersenghi (arXiv:2206.13211), which we adopt so our numbers are
directly comparable to the published critique:
- Upper bounds on MIS density (McKay 1987): rho_UB(d=3) = 0.45537, rho_UB(d=5) = 0.38443
- 1RSB (replica) optimal approximation ratios (Barbier et al. 2013):
  AR_1RSB(d=3) ~ 0.990, AR_1RSB(d=5) ~ 0.987
- Reference algorithm ARs from Angelini & Ricci-Tersenghi PRE 100, 013302 (2019):
  MCMC/SA: 0.984 (d=3), 0.981 (d=5); BP+reinforcement: 0.987 (d=3), 0.981 (d=5)
Primary metric: **approximation ratio AR = (|IS|/n) / rho_UB(d)**, matching the critique.
Secondary calibration: exact MIS via ILP at small n.

## D3 — d = 20 is included deliberately
The critique argues d < 16 is an easy regime (no clustering transition) and proposes
d > 16 as the hard benchmark, explicitly inviting the experiment:
"we hope to see future neural network optimizers tested on these problems before any
claim of superiority is made." Running d=20 is therefore the *requested* experiment,
not an extra. No published rho_UB is adopted for d=20; we report raw density and
relative standing there.

## D4 — Reimplementation policy (PI-GNN)
The official code (github.com/amazon-science/co-with-gnns-example) is DGL 0.5.3 /
torch 1.7.1 and does not run on this machine. We port to PyTorch Geometric.
The port is *faithful by default and charitable where it differs*:
- Architecture: 2-layer GCN, in=dim_embedding -> hidden -> 1, relu + dropout, sigmoid out.
- DGL GraphConv does NOT add self-loops; we set add_self_loops=False to match.
- Input: jointly-trained nn.Embedding(n, dim_embedding); dim_embedding=int(sqrt(n)),
  hidden_dim=dim_embedding//2 (exactly as the notebook).
- Adam lr=1e-4, epochs<=1e5, tol=1e-4, patience=100, dropout=0.0, threshold=0.5.
- Loss: the repo builds a DENSE n x n Q matrix (torch.zeros(n,n)), which is O(n^2) and
  cannot reach 1e6 nodes at all (1e12 entries). We implement the mathematically
  identical loss sparsely:
      L(p) = -sum_i p_i^2 + 2 * sum_{(u,v) in E} p_u p_v
  Note the diagonal term is -p_i^2, not -p_i, because the relaxation is continuous.
  This is a STRICT IMPROVEMENT for their side and is required for claim B to be
  testable at all.

## D5 — Scoring the GNN output fairly
The reference projection is a bare threshold (p >= 0.5) with no independence repair;
the notebook only *counts* violations. A thresholded output is therefore not
necessarily a valid independent set. We report:
- `raw_size` and `violations` (as the reference code does), and
- `repaired_size`: violations removed, then greedily maximalized.
The **repaired, maximalized size is the headline number for PI-GNN**, because DGA
output is maximal by construction and comparing a non-maximal set to a maximal one
would understate PI-GNN. Charity to the claim under test is deliberate.

## D6 — Two GNN arms
The published Reply's central defense is that the critics used an untuned network.
We pre-register two arms:
- `pignn_pub`: exactly the repo configuration (D4).
- `pignn_tuned`: a fixed, pre-declared hyperparameter budget, best-of-sweep reported.
An untuned model losing proves nothing; this removes that objection.

## D7 — Timing protocol
The published 10^4 speedup is a CROSS-MACHINE comparison: the critique states DGA was
timed on "a 2.3 GHz MacBook Pro" while GNN times were "extracted from Fig. 5 in Ref. [4]"
(different hardware, GPU). That number is therefore not a controlled measurement.
Our contribution: **same host, same instances, same seeds** (Apple M1 Pro, 10 core, 32 GB).
Every method reports wall clock measured by us. For the GNN we report training time and
total time separately, so the amortization argument can be evaluated rather than assumed.
Graph *loading* from disk is excluded for all methods; data-structure construction
(CSR build, tensor build) is included for all methods.

## D8 — Greedy baselines in C
DGA is the critique's champion and the timing comparison is the crux, so implementing it
in slow Python would inflate its runtime and bias the speedup toward PI-GNN. DGA and GA
are implemented in C with a bucket queue (near-linear, as the critique describes),
cross-validated against an independent Python implementation on small instances.

## D9 — Null controls (an addition to the published debate)
Neither the original paper, the critique, nor the Replies establish what the
*post-processing alone* achieves. We add two information-free controls that go through
the identical repair + maximalize pipeline as the GNN output:
- `null_ones`: the all-ones bitstring (every node selected).
- `null_rand`: an i.i.d. fair-coin bitstring.
Measured at n=1000, d=3: `null_ones` repairs to density 0.398 (AR 0.874). Any method
scoring at or below this floor has contributed nothing beyond its post-processing.
This turns "is the GNN worse than greedy" into the sharper and more decisive
"does the GNN's learned signal do measurable work at all". It is the single most
diagnostic measurement in this study and it is absent from all four papers.

## D10 — Measurements run serially
Wall-clock is the disputed quantity, so no two timed jobs run concurrently on this host.
Every phase is launched alone and allowed to finish before the next begins. Timings taken
under concurrent load would be uninterpretable and are discarded if they occur.

## D11 — Pre-registered tuning budget for `pignn_tuned` (declared before any sweep was run)
The Reply's central defense is that the critics used an untuned network, so the tuned arm
must be fixed in advance rather than chosen after seeing which setting wins.

Stage A (tune): n = 1000, each d in {3,5,20}, seeds {0,1,2}, grid
    lr      in {1e-4, 3e-4, 1e-3, 1e-2}      (1e-4 is the published value)
    penalty in {2, 3}                        (2 is the published value)
  = 8 configurations x 3 seeds per d. Selection criterion: highest mean repaired density.
Stage B (transfer): the winning (lr, penalty) per d is applied unchanged at
    n in {1e3, 1e4, 1e5}, seeds 0-4.
No further tuning is permitted after Stage A. If the tuned arm still loses to DGA, the
"it was untuned" objection is answered; if it wins, that is reported as such.
Architecture (2-layer GCN), dim_embedding=sqrt(n), hidden=dim/2, dropout=0, threshold=0.5
and the early-stopping rule are held fixed at the published values throughout.

## D12 — Device is chosen per size as whichever is FASTER for the GNN
Measured on this host at n=1e4, d=3: 300 epochs took 2.16 s on CPU vs 4.25 s on MPS —
the graphs are small and sparse enough that kernel-launch overhead dominates, so CPU
wins at small n. At large n the GPU should win. Running the GNN on a device that is
slower for it would inflate the speedup factor in the critique's favour, so for each n
we benchmark both and report the GNN on the faster one. Every reported GNN time is
therefore a best-of-two-devices time on this machine. The greedy baselines are
single-threaded C on the same host; they are given no such advantage.

## D13 — Phase 1 result and the guard it demands (recorded before the tuning sweep ran)
Phase 1 (n=1e3,1e4; seeds 0-4) reproduces the published critique at d=3,5:
  d=3  n=1e4: DGA AR 0.9502 vs PI-GNN AR 0.9138   (critique Fig.1: ~0.95 vs ~0.92)
  d=5  n=1e4: DGA AR 0.9279 vs PI-GNN AR 0.8882
At d=20 -- the hard benchmark the critique explicitly proposed and invited -- PI-GNN
falls BELOW the information-free null floor:
  d=20 n=1e4: DGA 0.17324, null_ones 0.15880, PI-GNN 0.13956  (PI-GNN is 12.1% BELOW null)
i.e. the trained network is worse than feeding a constant bitstring through its own
post-processing. This is a strong claim, so it must survive the strongest objections:
 (a) "untuned"       -> pre-registered Stage A sweep (D11) must be run at d=20 too.
 (b) "not converged" -> inspect epochs_run and loss traces at d=20; re-run with the
                        early-stopping patience relaxed if it is stopping early.
 (c) "model too small"-> dim_embedding=sqrt(n) is the published formula; test larger.
(a) is pre-registered. (b) and (c) are POST-HOC and will be reported under a separate
`pignn_posthoc` label, never merged into the pre-registered arm. If PI-GNN still loses
after every one of these, the objection set is exhausted.

## D14 — Device benchmark result: CPU (MPS gives no advantage here)
Measured alone on an idle host, n=1e5, 50 timed epochs after warmup:
  d=3 : CPU 155.9 ms/epoch, MPS 144.5 ms/epoch
  d=5 : CPU 212.0 ms/epoch, MPS 215.0 ms/epoch
The network is tiny (dim 316 -> 158 -> 1) and the cost is dominated by sparse scatter,
so the GPU cannot be exploited. Difference is <8%; all GNN runs use CPU. D12's
"faster of the two" rule is satisfied to within noise, and no meaningful advantage is
being withheld from PI-GNN.

## D15 — MECHANISM: the published objective has a trivial critical point at the origin
The reference loss is  L(p) = p^T Q p  with Q_ii = -1, Q_uv = +2. It is a PURE QUADRATIC
FORM with no linear term, therefore grad L(0) = 0 exactly and L(0) = 0. The all-zeros
assignment is a critical point of the published objective for EVERY degree d.
For a uniform assignment p_i = p:  L = -n p^2 + 2*(nd/2)*p^2 = n p^2 (d-1),
which is positive and minimised at p = 0 for every d > 1. Escaping the origin requires
symmetry breaking, and the basin of the origin grows with d.
Measured at n=1000 (12k epochs, published config):
  d=3 : loss -> -411.9,  mean p 0.414, max p 1.000, #(p>=0.5) = 414   (escapes)
  d=20: loss ->   +0.028, mean p 0.001, max p 0.004, #(p>=0.5) =   0   (collapses)
  d=20, penalty 1.1: identical collapse -- lowering the penalty does NOT rescue it.
This is an analytic property of the published relaxation, not an artefact of our port:
the origin is a critical point of THEIR objective as written. It is compounded by the
reference early-stopping rule, which initialises best_loss to the loss of the all-zeros
bitstring (0.0), so a run that never reaches negative loss returns the empty set by
construction. This explains the phase-1 d=20 numbers: raw_size = 0 in 10/10 runs.

## D16 — The collapse is probabilistic, not a hard threshold
First collapse-sweep results (n=1000, published config, 3 seeds per d):
  d=3  : 3/3 escape (raw 421, 419, 417)
  d=5  : 3/3 escape (raw 331, 333, 336)
  d=7  : 1/3 escape (raw 0, 257, 0)      <-- transition region
  d=10 : 0/2 escape so far (raw 0, 0)
So escaping the origin is a basin-of-attraction race decided by the random
initialisation, and the escape probability falls off sharply around d ~ 6-8. A run that
collapses stops early (~7.8k epochs, loss -> +0.14) while a run that escapes trains
much longer (~18.9k epochs, loss -> -256.9); the two outcomes are trivially separable.
This is a better characterisation than a hard threshold and needs more seeds to pin
down: added a follow-up sweep at d in {6,7,8,9} with 10 seeds each. Reporting an
escape PROBABILITY per degree is the honest form of this result.

## D17 — CORRECTION and generalisation of D15: it is failure to symmetry-break
D15 said the failure was the trivial critical point at the origin created by the
quadratic diagonal. The post-hoc `modified_linear` arm falsified that as the *general*
explanation: with the diagonal changed to -sum p_i the origin is NOT a critical point
(grad_i L(0) = -1 < 0), yet d=20 still returns the empty set. So the origin was the
symptom, not the cause.

The actual mechanism, verified quantitatively. For a uniform assignment p_i = p on a
d-regular graph (n nodes, m = nd/2 edges):
    quadratic diagonal (PUBLISHED): L(p) = n p^2 (Pd/2 - 1)  -> p* = 0,        L* = 0
    linear diagonal   (MODIFIED)  : L(p) = -n p + P(nd/2)p^2 -> p* = 1/(Pd),   L* = -n/(2Pd)
Observed final losses at n=1000, P=2, seeds 0-2:
    d=3  linear: predicted uniform L* = -83.333, observed -423.3   -> SYMMETRY-BROKEN
    d=5  linear: predicted uniform L* = -50.000, observed -341.3   -> SYMMETRY-BROKEN
    d=20 linear: predicted uniform L* = -12.500, observed -12.499  -> STUCK AT MEAN FIELD
The d=20 agreement is exact to four significant figures.

**Conclusion:** at high degree the GNN converges precisely to the uniform mean-field
solution of the relaxed objective -- it extracts no structure from the graph at all.
"Collapse to the empty set" and "stuck at the mean field" are the same event; the
published quadratic diagonal merely makes the symmetric solution BE the empty set,
which the fixed 0.5 projection then reports as nothing.
This is a property of the physics-inspired relaxation plus the projection rule, not of
our port, and it is not disclosed in the paper. It also explains why every post-hoc
rescue failed: lr, embedding width and patience do not cause symmetry breaking.

## D18 — Escape probability vs degree (n=1000, published config)
  d<=5 : 1.00   (3/3, 10/10, 3/3 at d=3,4,5)
  d=6  : 0.90   (9/10)
  d=7  : 0.23   (3/13)
  d>=8 : 0.00   (0/10 at d=8, 0/10 at d=9, 0/3 at each of d=10,12,14,16,18,20,25)
A sharp transition at d ~ 6-7. Above it the published method returns the empty set in
every run. The paper's own MIS experiments are at d=3, i.e. inside the narrow band
where the method functions at all. This ceiling is not disclosed in the paper, and it
sits well BELOW the d>16 clustering transition the critique proposed as the hard
regime -- the method fails long before the problem gets hard.

## D19 — Exact ILP calibration (HiGHS)
  d=3  n=100: rho=0.44000 (Optimal)          rho/rho_UB = 0.9662
  d=3  n=200: rho=0.44700 (Optimal)          rho/rho_UB = 0.9816
  d=5  n=100: rho=0.37000 (Optimal)          rho/rho_UB = 0.9625
  d=5  n=200: rho=0.37200 (4/5 Optimal, 1 hit the 240 s limit -> lower bound)
  d=20 n=100: rho=0.18000 (Optimal)
  d=20 n=200: rho=0.18000 (TIME LIMIT -- a lower bound, NOT a proven optimum)
The AR scale is sane: proven optima sit at 0.96-0.98 of McKay's bound at these small n,
consistent with the 1RSB asymptotic AR of 0.990. Runs that hit the time limit are
reported as lower bounds and never as exact optima.

## D20 — n=1e5 scope cut (explicit, not a silent truncation)
Measured at n=1e5: ALL baselines (dga, ga, null_ones, null_rand) at d=3,5,20, seeds 0-2,
plus PI-GNN at d=3 seed 0 (AR 0.9117, 13321 epochs, 2176 s vs DGA's 0.008 s = 272,000x).
NOT measured: PI-GNN at d=5 and d=20, n=1e5. Killed deliberately after 46 min of a
projected 2.4 h, because:
 - d=5 n=1e5 was predictable to three digits from n=1e4 (AR 0.888) and adds no new claim.
 - d=20 n=1e5 would confirm the all-zeros collapse a third time; it is already 10/10
   consistent at n=1e3 and n=1e4 AND has a closed-form mean-field explanation (D17).
 - the remaining unmeasured published claim is B (scale to 1e6), which had no data at all.
Reallocating that time to n=1e6 is the higher-value use of a fixed machine.
The report must state these cells as not-measured rather than leaving the gap implicit.

## D21 — n=1e6 baselines and the reference implementation's hard wall (measured)
On this host, n=1,000,000, d=3, seed 0:
    DGA       size=432,949  rho=0.43295  AR=0.9508  t=0.2671 s
    GA        size=375,147  rho=0.37515  AR=0.8238  t=0.1799 s
    null_ones size=395,270  rho=0.39527                t=0.0670 s
The reference implementation's dense Q = torch.zeros(1e6, 1e6) is 1.0e12 float32 entries
= 3.6 TiB. The published code cannot represent the problem size it claims to solve.
Our sparse reformulation (D4) is what makes any n=1e6 test possible.

## D22 — the sqrt(n) embedding rule is itself the scaling wall (analytic, to be confirmed)
The notebook sets dim_embedding = int(sqrt(n)) and hidden_dim = dim/2. The first GCN layer
is a dense (n x dim) @ (dim x hidden) matmul, so per-epoch cost grows as
    n * dim * hidden = n * sqrt(n) * sqrt(n)/2 = n^2 / 2
independently of graph sparsity. At n=1e6 that is dim=1000, hidden=500, an embedding table
of 1e9 parameters = 14.90 GiB with Adam states. This is a candidate explanation for the
near-quadratic runtime the critique observed, and it is a property of the paper's own
hyperparameter rule rather than of GNNs. STATUS: analytically clear, NOT yet confirmed
against measured ms/epoch at n=1e6. Do not state it as measured until the probe lands.
Measured so far: 8.4 ms/epoch (n=1e4), 155 ms/epoch (n=1e5) -> ratio 18.5, i.e. BELOW the
quadratic prediction of 100x, so the sparse edge term still dominates at n<=1e5. The
crossover matters; check it before claiming n^2.

## D23 — MEASURED scaling of PI-GNN per-epoch cost (refines D22)
Per-epoch wall clock, d=3, published config, this host:
    n=1e3      1.10 ms
    n=1e4      7.98 ms   ( 7.3x per decade, local exponent 0.86)
    n=1e5    163.22 ms   (20.4x per decade, local exponent 1.31)
    n=1e6 47,387.71 ms   (290.3x per decade, local exponent 2.46)
D22 predicted a quadratic FLOP count from the paper's own dim_embedding=sqrt(n) rule, since
n*dim*hidden = n^2/2. The FLOP count does grow exactly 100x per decade here
(4.65e5 -> 5.00e7 -> 4.99e9 -> 5.00e11). The MEASURED time does not track it:
 - below n=1e5 growth is SLOWER than the FLOP prediction (the sparse edge term still
   dominates and the small matmuls are latency-bound, not compute-bound);
 - in the last decade growth is ~3x FASTER than the FLOP prediction (290x vs 100x),
   because the 14.90 GiB working set exceeds cache and the matmul becomes memory-bound.
So D22's mechanism is the correct driver but the naive n^2 statement is wrong in both
directions. The honest claim: **the paper's own sqrt(n) embedding rule makes cost grow
super-linearly by construction, and the measured last-decade exponent is 2.46.**
Basis: single seed at n=1e6 from a 20-epoch probe; the other points are full runs.

## D24 — Claim B, quantified (projection, clearly labelled as such)
Epochs to convergence are stable across three decades (12,361 / 14,812 / 13,321 at
n=1e3/1e4/1e5), so the epoch count transfers. A full n=1e6 run at the published config:
    13,321 epochs x 47.39 s/epoch = 631,252 s = 175.3 h = 7.31 days
    DGA on the same host, same instance: 0.2671 s  ->  ratio 2,363,353x
The critique claimed 10^4. The same-host figure is ~236x larger than their claim, because
their 10^4 compared their own laptop's greedy against GPU timings read off the paper's
Figure 5. This is a PROJECTION from a measured per-epoch cost, not a completed run, and
must be labelled as such everywhere it appears.

## D25 — Claim B, MEASURED (not projected) at n=1e6, d=3
Budgeted run, published configuration, wall-clock cap 1500 s:
    36 epochs completed in 1503 s (0.27% of the ~13,321 epochs this model needs)
    raw thresholded output: 220 nodes of 1,000,000 (collapsing toward the empty set)
    after repair + maximalize: 374,704  rho=0.37470  AR=0.8229
    stop_reason = budget ; truncated = true
On the SAME instance and host:
    DGA            432,949  rho=0.43295  AR=0.9508  in 0.2671 s
    null_ones      395,270  rho=0.39527  AR=0.8680  in 0.0670 s
So after 25 minutes at n=1e6 PI-GNN sits BELOW the information-free null control, while the
greedy finished in a quarter of a second with a 15% larger set.
**Framing discipline:** this is a truncated run, so it is a lower bound on time-to-parity and
NOTHING more. It does not show the network would never converge. Every use of this number
must carry that qualifier. The 7.31-day figure (D24) remains a projection.

## D26 — Adversarial review round: port-equivalence findings (keystone audit)
Three independent review agents attacked the study (port equivalence / mechanism math /
data tracing). Port audit outcome: the keystone SURVIVES, with corrections we adopt:
 - `test_port.py` proves the FORWARD math only (sparse loss == dense p^T Q p to machine
   precision against the true asymmetric Q; GCN layer to float32 eps; mutation testing
   confirmed it catches symmetrized-penalty, wrong-diagonal, self-loop, dropped-bias and
   no-normalization mutants). The training loop, early stopping, and bitstring selection
   are OUTSIDE the test. Their fidelity rests on (a) line-by-line comparison against
   `run_gnn_training` in the reference (exact replication of the trigger, count-reset and
   order of operations), (b) 6/6 fresh published-config reruns reproducing our recorded
   numbers digit-for-digit, and (c) the DGL 0.5.x source confirming the transcribed
   GraphConv semantics. All wording of "proven equivalent" is narrowed accordingly.
 - UNDISCLOSED DEVIATION now disclosed: the port reports max(best, final) bitstring where
   the reference reports best only (`pignn.py` chosen = bb if bb.sum() >= fb.sum() else fb).
   Strictly in PI-GNN's favor; empirically inert (best == final in all 6 audit reruns).
 - The GCN normalization check is vacuous ON d-REGULAR GRAPHS (D^-1 A == D^-1/2 A D^-1/2
   there; a row-norm mutant passes it). Closed externally: the auditor verified the port
   against the dense symmetric-norm reference on an IRREGULAR G(30,0.2) graph (2.4e-7)
   and every study instance is d-regular, where the variants coincide.
 - CORRECTION to D4's prose: the port is a hand-rolled plain-PyTorch GCN layer; it does
   not use PyTorch Geometric (early plan, changed during implementation; D4's math
   bullets were and are accurate). Micro-deviations also now disclosed: deg.clamp(min=1)
   (inert, no isolated nodes in d-regular) and dropout gated on self.training (inert at
   dropout=0).

## D27 — Adversarial review round: mechanism corrections (SUPERSEDES the wording of D17)
D17's numbers all verify (closed forms re-derived by hand; observed losses match records
to the digit; the tolerance in verify_meanfield.py is stable over three orders of
magnitude). But D17's strongest sentence — "converges EXACTLY onto the uniform mean-field
solution … extracts no structure whatsoever" — is FALSE as literally written:
 - The loss is second-order blind to zero-mean scatter around p* (tr(A)=0, grad L(p*)=0),
   so loss agreement cannot certify uniformity. Demonstrated: 40% iid scatter moves the
   loss by < 0.05.
 - 5 of 6 recorded linear-diagonal losses at d=20 sit strictly BELOW L* = -12.500, which
   no uniform configuration can reach — the data itself proves small non-uniformity.
 - Direct measurement (results/pstats.jsonl, code/capture_pstats.py, reproducing the
   recorded losses): mean(p) = 0.0260-0.0261 (4% above p* = 0.025), std(p)/p* = 27-29%,
   max p up to 0.074 = 3 p*, fraction >= 0.5 threshold: exactly 0.
CORRECTED CLAIM: the network is TRAPPED AT THE UNIFORM MEAN-FIELD SADDLE in every
macroscopic sense — the loss sits at the mean-field level, every output is a factor ~7
below the projection threshold, and the ~28% graph-correlated fluctuations are never
amplified into symmetry breaking. The uniform point is the unique interior stationary
point (Ap = 1/P * 1 has only the uniform solution; nullity(A) = 0 measured on the d=20
instance), which strengthens the trapping story. What the analysis does NOT explain, and
we now say so explicitly: the LOCATION of the escape threshold at d ~ 6-7. The uniform
point is a strict saddle at every d (Hessian = P*A, lambda_min < 0 always), so the
mean-field analysis predicts where a non-breaking network lands, not why breaking fails
only above d ~ 6-7. The threshold is measured, not derived.
Also corrected: "exact to four significant figures" overstated a 3-seed mean whose s.e.
is ~0.02 (the 0.0014 agreement is partly averaging luck; ~3 digits is what a repeat
guarantees). And the sister arm modified_linear_lr1e-3 (mean -12.5189, below L*) is now
shown alongside modified_linear rather than omitted.

## D28 — Adversarial review round: data-tracing corrections
Every headline number recomputed independently traced to results/*.jsonl. Corrections:
 - QUOTE ATTRIBUTION: the Angelini quote used in the deliverables is verbatim from the
   arXiv LISTING ABSTRACT of 2206.13211 (on disk: refs/abs_2206.13211.html), including
   "The greedy algorithm is faster by a factor of 10^4 with respect to the GNN for
   problems with a million variables." The PDF's own abstract (refs/critique.txt) differs
   ("...much better quality than the GNN in a much shorter time", with the 10^4 claim
   appearing in the body as "larger than a factor 10^4"). We cited the journal version
   while quoting the arXiv abstract — attribution now names the arXiv abstract.
 - ESCAPE DOUBLE-COUNT at d=7: collapse.jsonl (seeds 0-2) and escape.jsonl (seeds 0-9)
   contain the SAME three runs (bit-identical losses); pooling gave 3/13 = 0.23. De-dup
   by (d, seed): 2/10 = 0.20. D18's d=7 row is corrected accordingly; the transition
   location is unaffected. All loaders now de-duplicate.
 - ms/epoch cells at n=1e3 and n=1e4 were last-seed values (dict overwrite bug in
   scaling_rows), not seed means: 1.13 -> 1.10, 7.95 -> 7.98 (max effect 2.7%). Fixed to
   means.
 - Claim A card said "327,659x faster" — that is the best measured cell (d=3, n=1e5),
   now stated as "up to".
 - D20's notebook prose contained a hand-written "0.008 s = 272,000x"; measured DGA
   seeds at n=1e5 are 0.0064-0.0070 s (mean 0.00664), ratio 327,659x. The generated
   deliverables were always correct; the D20 prose figure was sloppy and is retracted.
 - Cosmetics: projection ratio reported at false precision (now ~2.4 million x); the
   header "four papers" did not match the reference list (now "a paper, two Comments,
   two Replies"); the mean-field table now names its objective (modified linear
   diagonal); the tuning table now carries the d=20 density note.

## D29 — Erratum to D27: the below-L* count is 4 of 6, not 5 of 6
Recounted from results/posthoc.jsonl at d=20:
    modified_linear        : -12.4553, -12.5243, -12.5164  -> 2/3 strictly below -12.500
    modified_linear_lr1e-3 : -12.4753, -12.5444, -12.5370  -> 2/3 strictly below -12.500
D27 wrote "5 of 6", adopting the review agent's tally, whose "3/3 for lr1e-3" was wrong
(seed 0 sits above L*; the agent's reported MEAN of -12.5189 was correct). The argument is
unchanged — any loss strictly below L* is impossible for a uniform configuration, and four
independent seeds produce one. The generated deliverables never carried the 5-of-6 figure
(report and REVIEW say "two of the three seeds" for the arm they discuss, which is correct).

## D30 — Presentation-only motion pass on report.html (no data or prose-claim changes)
Entrance motion added to the findings page via code/build_report.py + code/svgchart.py:
scroll reveals (opacity + translateY(12px), 350 ms, cubic-bezier(0,0,.2,1)), hero load
entrance (0/60/140 ms stagger), verdict-card 70 ms stagger + 150 ms hover lift, chart
line draw-in (pathLength=1 + stroke-dashoffset, 800 ms) with dot pop (250 ms, 600 ms
delay), 150 ms table-row hover. All motion is gated behind BOTH prefers-reduced-motion:
no-preference AND an html.anim class set by script — no-JS and reduced-motion users get
the fully static page. transform/opacity only, exact transition properties, no layout
properties animated, no will-change.
Implementation note: IntersectionObserver callbacks never fire after load in the local
preview (the doctype-less file renders in quirks mode; the published Artifact gets a
doctype). Reveals therefore use a rAF-throttled scroll handler with a top-crossed-the-line
condition — an element scrolled PAST must also reveal, which an in-viewport-only
condition silently fails (first attempt left passed elements invisible; verified fixed:
52/52 reveal, none stranded).

## D31 — Writing-review round: two factual errors corrected on the live page
An academic-writing reviewer found two errors that had shipped:
 - RUN COUNT. report.html claimed "442 measured runs" while REVIEW.md said "439 records".
   Neither stated its convention. Recounted: 445 raw lines across results/*.jsonl, minus 3
   pstats records (instrumented RE-RUNS of cells already counted, not new measurements),
   minus 3 duplicated (d=7, seed 0-2) cells shared by collapse.jsonl and escape.jsonl and
   de-duplicated at load per D28 = **439 distinct measured runs**. That is now the single
   convention in all deliverables, with the reasoning in a code comment.
 - "FOUR PAPERS". D28 retired this in the page header ("a paper, two Comments and two
   Replies") but the Control section still opened "None of the four papers establishes...",
   and REVIEW.md repeated it. The exchange is FIVE documents. Corrected in both.
Both errors were live at gnnvsgreedy.vercel.app and are fixed in the redeploy.
The same review produced a structural revision plan (IMRaD skeleton, drafted abstract,
contributions list, limitations paragraph, figure/table numbering) which is advisory for a
future paper version and is NOT applied to report.html — the page is a findings artifact,
not the paper, and its prose is AI-written (see notebook/AI-USE-LOG.md).
One substantive gap it identified and we accept: the claim under test says "existing
solvers" PLURAL, but only greedy is benchmarked here. D2 records stronger published
baselines on the same metric (MCMC/SA AR 0.984 at d=3, BP+reinforcement 0.987; Angelini &
Ricci-Tersenghi, PRE 100, 013302 (2019)) which appear nowhere in report.html. Reporting
them from the literature would strengthen the refutation — PI-GNN is not merely below
greedy but ~7 points below the published state of the art. Deferred, not dismissed.

## D32 — PRIOR ART discovered; the novelty claim must be narrowed (verified, not relayed)
A venue-research review surfaced three papers published AFTER the NMI exchange. All three
abstracts were fetched and read directly (not taken on the reviewer's word):
 - Bother, Kissig, Taraz, Cohen, Seidel & Friedrich, ICLR 2022, arXiv:2201.10494,
   "What's Wrong with Deep Learning in Tree Search for Combinatorial Optimization":
   for Li et al.'s MIS guided tree search, "the graph convolution network ... does not learn
   a meaningful representation of the solution structure, and can in fact be replaced by
   random values." => THE NULL-CONTROL IDEA IS PRIOR ART, for a different method, same
   problem. Our contribution narrows to: the first such control for PI-GNN specifically,
   with a quantitative floor.
 - Ichikawa, NeurIPS 2024, arXiv:2309.16965, "Controlling Continuous Relaxation for
   Combinatorial Optimization": names both of our failure modes for UL-based solvers —
   "(I) an optimization issue, where UL-based solvers are easily trapped at local optima,
   and (II) a rounding issue, where UL-based solvers require artificial post-learning
   rounding." => our mechanism is SUBSTANTIALLY ANTICIPATED in kind.
 - Krutsky, Sir, Kungurtsev & Korpas, ECAI 2025, arXiv:2507.13703, "Binarizing
   Physics-Inspired GNNs for Combinatorial Optimization": "the performance of PI-GNNs
   systematically plummets with an increasing density of the combinatorial problem graphs.
   Our analysis reveals an interesting phase transition in the PI-GNNs' training dynamics,
   associated with degenerate solutions for the denser problems, highlighting a discrepancy
   between the relaxed, real-valued model outputs and the binary-valued problem solutions."
   => our degree-collapse finding is LARGELY ANTICIPATED.
Consequence: the honest novelty claim is now (a) the closed-form mean-field prediction
(the above report an informal/qualitative account, not L* = -n/(2Pd) fixed before
measurement), (b) the null control applied to PI-GNN with a measured floor, and (c) the
LOCATION of the transition — see D34. Any write-up that does not cite and position against
these three will be rejected on novelty. Nothing here is retracted; the framing is.

## D33 — Peer-review round: findings accepted, and one hole closed by experiment
A simulated hostile peer review (TMLR class, "major revision", confidence 4/5) raised:
 - THE NULL CONTROL IS NOT "INFORMATION-FREE" — ACCEPTED AND IMPORTANT. On the all-ones
   input every node has conflict degree = graph degree, so repair.c reduces to ITERATED
   MAXIMUM-DEGREE VERTEX DELETION, a classical MIS heuristic. The bitstring carries no
   information; the pipeline applied to it reads the graph. The published phrasing "zero
   information" is exploitable by a respondent ("your null control is a greedy algorithm
   wearing a costume").
   THE FIX MAKES THE RESULT STRONGER and needs no new compute. Measured decomposition of
   raw vs post-processed size:
       pignn d=3  n=1e3 : raw 417   -> final 418    (post-processing adds  0.3%)
       pignn d=3  n=1e5 : raw 41321 -> final 41516  (adds  0.5%)
       pignn d=5  n=1e3 : raw 332   -> final 340    (adds  2.2%)
       pignn d=20 n=1e3 : raw 0     -> final 135    (adds  100%)
   i.e. where the GNN works, the post-processing contributes ~nothing, so the "you gave it
   an unfair maximalization" objection cannot be made; where the GNN fails, the score is
   100% post-processing and the reported "PI-GNN" number at d>=8 is a relabelled first-fit
   greedy. This decomposition replaces the "zero information" framing.
 - "p = 0.00 for d >= 8 through 25" IS NOT SUPPORTED at 3 seeds (Wilson 95% CI on 0/3 is
   [0.00, 0.56]). ACCEPTED. Seeds added (D34) and Wilson intervals will be reported.
 - Phase-1 contains 5 unused published-config n=1000 runs at each of d=3 and d=5 that were
   never pooled into the escape table. ACCEPTED: pooling gives 8/8 at both, tightening the
   low-degree end at zero cost.
 - refs/ HOLDS NO FULL TEXT of Schuetz et al. and NEITHER REPLY. ACCEPTED as a real
   exposure: "undisclosed operating ceiling" and "the Replies' central defence is untuned"
   are claims about documents not in hand. Must be obtained and verified before publication.
 - Two per-epoch costs at n=1e6 exist and disagree: probe 47,387.7 ms vs budgeted run
   1503.09 s / 36 epochs = 41,752 ms (12% lower). Report as a RANGE (~6.4-8.1 days across
   plausible epoch counts), not a point estimate.
 - D12's device rule was never executed at n=1e6, the one cell where it could matter.
   ACCEPTED as an unexecuted protocol step; must be run or its failure recorded.

## D34 — The n=1e6 collapse concern, tested and REFUTED by experiment
The reviewer observed that the budgeted n=1e6 trace shows raw output decaying geometrically
(137,531 -> 220 over 36 epochs, halving every ~3) with the loss still POSITIVE (+81,089),
i.e. best_bits still all-zeros — which looks exactly like the d>=8 collapse, and would make
the 7.31-day figure a projection of time-to-EMPTY-SET rather than time-to-convergence.
Discriminating experiment (code/early_trace.py, code/find_escape_epoch.py):
   n=1e3 d=3: raw 823 -> 780 over 60 epochs (ratio 0.95)
   n=1e4 d=3: raw 2008 -> 1122 over 36 epochs (0.56); traced to 3000 epochs, the raw set
              reaches a MINIMUM OF 8 NODES AT EPOCH 450, then RECOVERS to 4,097 by epoch
              3000, with the loss crossing zero near epoch ~1000.
   n=1e6 d=3: raw 137,531 -> 220 over 36 epochs (0.0016).
CONCLUSION: descend-to-near-zero-then-recover is the NORMAL trajectory at a size where
escape is known to occur. At n=1e4 the minimum sits at epoch 450 of ~15,000 (3% in); the
n=1e6 run was stopped at epoch 36 of ~13,321 (0.27% in), i.e. far before even the minimum.
The trajectory is therefore consistent with the normal early transient and the projection
stands. NOTE HONESTLY: the decay RATE is strongly n-dependent (0.95 / 0.56 / 0.0016 over
comparable windows), which is not explained here and is worth reporting as an open
observation rather than smoothing over.

## D35 — Escape sweep strengthened; the d=10 contradiction with ECAI 2025 is now firm
Added 10-seed coverage at d=10,11,12,13,15 (code/escape_d10plus.py) and pooled the five
unused published-config n=1000 runs at each of d=3 and d=5 that already sat in phase1.jsonl
(D33). Final escape probabilities with Wilson 95% intervals, n=1000, published config:
    d=3   8/8   1.00 [0.68,1.00]      d=11  0/10  0.00 [0.00,0.28]
    d=4  10/10  1.00 [0.72,1.00]      d=12  0/10  0.00 [0.00,0.28]
    d=5   8/8   1.00 [0.68,1.00]      d=13  0/10  0.00 [0.00,0.28]
    d=6   9/10  0.90 [0.60,0.98]      d=14  0/3   0.00 [0.00,0.56]
    d=7   2/10  0.20 [0.06,0.51]      d=15  0/10  0.00 [0.00,0.28]
    d=8   0/10  0.00 [0.00,0.28]      d=16  0/3   0.00 [0.00,0.56]
    d=9   0/10  0.00 [0.00,0.28]      d=18  0/3   0.00 [0.00,0.56]
    d=10  0/10  0.00 [0.00,0.28]      d=20  0/8   0.00 [0.00,0.32]
                                      d=25  0/3   0.00 [0.00,0.56]
POOLED d>=8: **0/90, Wilson 95% CI [0.000, 0.041]** — this is the statistic to report; no
single degree row can carry the plateau claim, and the pooled one can. 136 escape-sweep
runs in total.
CONTRADICTION WITH PUBLISHED WORK, now firm: Krutsky et al. (ECAI 2025, arXiv:2507.13703)
locate the transition between d=10 and d=20 and report non-collapsed behaviour for d<=10.
We measure 0/10 at d=10 in the published configuration at n=1000. This is a direct
quantitative disagreement and is the study's sharpest genuinely-new claim (D32). It is NOT
resolved here: the likely explanation is a configuration difference (n, epochs, embedding
width, threshold rule) rather than an error on either side, and establishing which would
require running their configuration. Reported as a disagreement, not as a refutation.

## D36 — MY PRE-REGISTERED TUNING ARM DOES NOT TEST THE REPLY'S ACTUAL DEFENCE
The five documents were finally obtained in full (refs/*.pdf + .txt: schuetz_pignn_FULL,
reply_to_angelini = arXiv:2302.03602, reply_to_boettcher = arXiv:2303.12096, plus the three
prior-art papers). Reading the Reply to Angelini invalidates the premise of D6/D11.
I assumed the Replies' central defence was "the critics used an untuned network", and built
a pre-registered grid of lr in {1e-4,3e-4,1e-3,1e-2} x penalty in {2,3} against it.
THE ACTUAL DEFENCE IS TWO SPECIFIC CHANGES, quoted verbatim from the Reply:
  1. **GraphSAGE instead of GCN** — "In addition to the GCN-based results (dark, as
     previously reported) numerical (average) results are shown for a GraphSAGE
     architecture (light) ... GraphSAGE consistently provides slightly larger independent
     sets."
  2. **Penalty P = 10 instead of P = 2** — "by simply setting P = 10 we find even further
     consistent improvements. Specifically, for d = 3 at P = 10 GraphSAGE achieves
     approximation ratios of AR ~ 0.947, on par with the DGA-based results reported by
     Angelini and Ricci-Tersenghi."
  They conclude: "Given the improvements seen already with these two simple tweaks, we
  emphasize that the claims made in the comment do not hold in generality."
CONSEQUENCE: my penalty grid CAPPED AT 3 and I never implemented GraphSAGE at all. The
claim in report.html and REVIEW.md that "the objection is answered on its own terms" is
FALSE and must be retracted. The pre-registration was honest but aimed at the wrong target;
pre-registering against an assumed defence rather than the documented one is the error.
ALSO from the Reply, bearing on Claim B: they report an UPDATED post-processing whose
scaling improves from ~n^2.0 to ~n^1.0, and total runtime from ~n^1.7 to ~n^0.8. Our
timing analysis measured the ORIGINAL released implementation, not this updated one.
ALSO: the Reply dismisses the d>16 benchmark not on capability grounds but on relevance —
"While this is certainly an interesting academic exercise, we are not convinced about its
practical usefulness" because real-world graphs are not d-regular. That is a different
objection from the one our operating-ceiling result answers, and should be quoted as such.
CORRECTED (D28/peer-review item): the full paper contains ZERO occurrences of "dense" and
does not describe the QUBO construction used at n=1e6. The 3.6 TiB dense-Q wall is a
property of the RELEASED EXAMPLE NOTEBOOK (refs/pignn/utils.py), not of a published method
description. All wording must narrow to the released example implementation.
CONFIRMED: the paper's own MIS/MaxCut large-scale experiments use d = 3 and d = 5 only, so
the operating-ceiling claim (no non-empty output for d >= 8) does concern a regime the
paper never tested. That claim survives.

## D37 — Testing the Reply's ACTUAL defence (GraphSAGE + P=10): partial, honestly split
Implemented SAGELayer (mean aggregator) in code/pignn.py and ran a 2x2 of
{GCN, GraphSAGE} x {P=2, P=10}, n=1000, seeds 0-2 (code/reply_defence.py):

              d=3 AR    d=5 AR    d=20 non-empty
  gcn_P2      0.9223    0.8818    0/3
  gcn_P10     0.9289    0.8073    0/3
  sage_P2     0.8513    0.8151    0/3
  sage_P10    0.8777    0.8255    0/3
  DGA         0.9500    0.9182    (rho 0.170)

THREE FINDINGS, of different strengths:
1. STRONG — **the Reply's own configuration does not rescue the high-degree collapse.**
   All four arms, including GraphSAGE + P=10, return the empty set in 3/3 runs at d=20.
   The operating ceiling (D18/D35) survives the defence its authors published.
2. MODERATE — P=10 with GCN improves d=3 slightly (0.9223 -> 0.9289) in the direction the
   Reply claims, but remains below DGA's 0.9500. At d=5, P=10 makes it markedly WORSE
   (0.8818 -> 0.8073, and only 1/3 runs non-empty), which the Reply does not mention.
3. UNRESOLVED, AND WE DO NOT CLAIM OTHERWISE — the Reply states "for d = 3 at P = 10
   GraphSAGE achieves approximation ratios of AR ~ 0.947". Our GraphSAGE + P=10 reaches
   0.8777, far below both their figure and our own GCN. We therefore do NOT reproduce their
   headline number, and the honest reading is that OUR GraphSAGE is not equivalent to
   theirs (aggregator variant, normalization, or hyperparameters we cannot recover from the
   Reply's text, which gives no architecture detail and no code). This is recorded as a
   failure to reproduce a published claim, NOT as a refutation of it. Resolving it needs
   their implementation.
Consequence for the write-up: replace "the objection is answered on its own terms" (now
retracted per D36) with the precise, defensible statement — the published defence was
tested as described; it does not restore parity at d=3 or d=5 in our hands, we could not
reproduce its GraphSAGE figure, and it does not affect the d>=8 collapse at all.
