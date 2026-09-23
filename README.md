# Adjudicating PI-GNN vs. degree-based greedy for Maximum Independent Set

An independent re-run, on one machine, of a dispute in *Nature Machine Intelligence* that
two Comments and two Replies left unresolved.

**Live summary: https://gnnvsgreedy.vercel.app**

Schuetz, Brubaker & Katzgraber ([NMI 4, 367 (2022)](https://arxiv.org/abs/2107.01188))
reported that a physics-inspired graph neural network (PI-GNN) *"performs on par or
outperforms existing solvers"* for Maximum Independent Set and scales to 10⁶ variables.
Angelini & Ricci-Tersenghi ([NMI 5, 29 (2023)](https://arxiv.org/abs/2206.13211)) and
Boettcher (NMI 5, 24 (2023)) disputed it. The authors replied
([1](https://arxiv.org/abs/2302.03602), [2](https://arxiv.org/abs/2303.12096)) conceding
nothing.

**Bottom line.** Degree-based greedy (DGA, the Comment's baseline) finds larger independent
sets than PI-GNN in every cell tested, and in the released implementation's configuration the
network's raw output is non-empty in **0/87** runs pooled over every degree d ≥ 8 tested
(n = 1000) — an operating ceiling the paper never probed, not yet tested under the paper's
own ∛n embedding rule (D38). **This study has also withdrawn two of its own claims:** that its
tuning sweep answered the authors' defence, which it did not test ([D36](DECISIONS.md)) — the
defence was then run as the Reply describes it ([D37](DECISIONS.md)) — and a claimed
disagreement with Krutský et al. that compared against their MaxCut rather than their MIS
results ([D38](DECISIONS.md), which also lists the numerical errata).

This repository re-runs both sides: same host, same instances, same seeds. PI-GNN is a port of
the authors' reference code (D4, D26); DGA is written in C from the Comment's description
(D8). The Reply's GraphSAGE variant has no released code or architecture detail, so it was
reconstructed from the Reply's text (D37). **534 measurement records** on an Apple M1 Pro.

## What it finds

- **Replication.** The Comment's figures reproduce on independently written code. On
  uniform random d-regular graphs (d = degree) at d = 3, n = 10⁴ nodes, 5 seeds each, the
  approximation ratio (AR = independent-set density ÷ McKay's upper bound) is DGA 0.9502,
  PI-GNN 0.9138, against the Comment's published ≈0.95 / ≈0.92.
- **A decomposition nobody in the exchange reports.** Separating the network's raw output
  from its post-processing: where the network works (d = 3, 5) post-processing adds
  **0.3–2.2%**, so the gap to greedy is the network's own; at d = 20 the entire score is
  post-processing, because the raw output is empty and the reported set comes from a
  first-fit pass.
- **An operating ceiling.** Probability of a non-empty raw output falls from 1.0 at d≤5 to
  0.20 at d=7, and is **0/87 pooled across every d≥8 tested** (Wilson 95% CI [0.000, 0.042]),
  n = 1000 — below the d>16 clustering transition proposed as the genuinely hard regime. This
  is the released implementation's configuration (embedding width d0 = int(√n) at every n);
  the paper's text gives d0 = int(∛n) below n = 10⁵, and whether the ceiling holds under
  that setting is untested (D38). (It also uses patience 100 where the paper states 10³; a
  patience-5,000 arm is also empty at d=20, D17.)
- **A closed-form mechanism, checked against measurement.** Under a modified, post-hoc
  objective (linear diagonal, so the empty set is no longer a critical point) the uniform
  optimum is L\* = −n/(2Pd) = −12.500 at d=20, n=1000; the measured loss is −12.499 (3-seed
  mean), and the output vector stays near p\* = 0.025 with 27–29% scatter, never approaching
  the 0.5 threshold (D17, D27). Under the published objective (quadratic diagonal) the
  uniform optimum is p\* = 0, L\* = 0, so being trapped at the mean field shows up as an
  empty raw output.
- **The published defence, tested as written (D37).** GraphSAGE with P=10, the configuration
  the Reply proposes, does not restore parity with greedy where the Reply makes its claim:
  AR 0.8777 vs DGA 0.9479 at d=3 and 0.8255 vs 0.9148 at d=5 (n=1000, same seeds 0–2). At
  d=20 its raw output is empty in 3/3 runs (12/12 across all four GCN/GraphSAGE × P=2/P=10
  arms). The defence was tested at d = 3, 5 and 20 only, not at every d ≥ 8. The Reply sets
  d>16 aside on relevance rather than capability ("not convinced about its practical
  usefulness").
- **Scale (Claim B): a labelled projection.** The released example notebook builds a dense n×n
  QUBO (3.6 TiB at n = 10⁶); the paper never describes its n = 10⁶ construction, and a
  mathematically identical sparse reformulation is what makes the test possible at all. A
  budgeted n = 10⁶ run was stopped by its wall-clock budget after 36 epochs at AR 0.8229 —
  below the all-ones control's 0.8680 and DGA's 0.9508 in 0.27 s — which is a lower bound, not
  evidence of non-convergence. A full run projects to ~6.4–8.1 days, roughly 2.1–2.6 million×
  DGA's time (D33, D38).

## What it does not claim

- Maximum Independent Set only. Boettcher's Comment targets MaxCut; that is untested here.
- It does **not** show graph neural networks cannot do combinatorial optimisation. It shows
  this relaxation, with this projection, does not.
- Substantial prior art exists and is credited in `DECISIONS.md` D32:
  [Böther et al. ICLR 2022](https://arxiv.org/abs/2201.10494) (the null-control idea, for a
  different method), [Ichikawa NeurIPS 2024](https://arxiv.org/abs/2309.16965) (the failure
  modes), and [Krutský et al. ECAI 2025](https://arxiv.org/abs/2507.13703) (the density
  transition). Their MIS tables show the baseline collapsed at d=10, consistent with the 0/10
  here. An earlier version of this repository claimed a disagreement at d=10; it compared
  against their **MaxCut** results and is withdrawn (D38). What this study adds on that axis is
  resolution: its degree grid places the MIS transition at d≈6–7, between their grid points
  d=5 and d=10.
- The GraphSAGE+P=10 arm does **not** reproduce the Reply's reported AR ≈ 0.947 (measured
  0.8777). That is recorded as a failure to reproduce, not a refutation — see D37.
- The timings are of the originally released implementation. The Reply reports an updated
  post-processing (scaling ~n^2.0 → ~n^1.0; total runtime ~n^1.7 → ~n^0.8) that is not
  measured here (D36). The n = 10⁶ projection counts training epochs only, so it does not
  depend on post-processing.

## Start here

| File | Why |
|---|---|
| [`review/REVIEW.md`](review/REVIEW.md) | Orientation, verdicts, and a ranked list of where to attack the work |
| [`DECISIONS.md`](DECISIONS.md) | Append-only log, D1–D38: every design choice, and every correction — including the ones against my own conclusions |
| [`code/test_port.py`](code/test_port.py) | The keystone. Run it first: it checks that the PI-GNN port's forward mathematics (QUBO loss, GCN layer) matches the reference's, re-derived inline |
| [`review/DATA-APPENDIX.md`](review/DATA-APPENDIX.md) | Every measured configuration |
| [`notebook/AI-USE-LOG.md`](notebook/AI-USE-LOG.md) | **Provenance.** Which parts were AI-written |

## Reproducing

Create a virtual environment and activate it, then:

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
# collapse_sweep.py, escape_prob.py, escape_d10plus.py and reply_defence.py take no --out: each
# appends to a fixed file in results/ and skips cells already there, so move that file aside first.
# mv -n never overwrites an earlier move-aside; the published files are tracked by git, so
# `git checkout -- results/` restores them.
mkdir -p results.published
mv -n results/collapse.jsonl results/escape.jsonl results.published/
python code/collapse_sweep.py && python code/escape_prob.py && python code/escape_d10plus.py   # escape table: 121 of its 127 runs (the other 6 are phase-1 runs, re-measured by runner.py above)
mv -n results/reply_defence.jsonl results.published/ && python code/reply_defence.py             # the Reply's defence, GraphSAGE + P=10 (D37)
# Open (D38), not yet run: the ceiling under the paper's d0 = int(∛n) at n = 1000
python code/runner.py --out results/cbrt_d0.jsonl --methods pignn_pub --tag pignn_cbrt --dim-embedding 10 --n 1000 --d 3 5 8 12 20 --seeds 0 1 2 3 4
```

`data/` is not committed: instances are regenerated exactly from their seeds by
`code/gen_graphs.py`, which every driver calls automatically (igraph's seeded sampler, so
keep the pinned igraph version). `refs/` is not committed either: the papers are
copyrighted, and `code/fetch_refs.py` retrieves them from arXiv and clones the reference
implementation into `refs/pignn/`. `code/test_port.py` does not need `refs/`.

## Provenance

Every file in `code/` was written by Claude (Anthropic) at the author's direction; the
measurements are the author's own, produced by running that code on his machine. The prose
in `report.html`, `site/index.html`, `review/` and this README is AI-written and is **not**
suitable for reuse in any submission where authorship is attested. See
[`notebook/AI-USE-LOG.md`](notebook/AI-USE-LOG.md) for the provenance record.

`refs/pignn/` (cloned by `code/fetch_refs.py`) is the authors' own reference implementation,
[amazon-science/co-with-gnns-example](https://github.com/amazon-science/co-with-gnns-example),
Apache-2.0. The port in `code/pignn.py` is an independent reimplementation, tested rather
than assumed: `code/test_port.py` checks its forward mathematics (the QUBO loss and the GCN
layer) against the reference's formulation, re-derived inline. The training loop and early
stopping rest on a line-by-line comparison with the reference and 6/6 fresh reruns that
matched the recorded results (D26); training is not bit-deterministic, so re-executing a
seeded cell can shift its final loss and epoch count slightly (D38). One disclosed deviation: the port
reports the larger of the best-loss and final bitstrings where the reference reports
best-loss only, which favours PI-GNN and was inert in all 6 audit reruns (D26). The test's
normalisation check is vacuous on d-regular graphs (D26), and the GraphSAGE layer has no
reference implementation, so `test_port.py` does not test it.

## Licence

Code and data: MIT (see `LICENSE`). The third-party papers this analyses are not included
and remain under their authors' terms.
