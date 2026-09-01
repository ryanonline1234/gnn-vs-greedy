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

This repository re-runs both sides: same host, same instances, same seeds, each method
built from its published source. **483 measured runs** on an Apple M1 Pro.

## What it finds

- **Replication.** The Comment's figures reproduce on independently written code
  (d=3, n=10⁴: DGA 0.9502, PI-GNN 0.9138, against published ≈0.95 / ≈0.92).
- **A decomposition nobody in the exchange reports.** Separating the network's raw output
  from its post-processing: at d=3 the post-processing adds **0.3%**, so the gap to greedy
  is the network's own; at d=20 it adds **100%**, because the raw output is empty and the
  reported score comes entirely from a first-fit pass.
- **An operating ceiling.** Probability of a non-empty output falls from 1.0 at d≤5 to 0.20
  at d=7, and is **0/90 pooled across every d≥8 tested** (Wilson 95% CI [0.000, 0.041]) —
  below the d>16 clustering transition proposed as the genuinely hard regime.
- **A mechanism, predicted before measurement.** The relaxed objective's uniform optimum is
  L\* = −n/(2Pd) = −12.500 at d=20, n=1000; the measured loss is −12.499, and the output
  vector stays near p\* = 0.025 with 27–29% scatter, never approaching the 0.5 threshold.
- **The published defence, tested as written.** GraphSAGE with P=10 — the configuration the
  Reply proposes — still returns the empty set in every run at d=20.

## What it does not claim

- Maximum Independent Set only. Boettcher's Comment targets MaxCut; that is untested here.
- It does **not** show graph neural networks cannot do combinatorial optimisation. It shows
  this relaxation, with this projection, does not.
- Substantial prior art exists and is credited in `DECISIONS.md` D32:
  [Böther et al. ICLR 2022](https://arxiv.org/abs/2201.10494) (the null-control idea, for a
  different method), [Ichikawa NeurIPS 2024](https://arxiv.org/abs/2309.16965) (the failure
  modes), and [Krutský et al. ECAI 2025](https://arxiv.org/abs/2507.13703) (the density
  transition). This work's degree result **disagrees** with the last of these at d=10, and
  that disagreement is unresolved.
- The GraphSAGE+P=10 arm does **not** reproduce the Reply's reported AR ≈ 0.947 (measured
  0.8777). That is recorded as a failure to reproduce, not a refutation — see D37.

## Start here

| File | Why |
|---|---|
| [`review/REVIEW.md`](review/REVIEW.md) | Orientation, verdicts, and a ranked list of where to attack the work |
| [`DECISIONS.md`](DECISIONS.md) | Append-only log, D1–D37: every design choice, and every correction — including the ones against my own conclusions |
| [`code/test_port.py`](code/test_port.py) | The keystone. Run it first: it proves the PI-GNN port's forward mathematics matches the reference |
| [`review/DATA-APPENDIX.md`](review/DATA-APPENDIX.md) | Every measured configuration |
| [`notebook/AI-USE-LOG.md`](notebook/AI-USE-LOG.md) | **Provenance.** Which parts were AI-written |

## Reproducing

Create a Python 3.12 virtual environment and activate it, then:

```bash
pip install torch numpy scipy igraph pandas pyarrow matplotlib highspy pypdf
cc -O3 -march=native -o code/greedy code/greedy.c
cc -O3 -march=native -o code/repair code/repair.c
python code/fetch_refs.py        # downloads the papers (not redistributed here)
python code/test_port.py         # equivalence proof — run this first
python code/runner.py --out results/phase1.jsonl --d 3 5 20 --n 1000 10000 --seeds 0 1 2 3 4
python code/analyze.py
```

`data/` is not committed: instances are regenerated exactly from their seeds by
`code/gen_graphs.py`, which every driver calls automatically. `refs/` is not committed
either — those are copyrighted papers; `code/fetch_refs.py` retrieves them from arXiv.

## Provenance

The code in `code/` was written with AI assistance (Claude), directed and reviewed by the
author; the measurements are the author's own, produced by running it. The prose in
`report.html` is AI-written and is **not** suitable for reuse in any submission where
authorship is attested. See [`notebook/AI-USE-LOG.md`](notebook/AI-USE-LOG.md) for a
file-by-file record.

When fetched, `refs/pignn/` is the authors' own reference implementation,
[amazon-science/co-with-gnns-example](https://github.com/amazon-science/co-with-gnns-example),
Apache-2.0. The port in `code/pignn.py` is an independent reimplementation whose equivalence
to it is proven by `code/test_port.py`, not assumed.

## Licence

Code and data: MIT (see `LICENSE`). The third-party papers this analyses are not included
and remain under their authors' terms.
