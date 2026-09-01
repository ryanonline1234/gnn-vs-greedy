# Protocol — adjudicating the PI-GNN claim

## The dispute
| Paper | Position |
|---|---|
| Schuetz, Brubaker & Katzgraber, *Nat Mach Intell* **4**, 367 (2022) | PI-GNN "performs on par or outperforms existing solvers, with the ability to scale beyond the state of the art to problems with millions of variables." |
| Angelini & Ricci-Tersenghi, *Nat Mach Intell* **5**, 29 (2023) | A degree-based greedy finds "much better quality" solutions, "faster by a factor of 10^4" at n=10^6. |
| Boettcher, *Nat Mach Intell* **5**, 24 (2023) | Concurrent critique on MaxCut. |
| Two formal Replies | Concede nothing; the central defense is that the network was not tuned. |

## What is measured
The validity of a published claim against an exact objective. Not model performance.
The claim splits into two independently scored parts:
- **A (parity):** on par with or better than existing solvers.
- **B (scale):** reaches ~10^6 variables.
They are never averaged. B can hold while A fails.

## Instances
Uniform random d-regular graphs (igraph `K_Regular`), verified simple and exactly
d-regular at generation. d in {3, 5, 20} for the main sweep; d in {3,5,7,10,12,14,16,18,20,25}
for the collapse sweep. n in {10^3, 10^4, 10^5, 10^6}. Seeds control graph, network init,
and the random-greedy tie-break together.

## Methods
| id | what |
|---|---|
| `dga` | degree-based greedy — the critique's champion; C, bucket queue, near-linear |
| `ga` | random greedy (Karp–Sipser style) |
| `pignn_pub` | PI-GNN at the exact published configuration |
| `pignn_tuned` | PI-GNN, pre-registered tuning budget (D11) |
| `null_ones` | all-ones bitstring through the GNN's own post-processing |
| `null_rand` | fair-coin bitstring through the same |
| `modified_linear` | PI-GNN with the diagonal changed from −Σp² to −Σp (post-hoc, labelled) |

## Metric
Approximation ratio `AR = (|IS|/n) / rho_UB(d)`, matching the critique so numbers are
directly comparable. `rho_UB(3) = 0.45537`, `rho_UB(5) = 0.38443` (McKay 1987).
1RSB optima: `AR = 0.990` (d=3), `0.987` (d=5) (Barbier et al. 2013).
No published `rho_UB` is adopted at d=20; raw density is reported there.
Every reported set is verified independent and maximal by an independent checker.

## Timing
Same host (Apple M1 Pro, 32 GB), same instances, same seeds, one job at a time.
Graph loading excluded for all methods; data-structure construction included for all.
GNN reports build / train / post-processing separately so the amortization argument can
be evaluated rather than assumed.

This is the part the published debate does not have: the critique's 10^4 figure compares
DGA timed on the authors' own laptop against GNN times *read off Figure 5 of the paper
under test*, i.e. different hardware. Nothing here is cross-machine.

## Reproduce
```
./pyrt/bin/python code/test_port.py                       # port equivalence proof
./pyrt/bin/python code/runner.py --out results/phase1.jsonl \
    --d 3 5 20 --n 1000 10000 --seeds 0 1 2 3 4 --device cpu
./pyrt/bin/python code/collapse_sweep.py                  # collapse threshold in d
./pyrt/bin/python code/sweep_tuned.py --out results/tuning.jsonl --d 3 5 20 --n 1000 --seeds 0 1 2
./pyrt/bin/python code/posthoc_d20.py                     # escape attempts
./pyrt/bin/python code/scale_test.py --n 1000000 --d 3    # claim B
./pyrt/bin/python code/analyze.py
```
