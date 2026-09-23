# DATA APPENDIX

Every measured configuration in `results/`: 534 measurement records across 11 files (collapse 30 · early_trace 3 · escape 100 · exact 30 · phase1 150 · phase2 37 · posthoc 63 · pstats 3 · reply_defence 36 · scale 10 · tuning 72). Some cells were executed by more than one driver (e.g. n=1000, d ∈ {3, 5, 20}, seeds 0–2), so records are not distinct runs; each table states its own de-duplication rule.
Mean ± s.e.m. over seeds where more than one seed was run.
Host: Apple M1 Pro, 10 core, 32 GB. Greedy baselines in C (`-O3 -march=native`); PI-GNN in PyTorch 2.13.0 on CPU.

Approximation ratio AR = density / ρ_UB(d), with ρ_UB(3)=0.45537, ρ_UB(5)=0.38443 (McKay 1987). No ρ_UB is adopted for d=20.

`pignn_pub` = PI-GNN at the released implementation's configuration: embedding size d0 = int(√n) at every n, as in the authors' released example notebook. The paper's text sets d0 = int(√n) only for n ≥ 10^5 and d0 = int(∛n) below that, so at n = 1000 and n = 10^4 these runs use d0 = 31 / 100 where the text gives 10 / 21; at n ≥ 10^5 the two agree (D38).

## Main sweep — quality and time

| d | n | method | trials | density | AR | wall clock (s) | violations |
|---|---|---|---|---|---|---|---|
| 3 | 1,000 | dga | 5 | 0.43260 ± 0.00121 | 0.9500 | 4.98e-05 | 0 |
| 3 | 1,000 | ga | 5 | 0.37880 ± 0.00191 | 0.8319 | 5.34e-05 | 0 |
| 3 | 1,000 | null_ones | 5 | 0.39660 ± 0.00098 | 0.8709 | 0.000102846 | 0 |
| 3 | 1,000 | null_rand | 5 | 0.37680 ± 0.00196 | 0.8275 | 0.00425367 | 0 |
| 3 | 1,000 | pignn_pub | 5 | 0.41820 ± 0.00124 | 0.9184 | 13.7412 | 0 |
| 3 | 10,000 | dga | 5 | 0.43270 ± 0.00044 | 0.9502 | 0.0005792 | 0 |
| 3 | 10,000 | ga | 5 | 0.37442 ± 0.00072 | 0.8222 | 0.0005266 | 0 |
| 3 | 10,000 | null_ones | 5 | 0.39386 ± 0.00007 | 0.8649 | 0.000473363 | 0 |
| 3 | 10,000 | null_rand | 5 | 0.37630 ± 0.00057 | 0.8264 | 0.000724905 | 0 |
| 3 | 10,000 | pignn_pub | 5 | 0.41612 ± 0.00044 | 0.9138 | 118.277 | 0 |
| 3 | 100,000 | dga | 3 | 0.43254 ± 0.00018 | 0.9499 | 0.00664 | 0 |
| 3 | 100,000 | ga | 3 | 0.37547 ± 0.00035 | 0.8245 | 0.006074 | 0 |
| 3 | 100,000 | null_ones | 3 | 0.39588 ± 0.00017 | 0.8694 | 0.00456294 | 0 |
| 3 | 100,000 | null_rand | 3 | 0.37653 ± 0.00037 | 0.8269 | 0.018617 | 0 |
| 3 | 100,000 | pignn_pub | 1 | 0.41516 | 0.9117 | 2,175.66 | 0 |
| 5 | 1,000 | dga | 5 | 0.35300 ± 0.00114 | 0.9182 | 7.74e-05 | 0 |
| 5 | 1,000 | ga | 5 | 0.29900 ± 0.00170 | 0.7778 | 7.26e-05 | 0 |
| 5 | 1,000 | null_ones | 5 | 0.32800 ± 0.00217 | 0.8532 | 0.000130035 | 0 |
| 5 | 1,000 | null_rand | 5 | 0.31140 ± 0.00186 | 0.8100 | 0.000290801 | 0 |
| 5 | 1,000 | pignn_pub | 5 | 0.34000 ± 0.00114 | 0.8844 | 20.7144 | 0 |
| 5 | 10,000 | dga | 5 | 0.35672 ± 0.00048 | 0.9279 | 0.0007978 | 0 |
| 5 | 10,000 | ga | 5 | 0.30190 ± 0.00075 | 0.7853 | 0.0007272 | 0 |
| 5 | 10,000 | null_ones | 5 | 0.32670 ± 0.00043 | 0.8498 | 0.000718303 | 0 |
| 5 | 10,000 | null_rand | 5 | 0.30698 ± 0.00073 | 0.7985 | 0.000906086 | 0 |
| 5 | 10,000 | pignn_pub | 5 | 0.34144 ± 0.00032 | 0.8882 | 191.161 | 0 |
| 5 | 100,000 | dga | 3 | 0.35665 ± 0.00022 | 0.9277 | 0.00904167 | 0 |
| 5 | 100,000 | ga | 3 | 0.30152 ± 0.00035 | 0.7843 | 0.00877167 | 0 |
| 5 | 100,000 | null_ones | 3 | 0.32689 ± 0.00020 | 0.8503 | 0.00735271 | 0 |
| 5 | 100,000 | null_rand | 3 | 0.30695 ± 0.00035 | 0.7984 | 0.00770734 | 0 |
| 20 | 1,000 | dga | 5 | 0.17000 ± 0.00158 | — | 0.0001952 | 0 |
| 20 | 1,000 | ga | 5 | 0.13820 ± 0.00177 | — | 0.0001858 | 0 |
| 20 | 1,000 | null_ones | 5 | 0.15720 ± 0.00066 | — | 0.000268025 | 0 |
| 20 | 1,000 | null_rand | 5 | 0.14880 ± 0.00159 | — | 0.00036922 | 0 |
| 20 | 1,000 | pignn_pub | 5 | 0.13540 ± 0.00103 | — | 20.6525 | 0 |
| 20 | 10,000 | dga | 5 | 0.17324 ± 0.00023 | — | 0.0021028 | 0 |
| 20 | 10,000 | ga | 5 | 0.13956 ± 0.00032 | — | 0.0020196 | 0 |
| 20 | 10,000 | null_ones | 5 | 0.15880 ± 0.00019 | — | 0.00234027 | 0 |
| 20 | 10,000 | null_rand | 5 | 0.14752 ± 0.00051 | — | 0.00214902 | 0 |
| 20 | 10,000 | pignn_pub | 5 | 0.13956 ± 0.00072 | — | 272.733 | 0 |
| 20 | 100,000 | dga | 3 | 0.17371 ± 0.00021 | — | 0.0298723 | 0 |
| 20 | 100,000 | ga | 3 | 0.13948 ± 0.00036 | — | 0.028876 | 0 |
| 20 | 100,000 | null_ones | 3 | 0.15873 ± 0.00013 | — | 0.0301533 | 0 |
| 20 | 100,000 | null_rand | 3 | 0.14834 ± 0.00019 | — | 0.0225831 | 0 |

## Pre-registered tuning grid (Stage A, n=1000, seeds 0–2)

| d | lr | penalty | non-empty | density | AR |
|---|---|---|---|---|---|
| 3 | 0.0001 | 2 | 3/3 | 0.42000 ± 0.00100 | 0.9223 |
| 3 | 0.0001 | 3 | 3/3 | 0.42033 ± 0.00145 | 0.9231 |
| 3 | 0.0003 | 2 | 3/3 | 0.42033 ± 0.00088 | 0.9231 |
| 3 | 0.0003 | 3 | 3/3 | 0.41967 ± 0.00176 | 0.9216 |
| 3 | 0.001 | 2 | 3/3 | 0.42033 ± 0.00167 | 0.9231 |
| 3 | 0.001 | 3 | 3/3 | 0.42000 ± 0.00200 | 0.9223 |
| 3 | 0.01 | 2 | 3/3 | 0.42000 ± 0.00173 | 0.9223 |
| 3 | 0.01 | 3 | 3/3 | 0.42000 ± 0.00361 | 0.9223 |
| 5 | 0.0001 | 2 | 3/3 | 0.33900 ± 0.00153 | 0.8818 |
| 5 | 0.0001 | 3 | 3/3 | 0.34100 ± 0.00100 | 0.8870 |
| 5 | 0.0003 | 2 | 3/3 | 0.33700 ± 0.00208 | 0.8766 |
| 5 | 0.0003 | 3 | 3/3 | 0.33733 ± 0.00120 | 0.8775 |
| 5 | 0.001 | 2 | 3/3 | 0.33567 ± 0.00120 | 0.8732 |
| 5 | 0.001 | 3 | 3/3 | 0.33800 ± 0.00058 | 0.8792 |
| 5 | 0.01 | 2 | 3/3 | 0.33933 ± 0.00176 | 0.8827 |
| 5 | 0.01 | 3 | 3/3 | 0.33900 ± 0.00231 | 0.8818 |
| 20 | 0.0001 | 2 | 0/3 | 0.13500 ± 0.00058 | — |
| 20 | 0.0001 | 3 | 0/3 | 0.13500 ± 0.00058 | — |
| 20 | 0.0003 | 2 | 0/3 | 0.13500 ± 0.00058 | — |
| 20 | 0.0003 | 3 | 0/3 | 0.13500 ± 0.00058 | — |
| 20 | 0.001 | 2 | 0/3 | 0.13500 ± 0.00058 | — |
| 20 | 0.001 | 3 | 0/3 | 0.13500 ± 0.00058 | — |
| 20 | 0.01 | 2 | 0/3 | 0.13500 ± 0.00058 | — |
| 20 | 0.01 | 3 | 0/3 | 0.13500 ± 0.00058 | — |

## Post-hoc escape arms (n=1000, seeds 0–2)

| d | arm | non-empty | mean raw size | density | mean final loss |
|---|---|---|---|---|---|
| 3 | modified_linear | 3/3 | 423.3 | 0.42333 ± 0.00120 | -423.254 |
| 3 | modified_linear_lr1e-3 | 3/3 | 422.0 | 0.42200 ± 0.00153 | -421.949 |
| 3 | posthoc_combined | 3/3 | 412.0 | 0.41300 ± 0.00208 | -411.333 |
| 3 | posthoc_dim256 | 3/3 | 411.7 | 0.41267 ± 0.00186 | -411.595 |
| 3 | posthoc_lr1e-2 | 3/3 | 419.7 | 0.42000 ± 0.00173 | -417.650 |
| 3 | posthoc_lr1e-3 | 3/3 | 420.0 | 0.42033 ± 0.00167 | -417.943 |
| 3 | posthoc_patience5000 | 3/3 | 420.0 | 0.42033 ± 0.00088 | -419.332 |
| 5 | modified_linear | 3/3 | 341.3 | 0.34133 ± 0.00033 | -341.256 |
| 5 | modified_linear_lr1e-3 | 3/3 | 343.0 | 0.34333 ± 0.00088 | -342.936 |
| 5 | posthoc_combined | 3/3 | 331.3 | 0.33800 ± 0.00115 | -331.333 |
| 5 | posthoc_dim256 | 3/3 | 328.7 | 0.33800 ± 0.00115 | -328.578 |
| 5 | posthoc_lr1e-2 | 3/3 | 333.3 | 0.33933 ± 0.00176 | -331.313 |
| 5 | posthoc_lr1e-3 | 3/3 | 327.0 | 0.33567 ± 0.00120 | -326.269 |
| 5 | posthoc_patience5000 | 3/3 | 333.7 | 0.33900 ± 0.00153 | -333.664 |
| 20 | modified_linear | 0/3 | 0.0 | 0.13500 ± 0.00058 | -12.499 |
| 20 | modified_linear_lr1e-3 | 0/3 | 0.0 | 0.13500 ± 0.00058 | -12.519 |
| 20 | posthoc_combined | 0/3 | 0.0 | 0.13500 ± 0.00058 | 0.001 |
| 20 | posthoc_dim256 | 0/3 | 0.0 | 0.13500 ± 0.00058 | 0.135 |
| 20 | posthoc_lr1e-2 | 0/3 | 0.0 | 0.13500 ± 0.00058 | 0.019 |
| 20 | posthoc_lr1e-3 | 0/3 | 0.0 | 0.13500 ± 0.00058 | 0.100 |
| 20 | posthoc_patience5000 | 0/3 | 0.0 | 0.13500 ± 0.00058 | 0.009 |

## The Reply's published defence — {GCN, GraphSAGE} × P ∈ {2, 10} (n=1000, seeds 0–2; D37)

The GraphSAGE layer is this study's mean-aggregator implementation; the Reply gives no architecture detail and no reference code exists (D37). `dga` rows are phase-1 DGA on the same graphs and seeds, for a like-for-like comparison. non-empty = the network's raw output (raw_size > 0), before post-processing.

| d | arm | runs | non-empty | mean raw size | density | AR | mean final loss |
|---|---|---|---|---|---|---|---|
| 3 | gcn_P2 | 3 | 3/3 | 419.0 | 0.42000 ± 0.00100 | 0.9223 | -418.255 |
| 3 | gcn_P10 | 3 | 3/3 | 418.0 | 0.42300 ± 0.00346 | 0.9289 | -417.933 |
| 3 | sage_P2 | 3 | 3/3 | 388.0 | 0.38767 ± 0.00203 | 0.8513 | -387.250 |
| 3 | sage_P10 | 3 | 3/3 | 399.7 | 0.39967 ± 0.00410 | 0.8777 | -399.584 |
| 3 | dga | 3 | — | — | 0.43167 ± 0.00088 | 0.9479 | — |
| 5 | gcn_P2 | 3 | 3/3 | 333.3 | 0.33900 ± 0.00153 | 0.8818 | -333.238 |
| 5 | gcn_P10 | 3 | 1/3 | 104.0 | 0.31033 ± 0.01239 | 0.8073 | -103.877 |
| 5 | sage_P2 | 3 | 3/3 | 313.0 | 0.31333 ± 0.00376 | 0.8151 | -312.917 |
| 5 | sage_P10 | 3 | 3/3 | 217.3 | 0.31733 ± 0.01068 | 0.8255 | -217.226 |
| 5 | dga | 3 | — | — | 0.35167 ± 0.00145 | 0.9148 | — |
| 20 | gcn_P2 | 3 | 0/3 | 0.0 | 0.13500 ± 0.00058 | — | 0.155 |
| 20 | gcn_P10 | 3 | 0/3 | 0.0 | 0.13500 ± 0.00058 | — | 0.162 |
| 20 | sage_P2 | 3 | 0/3 | 0.0 | 0.13500 ± 0.00058 | — | 0.144 |
| 20 | sage_P10 | 3 | 0/3 | 0.0 | 0.13500 ± 0.00058 | — | 0.154 |
| 20 | dga | 3 | — | — | 0.16967 ± 0.00120 | — | — |

## Escape probability vs degree (n=1000, released implementation's configuration, d0 = int(√n) = 31)

Sources: collapse.jsonl + escape.jsonl, overlapping (d, seed) cells de-duplicated per D28; phase-1 `pignn_pub` n=1000 runs pooled only where (d, seed) is not already present (D38). non-empty = the network's raw output (raw_size > 0), before post-processing. Not tested: whether the d ≥ 8 ceiling holds under the paper's text setting d0 = int(∛n) = 10 at n = 1000 (D38).

| d | runs | non-empty | p(non-empty) | Wilson 95% CI | mean epochs |
|---|---|---|---|---|---|
| 3 | 5 | 5 | 1.00 | [0.57, 1.00] | 12,387 |
| 4 | 10 | 10 | 1.00 | [0.72, 1.00] | 14,151 |
| 5 | 5 | 5 | 1.00 | [0.57, 1.00] | 15,694 |
| 6 | 10 | 9 | 0.90 | [0.60, 0.98] | 16,840 |
| 7 | 10 | 2 | 0.20 | [0.06, 0.51] | 10,221 |
| 8 | 10 | 0 | 0.00 | [0.00, 0.28] | 8,002 |
| 9 | 10 | 0 | 0.00 | [0.00, 0.28] | 8,178 |
| 10 | 10 | 0 | 0.00 | [0.00, 0.28] | 8,309 |
| 11 | 10 | 0 | 0.00 | [0.00, 0.28] | 8,438 |
| 12 | 10 | 0 | 0.00 | [0.00, 0.28] | 8,592 |
| 13 | 10 | 0 | 0.00 | [0.00, 0.28] | 8,692 |
| 14 | 3 | 0 | 0.00 | [0.00, 0.56] | 8,626 |
| 15 | 10 | 0 | 0.00 | [0.00, 0.28] | 8,908 |
| 16 | 3 | 0 | 0.00 | [0.00, 0.56] | 8,819 |
| 18 | 3 | 0 | 0.00 | [0.00, 0.56] | 8,991 |
| 20 | 5 | 0 | 0.00 | [0.00, 0.43] | 9,251 |
| 25 | 3 | 0 | 0.00 | [0.00, 0.56] | 9,500 |
| pooled d ≥ 8 | 87 | 0 | 0.00 | [0.000, 0.042] | — |

Total: 127 runs in this table.

## Exact MIS via HiGHS ILP

| d | n | instances | density | /ρ_UB | all proven optimal |
|---|---|---|---|---|---|
| 3 | 100 | 5 | 0.44000 ± 0.00316 | 0.9662 | yes |
| 3 | 200 | 5 | 0.44700 ± 0.00200 | 0.9816 | yes |
| 5 | 100 | 5 | 0.37000 ± 0.00000 | 0.9625 | yes |
| 5 | 200 | 5 | 0.37200 ± 0.00122 | 0.9677 | **no — time limit** |
| 20 | 100 | 5 | 0.18000 ± 0.00000 | — | yes |
| 20 | 200 | 5 | 0.18000 ± 0.00158 | — | **no — time limit** |

## n = 10^6 (scale test)

All n=10^6 rows are a single instance/seed (seed 0).

| d | method | size | density | AR | wall clock | note |
|---|---|---|---|---|---|---|
| 3 | reference dense Q | — | — | — | — | 1.00e+12 entries = 3.6 TiB — **not representable** |
| 3 | dga | 432,949 | 0.43295 | 0.9508 | 0.2671 s | |
| 3 | ga | 375,147 | 0.37515 | 0.8238 | 0.1799 s | |
| 3 | null_ones | 395,270 | 0.39527 | 0.8680 | 0.0670 s | |
| 3 | PI-GNN probe | — | — | — | 47,387.7 ms/epoch | 20 epochs; 1.0e+09 embedding params, 14.9 GiB |
| 5 | dga | 356,710 | 0.35671 | 0.9279 | 0.2864 s | |
| 5 | ga | 301,917 | 0.30192 | 0.7854 | 0.2418 s | |
| 5 | null_ones | 327,034 | 0.32703 | 0.8507 | 0.1262 s | |
| 5 | reference dense Q | — | — | — | — | 1.00e+12 entries = 3.6 TiB — **not representable** |
| 3 | PI-GNN budgeted | 374,704 | 0.37470 | 0.8229 | 1,503 s | 36 epochs; stop=budget; truncated=True |

## Final p-vector statistics, modified linear diagonal, d=20 (D27)

| seed | final loss | L* | mean p | p* | std(p)/p* | max p | frac >= 0.5 |
|---|---|---|---|---|---|---|---|
| 0 | -12.4554 | -12.5 | 0.02600 | 0.025 | 0.273 | 0.0570 | 0 |
| 1 | -12.5242 | -12.5 | 0.02611 | 0.025 | 0.289 | 0.0737 | 0 |
| 2 | -12.5164 | -12.5 | 0.02612 | 0.025 | 0.276 | 0.0534 | 0 |

## Early-epoch raw-output traces (D34) — instrumented diagnostic re-runs

Instrumented re-runs (code/early_trace.py, code/find_escape_epoch.py) that record the raw thresholded set size during training; they are diagnostics, not additional sweep cells. The n = 10^6 row is the checkpoint trace of the budgeted scale run above (scale.jsonl), not a separate record.

| source | n | d | seed | d0 | epochs traced | raw at first traced epoch | min raw in window (epoch) | raw at last traced epoch | loss at last traced epoch |
|---|---|---|---|---|---|---|---|---|---|
| early_trace.jsonl (early_trace) | 1,000 | 3 | 0 | 31 | 60 (max 60) | 823 (epoch 0) | 760 (epoch 57) | 760 | 629.9 |
| early_trace.jsonl (early_trace) | 10,000 | 3 | 0 | 100 | 60 (max 60) | 2,008 (epoch 0) | 727 (epoch 60 — end of window, still falling) | 727 | 2,472.4 |
| early_trace.jsonl (escape_epoch) | 10,000 | 3 | 0 | 100 | 3000 (max 3000) | 2,008 (epoch 0) | 8 (epoch 450) | 4,097 | -4,034.3 |
| scale.jsonl (budgeted checkpoints) | 1,000,000 | 3 | 0 | 1000 | 36 (max 36) | 137,531 (epoch 3) | 220 (epoch 36 — end of window, still falling) | 220 | 81,089.4 |

Full traces (epoch: raw):

- n=1,000, early_trace.jsonl (early_trace): 0: 823, 3: 819, 6: 815, 9: 813, 12: 811, 15: 806, 18: 803, 21: 801, 24: 794, 27: 793, 30: 788, 33: 783, 36: 780, 39: 780, 42: 777, 45: 770, 48: 768, 51: 765, 54: 762, 57: 760, 60: 760
- n=10,000, early_trace.jsonl (early_trace): 0: 2,008, 3: 1,928, 6: 1,851, 9: 1,764, 12: 1,682, 15: 1,599, 18: 1,530, 21: 1,437, 24: 1,363, 27: 1,290, 30: 1,237, 33: 1,174, 36: 1,122, 39: 1,060, 42: 1,018, 45: 964, 48: 916, 51: 861, 54: 810, 57: 775, 60: 727
- n=10,000, early_trace.jsonl (escape_epoch): 0: 2,008, 150: 174, 300: 26, 450: 8, 600: 13, 750: 29, 900: 65, 1050: 174, 1200: 408, 1350: 929, 1500: 1,633, 1650: 2,403, 1800: 3,059, 1950: 3,502, 2100: 3,767, 2250: 3,909, 2400: 4,000, 2550: 4,043, 2700: 4,069, 2850: 4,085, 3000: 4,097
- n=1,000,000, scale.jsonl (budgeted checkpoints): 3: 137,531, 6: 87,765, 9: 53,531, 12: 31,007, 15: 17,433, 18: 9,487, 21: 4,992, 24: 2,651, 27: 1,388, 30: 736, 33: 384, 36: 220
