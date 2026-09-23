# MANIFEST — review package

SHA-256 (first 16 hex) and byte size for every file a reviewer needs.
Regenerate with `python code/build_manifest.py` (after `python code/fetch_refs.py`, so the refs/ rows hash) — after every other edit, since it hashes the other files.

## Read first

| file | bytes | sha256[:16] |
|---|---|---|
| `review/REVIEW.md` | 24,277 | `22499e397f5eb0bd` |
| `STATE.md` | 6,851 | `c1df96ac765b22f5` |
| `DECISIONS.md` | 54,266 | `9c923dacdb4df6d2` |

## The keystone — verify this before anything else

| file | bytes | sha256[:16] |
|---|---|---|
| `code/test_port.py` | 3,432 | `03201ceea946ade8` |

## Implementations under test

| file | bytes | sha256[:16] |
|---|---|---|
| `code/pignn.py` | 8,568 | `5dd15cbc53dd333d` |
| `code/greedy.c` | 5,984 | `ba70db30ce2a898b` |
| `code/repair.c` | 4,392 | `3d8902aca82fd876` |

## Analysis + mechanism

| file | bytes | sha256[:16] |
|---|---|---|
| `code/verify_meanfield.py` | 3,048 | `deb52e2ad3455b88` |
| `code/capture_pstats.py` | 1,836 | `e947d4870bfb6983` |
| `code/analyze.py` | 4,762 | `5764fd2952760a50` |
| `code/exact_mis.py` | 2,255 | `8fb5090065289879` |

## Experiment drivers

| file | bytes | sha256[:16] |
|---|---|---|
| `code/runner.py` | 6,480 | `66f027d357ead96c` |
| `code/gen_graphs.py` | 1,847 | `828a2a9bde489a61` |
| `code/sweep_tuned.py` | 1,446 | `4f6e60151e80c766` |
| `code/posthoc_d20.py` | 3,256 | `1b38499920bb1fa0` |
| `code/collapse_sweep.py` | 1,421 | `8dab42a29b553df5` |
| `code/escape_prob.py` | 1,128 | `44d8fd9f5566926c` |
| `code/escape_d10plus.py` | 1,958 | `2e79638a1640dd5b` |
| `code/early_trace.py` | 2,624 | `ff986acd3644c3a4` |
| `code/find_escape_epoch.py` | 2,162 | `168b16f32a51aca9` |
| `code/reply_defence.py` | 2,634 | `5dec8c7195951a73` |
| `code/scale_test.py` | 5,349 | `a4c356156bfacb7a` |
| `code/scale_budget.py` | 3,920 | `2882e15795406ca8` |

## Diagnostics and figures

| file | bytes | sha256[:16] |
|---|---|---|
| `code/diag_collapse.py` | 1,957 | `3fb0c28c3ae9b92c` |
| `code/bench_device.py` | 1,308 | `6fe782b635e084ac` |
| `code/fig_collapse.py` | 3,306 | `faa733fc67c72c83` |
| `code/make_plots.py` | 3,717 | `39ea47575e9e1da2` |

## Raw measurements

| file | bytes | sha256[:16] |
|---|---|---|
| `results/phase1.jsonl` | 43,203 | `8860c8cbd0d54915` |
| `results/phase2.jsonl` | 9,018 | `9f85b18fe5a23315` |
| `results/tuning.jsonl` | 38,692 | `166ed4b65eef78f8` |
| `results/posthoc.jsonl` | 23,215 | `8b045514bda397c9` |
| `results/collapse.jsonl` | 5,181 | `abd9d7ad1c2e7ac5` |
| `results/escape.jsonl` | 11,547 | `77fd63afa4df2f48` |
| `results/exact.jsonl` | 4,155 | `beb980300316f5dc` |
| `results/scale.jsonl` | 3,364 | `13d91b2ef465c8fa` |
| `results/pstats.jsonl` | 1,151 | `21bd34d3dbf2ce10` |
| `results/early_trace.jsonl` | 4,388 | `1dab0b7445b02817` |
| `results/reply_defence.jsonl` | 9,715 | `8a731ad20e237c20` |
| `results/cbrt_d0.jsonl` | 12,703 | `87fa40e277bbddfc` |

## Findings page (AI-written prose — see the AI use log)

| file | bytes | sha256[:16] |
|---|---|---|
| `report.html` | 77,645 | `66f773eb5e020a65` |
| `site/index.html` | 77,990 | `1afe6b94926d868d` |
| `code/build_report.py` | 72,691 | `42e3fa91d8e63f3f` |
| `code/make_site.py` | 1,495 | `db151d78bd3fa2f3` |
| `code/svgchart.py` | 3,318 | `555766f7ad442591` |

## Review package generators

| file | bytes | sha256[:16] |
|---|---|---|
| `code/build_review.py` | 32,783 | `78ea6430ea591aa8` |
| `code/build_appendix.py` | 14,181 | `4a7fa459d792dc2d` |
| `code/build_manifest.py` | 5,487 | `0a72e9ede5f874d3` |
| `review/DATA-APPENDIX.md` | 15,199 | `f97f3706daaf8e1e` |

## Reproduction

| file | bytes | sha256[:16] |
|---|---|---|
| `README.md` | 11,301 | `05b4964cb751bce1` |
| `requirements.txt` | 497 | `08317517612f4d44` |
| `code/fetch_refs.py` | 4,303 | `e914232f87b0ce0d` |

## Competition compliance

| file | bytes | sha256[:16] |
|---|---|---|
| `notebook/AI-USE-LOG.md` | 8,866 | `665d7a669ef1f831` |

## Source material — NOT committed (copyrighted); retrieved by code/fetch_refs.py. Hashes identify the exact local versions this study used; refs/critique.txt is text extracted from arXiv:2206.13211.

| file | bytes | sha256[:16] |
|---|---|---|
| `refs/critique.txt` | 15,231 | `93d1f3b357c2e895` |
| `refs/pignn/utils.py` | 8,324 | `26cf7392a298aacf` |
| `refs/pignn/README.md` | 6,000 | `8b8181fe3d29ce93` |

## Totals

- **559 measurement records** across 12 files in `results/*.jsonl`: cbrt_d0 25, collapse 30, early_trace 3, escape 100, exact 30, phase1 150, phase2 37, posthoc 63, pstats 3, reply_defence 36, scale 10, tuning 72.
- Of these, 3 (pstats) and 3 (early_trace) are instrumented diagnostic re-runs of cells measured elsewhere, and 9 (d, seed) cells appear in both collapse.jsonl and escape.jsonl (d = 7, 10, 12; re-executions of the same cell, de-duplicated at load per D28/D38).
- No "distinct measured runs" total is published: D38 retired that convention because the same published-config cell was executed by several drivers, so any single figure depends on an equivalence rule the study never fixed. Per-experiment counts are computed where they are reported.
- host: Apple M1 Pro, 10 core (8P/2E), 32 GB unified; macOS; torch 2.13.0, CPU
- greedy baselines: C, `cc -O3 -march=native`
