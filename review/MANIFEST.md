# MANIFEST — review package

SHA-256 (first 16 hex) and byte size for every file a reviewer needs.
Regenerate with `./pyrt/bin/python code/build_manifest.py`.

## Read first

| file | bytes | sha256[:16] |
|---|---|---|
| `review/REVIEW.md` | 12,982 | `100ac4db99eeb95f` |
| `STATE.md` | 3,368 | `6ffe0ecf225aac0c` |
| `DECISIONS.md` | 43,017 | `0b7b8900f03c21ca` |

## The keystone — verify this before anything else

| file | bytes | sha256[:16] |
|---|---|---|
| `code/test_port.py` | 2,472 | `2b1157b11cb9d7b9` |

## Implementations under test

| file | bytes | sha256[:16] |
|---|---|---|
| `code/pignn.py` | 8,568 | `5dd15cbc53dd333d` |
| `code/greedy.c` | 5,984 | `ba70db30ce2a898b` |
| `code/repair.c` | 4,392 | `3d8902aca82fd876` |

## Analysis + mechanism

| file | bytes | sha256[:16] |
|---|---|---|
| `code/verify_meanfield.py` | 2,294 | `8adf1be6296cf44f` |
| `code/capture_pstats.py` | 1,836 | `e947d4870bfb6983` |
| `code/analyze.py` | 3,513 | `69fe55655fed99af` |
| `code/exact_mis.py` | 2,255 | `8fb5090065289879` |

## Experiment drivers

| file | bytes | sha256[:16] |
|---|---|---|
| `code/runner.py` | 6,480 | `66f027d357ead96c` |
| `code/gen_graphs.py` | 1,847 | `828a2a9bde489a61` |
| `code/sweep_tuned.py` | 1,446 | `4f6e60151e80c766` |
| `code/posthoc_d20.py` | 3,256 | `1b38499920bb1fa0` |
| `code/collapse_sweep.py` | 1,421 | `8dab42a29b553df5` |
| `code/scale_test.py` | 5,349 | `a4c356156bfacb7a` |
| `code/scale_budget.py` | 3,920 | `2882e15795406ca8` |

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

## Findings page (AI-written prose — see the AI use log)

| file | bytes | sha256[:16] |
|---|---|---|
| `report.html` | 67,961 | `b7bebf0911670c93` |
| `site/index.html` | 68,300 | `a5a894c5fb46db1a` |
| `code/build_report.py` | 55,262 | `6dc7de055f41283b` |
| `code/make_site.py` | 1,124 | `4df01946b611a619` |
| `code/svgchart.py` | 3,318 | `555766f7ad442591` |

## Competition compliance

| file | bytes | sha256[:16] |
|---|---|---|
| `notebook/AI-USE-LOG.md` | 4,730 | `676cc3eb6db462e4` |

## Source material

| file | bytes | sha256[:16] |
|---|---|---|
| `refs/critique.txt` | 15,231 | `93d1f3b357c2e895` |
| `refs/pignn/utils.py` | 8,324 | `26cf7392a298aacf` |
| `refs/pignn/README.md` | 6,000 | `8b8181fe3d29ce93` |

## Totals

- 534 measurement records across `results/*.jsonl` (3 of these are the d=7 duplicates documented in D28, so 442 unique runs; the deliverables de-duplicate at load time)
- host: Apple M1 Pro, 10 core (8P/2E), 32 GB unified; macOS; torch 2.13.0, CPU
- greedy baselines: C, `cc -O3 -march=native`
