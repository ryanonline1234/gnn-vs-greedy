import hashlib, os, json, subprocess
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GROUPS = [
    ("Read first", ["review/REVIEW.md", "STATE.md", "DECISIONS.md"]),
    ("The keystone — verify this before anything else", ["code/test_port.py"]),
    ("Implementations under test", ["code/pignn.py", "code/greedy.c", "code/repair.c"]),
    ("Analysis + mechanism", ["code/verify_meanfield.py", "code/capture_pstats.py",
                              "code/analyze.py", "code/exact_mis.py"]),
    ("Experiment drivers", ["code/runner.py", "code/gen_graphs.py", "code/sweep_tuned.py",
                            "code/posthoc_d20.py", "code/collapse_sweep.py",
                            "code/escape_prob.py", "code/escape_d10plus.py",
                            "code/early_trace.py", "code/find_escape_epoch.py",
                            "code/reply_defence.py",
                            "code/scale_test.py", "code/scale_budget.py"]),
    ("Diagnostics and figures", ["code/diag_collapse.py", "code/bench_device.py",
                                 "code/fig_collapse.py", "code/make_plots.py"]),
    ("Raw measurements", ["results/phase1.jsonl", "results/phase2.jsonl", "results/tuning.jsonl",
                          "results/posthoc.jsonl", "results/collapse.jsonl", "results/escape.jsonl",
                          "results/exact.jsonl", "results/scale.jsonl", "results/pstats.jsonl",
                          "results/early_trace.jsonl", "results/reply_defence.jsonl"]),
    ("Findings page (AI-written prose — see the AI use log)", ["report.html",
                                                              "site/index.html",
                                                              "code/build_report.py",
                                                              "code/make_site.py",
                                                              "code/svgchart.py"]),
    ("Review package generators", ["code/build_review.py", "code/build_appendix.py",
                                   "code/build_manifest.py", "review/DATA-APPENDIX.md"]),
    ("Reproduction", ["README.md", "requirements.txt", "code/fetch_refs.py"]),
    ("Competition compliance", ["notebook/AI-USE-LOG.md"]),
    ("Source material — NOT committed (copyrighted); retrieved by code/fetch_refs.py. "
     "Hashes identify the exact local versions this study used; refs/critique.txt is text "
     "extracted from arXiv:2206.13211.",
     ["refs/critique.txt", "refs/pignn/utils.py", "refs/pignn/README.md"]),
]
def h(p):
    try:
        return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    except Exception:
        return None
lines = ["# MANIFEST — review package", "",
         "SHA-256 (first 16 hex) and byte size for every file a reviewer needs.",
         "Regenerate with `python code/build_manifest.py` (after `python code/fetch_refs.py`, so the refs/ rows hash) — after every other edit, "
         "since it hashes the other files.", ""]
missing = []
for title, files in GROUPS:
    lines += [f"## {title}", "", "| file | bytes | sha256[:16] |", "|---|---|---|"]
    for f in files:
        p = os.path.join(ROOT, f)
        if not os.path.exists(p):
            missing.append(f); lines.append(f"| `{f}` | **MISSING** | — |"); continue
        lines.append(f"| `{f}` | {os.path.getsize(p):,} | `{h(p)}` |")
    lines.append("")

# ---- totals, computed from the data (D38: never hardcode a count the data determines) ----
RES = os.path.join(ROOT, "results")
def load(name):
    p = os.path.join(RES, name)
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []
per_file = {f: sum(1 for l in open(os.path.join(RES, f)) if l.strip())
            for f in sorted(os.listdir(RES)) if f.endswith(".jsonl")}
nrec = sum(per_file.values())
coll_keys = {(r["d"], r["seed"]) for r in load("collapse.jsonl")}
esc_keys = {(r["d"], r["seed"]) for r in load("escape.jsonl")}
overlap = sorted(coll_keys & esc_keys)
ov_ds = sorted({d for d, _ in overlap})
lines += ["## Totals", "",
          f"- **{nrec:,} measurement records** across {len(per_file)} files in `results/*.jsonl`: "
          + ", ".join(f"{f.removesuffix('.jsonl')} {n}" for f, n in per_file.items()) + ".",
          f"- Of these, {per_file.get('pstats.jsonl', 0)} (pstats) and "
          f"{per_file.get('early_trace.jsonl', 0)} (early_trace) are instrumented diagnostic "
          f"re-runs of cells measured elsewhere, and {len(overlap)} (d, seed) cells appear in both "
          f"collapse.jsonl and escape.jsonl (d = {', '.join(map(str, ov_ds))}; re-executions of the "
          f"same cell, de-duplicated at load per D28/D38).",
          "- No \"distinct measured runs\" total is published: D38 retired that convention because "
          "the same published-config cell was executed by several drivers, so any single figure "
          "depends on an equivalence rule the study never fixed. Per-experiment counts are computed "
          "where they are reported.",
          f"- host: Apple M1 Pro, 10 core (8P/2E), 32 GB unified; macOS; torch 2.13.0, CPU",
          f"- greedy baselines: C, `cc -O3 -march=native`", ""]
if missing:
    lines += ["## Missing", ""] + [f"- `{m}`" for m in missing]
out = os.path.join(ROOT, "review", "MANIFEST.md")
open(out, "w").write("\n".join(lines))
print(f"wrote {out}; {nrec:,} records; overlap={len(overlap)}; missing={len(missing)}")
