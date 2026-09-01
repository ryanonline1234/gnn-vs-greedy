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
                            "code/scale_test.py", "code/scale_budget.py"]),
    ("Raw measurements", ["results/phase1.jsonl", "results/phase2.jsonl", "results/tuning.jsonl",
                          "results/posthoc.jsonl", "results/collapse.jsonl", "results/escape.jsonl",
                          "results/exact.jsonl", "results/scale.jsonl", "results/pstats.jsonl"]),
    ("Findings page (AI-written prose — see the AI use log)", ["report.html",
                                                              "site/index.html",
                                                              "code/build_report.py",
                                                              "code/make_site.py",
                                                              "code/svgchart.py"]),
    ("Competition compliance", ["notebook/AI-USE-LOG.md"]),
    ("Source material", ["refs/critique.txt", "refs/pignn/utils.py", "refs/pignn/README.md"]),
]
def h(p):
    try:
        return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    except Exception:
        return None
lines = ["# MANIFEST — review package", "",
         "SHA-256 (first 16 hex) and byte size for every file a reviewer needs.",
         "Regenerate with `./pyrt/bin/python code/build_manifest.py`.", ""]
missing = []
for title, files in GROUPS:
    lines += [f"## {title}", "", "| file | bytes | sha256[:16] |", "|---|---|---|"]
    for f in files:
        p = os.path.join(ROOT, f)
        if not os.path.exists(p):
            missing.append(f); lines.append(f"| `{f}` | **MISSING** | — |"); continue
        lines.append(f"| `{f}` | {os.path.getsize(p):,} | `{h(p)}` |")
    lines.append("")
nrec = 0
for f in os.listdir(os.path.join(ROOT, "results")):
    if f.endswith(".jsonl"):
        nrec += sum(1 for _ in open(os.path.join(ROOT, "results", f)))
lines += ["## Totals", "",
          f"- {nrec:,} measurement records across `results/*.jsonl` "
          f"(3 of these are the d=7 duplicates documented in D28, so 442 unique runs; "
          f"the deliverables de-duplicate at load time)",
          f"- host: Apple M1 Pro, 10 core (8P/2E), 32 GB unified; macOS; torch 2.13.0, CPU",
          f"- greedy baselines: C, `cc -O3 -march=native`", ""]
if missing:
    lines += ["## Missing", ""] + [f"- `{m}`" for m in missing]
out = os.path.join(ROOT, "review", "MANIFEST.md")
open(out, "w").write("\n".join(lines))
print(f"wrote {out}; {nrec:,} records; missing={len(missing)}")
