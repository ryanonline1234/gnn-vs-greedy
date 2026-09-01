"""Wrap report.html (Artifact-style fragment, no doctype) into a standalone document at
site/index.html for Vercel hosting. The doctype matters: the fragment renders in quirks
mode on its own (D30). Run after any build_report.py rebuild that should go live."""
import os, re
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src = open(os.path.join(ROOT, "report.html")).read()
cut = src.index("</style>") + len("</style>")
head_part, body_part = src[:cut], src[cut:]
doc = ("<!doctype html>\n<html lang=\"en\">\n<head>\n"
       "<meta charset=\"utf-8\">\n"
       "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
       "<meta name=\"description\" content=\"A same-machine adjudication of the PI-GNN vs "
       "greedy dispute in Nature Machine Intelligence: 442 measured runs on Maximum "
       "Independent Set.\">\n"
       + head_part + "\n</head>\n<body>\n" + body_part + "\n</body>\n</html>\n")
os.makedirs(os.path.join(ROOT, "site"), exist_ok=True)
out = os.path.join(ROOT, "site", "index.html")
open(out, "w").write(doc)
print(f"wrote {out} ({len(doc):,} bytes)")
