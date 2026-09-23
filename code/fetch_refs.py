"""Download the papers this study analyses, and the reference implementation it ports. They
are NOT redistributed in this repository — the papers are copyrighted works by their authors.
This fetches, into refs/:
  - the seven arXiv PDFs below, and a plain-text copy of each (<name>.txt, extracted with
    pypdf; e.g. refs/critique.txt is the text of arXiv:2206.13211);
  - the authors' reference implementation, amazon-science/co-with-gnns-example (Apache-2.0),
    cloned into refs/pignn/ and checked out at the commit this study used (PIGNN_COMMIT).
Anything already present locally is kept, never re-downloaded or overwritten. Needs `git` on
PATH for the clone. code/test_port.py does not need any of this.

Run once before reading the decision log, which cites them heavily.
"""
import os, subprocess, time, urllib.request

DOCS = {
    "2107.01188": ("schuetz_pignn_FULL", "Schuetz, Brubaker & Katzgraber, Nat Mach Intell 4, 367 (2022) — the paper under test"),
    "2206.13211": ("critique",           "Angelini & Ricci-Tersenghi, NMI 5, 29 (2023) — the Comment"),
    "2302.03602": ("reply_to_angelini",  "Schuetz et al., Reply to Angelini & Ricci-Tersenghi"),
    "2303.12096": ("reply_to_boettcher", "Schuetz et al., Reply to Boettcher"),
    "2201.10494": ("bother_iclr22_tree_search",  "Bother et al., ICLR 2022 — prior art for the null-control idea"),
    "2309.16965": ("ichikawa_neurips24_CRA",     "Ichikawa, NeurIPS 2024 — prior art for the failure modes"),
    "2507.13703": ("krutsky_ecai25_binarizing",  "Krutsky et al., ECAI 2025 — prior art for the density transition"),
}
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "refs")
PIGNN_URL = "https://github.com/amazon-science/co-with-gnns-example.git"
PIGNN_COMMIT = "955489d90df286ed9a29e939f364a7cad3706dd1"   # the version this study ported

def fetch_pignn():
    dest = os.path.join(HERE, "pignn")
    if os.path.exists(dest):
        try:
            have = subprocess.run(["git", "-C", dest, "log", "-1", "--format=%H"],
                                  capture_output=True, text=True).stdout.strip()
        except Exception:
            have = ""
        note = ("" if have == PIGNN_COMMIT else
                f"  (WARNING: at {have or 'unknown commit'}; the study used {PIGNN_COMMIT})")
        print(f"  have pignn/{note}"); return
    try:
        subprocess.run(["git", "clone", "--quiet", PIGNN_URL, dest], check=True)
        subprocess.run(["git", "-C", dest, "-c", "advice.detachedHead=false",
                        "checkout", "--quiet", PIGNN_COMMIT], check=True)
        print(f"  got  pignn/ at {PIGNN_COMMIT}  — the authors' reference implementation")
    except Exception as e:
        print(f"  FAIL pignn clone: {e}")

def extract_text(name):
    pdf, txt = (os.path.join(HERE, name + ext) for ext in (".pdf", ".txt"))
    if os.path.exists(txt) or not os.path.exists(pdf): return
    try:
        import pypdf
    except ImportError:
        print(f"  skip {name}.txt (pip install pypdf to extract text)"); return
    try:
        text = "\n".join((pg.extract_text() or "") for pg in pypdf.PdfReader(pdf).pages)
        with open(txt, "w", encoding="utf-8") as f: f.write(text)
        print(f"  made {name}.txt")
    except Exception as e:
        print(f"  FAIL text of {name}.pdf: {e}")

def main():
    os.makedirs(HERE, exist_ok=True)
    for aid, (name, why) in DOCS.items():
        out = os.path.join(HERE, name + ".pdf")
        if os.path.exists(out) and os.path.getsize(out) > 20000:
            print(f"  have {name}.pdf"); continue
        try:
            r = urllib.request.urlopen(urllib.request.Request(
                f"https://arxiv.org/pdf/{aid}", headers={"User-Agent": "Mozilla/5.0"}), timeout=60)
            with open(out, "wb") as f: f.write(r.read())
            print(f"  got  {name}.pdf  ({os.path.getsize(out):,} B)  — {why}")
            time.sleep(2)   # be polite to arXiv
        except Exception as e:
            print(f"  FAIL {aid} ({name}): {e}")
    for name, _ in DOCS.values():
        extract_text(name)
    fetch_pignn()
    print("\nBoettcher's Comment (Nat Mach Intell 5, 24 (2023)) has no arXiv preprint I could\n"
          "find; obtain it from the journal if you need it.")

if __name__ == "__main__":
    main()
