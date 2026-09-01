"""Download the papers this study analyses. They are NOT redistributed in this repository —
they are copyrighted works by their authors. This fetches them from arXiv into refs/.

Run once before reading the decision log, which cites them heavily.
"""
import os, time, urllib.request

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
    print("\nBoettcher's Comment (Nat Mach Intell 5, 24 (2023)) has no arXiv preprint I could\n"
          "find; obtain it from the journal if you need it.")

if __name__ == "__main__":
    main()
