# STATE — where this is now

**Goal:** adjudicate the PI-GNN claim (Schuetz et al., NMI 4, 367 (2022)) against the
Angelini & Ricci-Tersenghi critique (NMI 5, 29 (2023)) by same-machine measurement.
**Host:** Apple M1 Pro, 10 core, 32 GB. torch 2.13.0, CPU. 442 records in `results/`.
**Env:** `pyrt/`, `runlog/` (a repo hook blocks `.venv`, `venv`, `logs/` paths).

## STATUS: experiment COMPLETE, adversarially reviewed (D26-D29), motion pass done (D30),
## PUBLISHED: site https://gnnvsgreedy.vercel.app + public repo
## https://github.com/ryanonline1234/gnn-vs-greedy (MIT; refs/ data/ outreach/ runlog/ excluded).
## Deploy flow: code/build_report.py -> code/make_site.py (adds doctype/head wrapper;
## the raw fragment renders in quirks mode) -> cd site && vercel deploy --prod --yes

## Verdicts (all measured; see DECISIONS.md D1–D25)
- **Claim A (parity) — REFUTED.** DGA wins at every d and n tested.
- **Claim B (scale) — upheld only via a reformulation the paper does not contain.**
  Reference dense Q = 3.6 TiB at n=1e6. Measured budgeted run: 36 epochs / 1503 s lands at
  AR 0.8229, BELOW the null control's 0.8680; DGA got AR 0.9508 in 0.2671 s.
  Projection (labelled): 7.31 days, 2,363,318x. Truncated run = lower bound only.
- **Undisclosed operating ceiling:** p(non-empty) = 0 for d >= 8 through d=25.
- **"Untuned" defence answered:** 0/24 pre-registered and 0/21 post-hoc non-empty at d=20.
- **Mechanism (corrected in review, D27):** trapped at the uniform mean-field saddle —
  loss at L* (-12.499 vs -12.500) AND measured p-vector: mean p 4% above p*, std/p* 27-29%,
  max p = 3p*, factor ~7 below threshold (results/pstats.jsonl). NOT "exactly uniform" —
  the loss is blind to scatter; 2/3 seeds land below L*. d~6-7 threshold measured, not derived.
  Escape at d=7 corrected to 2/10 (de-dup, D28).

## Deliverables
- `report.html` — findings page. Browser-verified (fonts, no overflow, 3 theme states, AA).
  **AI-written prose — must NOT be copied into an STS report.**
- `review/REVIEW.md` — orientation + ranked attack surface for a reviewing agent.
- `review/DATA-APPENDIX.md` — every measured configuration, data only.
- `review/MANIFEST.md` — file inventory with sha256.
- `notebook/AI-USE-LOG.md` — Regeneron STS Appendix 4 compliance record.

## Regeneron STS constraint (verified against the user's local rules copy)
Appendix 4: using AI to **initially write** the report/abstract/poster is *never acceptable*;
the ethics attestation says "I did not use AI tools to draft the paper." AI-written **code** is
explicitly permitted with citation + prompt log. Bibliography must be student-generated.
Therefore: no STS report draft has been or will be produced here.
Also relevant: STS is one entry, one topic per student — this would REPLACE the
`~/asr-soil-water` entry, not supplement it.

## Review round (complete)
Three adversarial agents: port audit (keystone survives; wording narrowed to forward-math
proof + reproduction, D26), mechanism audit (exactness claim refuted and corrected with
direct p-measurement, D27), fact-check (all 44 head-to-head cells exact; quote attribution,
ms/epoch seed-means, and cosmetics fixed, D28).

## Open items (D36/D37, highest value first)
1. The GraphSAGE non-reproduction (measured AR 0.8777 vs the Reply's ~0.947 at d=3) is
   unresolved and is the weakest point in the study. Needs their implementation.
2. The d=10 disagreement with Krutsky et al. (ECAI 2025) is unresolved — likely a
   configuration difference; email draft prepared in outreach/ (local only).
3. Prior art (D32) must be cited in any write-up or it will be rejected on novelty.
4. arXiv endorsement needed before preprinting; candidates + drafts in outreach/.

## Next action
None blocking. Redeploys: rebuild report + make_site, then vercel deploy --prod --yes from site/. When cleared to publish, `report.html` is ready for the Artifact tool
(title "The PI-GNN Verdict", favicon a single scales emoji).
