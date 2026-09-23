# STATE — where this is now (updated 2026-09-22)

**Goal:** adjudicate the PI-GNN claim (Schuetz et al., NMI 4, 367 (2022)) against the
Angelini & Ricci-Tersenghi critique (NMI 5, 29 (2023)) by same-machine measurement.
**Host:** Apple M1 Pro, 10 core, 32 GB. torch 2.13.0, CPU. 534 measurement records in `results/` (11 files; counted in code at
build time — D38 retired the D31 "distinct measured runs" count).
**Env:** `pyrt/`, `runlog/` (a repo hook blocks `.venv`, `venv`, `logs/` paths).

## STATUS: experiment COMPLETE, adversarially reviewed (D26-D29, D31, D33, D35-D37; audit
## errata D38), motion pass done (D30),
## PUBLISHED: site https://gnnvsgreedy.vercel.app + public repo
## https://github.com/ryanonline1234/gnn-vs-greedy (MIT; refs/ data/ outreach/ runlog/ excluded).
## Deploy flow: code/build_report.py -> code/make_site.py (adds doctype/head wrapper;
## the raw fragment renders in quirks mode) -> cd site && vercel deploy --prod --yes

## Now (2026-09-22)
Audit fix (113-agent audit) on branch `audit-fixes-2026-09-22`: **not yet pushed or deployed**;
the live site and public repo still show the pre-audit numbers. Errata are logged in D38
(escape-pool double count, record count, DGA seed basis, √n-vs-∛n configuration). Open
experiment from D38: does the d >= 8 ceiling hold under the paper's d0 = int(∛n) at n = 1000?

## Verdicts (see DECISIONS.md D1–D38; D24 is a labelled projection)
- **Claim A (parity) — REFUTED.** DGA wins in every (d, n) cell where both were run, and in
  every seed-matched pair. Replication: d=3, n=1e4, 5 seeds: DGA AR 0.9502, PI-GNN 0.9138.
- **Claim B (scale) — upheld only via a reformulation the paper does not describe.**
  The 3.6 TiB dense-Q wall at n=1e6 is a property of the released example notebook
  (refs/pignn/utils.py); the paper never describes its n=1e6 QUBO construction (D36).
  Measured budgeted run: 36 epochs / 1503 s lands at AR 0.8229, BELOW the null control's
  0.8680; DGA got AR 0.9508 in 0.2671 s.
  Projection (labelled, D33 range): ~6.4-8.1 days, roughly 2.1-2.6 million x DGA.
  Truncated run = lower bound only (D34: its raw-output decay is the normal early transient).
  D12's faster-device rule was never executed at n=1e6.
- **Operating ceiling (a regime the paper never tested, D36):** raw output non-empty in
  0/87 runs pooled over d = 8-25, n = 1000 (Wilson 95% CI [0.000, 0.042]); d=7 escapes 2/10;
  d=3 and d=5 5/5, d=20 0/5 (D38 corrects D33/D35's 8/8 and 0/8). Released-implementation
  configuration (d0 = int(√n)); untested under the paper's ∛n rule below n=1e5 (D38).
- **Reply's defence (D36/D37):** the pre-registered lr × P∈{2,3} grid (0/24 pre-registered,
  0/21 post-hoc raw non-empty at d=20) did NOT test the published defence, so the
  "answered" claim is retracted (D36). The actual defence, GraphSAGE + P=10 (D37), n=1000,
  seeds 0-2, tested at d=3, 5, 20 only: raw output empty at d=20 in 12/12 runs across all
  four {GCN,SAGE}×{P=2,P=10} arms; no arm reaches DGA (same seeds: 0.9479 at d=3, 0.9148
  at d=5); SAGE+P=10 reaches AR 0.8777 at d=3 against the Reply's ≈0.947 — not reproduced,
  recorded as a failure to reproduce, not a refutation.
- **Mechanism (corrected in review, D27; closed form checked against measurement, D17):**
  under the modified post-hoc objective (linear diagonal) the run sits at the uniform
  mean-field point — loss at L* (-12.499 vs -12.500) AND measured p-vector: mean p 4% above
  p*, std/p* 27-29%, max p = 3p*, factor ~7 below threshold (results/pstats.jsonl). Under
  the published (quadratic-diagonal) objective the uniform optimum is p* = 0, L* = 0, which
  shows up as an empty raw output. NOT "exactly uniform" — the loss is blind to scatter;
  2/3 seeds land below L*. d~6-7 threshold measured, not derived.
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
Audit round 2026-09-22 (113 agents, four lenses, adversarially verified): number errata,
propagation of D26/D32/D33/D36/D37, and the √n-vs-∛n configuration disclosure, all in D38.

## Open items (D36-D38, highest value first)
1. The GraphSAGE non-reproduction (measured AR 0.8777 vs the Reply's ~0.947 at d=3) is
   unresolved and is the weakest point in the study. Needs their implementation.
2. WITHDRAWN (D38): the "d=10 disagreement" with Krutsky et al. (ECAI 2025) compared their
   MaxCut result with our MIS one; their MIS tables agree with ours. Do NOT send
   outreach/email-draft-krutsky.md, which leads with it. The Böther and Angelini emails sent
   2026-09-01 stated it (plus 0/90 and "predicted before measurement") — corrections pending.
3. Prior art (D32) must be cited in any write-up or it will be rejected on novelty.
4. arXiv endorsement needed before preprinting; candidates + drafts in outreach/.
5. The ∛n configuration (D38): the paper's text sets d0 = int(∛n) below n = 1e5; this study
   ran int(√n) (released notebook). Re-run the escape sweep at n = 1000 with d0 = 10 to see
   whether the d >= 8 ceiling holds there.

## Next action
Review the audit-fix branch `audit-fixes-2026-09-22`, then merge, push and redeploy
(rebuild report + make_site, then vercel deploy --prod --yes from site/); update the GitHub
repo description (still says 483). Then the ∛n experiment (open item 5; command in D38 item 11). When cleared to publish, `report.html` is ready for the Artifact tool
(title "The PI-GNN Verdict", favicon a single scales emoji).
