"""Is the high-degree failure 'collapse to the origin', or the more general
'failure to symmetry-break'? Compare the observed final loss against the exact
UNIFORM (mean-field) optimum of each relaxed objective.

For a uniform assignment p_i = p on a d-regular graph with n nodes (m = nd/2 edges):
  quadratic diagonal (PUBLISHED):  L(p) = -n p^2 + P (nd/2) p^2 = n p^2 (Pd/2 - 1)
        -> for Pd/2 > 1 this is minimised at p* = 0 with L* = 0
  linear diagonal (MODIFIED):      L(p) = -n p   + P (nd/2) p^2
        -> dL/dp = -n + P n d p = 0  ->  p* = 1/(P d),  L* = -n / (2 P d)
A network that has learned nothing beyond the mean field lands exactly on L*.
A network that has symmetry-broken lands far below it.
"""
import json, numpy as np

def uniform_opt(n, d, P, diag):
    if diag == "quadratic":
        return (0.0, 0.0) if (P*d/2.0) > 1 else (1.0, n*(P*d/2.0-1))
    p = 1.0/(P*d)
    return p, -n/(2.0*P*d)

rs = [json.loads(l) for l in open("results/posthoc.jsonl")]
print(f"{'arm':<24} {'d':>3} {'diag':<10} {'uniform p*':>11} {'uniform L*':>11} "
      f"{'observed L':>12} {'verdict':>22}")
print("-"*100)
# pignn_pub (quadratic) rows live in phase1.jsonl, not posthoc.jsonl; the quadratic
# uniform optimum is p*=0, L*=0 and the empty-set collapse is documented in D18/D27.
for m, diag in [("modified_linear", "linear"),
                ("modified_linear_lr1e-3", "linear")]:
    for d in (3, 5, 20):
        v = [r for r in rs if r["d"] == d and r["method"] == m]
        if not v:
            continue
        n, P = 1000, 2.0
        ps, Ls = uniform_opt(n, d, P, diag)
        obs = np.mean([x["final_loss"] for x in v])
        # "symmetry-broken" = observed loss meaningfully below the uniform optimum
        broke = obs < Ls - 1e-6 - 0.02*abs(Ls)
        verdict = "SYMMETRY-BROKEN" if broke else "stuck at mean field"
        print(f"{m:<24} {d:>3} {diag:<10} {ps:>11.5f} {Ls:>11.3f} {obs:>12.3f} {verdict:>22}")

# pignn_pub d=20 observed loss comes from phase1/posthoc; quadratic uniform optimum is 0
print("\nNote: for the PUBLISHED quadratic diagonal the uniform optimum IS p*=0, L*=0.")
print("So 'collapse to the empty set' and 'stuck at the mean field' are the SAME event;")
print("the quadratic diagonal merely makes the symmetric solution be the empty set.")
