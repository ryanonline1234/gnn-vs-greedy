"""Exact MIS via ILP (HiGHS) on small instances, to calibrate the approximation-ratio
scale against a true optimum rather than only against McKay's upper bound.

max sum x_v  s.t.  x_u + x_v <= 1 for every edge,  x binary.
"""
import argparse, json, os, time
import numpy as np, highspy, gen_graphs

def exact(edges, n, time_limit=300.0):
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", time_limit)
    inf = highspy.kHighsInf
    for _ in range(n):
        h.addVar(0.0, 1.0)
    h.changeColsIntegrality(n, np.arange(n, dtype=np.int32),
                            np.array([highspy.HighsVarType.kInteger]*n))
    # maximise sum x  ->  minimise -sum x
    h.changeColsCost(n, np.arange(n, dtype=np.int32), -np.ones(n))
    for (u, v) in edges:
        h.addRow(-inf, 1.0, 2, np.array([int(u), int(v)], dtype=np.int32),
                 np.array([1.0, 1.0]))
    t0 = time.time()
    h.run()
    t = time.time() - t0
    st = h.getModelStatus()
    sol = np.array(h.getSolution().col_value)
    size = int(round(sol.sum()))
    return size, t, str(h.modelStatusToString(st))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, nargs="+", default=[100, 200])
    ap.add_argument("--d", type=int, nargs="+", default=[3, 5, 20])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--time-limit", type=float, default=300.0)
    a = ap.parse_args()
    for d in a.d:
        for n in a.n:
            for s in a.seeds:
                import random as _r; _r.seed(s)
                import igraph as ig; ig.set_random_number_generator(_r)
                binp, npzp = gen_graphs.generate(n, d, s)
                z = np.load(npzp)
                size, t, st = exact(z["edges"], n, a.time_limit)
                rec = dict(method="exact_ilp", d=d, n=n, seed=s, size=size,
                           density=size/n, t_total_s=t, status=st)
                with open(a.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"  exact d={d} n={n} s={s}: {size} (rho={size/n:.4f}) "
                      f"{t:.1f}s [{st}]", flush=True)
