"""Main experiment runner. Appends one JSON record per (method, d, n, seed) to a JSONL
file and skips work already recorded, so it is restart-safe.

Timing protocol (DECISIONS.md D7): all methods timed on this host. Graph loading from
disk excluded for every method; data-structure construction included for every method.
"""
import argparse, json, os, subprocess, sys, time
import numpy as np
import gen_graphs, pignn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GREEDY = os.path.join(ROOT, "code", "greedy")
REPAIR = os.path.join(ROOT, "code", "repair")

RHO_UB = {3: 0.45537, 5: 0.38443}       # McKay 1987, as used by the critique
AR_1RSB = {3: 0.990, 5: 0.987}          # Barbier et al. 2013


def load_done(path):
    done = set()
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add((r["method"], r["d"], r["n"], r["seed"]))
            except Exception:
                pass
    return done


def emit(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    ar = rec.get("ar")
    print(f"  -> {rec['method']:14s} d={rec['d']:2d} n={rec['n']:>7d} s={rec['seed']} "
          f"size={rec['size']:>7d} rho={rec['density']:.4f}"
          + (f" AR={ar:.4f}" if ar else "") + f" t={rec['t_total_s']:.3f}s", flush=True)


def with_ar(rec):
    ub = RHO_UB.get(rec["d"])
    rec["ar"] = (rec["density"] / ub) if ub else None
    return rec


def run_greedy(binp, method, d, n, seed):
    out = subprocess.run([GREEDY, binp, method, str(seed)], capture_output=True, text=True)
    j = json.loads(out.stdout)
    return with_ar(dict(method=method, d=d, n=n, seed=seed, size=j["size"],
                        density=j["density"], t_total_s=j["t_solve_s"],
                        t_solve_s=j["t_solve_s"], violations=j["violations"],
                        nonmaximal=j["nonmaximal"]))


def run_control(binp, kind, d, n, seed, tmpdir):
    """Information-free bitstrings put through the SAME post-processing as the GNN.
    Establishes the floor that any learned signal must clear."""
    t0 = time.time()
    if kind == "null_ones":
        b = np.ones(n, dtype=np.int8)
    else:
        b = (np.random.default_rng(seed).random(n) < 0.5).astype(np.int8)
    t_make = time.time() - t0
    bp = os.path.join(tmpdir, f"ctl_{kind}_{d}_{n}_{seed}.bin")
    with open(bp, "wb") as f:
        f.write(np.array([n], dtype=np.int32).tobytes()); f.write(b.tobytes())
    out = subprocess.run([REPAIR, binp, bp], capture_output=True, text=True)
    j = json.loads(out.stdout)
    os.remove(bp)
    return with_ar(dict(method=kind, d=d, n=n, seed=seed, size=j["repaired_size"],
                        density=j["density"], t_total_s=t_make + j["t_post_s"],
                        t_post_s=j["t_post_s"], raw_size=j["raw_size"],
                        violations=j["violations_after"], nonmaximal=j["nonmaximal_after"]))


def run_gnn(npzp, binp, d, n, seed, device, tag, tmpdir, **hp):
    bp = os.path.join(tmpdir, f"gnn_{tag}_{d}_{n}_{seed}.bin")
    r = pignn.run(npzp, device, seed=seed, dump=bp, log_every=hp.pop("log_every", 0), **hp)
    out = subprocess.run([REPAIR, binp, bp], capture_output=True, text=True)
    j = json.loads(out.stdout)
    os.remove(bp)
    rec = dict(method=tag, d=d, n=n, seed=seed,
               size=j["repaired_size"], density=j["density"],
               raw_size=j["raw_size"], raw_violations_fixed=j["dropped"],
               added_by_maximalize=j["added"],
               t_build_s=r["t_build_s"], t_train_s=r["t_train_s"], t_post_s=j["t_post_s"],
               t_total_s=r["t_build_s"] + r["t_train_s"] + j["t_post_s"],
               epochs_run=r["epochs_run"], dim_embedding=r["dim_embedding"],
               hidden_dim=r["hidden_dim"], lr=r["lr"], device=device,
               final_loss=r["final_loss"], best_loss=r["best_loss"],
               violations=j["violations_after"], nonmaximal=j["nonmaximal_after"])
    return with_ar(rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--d", type=int, nargs="+", required=True)
    ap.add_argument("--n", type=int, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--methods", nargs="+",
                    default=["dga", "ga", "null_ones", "null_rand", "pignn_pub"])
    ap.add_argument("--epochs", type=int, default=100_000)
    ap.add_argument("--dim-embedding", type=int, default=None)
    ap.add_argument("--tag", default="pignn_pub")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--log-every", type=int, default=0)
    a = ap.parse_args()

    tmpdir = os.path.join(ROOT, "data", "tmp"); os.makedirs(tmpdir, exist_ok=True)
    done = load_done(a.out)

    for d in a.d:
        for n in a.n:
            for seed in a.seeds:
                import random as _r
                _r.seed(seed)
                import igraph as ig; ig.set_random_number_generator(_r)
                binp, npzp = gen_graphs.generate(n, d, seed)
                for mth in a.methods:
                    key = (mth if mth != "pignn_pub" else a.tag, d, n, seed)
                    if key in done:
                        print(f"  .. skip {key}", flush=True); continue
                    t0 = time.time()
                    try:
                        if mth in ("dga", "ga"):
                            rec = run_greedy(binp, mth, d, n, seed)
                        elif mth in ("null_ones", "null_rand"):
                            rec = run_control(binp, mth, d, n, seed, tmpdir)
                        elif mth == "pignn_pub":
                            rec = run_gnn(npzp, binp, d, n, seed, a.device, a.tag, tmpdir,
                                          epochs=a.epochs, dim_embedding=a.dim_embedding,
                                          lr=a.lr, log_every=a.log_every)
                        else:
                            raise ValueError(mth)
                    except Exception as e:
                        print(f"  !! FAILED {key}: {type(e).__name__}: {e}", flush=True)
                        continue
                    rec["wall_s"] = time.time() - t0
                    emit(a.out, rec)


if __name__ == "__main__":
    main()
