"""Equivalence tests: our sparse port vs the reference implementation's dense math.

Reference (amazon-science/co-with-gnns-example):
  qubo_dict_to_torch + gen_q_dict_mis  ->  dense Q, Q[u][v]=2 per nx edge (one direction),
                                           Q[i][i] = -1
  loss_func(probs, Q)                  ->  probs.T @ Q @ probs
  dgl GraphConv(norm='both')           ->  D^-1/2 A D^-1/2 X W + b, no self-loops
"""
import numpy as np, torch, sys
from collections import defaultdict
from pignn import GCNLayer, build_tensors, loss_fn

torch.manual_seed(0); np.random.seed(0)
ok = True

for (n, d) in [(30, 3), (60, 5), (50, 20)]:
    import igraph as ig, random
    random.seed(7); ig.set_random_number_generator(random)
    g = ig.Graph.K_Regular(n, d)
    edges = np.array(g.get_edgelist(), dtype=np.int32)

    # ---------- 1. dense Q loss (reference) vs sparse loss (ours) ----------
    Q = torch.zeros(n, n, dtype=torch.float64)
    Q_dic = defaultdict(int)
    for (u, v) in edges:
        Q_dic[(int(u), int(v))] = 2            # penalty, one direction, as in gen_q_dict_mis
    for u in range(n):
        Q_dic[(u, u)] = -1
    for (x, y), val in Q_dic.items():
        Q[x][y] = val

    p = torch.rand(n, dtype=torch.float64)
    ref = (p.unsqueeze(1).T @ Q @ p.unsqueeze(1)).squeeze()
    src, dst, norm, eu, ev = build_tensors(edges, n, torch.device("cpu"))
    ours = loss_fn(p, eu.long(), ev.long(), penalty=2.0)
    dl = abs(ref.item() - ours.item())
    print(f"n={n:3d} d={d:2d}  loss  ref={ref.item():.10f} ours={ours.item():.10f} |d|={dl:.3e}")
    if dl > 1e-9: ok = False

    # ---------- 2. dense GCN propagation vs our index_add layer ----------
    A = np.zeros((n, n))
    for (u, v) in edges: A[u][v] = 1; A[v][u] = 1
    deg = A.sum(1)
    Dm = np.diag(1.0 / np.sqrt(deg))
    An = torch.tensor(Dm @ A @ Dm, dtype=torch.float32)

    in_f, out_f = 8, 4
    layer = GCNLayer(in_f, out_f)
    with torch.no_grad():
        layer.bias.copy_(torch.randn(out_f))
    x = torch.randn(n, in_f)
    ref2 = An @ (x @ layer.weight) + layer.bias
    ours2 = layer(x, src, dst, norm)
    dg = (ref2 - ours2).abs().max().item()
    print(f"           gcn   max|ref-ours| = {dg:.3e}")
    if dg > 1e-4: ok = False

    # ---------- 3. self-loop check: DGL GraphConv adds none ----------
    assert np.allclose(np.diag(A), 0), "adjacency must have no self-loops"

print("\nPORT EQUIVALENCE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
