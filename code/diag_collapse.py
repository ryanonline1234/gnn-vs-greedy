"""Diagnose the d=20 collapse: is the published loss driving p -> 0?

The reference loss is the pure quadratic form  L(p) = p^T Q p  with Q_ii=-1, Q_uv=+2.
It has NO linear term, so grad L(0) = 0 exactly: the origin is a critical point of the
published objective for every d. The question is whether training falls into it.
"""
import numpy as np, torch, math, sys
from pignn import PIGNN, build_tensors, loss_fn

def probe(npz, penalty=2.0, seed=1, epochs=12000, device="cpu"):
    z = np.load(npz); edges, n, d = z["edges"], int(z["n"]), int(z["d"])
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device(device)
    src, dst, norm, eu, ev = build_tensors(edges, n, dev)
    dim = int(math.sqrt(n)); net = PIGNN(n, dim, max(1, dim//2), 0.0).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    trace = []
    for ep in range(epochs):
        p = net(src, dst, norm)
        loss = loss_fn(p, eu, ev, penalty)
        if ep % 2000 == 0 or ep == epochs-1:
            pd = p.detach()
            trace.append((ep, loss.item(), pd.mean().item(), pd.max().item(),
                          int((pd >= 0.5).sum().item())))
        opt.zero_grad(); loss.backward(); opt.step()
    return d, n, trace

# analytic check: uniform-p direction
print("Analytic: for uniform p, L(p) = n*p^2*(d-1)  -> minimised at p=0 for all d>1.")
print("          grad L(0) = 0 exactly (L is a pure quadratic form, no linear term),")
print("          so the all-zeros point is a critical point of the PUBLISHED objective.\n")
for f, pen in [("data/rrg_n1000_d3_s0.npz", 2.0), ("data/rrg_n1000_d20_s0.npz", 2.0),
               ("data/rrg_n1000_d20_s0.npz", 1.1)]:
    d, n, tr = probe(f, penalty=pen)
    print(f"--- d={d} n={n} penalty={pen}")
    print(f"    {'epoch':>7} {'loss':>12} {'mean p':>9} {'max p':>9} {'#(p>=0.5)':>10}")
    for (ep, l, mu, mx, c) in tr:
        print(f"    {ep:>7} {l:>12.3f} {mu:>9.5f} {mx:>9.5f} {c:>10}")
    print()
