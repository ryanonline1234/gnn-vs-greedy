import time, sys, numpy as np, torch, math
from pignn import PIGNN, build_tensors, loss_fn
for npz in sys.argv[1:]:
    z = np.load(npz); edges, n, d = z["edges"], int(z["n"]), int(z["d"])
    row = [f"n={n:>7d} d={d:2d}"]
    for dev_s in ("cpu", "mps"):
        torch.manual_seed(1)
        dev = torch.device(dev_s)
        try:
            src, dst, norm, eu, ev = build_tensors(edges, n, dev)
            dim = int(math.sqrt(n)); net = PIGNN(n, dim, max(1, dim//2), 0.0).to(dev)
            opt = torch.optim.Adam(net.parameters(), lr=1e-4)
            for _ in range(5):   # warmup
                loss = loss_fn(net(src, dst, norm), eu, ev); opt.zero_grad(); loss.backward(); opt.step()
            if dev_s == "mps": torch.mps.synchronize()
            t0 = time.time(); K = 50
            for _ in range(K):
                loss = loss_fn(net(src, dst, norm), eu, ev); opt.zero_grad(); loss.backward(); opt.step()
            if dev_s == "mps": torch.mps.synchronize()
            ms = 1000*(time.time()-t0)/K
            row.append(f"{dev_s}={ms:8.2f} ms/epoch")
            del src, dst, norm, eu, ev, net, opt
            if dev_s == "mps": torch.mps.empty_cache()
        except Exception as e:
            row.append(f"{dev_s}=FAILED({type(e).__name__})")
    print("  ".join(row), flush=True)
