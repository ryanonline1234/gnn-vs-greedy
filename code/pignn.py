"""Faithful port of PI-GNN (Schuetz, Brubaker & Katzgraber, Nat Mach Intell 4, 367 (2022))
from the official DGL implementation (amazon-science/co-with-gnns-example) to PyTorch.

Faithfulness notes (see DECISIONS.md D4):
 * 2-layer GCN, symmetric normalisation D^-1/2 A D^-1/2, NO self-loops (DGL GraphConv
   default), relu + dropout between layers, sigmoid output.
 * Jointly-trained nn.Embedding(n, dim_embedding) as input; N(0,1) init as in torch.
 * xavier_uniform_ weights / zero bias, matching dgl.nn.pytorch.GraphConv.
 * Adam, lr=1e-4, <=1e5 epochs, tol=1e-4, patience=100, dropout=0.0, threshold=0.5.
 * Loss is the SPARSE form of the repo's dense p^T Q p with Q_ii=-1, Q_uv=+2 per edge:
       L(p) = -sum_i p_i^2 + 2 * sum_{(u,v) in E} p_u p_v
   The repo materialises a dense n x n Q (torch.zeros(n,n)), which is O(n^2) and cannot
   represent n=1e6 at all. The sparse form is mathematically identical and strictly
   more capable -- a deliberate improvement in the claim's favour.
 * Early-stopping bookkeeping replicated exactly, including best_loss initialised to the
   loss of the all-zeros bitstring.
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn


class GCNLayer(nn.Module):
    """D^-1/2 A D^-1/2 X W + b, no self-loops (matches dgl GraphConv norm='both')."""
    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_f, out_f))
        self.bias = nn.Parameter(torch.zeros(out_f))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, src, dst, norm):
        xw = x @ self.weight
        out = torch.zeros_like(xw)
        out.index_add_(0, dst, xw[src] * norm.unsqueeze(1))
        return out + self.bias


class SAGELayer(nn.Module):
    """GraphSAGE (mean aggregator), the architecture the Reply to Angelini adopts:
       h_i' = W_self h_i + W_neigh * mean_{j in N(i)} h_j + b
    Implemented to match dgl.nn.SAGEConv(aggregator_type='mean') semantics."""
    def __init__(self, in_f, out_f):
        super().__init__()
        self.w_self = nn.Parameter(torch.empty(in_f, out_f))
        self.w_neigh = nn.Parameter(torch.empty(in_f, out_f))
        self.bias = nn.Parameter(torch.zeros(out_f))
        nn.init.xavier_uniform_(self.w_self); nn.init.xavier_uniform_(self.w_neigh)

    def forward(self, x, src, dst, norm):
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])
        deg = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        agg = agg / deg.clamp(min=1).unsqueeze(1)
        return x @ self.w_self + agg @ self.w_neigh + self.bias


class PIGNN(nn.Module):
    def __init__(self, n, dim_embedding, hidden_dim, dropout, arch="gcn"):
        super().__init__()
        self.embed = nn.Embedding(n, dim_embedding)
        L = SAGELayer if arch == "sage" else GCNLayer
        self.conv1 = L(dim_embedding, hidden_dim)
        self.conv2 = L(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, src, dst, norm):
        h = self.conv1(self.embed.weight, src, dst, norm)
        h = torch.relu(h)
        if self.dropout > 0:
            h = torch.nn.functional.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, src, dst, norm)
        return torch.sigmoid(h).squeeze(1)


def build_tensors(edges, n, device):
    """edges: (m,2) int32 unique undirected edges."""
    e = torch.from_numpy(edges.astype(np.int64))
    eu, ev = e[:, 0], e[:, 1]
    src = torch.cat([eu, ev]).to(device)          # both directions for propagation
    dst = torch.cat([ev, eu]).to(device)
    deg = torch.zeros(n, dtype=torch.float32, device=device)
    deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
    deg = deg.clamp(min=1)
    norm = (deg[src].rsqrt() * deg[dst].rsqrt())
    return src, dst, norm, eu.to(device), ev.to(device)


def loss_fn(p, eu, ev, penalty=2.0, diag="quadratic"):
    """diag='quadratic' is the PUBLISHED loss: p^T Q p with Q_ii=-1, i.e. -sum p_i^2.
    It is a pure quadratic form, so grad L(0)=0 and the origin is a critical point.
    diag='linear' replaces the diagonal with -sum p_i. For BINARY x the two are
    identical (x^2 = x); for the CONTINUOUS relaxation they differ, and the linear
    form has grad_i L(0) = -1 < 0, removing the trivial critical point at the origin.
    This is a MODIFICATION of the published objective and is always reported as such."""
    pair = penalty * (p[eu] * p[ev]).sum()
    return (-(p * p).sum() if diag == "quadratic" else -p.sum()) + pair


def run(npz, device_str, dim_embedding=None, hidden_dim=None, lr=1e-4, epochs=100_000,
        tol=1e-4, patience=100, dropout=0.0, threshold=0.5, seed=1, penalty=2.0,
        dump=None, log_every=2000, diag="quadratic", arch="gcn"):
    z = np.load(npz)
    edges, n, d = z["edges"], int(z["n"]), int(z["d"])
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device(device_str)

    if dim_embedding is None:
        dim_embedding = int(math.sqrt(n))          # notebook formula
    if hidden_dim is None:
        hidden_dim = max(1, int(dim_embedding / 2))

    t_build0 = time.time()
    src, dst, norm, eu, ev = build_tensors(edges, n, device)
    net = PIGNN(n, dim_embedding, hidden_dim, dropout, arch=arch).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    if device_str == "mps":
        torch.mps.synchronize()
    t_build = time.time() - t_build0

    prev_loss, count = 1.0, 0
    best_loss = torch.tensor(0.0, device=device)   # loss of all-zeros bitstring
    best_bits = torch.zeros(n, dtype=torch.int8, device=device)
    stopped_epoch = epochs - 1

    t0 = time.time()
    net.train()
    for epoch in range(epochs):
        p = net(src, dst, norm)
        loss = loss_fn(p, eu, ev, penalty, diag)
        loss_ = loss.item()
        bits = (p.detach() >= threshold).to(torch.int8)
        if loss < best_loss:
            best_loss = loss.detach()
            best_bits = bits
        if (abs(loss_ - prev_loss) <= tol) or ((loss_ - prev_loss) > 0):
            count += 1
        else:
            count = 0
        if count >= patience:
            stopped_epoch = epoch
            break
        prev_loss = loss_
        opt.zero_grad(); loss.backward(); opt.step()
        if log_every and epoch % log_every == 0:
            print(f"    epoch {epoch} loss {loss_:.4f}", flush=True)
    if device_str == "mps":
        torch.mps.synchronize()
    t_train = time.time() - t0

    final_bits = (p.detach() >= threshold).to(torch.int8)
    bb = best_bits.cpu().numpy().astype(np.int8)
    fb = final_bits.cpu().numpy().astype(np.int8)
    # the reference reports the best-continuous-loss bitstring; keep whichever is larger
    chosen = bb if bb.sum() >= fb.sum() else fb

    if dump:
        with open(dump, "wb") as f:
            f.write(np.array([n], dtype=np.int32).tobytes())
            f.write(chosen.tobytes())

    return dict(n=n, d=d, dim_embedding=dim_embedding, hidden_dim=hidden_dim, lr=lr,
                penalty=penalty, dropout=dropout, seed=seed, device=device_str, diag=diag, arch=arch,
                patience=patience,
                epochs_run=stopped_epoch + 1, t_build_s=t_build, t_train_s=t_train,
                raw_size_best=int(bb.sum()), raw_size_final=int(fb.sum()),
                raw_size=int(chosen.sum()), final_loss=float(loss_),
                best_loss=float(best_loss.item()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dim-embedding", type=int, default=None)
    ap.add_argument("--hidden-dim", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=100_000)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--penalty", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dump", default=None)
    ap.add_argument("--log-every", type=int, default=2000)
    ap.add_argument("--diag", default="quadratic", choices=["quadratic","linear"])
    ap.add_argument("--patience", type=int, default=100)
    a = ap.parse_args()
    r = run(a.npz, a.device, a.dim_embedding, a.hidden_dim, a.lr, a.epochs,
            patience=a.patience, dropout=a.dropout, penalty=a.penalty, seed=a.seed,
            dump=a.dump, log_every=a.log_every, diag=a.diag)
    print(json.dumps(r))
