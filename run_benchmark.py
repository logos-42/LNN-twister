"""
极简基准测试
"""

import torch, torch.nn.functional as F, numpy as np, time, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twistor_lnn.core import TwistorLNN
from twistor_lnn.growable import GrowableTwistorLNN
from twistor_lnn.datasets import create_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"设备: {device}")


def train1(model, X, y, epochs=10, bs=16, lr=0.01):
    n = len(X)
    nv = int(n * 0.2)
    idx = torch.randperm(n)
    Xt, Xv = X[idx[nv:]].to(device), X[idx[:nv]].to(device)
    yt, yv = y[idx[nv:]].to(device), y[idx[:nv]].to(device)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = float("inf")
    for _ in range(epochs):
        model.train()
        p = torch.randperm(len(Xt))
        Xs, ys = Xt[p], yt[p]
        nb = max(len(Xs) // bs, 1)
        for i in range(nb):
            s, e = i * bs, (i + 1) * bs
            xb, yb = Xs[s:e].transpose(0, 1), ys[s:e].transpose(0, 1)
            opt.zero_grad()
            loss = F.mse_loss(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = F.mse_loss(model(Xv.transpose(0, 1)), yv.transpose(0, 1)).item()
        if vl < best:
            best = vl
    return model, best


def speed(m, x, n=5):
    m.eval()
    ts = []
    with torch.no_grad():
        for _ in range(n):
            t0 = time.perf_counter()
            _ = m(x)
            ts.append(time.perf_counter() - t0)
    return np.mean(ts) * 1000


for task in ["sine"]:
    print(f"\n{'=' * 60}\n{task.upper()}\n{'=' * 60}")
    X, y = create_dataset(task, n_samples=100, seq_len=30, device=device)
    idim, odim, sl = X.shape[2], y.shape[2], X.shape[1]
    print(f"数据: {X.shape}")

    for name, mk in [
        ("Base-16", lambda: TwistorLNN(idim, 16, odim)),
        ("Base-32", lambda: TwistorLNN(idim, 32, odim)),
    ]:
        t0 = time.perf_counter()
        m = mk()
        m, b = train1(m, X, y, epochs=10)
        tt = time.perf_counter() - t0
        p = sum(pp.numel() for pp in m.parameters())
        ms = speed(m, torch.randn(sl, 8, idim, device=device))
        print(f"  {name:<12s} MSE={b:.6f} p={p:,} {ms:.0f}ms train={tt:.1f}s")

    for name, hd in [("MR-16", 16), ("MR-32", 32)]:
        t0 = time.perf_counter()
        m = TwistorLNN(idim, hd, odim)
        m.enable_mobius_resonance(mobius_strength=0.05, resonance_strength=0.05)
        m, b = train1(m, X, y, epochs=10)
        tt = time.perf_counter() - t0
        p = sum(pp.numel() for pp in m.parameters())
        ms = speed(m, torch.randn(sl, 8, idim, device=device))
        info = m.get_mobius_info() if m.mobius else {}
        print(
            f"  {name:<12s} MSE={b:.6f} p={p:,} {ms:.0f}ms train={tt:.1f}s mode={info.get('mode', '?')}"
        )

    for name, gd in [("Grow-8", 8), ("GrowMR-8", 8)]:
        t0 = time.perf_counter()
        m = GrowableTwistorLNN(
            input_dim=idim,
            hidden_dim=0,
            output_dim=odim,
            enable_growth=True,
            enable_mobius="MR" in name,
            enable_resonance="MR" in name,
            mobius_strength=0.05,
            resonance_strength=0.05,
        )
        m.force_grow_to(gd)
        print(f"  {name:<12s} grow_to({gd}) -> dim={m.hidden_dim}")
        if m.hidden_dim < 2:
            m = GrowableTwistorLNN(
                input_dim=idim,
                hidden_dim=4,
                output_dim=odim,
                enable_growth=False,
                enable_mobius="MR" in name,
                enable_resonance="MR" in name,
                mobius_strength=0.05,
                resonance_strength=0.05,
            )
        m, b = train1(m, X, y, epochs=10, lr=0.005)
        tt = time.perf_counter() - t0
        p = sum(pp.numel() for pp in m.parameters())
        ms = speed(m, torch.randn(sl, 8, idim, device=device))
        ds = f" dim={m.hidden_dim}"
        if m.mobius:
            info = m.get_mobius_info()
            ds += f" mode={info['mode']}"
        print(f"  {name:<12s} MSE={b:.6f} p={p:,} {ms:.0f}ms train={tt:.1f}s{ds}")
