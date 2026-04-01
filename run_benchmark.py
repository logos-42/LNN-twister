"""
快速基准测试 - 验证模型效果和增长能力
=========================================
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from twistor_lnn.core import TwistorLNN
from twistor_lnn.growable import GrowableTwistorLNN
from twistor_lnn.datasets import create_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"设备: {device}")
print(f"PyTorch: {torch.__version__}")


def quick_train(model, X, y, epochs=10, batch_size=16, lr=0.01):
    """快速训练"""
    n = len(X)
    n_val = int(n * 0.2)
    idx = torch.randperm(n)
    Xt, Xv = X[idx[n_val:]].to(device), X[idx[:n_val]].to(device)
    yt, yv = y[idx[n_val:]].to(device), y[idx[:n_val]].to(device)

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        Xt_s, yt_s = Xt[perm], yt[perm]
        nb = max(len(Xt_s) // batch_size, 1)
        for i in range(nb):
            s, e = i * batch_size, (i + 1) * batch_size
            xb = Xt_s[s:e].transpose(0, 1)
            yb = yt_s[s:e].transpose(0, 1)
            opt.zero_grad()
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            xv = Xv.transpose(0, 1)
            yv_t = yv.transpose(0, 1)
            pv = model(xv)
            vl = F.mse_loss(pv, yv_t).item()
        if vl < best_val:
            best_val = vl

    return model, best_val


def speed_test(model, x, n=10):
    """速度测试"""
    model.eval()
    ts = []
    with torch.no_grad():
        for _ in range(n):
            t0 = time.perf_counter()
            _ = model(x)
            ts.append(time.perf_counter() - t0)
    return np.mean(ts) * 1000


def run_task(task):
    print(f"\n{'=' * 70}")
    print(f"任务: {task.upper()}")
    print(f"{'=' * 70}")

    ns = 100
    sl = 25
    X, y = create_dataset(task, n_samples=ns, seq_len=sl, device=device)
    idim, odim = X.shape[2], y.shape[2]
    print(f"数据: X={X.shape}, y={y.shape}")

    results = {}

    # 1. Base-16
    print(f"\n  [Base-16]")
    m = TwistorLNN(idim, 16, odim)
    m, best = quick_train(m, X, y, epochs=10)
    params = sum(p.numel() for p in m.parameters())
    ms = speed_test(m, torch.randn(sl, 8, idim, device=device))
    results["Base-16"] = {"mse": best, "params": params, "ms": ms}
    print(f"    MSE={best:.6f} | params={params:,} | {ms:.1f}ms")

    # 2. Base-32
    print(f"\n  [Base-32]")
    m = TwistorLNN(idim, 32, odim)
    m, best = quick_train(m, X, y, epochs=10)
    params = sum(p.numel() for p in m.parameters())
    ms = speed_test(m, torch.randn(sl, 8, idim, device=device))
    results["Base-32"] = {"mse": best, "params": params, "ms": ms}
    print(f"    MSE={best:.6f} | params={params:,} | {ms:.1f}ms")

    # 3. MR-16
    print(f"\n  [MR-16]")
    m = TwistorLNN(idim, 16, odim)
    m.enable_mobius_resonance(mobius_strength=0.05, resonance_strength=0.05)
    m, best = quick_train(m, X, y, epochs=10)
    params = sum(p.numel() for p in m.parameters())
    ms = speed_test(m, torch.randn(sl, 8, idim, device=device))
    results["MR-16"] = {"mse": best, "params": params, "ms": ms}
    print(f"    MSE={best:.6f} | params={params:,} | {ms:.1f}ms")
    if m.mobius:
        info = m.get_mobius_info()
        print(f"    流形: mode={info['mode']}, dim={info['manifold_dim']}")

    # 4. MR-32
    print(f"\n  [MR-32]")
    m = TwistorLNN(idim, 32, odim)
    m.enable_mobius_resonance(mobius_strength=0.05, resonance_strength=0.05)
    m, best = quick_train(m, X, y, epochs=10)
    params = sum(p.numel() for p in m.parameters())
    ms = speed_test(m, torch.randn(sl, 8, idim, device=device))
    results["MR-32"] = {"mse": best, "params": params, "ms": ms}
    print(f"    MSE={best:.6f} | params={params:,} | {ms:.1f}ms")

    # 5. Growable
    print(f"\n  [Growable]")
    m = GrowableTwistorLNN(
        input_dim=idim, hidden_dim=0, output_dim=odim, enable_growth=True
    )
    for _ in range(30):
        m.growth_step()
    if m.hidden_dim == 0:
        m = GrowableTwistorLNN(
            input_dim=idim, hidden_dim=2, output_dim=odim, enable_growth=False
        )
    print(f"    初始 dim={m.hidden_dim}")
    m, best = quick_train(m, X, y, epochs=10, lr=0.005)
    params = sum(p.numel() for p in m.parameters())
    ms = speed_test(m, torch.randn(sl, 8, idim, device=device))
    results["Growable"] = {"mse": best, "params": params, "ms": ms, "dim": m.hidden_dim}
    print(f"    MSE={best:.6f} | params={params:,} | {ms:.1f}ms | dim={m.hidden_dim}")

    # 6. Growable + MR
    print(f"\n  [GrowableMR]")
    m = GrowableTwistorLNN(
        input_dim=idim,
        hidden_dim=0,
        output_dim=odim,
        enable_growth=True,
        enable_mobius=True,
        enable_resonance=True,
        mobius_strength=0.05,
        resonance_strength=0.05,
    )
    for _ in range(30):
        m.growth_step()
    if m.hidden_dim == 0:
        m = GrowableTwistorLNN(
            input_dim=idim,
            hidden_dim=2,
            output_dim=odim,
            enable_growth=False,
            enable_mobius=True,
            enable_resonance=True,
            mobius_strength=0.05,
            resonance_strength=0.05,
        )
    print(f"    初始 dim={m.hidden_dim}")
    m, best = quick_train(m, X, y, epochs=10, lr=0.005)
    params = sum(p.numel() for p in m.parameters())
    ms = speed_test(m, torch.randn(sl, 8, idim, device=device))
    results["GrowableMR"] = {
        "mse": best,
        "params": params,
        "ms": ms,
        "dim": m.hidden_dim,
    }
    print(f"    MSE={best:.6f} | params={params:,} | {ms:.1f}ms | dim={m.hidden_dim}")
    if m.mobius:
        info = m.get_mobius_info()
        print(f"    流形: mode={info['mode']}, dim={info['manifold_dim']}")

    # Summary
    print(f"\n  {'模型':<18s} {'MSE':>10s} {'参数':>8s} {'ms':>6s}")
    print(f"  {'-' * 46}")
    for name, r in results.items():
        ds = f"(d={r['dim']})" if "dim" in r else ""
        print(f"  {name + ds:<18s} {r['mse']:>10.6f} {r['params']:>8,} {r['ms']:>6.1f}")

    return results


if __name__ == "__main__":
    all_res = {}
    for task in ["sine", "lorenz"]:
        all_res[task] = run_task(task)

    print(f"\n{'=' * 70}")
    print("总结")
    print(f"{'=' * 70}")
    for task, res in all_res.items():
        print(f"\n{task}:")
        for name, r in res.items():
            ds = f" (dim={r['dim']})" if "dim" in r else ""
            print(
                f"  {name}{ds}: MSE={r['mse']:.6f}, {r['params']:,}p, {r['ms']:.1f}ms"
            )
