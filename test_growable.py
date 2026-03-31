"""
测试可增长扭量神经网络
=====================
验证神经元分裂、剪枝机制是否正常工作
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

sys.path.insert(0, ".")

from twistor_lnn.growable import GrowableTwistorLNN, GrowthConfig


def generate_sine_data(n_samples=200, seq_len=30, input_dim=1):
    """生成正弦波数据"""
    X, y = [], []

    for _ in range(n_samples):
        freq = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2 * np.pi)
        t = np.linspace(0, 4 * np.pi, seq_len + 1)
        signal = np.sin(freq * t + phase)

        X.append(signal[:-1].reshape(seq_len, input_dim))
        y.append(signal[1:].reshape(seq_len, 1))

    X = torch.FloatTensor(np.stack(X))
    y = torch.FloatTensor(np.stack(y))

    return X, y


def train_growable_model():
    """训练可增长模型"""
    print("=" * 60)
    print("可增长扭量神经网络测试")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    config = GrowthConfig(
        min_hidden_dim=8,
        max_hidden_dim=64,
        split_threshold_grad=0.2,
        split_threshold_var=0.3,
        split_threshold_sens=0.2,
        prune_threshold=0.05,
        growth_interval=30,
        prune_interval=15,
        noise_scale=0.1,
        topology_penalty=0.001,
        max_split_per_step=2,
        max_prune_per_step=2,
    )

    model = GrowableTwistorLNN(
        input_dim=1,
        hidden_dim=8,
        output_dim=1,
        growth_config=config,
        enable_growth=True,
        sparsity=0.3,
        multi_scale_tau=True,
    ).to(device)

    print(f"\n初始参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"初始 hidden_dim: {model.hidden_dim}")
    print(f"激活神经元: {len(model.active_neurons)}")

    X, y = generate_sine_data(n_samples=500, seq_len=30)
    X, y = X.to(device), y.to(device)

    n_epochs = 200
    batch_size = 32
    lr = 0.01

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\n开始训练 ({n_epochs} epochs)...")
    print("-" * 60)

    history = {
        "loss": [],
        "mse": [],
        "hidden_dim": [],
        "growth_events": [],
    }

    for epoch in range(n_epochs):
        model.train()

        perm = torch.randperm(len(X))
        X = X[perm]
        y = y[perm]

        epoch_loss = 0.0
        epoch_mse = 0.0
        n_batches = len(X) // batch_size

        growth_event = None

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size

            x_batch = X[start:end].transpose(0, 1)
            y_batch = y[start:end].transpose(0, 1)

            optimizer.zero_grad()

            pred = model(x_batch)
            mse_loss = F.mse_loss(pred, y_batch)

            topology_loss = model.compute_topology_penalty()

            loss = mse_loss + topology_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mse += mse_loss.item()

        epoch_loss /= n_batches
        epoch_mse /= n_batches

        result = model.growth_step()

        if result["action"] != "none":
            growth_event = result
            print(f"Epoch {epoch + 1}: {result}")

        history["loss"].append(epoch_loss)
        history["mse"].append(epoch_mse)
        history["hidden_dim"].append(model.hidden_dim)

        if (epoch + 1) % 20 == 0:
            diag = model.get_diagnostics()
            print(
                f"Epoch {epoch + 1:3d} | Loss: {epoch_loss:.4f} | MSE: {epoch_mse:.4f} | "
                f"Dim: {diag['hidden_dim']} | Active: {diag['active_count']}"
            )

    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最终参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"最终 hidden_dim: {model.hidden_dim}")

    diag = model.get_diagnostics()
    print(f"激活神经元: {diag['active_count']}")
    print(f"训练步数: {diag['training_step']}")

    print("\n" + "=" * 60)
    print("测试前向传播...")

    model.eval()
    with torch.no_grad():
        test_x = torch.randn(20, 4, 1).to(device)
        test_y = model(test_x)
        print(f"输入形状: {test_x.shape}")
        print(f"输出形状: {test_y.shape}")

    print("\n测试完成!")

    return model, history


if __name__ == "__main__":
    train_growable_model()
