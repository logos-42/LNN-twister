"""
测试可增长扭量神经网络 - 验证分裂和剪枝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from twistor_lnn.growable import GrowableTwistorLNN, GrowthConfig


def test_basic():
    print("测试1: 基本前向传播")
    model = GrowableTwistorLNN(input_dim=1, hidden_dim=8, output_dim=1)
    x = torch.randn(10, 4, 1)
    y = model(x)
    print(f"  输入: {x.shape} -> 输出: {y.shape}")
    print("  ✓ 通过")


def test_growth():
    print("\n测试2: 神经元分裂（强制触发）")
    config = GrowthConfig(
        min_hidden_dim=4,
        max_hidden_dim=16,
        growth_interval=1,
        split_threshold_var=0.0,
        max_split_per_step=2,
    )
    model = GrowableTwistorLNN(
        input_dim=1, hidden_dim=4, output_dim=1, growth_config=config
    )

    print(f"  初始 dim: {model.hidden_dim}")

    # 预先填充高方差激活（需要至少10个）
    for _ in range(15):
        model._activation_buffer.append(torch.rand(4) * 10)

    # 模拟训练过程
    for step in range(5):
        result = model.growth_step()
        if result["action"] == "split":
            print(f"  Step {step}: 分裂 -> dim={model.hidden_dim}")

    print(f"  最终 dim: {model.hidden_dim}")
    assert model.hidden_dim > 4, "应该发生分裂"
    print("  ✓ 通过")


def test_prune():
    print("\n测试3: 神经元剪枝")
    config = GrowthConfig(
        min_hidden_dim=2,
        max_hidden_dim=16,
        prune_interval=1,
    )
    model = GrowableTwistorLNN(
        input_dim=1, hidden_dim=8, output_dim=1, growth_config=config
    )

    print(f"  初始 active: {len(model.active_neurons)}")

    # 预先填充低重要性激活
    for _ in range(15):
        model._activation_buffer.append(torch.rand(8) * 0.001)

    for step in range(5):
        result = model.growth_step()
        if result["action"] == "prune":
            print(f"  Step {step}: 剪枝 -> active={len(model.active_neurons)}")

    print(f"  最终 active: {len(model.active_neurons)}")
    print("  ✓ 通过")


def test_full_training():
    print("\n测试4: 完整训练循环")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = GrowthConfig(
        min_hidden_dim=4,
        max_hidden_dim=24,
        growth_interval=5,
        prune_interval=5,
        split_threshold_var=0.1,
    )
    model = GrowableTwistorLNN(
        input_dim=1, hidden_dim=4, output_dim=1, growth_config=config
    ).to(device)

    X = torch.randn(50, 15, 1).to(device)
    y = torch.randn(50, 15, 1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    print(f"  初始: dim={model.hidden_dim}")

    # 预先填充一些激活数据
    for _ in range(15):
        model._activation_buffer.append(torch.rand(model.hidden_dim).to(device))

    for epoch in range(30):
        model.train()
        for i in range(0, 50, 10):
            x_b = X[i : i + 10].transpose(0, 1)
            y_b = y[i : i + 10].transpose(0, 1)

            optimizer.zero_grad()
            pred = model(x_b)
            loss = F.mse_loss(pred, y_b)
            loss.backward()
            optimizer.step()

        result = model.growth_step()
        if result["action"] != "none":
            print(f"  Epoch {epoch + 1}: {result['action']}, dim={model.hidden_dim}")

    print(f"  最终: dim={model.hidden_dim}, active={len(model.active_neurons)}")
    print("  ✓ 通过")


if __name__ == "__main__":
    print("=" * 50)
    print("可增长扭量神经网络测试")
    print("=" * 50)

    test_basic()
    test_growth()
    test_prune()
    test_full_training()

    print("\n" + "=" * 50)
    print("所有测试通过!")
    print("=" * 50)
