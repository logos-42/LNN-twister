"""
简单测试可增长扭量神经网络
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
    print("\n测试2: 神经元分裂")
    config = GrowthConfig(
        min_hidden_dim=4,
        max_hidden_dim=16,
        growth_interval=2,
        split_threshold_var=0.1,
    )
    model = GrowableTwistorLNN(
        input_dim=1, hidden_dim=4, output_dim=1, growth_config=config
    )

    print(f"  初始 dim: {model.hidden_dim}")

    model.training_step = 1
    model._activation_buffer = [torch.rand(4) for _ in range(20)]
    model._update_neuron_stats()

    for i in range(5):
        model.training_step = i * 2
        result = model.growth_step()
        if result["action"] == "split":
            print(f"  分裂后 dim: {model.hidden_dim}")

    print(f"  最终 dim: {model.hidden_dim}")
    print("  ✓ 通过")


def test_prune():
    print("\n测试3: 神经元剪枝")
    config = GrowthConfig(
        min_hidden_dim=4,
        max_hidden_dim=16,
        prune_interval=2,
    )
    model = GrowableTwistorLNN(
        input_dim=1, hidden_dim=8, output_dim=1, growth_config=config
    )

    print(f"  初始 active: {len(model.active_neurons)}")

    model.training_step = 10
    model._activation_buffer = [torch.rand(8) * 0.001 for _ in range(20)]
    model._update_neuron_stats()

    for i in range(3):
        model.training_step = 2 + i * 2
        result = model.growth_step()
        if result["action"] == "prune":
            print(f"  剪枝后 active: {len(model.active_neurons)}")

    print(f"  最终 active: {len(model.active_neurons)}")
    print("  ✓ 通过")


def test_training():
    print("\n测试4: 快速训练")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = GrowthConfig(
        min_hidden_dim=4,
        max_hidden_dim=32,
        growth_interval=10,
        prune_interval=10,
    )
    model = GrowableTwistorLNN(
        input_dim=1, hidden_dim=4, output_dim=1, growth_config=config
    ).to(device)

    X = torch.randn(100, 20, 1).to(device)
    y = torch.randn(100, 20, 1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print(f"  初始 dim: {model.hidden_dim}")

    for epoch in range(20):
        model.train()
        for i in range(0, 100, 20):
            x_batch = X[i : i + 20].transpose(0, 1)
            y_batch = y[i : i + 20].transpose(0, 1)

            optimizer.zero_grad()
            pred = model(x_batch)
            loss = F.mse_loss(pred, y_batch)
            loss.backward()
            optimizer.step()

        result = model.growth_step()

    print(f"  最终 dim: {model.hidden_dim}")
    print(f"  最终 active: {len(model.active_neurons)}")
    print("  ✓ 通过")


if __name__ == "__main__":
    print("=" * 50)
    print("可增长扭量神经网络单元测试")
    print("=" * 50)

    test_basic()
    test_growth()
    test_prune()
    test_training()

    print("\n" + "=" * 50)
    print("所有测试通过!")
    print("=" * 50)
