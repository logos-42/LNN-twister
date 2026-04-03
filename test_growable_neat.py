"""
测试 GrowableTwistorLNN - NEAT风格实现 - 增强版
"""

import torch
import torch.nn as nn
import torch.optim as optim
from twistor_lnn.growable import GrowableTwistorLNN, GrowthConfig


def test_forward_pass():
    """测试前向传播"""
    print("=" * 60)
    print("测试前向传播")
    print("=" * 60)

    model = GrowableTwistorLNN(
        input_dim=2,
        hidden_dim=4,
        output_dim=1,
    )

    T, B = 10, 4
    x = torch.randn(T, B, 2)

    y = model(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")

    assert y.shape == (T, B, 1), f"输出形状错误: {y.shape}"
    print("✓ 前向传播测试通过")


def test_minimal_to_network():
    """测试从最小结构增长到网络"""
    print("\n" + "=" * 60)
    print("测试从最小结构增长到网络")
    print("=" * 60)

    config = GrowthConfig(
        min_hidden_dim=0,
        max_hidden_dim=16,
        prob_add_node=0.8,
        prob_add_connection=0.5,
        growth_interval=10,
    )

    model = GrowableTwistorLNN(
        input_dim=2,
        hidden_dim=0,
        output_dim=1,
        growth_config=config,
        enable_growth=True,
    )

    print(f"初始: hidden_dim={model.hidden_dim}")

    for step in range(500):
        if step > 60 and model.hidden_dim > 0:
            x = torch.randn(5, 4, 2)
            with torch.no_grad():
                model(x)

        result = model.growth_step()
        if step % 50 == 0:
            diag = model.get_diagnostics()
            print(
                f"Step {step}: action={result['action']}, dim={diag['hidden_dim']}, conn={diag['connection_count']}"
            )

    print(f"\n最终hidden_dim: {model.hidden_dim}")
    print(f"最终连接基因数: {model.get_diagnostics()['connection_count']}")

    return model.hidden_dim > 0


def test_training_with_growth():
    """测试带增长的训练"""
    print("\n" + "=" * 60)
    print("测试带增长的训练")
    print("=" * 60)

    config = GrowthConfig(
        min_hidden_dim=0,
        max_hidden_dim=16,
        prob_add_node=0.5,
        prob_add_connection=0.3,
        growth_interval=20,
    )

    model = GrowableTwistorLNN(
        input_dim=2,
        hidden_dim=1,
        output_dim=1,
        growth_config=config,
        enable_growth=True,
    )

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    T, B = 5, 8

    for epoch in range(200):
        x = torch.randn(T, B, 2)
        y_target = torch.randn(T, B, 1)

        optimizer.zero_grad()

        y_pred = model(x)

        if model.hidden_dim == 0:
            loss = torch.tensor(0.0, requires_grad=True)
        else:
            loss = criterion(y_pred, y_target)
            loss.backward()
            optimizer.step()

        model.growth_step()

        if epoch % 40 == 0:
            diag = model.get_diagnostics()
            loss_val = loss.item() if model.hidden_dim > 0 else 0.0
            print(
                f"Epoch {epoch}: dim={diag['hidden_dim']}, conn={diag['connection_count']}, loss={loss_val:.4f}"
            )

    print(f"\n最终hidden_dim: {model.hidden_dim}")
    print(f"最终连接基因数: {model.get_diagnostics()['connection_count']}")

    return model.hidden_dim >= 1


def test_neuron_split():
    """测试神经元分裂 - 强制增长"""
    print("\n" + "=" * 60)
    print("测试神经元分裂")
    print("=" * 60)

    config = GrowthConfig(
        min_hidden_dim=0,
        max_hidden_dim=8,
        prob_add_node=1.0,
        growth_interval=5,
    )

    model = GrowableTwistorLNN(
        input_dim=2,
        hidden_dim=0,
        output_dim=1,
        growth_config=config,
        enable_growth=True,
    )

    print(f"初始: hidden_dim={model.hidden_dim}")

    for step in range(500):
        model.training_step = step
        model.growth_step()

        if step % 50 == 0:
            print(
                f"Step {step}: hidden_dim={model.hidden_dim}, conn={len(model.connection_genes)}"
            )

    print(f"\n最终: hidden_dim={model.hidden_dim}")
    print(f"连接基因: {len(model.connection_genes)}")

    enabled = len([g for g in model.connection_genes if g.enabled])
    print(f"启用连接: {enabled}")

    return model.hidden_dim >= 2


def test_pruning():
    """测试剪枝"""
    print("\n" + "=" * 60)
    print("测试剪枝")
    print("=" * 60)

    config = GrowthConfig(
        min_hidden_dim=1,
        max_hidden_dim=8,
        connection_threshold=0.5,
    )

    model = GrowableTwistorLNN(
        input_dim=2,
        hidden_dim=4,
        output_dim=1,
        growth_config=config,
    )

    print(f"初始hidden_dim: {model.hidden_dim}")

    with torch.no_grad():
        model.sparse_mask_real.data[:, :] = -10
        model.sparse_mask_imag.data[:, :] = -10

    n_pruned = model.prune_connections()
    print(f"剪枝连接数: {n_pruned}")

    return True


def test_full_training():
    """完整训练测试"""
    print("\n" + "=" * 60)
    print("完整训练测试")
    print("=" * 60)

    config = GrowthConfig(
        min_hidden_dim=0,
        max_hidden_dim=8,
        prob_add_node=0.4,
        prob_add_connection=0.3,
        growth_interval=30,
    )

    model = GrowableTwistorLNN(
        input_dim=2,
        hidden_dim=0,
        output_dim=1,
        growth_config=config,
        enable_growth=True,
    )

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    T, B = 5, 8

    print("开始训练...")
    print(f"初始: hidden_dim={model.hidden_dim}")

    for epoch in range(300):
        x = torch.randn(T, B, 2)
        y_target = x[:, :, 0:1] * 0.5 + x[:, :, 1:2] * 0.3 + torch.randn(T, B, 1) * 0.1

        optimizer.zero_grad()

        y_pred = model(x)

        if model.hidden_dim > 0:
            loss = criterion(y_pred, y_target)
            loss.backward()
            optimizer.step()

        model.growth_step()

        if epoch % 60 == 0:
            diag = model.get_diagnostics()
            loss_val = loss.item() if model.hidden_dim > 0 else 0.0
            print(
                f"Epoch {epoch}: dim={diag['hidden_dim']}, conn={diag['connection_count']}, loss={loss_val:.4f}"
            )

    print(f"\n最终hidden_dim: {model.hidden_dim}")
    print(f"最终连接基因数: {model.get_diagnostics()['connection_count']}")

    print("\n测试预测...")
    x_test = torch.randn(1, 1, 2)
    with torch.no_grad():
        y_test = model(x_test)
    print(f"输入: {x_test[0, 0].numpy()}")
    print(f"输出: {y_test[0, 0].numpy()}")

    return model.hidden_dim >= 1


def test_developmental_phases():
    """测试大脑发育时间表机制"""
    print("\n" + "=" * 60)
    print("测试大脑发育时间表 (5阶段, ×100量级)")
    print("=" * 60)

    config = GrowthConfig(
        min_hidden_dim=0,
        max_hidden_dim=8192,
        prob_add_node=0.5,
        prob_add_connection=0.3,
        growth_interval=5,
        prune_interval=10,
        enable_developmental_schedule=True,
    )

    model = GrowableTwistorLNN(
        input_dim=2,
        hidden_dim=0,
        output_dim=1,
        growth_config=config,
        enable_growth=True,
    )

    phase_history = []
    current_phase_name = None

    for step in range(1500):
        if model.hidden_dim > 0 and model.hidden_dim < 256:
            x = torch.randn(3, 2, 2)
            with torch.no_grad():
                model(x)

        result = model.growth_step()
        phase = model._get_current_developmental_phase()

        if phase.name != current_phase_name:
            if current_phase_name is not None:
                phase_history.append(
                    {
                        "phase": current_phase_name,
                        "end_dim": model.hidden_dim,
                        "end_conn": model.get_diagnostics()["connection_count"],
                        "end_active": len(
                            [
                                s
                                for s in model.neuron_states
                                if s.active and s.neuron_type == "hidden"
                            ]
                        ),
                    }
                )
            current_phase_name = phase.name
            print(f"\n{'=' * 60}")
            print(f"🧠 阶段: {phase.name}")
            print(f"📝 {phase.description}")
            print(
                f"🎯 目标: {phase.target_neurons}神经元, {phase.target_connections_per_neuron}连接/神经元"
            )
            print(
                f"📊 生长率={phase.growth_rate}, 修剪率={phase.prune_rate}, 可塑性={phase.plasticity}"
            )
            print(f"{'=' * 60}")

        if step % 50 == 0 or step in [
            p.start_step for p in config.developmental_phases
        ]:
            diag = model.get_diagnostics()
            active = len(
                [
                    s
                    for s in model.neuron_states
                    if s.active and s.neuron_type == "hidden"
                ]
            )
            conn_per_neuron = model._get_avg_connections_per_neuron()
            print(
                f"Step {step:4d}: [{phase.name:12s}] dim={diag['hidden_dim']:5d}, "
                f"active={active:5d}, conn={diag['connection_count']:6d}, "
                f"conn/neuron={conn_per_neuron:.1f}, action={result['action']}"
            )

    if current_phase_name is not None:
        phase_history.append(
            {
                "phase": current_phase_name,
                "end_dim": model.hidden_dim,
                "end_conn": model.get_diagnostics()["connection_count"],
                "end_active": len(
                    [
                        s
                        for s in model.neuron_states
                        if s.active and s.neuron_type == "hidden"
                    ]
                ),
            }
        )

    print("\n" + "=" * 60)
    print("📊 发育阶段总结")
    print("=" * 60)
    for ph in phase_history:
        print(
            f"  {ph['phase']:12s}: dim={ph['end_dim']:5d}, active={ph['end_active']:5d}, conn={ph['end_conn']:6d}"
        )

    consolidated = [
        s
        for s in model.neuron_states
        if s.neuron_type == "hidden" and s.consolidation_score > 0.3
    ]
    decayed = [
        s
        for s in model.neuron_states
        if s.neuron_type == "hidden" and s.decay_counter > 0.5
    ]
    pruned = [
        s for s in model.neuron_states if s.neuron_type == "hidden" and not s.active
    ]

    print(f"\n💾 巩固神经元 (长期记忆): {len(consolidated)}")
    print(f"📉 衰减神经元 (待遗忘): {len(decayed)}")
    print(f"✂️ 已修剪神经元 (已遗忘): {len(pruned)}")

    if consolidated:
        print("\n🏆 核心神经元 (巩固分数 Top 5):")
        top5 = sorted(consolidated, key=lambda s: s.consolidation_score, reverse=True)[
            :5
        ]
        for s in top5:
            print(
                f"  Neuron {s.index}: cons={s.consolidation_score:.3f}, decay={s.decay_counter:.3f}, peak={s.peak_activation:.3f}, life={s.life_span}"
            )

    return model.hidden_dim > 0


if __name__ == "__main__":
    print("运行NEAT风格GrowableTwistorLNN测试")
    print("=" * 60)

    test_forward_pass()

    test_minimal_to_network()

    test_training_with_growth()

    test_neuron_split()

    test_pruning()

    test_full_training()

    test_developmental_phases()

    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)
