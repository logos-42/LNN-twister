"""
测试莫比乌斯约束和扭量共振注意力
=================================
验证:
1. 莫比乌斯约束基础功能
2. 共振注意力基础功能
3. 集成到 TwistorLNN
4. 集成到 GrowableTwistorLNN (自生长+升维联动)
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from twistor_lnn.core import TwistorLNN
from twistor_lnn.mobius import MobiusConstraint
from twistor_lnn.resonance import TwistorResonance
from twistor_lnn.growable import GrowableTwistorLNN


def test_mobius_basic():
    """测试莫比乌斯约束基础功能"""
    print("\n1. 莫比乌斯约束基础测试")
    print("-" * 40)

    mobius = MobiusConstraint(max_dim=256, enable_learning=True)

    for n in [4, 16, 64, 128]:
        dim = mobius.compute_manifold_dimension(n)
        print(f"  hidden_dim={n:4d} → manifold_dim={dim}")

    z = torch.randn(2, 32, dtype=torch.complex64)
    z_proj = mobius.project_state(z)
    diff = (z - z_proj).abs().mean().item()
    print(f"  投影前后差异: {diff:.6f}")

    W = mobius.topology_weight_matrix(32)
    print(f"  拓扑权重形状: {W.shape}, 范围: [{W.min():.4f}, {W.max():.4f}]")

    info = mobius.get_manifold_info(32)
    print(f"  流形模式: {info['mode']}, 维度: {info['manifold_dim']}")

    print("  ✅ 莫比乌斯约束测试通过")


def test_resonance_basic():
    """测试共振注意力基础功能"""
    print("\n2. 扭量共振注意力基础测试")
    print("-" * 40)

    resonance = TwistorResonance(hidden_dim=32, resonance_strength=0.1)

    z = torch.randn(2, 32, dtype=torch.complex64)
    dzdt = resonance(z)

    print(f"  共振输出形状: {dzdt.shape}")
    print(f"  共振强度: {resonance.resonance_strength.item():.4f}")

    R = resonance.compute_resonance_matrix(z)
    print(f"  共振矩阵形状: {R.shape}")
    print(f"  共振分数范围: [{R.min():.4f}, {R.max():.4f}]")

    print("  ✅ 共振注意力测试通过")


def test_integrated_core():
    """测试集成到 TwistorLNN"""
    print("\n3. 集成到 TwistorLNN 测试")
    print("-" * 40)

    model = TwistorLNN(input_dim=4, hidden_dim=32, output_dim=2)

    print(f"  基础模型参数: {sum(p.numel() for p in model.parameters()):,}")

    x = torch.randn(20, 2, 4)
    y_base = model(x)
    print(f"  基础输出形状: {y_base.shape}")

    model.enable_mobius_resonance(
        enable_mobius=True,
        enable_resonance=True,
        mobius_strength=0.1,
        resonance_strength=0.05,
    )

    print(f"  启用后参数: {sum(p.numel() for p in model.parameters()):,}")

    y_mr = model(x)
    print(f"  莫比乌斯+共振输出形状: {y_mr.shape}")

    info = model.get_mobius_info()
    if info:
        print(f"  流形模式: {info['mode']}, 维度: {info['manifold_dim']}")

    diff = (y_base - y_mr).abs().mean().item()
    print(f"  基础 vs 增强差异: {diff:.6f}")

    print("  ✅ 集成模型测试通过")


def test_integrated_growable():
    """测试集成到 GrowableTwistorLNN (自生长+升维联动)"""
    print("\n4. 集成到 GrowableTwistorLNN 测试")
    print("-" * 40)

    model = GrowableTwistorLNN(
        input_dim=4,
        hidden_dim=0,
        output_dim=2,
        enable_growth=True,
        enable_mobius=True,
        enable_resonance=True,
        mobius_strength=0.1,
        resonance_strength=0.05,
    )

    print(f"  初始 hidden_dim: {model.hidden_dim}")
    print(f"  莫比乌斯启用: {model.mobius is not None}")
    print(f"  共振启用: {model.resonance is not None}")

    x = torch.randn(30, 2, 4)
    y = model(x)
    print(f"  输出形状: {y.shape}")

    for i in range(5):
        model.growth_step()

    print(f"  增长后 hidden_dim: {model.hidden_dim}")

    if model.mobius is not None:
        info = model.mobius.get_manifold_info(model.hidden_dim)
        print(f"  流形模式: {info['mode']}, 维度: {info['manifold_dim']}")

    print("  ✅ 可增长模型集成测试通过")


def test_mobius_dimension_evolution():
    """测试莫比乌斯维度进化"""
    print("\n5. 莫比乌斯维度进化测试")
    print("-" * 40)

    mobius = MobiusConstraint(max_dim=1024, enable_learning=True)

    test_dims = [1, 4, 8, 16, 32, 64, 128, 256, 512]

    prev_dim = 0
    for n in test_dims:
        m_dim = mobius.compute_manifold_dimension(n)
        changed = " ↑" if m_dim > prev_dim else ""
        print(f"  neurons={n:4d} → manifold_dim={m_dim}{changed}")
        prev_dim = m_dim

    print("  ✅ 维度进化测试通过")


def test_mobius_klein_transition():
    """测试莫比乌斯→克莱因混合模式"""
    print("\n6. 莫比乌斯→克莱因混合模式测试")
    print("-" * 40)

    from twistor_lnn.mobius import AdaptiveMobiusConstraint

    mobius = AdaptiveMobiusConstraint(max_dim=128, mobius_weight=1.0, klein_weight=0.0)

    for step_pct in [0.0, 0.25, 0.5, 0.75, 1.0]:
        mobius.update_transition(int(step_pct * 100), 100)

        alpha = torch.sigmoid(mobius.mobius_weight).item()
        beta = torch.sigmoid(mobius.klein_weight).item()
        total = alpha + beta + 1e-6

        print(
            f"  progress={step_pct:.2f} → mobius={alpha / total:.3f}, klein={beta / total:.3f}"
        )

    print("  ✅ 混合模式过渡测试通过")


def test_resonance_modes():
    """测试共振注意力的不同应用模式"""
    print("\n7. 共振注意力模式测试")
    print("-" * 40)

    resonance = TwistorResonance(hidden_dim=16, resonance_strength=0.1)
    z = torch.randn(2, 16, dtype=torch.complex64)

    for mode in ["additive", "multiplicative", "gating"]:
        dzdt = resonance(z, mode=mode)
        print(
            f"  mode={mode:15s} → output shape: {dzdt.shape}, mean: {dzdt.abs().mean():.4f}"
        )

    print("  ✅ 共振模式测试通过")


if __name__ == "__main__":
    print("=" * 50)
    print("莫比乌斯约束 + 扭量共振 完整测试")
    print("=" * 50)

    test_mobius_basic()
    test_resonance_basic()
    test_integrated_core()
    test_integrated_growable()
    test_mobius_dimension_evolution()
    test_mobius_klein_transition()
    test_resonance_modes()

    print("\n" + "=" * 50)
    print("所有测试通过!")
    print("=" * 50)
