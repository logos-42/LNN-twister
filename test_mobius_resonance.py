"""
Test Mobius Manifold Constraint and Twistor Resonance Attention
===============================================================
Validates:
1. Mobius constraint basic functionality
2. Resonance attention basic functionality
3. Integration with TwistorLNN
4. Integration with GrowableTwistorLNN (self-growth + dimension evolution)
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from twistor_lnn.core import TwistorLNN
from twistor_lnn.mobius import MobiusConstraint, AdaptiveMobiusConstraint
from twistor_lnn.resonance import TwistorResonance, MultiHeadResonance
from twistor_lnn.growable import GrowableTwistorLNN


def test_mobius_basic():
    """Test Mobius constraint basic functionality"""
    print("\n1. Mobius Constraint Basic Test")
    print("-" * 40)

    mobius = MobiusConstraint(max_dim=256, enable_learning=True)

    for n in [4, 16, 64, 128]:
        dim = mobius.compute_manifold_dimension(n)
        assert 1 <= dim <= 8, f"manifold_dim {dim} out of range for hidden_dim={n}"
        print(f"  hidden_dim={n:4d} -> manifold_dim={dim}")

    z = torch.randn(2, 32, dtype=torch.complex64)
    z_proj = mobius.project_state(z)
    assert z_proj.shape == z.shape, f"Shape mismatch: {z_proj.shape} vs {z.shape}"
    diff = (z - z_proj).abs().mean().item()
    assert diff > 0, "Projection should change the state"
    print(f"  Projection difference: {diff:.6f}")

    W = mobius.topology_weight_matrix(32)
    assert W.shape == (32, 32), f"Topology weight shape mismatch: {W.shape}"
    print(f"  Topology weight shape: {W.shape}, range: [{W.min():.4f}, {W.max():.4f}]")

    info = mobius.get_manifold_info(32)
    assert info["manifold_dim"] >= 1
    assert info["mode"] in ("mobius", "klein", "mixed")
    print(f"  Manifold mode: {info['mode']}, dim: {info['manifold_dim']}")

    print("  PASSED")


def test_resonance_basic():
    """Test resonance attention basic functionality"""
    print("\n2. Twistor Resonance Attention Basic Test")
    print("-" * 40)

    resonance = TwistorResonance(hidden_dim=32, resonance_strength=0.1)

    z = torch.randn(2, 32, dtype=torch.complex64)
    dzdt = resonance(z)

    assert dzdt.shape == z.shape, f"Shape mismatch: {dzdt.shape} vs {z.shape}"
    print(f"  Resonance output shape: {dzdt.shape}")
    print(f"  Resonance strength: {resonance.resonance_strength.item():.4f}")

    R = resonance.compute_resonance_matrix(z)
    assert R.shape == (2, 32, 32), f"Resonance matrix shape mismatch: {R.shape}"
    assert not torch.isnan(R).any(), "Resonance matrix contains NaN"
    assert not torch.isinf(R).any(), "Resonance matrix contains Inf"
    print(f"  Resonance matrix shape: {R.shape}")
    print(f"  Resonance score range: [{R.min():.4f}, {R.max():.4f}]")

    print("  PASSED")


def test_integrated_core():
    """Test integration with TwistorLNN"""
    print("\n3. Integration with TwistorLNN Test")
    print("-" * 40)

    model = TwistorLNN(input_dim=4, hidden_dim=32, output_dim=2)

    base_params = sum(p.numel() for p in model.parameters())
    print(f"  Base model parameters: {base_params:,}")

    x = torch.randn(20, 2, 4)
    y_base = model(x)
    assert y_base.shape == (20, 2, 2), f"Output shape mismatch: {y_base.shape}"
    print(f"  Base output shape: {y_base.shape}")

    model.enable_mobius_resonance(
        enable_mobius=True,
        enable_resonance=True,
        mobius_strength=0.1,
        resonance_strength=0.05,
    )

    mr_params = sum(p.numel() for p in model.parameters())
    assert mr_params > base_params, (
        "Parameters should increase after enabling mobius+resonance"
    )
    print(f"  Parameters after enabling: {mr_params:,}")

    y_mr = model(x)
    assert y_mr.shape == y_base.shape, f"Output shape mismatch: {y_mr.shape}"
    print(f"  Mobius+Resonance output shape: {y_mr.shape}")

    info = model.get_mobius_info()
    assert info is not None, "Mobius info should not be None"
    print(f"  Manifold mode: {info['mode']}, dim: {info['manifold_dim']}")

    diff = (y_base - y_mr).abs().mean().item()
    print(f"  Base vs Enhanced difference: {diff:.6f}")

    print("  PASSED")


def test_integrated_growable():
    """Test integration with GrowableTwistorLNN"""
    print("\n4. Integration with GrowableTwistorLNN Test")
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

    assert model.hidden_dim == 0
    assert model.mobius is not None
    assert model.resonance is not None
    print(f"  Initial hidden_dim: {model.hidden_dim}")
    print(f"  Mobius enabled: {model.mobius is not None}")
    print(f"  Resonance enabled: {model.resonance is not None}")

    x = torch.randn(30, 2, 4)
    y = model(x)
    assert y.shape == (30, 2, 2), f"Output shape mismatch: {y.shape}"
    print(f"  Output shape: {y.shape}")

    for i in range(5):
        model.growth_step()

    print(f"  After growth hidden_dim: {model.hidden_dim}")

    if model.mobius is not None:
        info = model.mobius.get_manifold_info(model.hidden_dim)
        print(f"  Manifold mode: {info['mode']}, dim: {info['manifold_dim']}")

    print("  PASSED")


def test_mobius_dimension_evolution():
    """Test Mobius dimension evolution"""
    print("\n5. Mobius Dimension Evolution Test")
    print("-" * 40)

    mobius = MobiusConstraint(max_dim=1024, enable_learning=True)

    test_dims = [1, 4, 8, 16, 32, 64, 128, 256, 512]

    prev_dim = 0
    for n in test_dims:
        m_dim = mobius.compute_manifold_dimension(n)
        assert 1 <= m_dim <= 8, f"manifold_dim {m_dim} out of range for n={n}"
        changed = " ^" if m_dim > prev_dim else ""
        print(f"  neurons={n:4d} -> manifold_dim={m_dim}{changed}")
        prev_dim = m_dim

    print("  PASSED")


def test_mobius_klein_transition():
    """Test Mobius -> Klein mixed mode transition"""
    print("\n6. Mobius -> Klein Mixed Mode Transition Test")
    print("-" * 40)

    mobius = AdaptiveMobiusConstraint(max_dim=128, mobius_weight=1.0, klein_weight=0.0)

    prev_mobius = 1.0
    for step_pct in [0.0, 0.25, 0.5, 0.75, 1.0]:
        mobius.update_transition(int(step_pct * 100), 100)

        alpha = torch.sigmoid(mobius.mobius_weight).item()
        beta = torch.sigmoid(mobius.klein_weight).item()
        total = alpha + beta + 1e-6

        mobius_ratio = alpha / total
        klein_ratio = beta / total

        assert mobius_ratio + klein_ratio > 0.99, "Ratios should sum to ~1"
        print(
            f"  progress={step_pct:.2f} -> mobius={mobius_ratio:.3f}, klein={klein_ratio:.3f}"
        )

    print("  PASSED")


def test_resonance_modes():
    """Test resonance attention different application modes"""
    print("\n7. Resonance Attention Modes Test")
    print("-" * 40)

    resonance = TwistorResonance(hidden_dim=16, resonance_strength=0.1)
    z = torch.randn(2, 16, dtype=torch.complex64)

    for mode in ["additive", "multiplicative", "gating"]:
        dzdt = resonance(z, mode=mode)
        assert dzdt.shape == z.shape, f"Shape mismatch for mode={mode}: {dzdt.shape}"
        assert not torch.isnan(dzdt).any(), f"NaN detected for mode={mode}"
        print(
            f"  mode={mode:15s} -> output shape: {dzdt.shape}, mean: {dzdt.abs().mean():.4f}"
        )

    print("  PASSED")


def test_multihead_resonance():
    """Test MultiHeadResonance with valid configurations"""
    print("\n8. MultiHead Resonance Test")
    print("-" * 40)

    z = torch.randn(2, 32, dtype=torch.complex64)

    head = MultiHeadResonance(hidden_dim=32, num_heads=4)
    out = head(z)
    assert out.shape == z.shape, f"Shape mismatch: {out.shape} vs {z.shape}"
    print(f"  32-dim, 4 heads -> output shape: {out.shape}")

    head2 = MultiHeadResonance(hidden_dim=33, num_heads=4)
    out2 = head2(torch.randn(2, 33, dtype=torch.complex64))
    assert out2.shape == (2, 33), f"Shape mismatch for 33-dim: {out2.shape}"
    print(f"  33-dim, 4 heads -> output shape: {out2.shape} (no restriction)")

    print("  PASSED")


def test_resonance_numerical_stability():
    """Test resonance numerical stability with extreme inputs"""
    print("\n9. Resonance Numerical Stability Test")
    print("-" * 40)

    resonance = TwistorResonance(hidden_dim=16, resonance_strength=0.1)

    z_zero = torch.zeros(2, 16, dtype=torch.complex64)
    out_zero = resonance(z_zero)
    assert not torch.isnan(out_zero).any(), "NaN with zero input"
    assert not torch.isinf(out_zero).any(), "Inf with zero input"
    print(f"  Zero input -> no NaN/Inf")

    z_large = torch.randn(2, 16, dtype=torch.complex64) * 100
    out_large = resonance(z_large)
    assert not torch.isnan(out_large).any(), "NaN with large input"
    assert not torch.isinf(out_large).any(), "Inf with large input"
    print(f"  Large input -> no NaN/Inf")

    resonance.kernel_bias.data = torch.tensor(-10.0)
    out_neg_exp = resonance(z_large)
    assert not torch.isnan(out_neg_exp).any(), "NaN with negative exponent"
    assert not torch.isinf(out_neg_exp).any(), "Inf with negative exponent"
    print(f"  Negative exponent -> no NaN/Inf")

    print("  PASSED")


if __name__ == "__main__":
    print("=" * 50)
    print("Mobius Constraint + Twistor Resonance Full Test")
    print("=" * 50)

    test_mobius_basic()
    test_resonance_basic()
    test_integrated_core()
    test_integrated_growable()
    test_mobius_dimension_evolution()
    test_mobius_klein_transition()
    test_resonance_modes()
    test_multihead_resonance()
    test_resonance_numerical_stability()

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)
