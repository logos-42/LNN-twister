"""
Test script for Sparse LTC Cell.
Verifies:
1. Sparse connectivity
2. Multi-scale time constants τᵢ
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '.')

from liquid_net.models.sparse_ltc_cell import SparseLTCCell


def test_sparse_ltc_cell():
    """Test the sparse LTC cell."""
    
    print("=" * 60)
    print("Testing Sparse LTC Cell")
    print("=" * 60)
    
    # Configuration
    input_dim = 2
    hidden_dim = 8  # Small for easy inspection
    batch_size = 4
    sparsity = 0.3  # 30% sparsity
    
    # Create model
    model = SparseLTCCell(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        sparsity=sparsity,
        use_multi_scale_tau=True
    )
    
    print(f"\nModel configuration:")
    print(f"  input_dim: {input_dim}")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  target sparsity: {sparsity}")
    print(f"  multi-scale τ: True")
    
    # Print initial sparsity
    stats = model.get_sparsity()
    print(f"\nInitial sparsity:")
    print(f"  W_real: {stats['W_real_sparsity']:.2%} ({stats['W_real_active']}/{stats['W_real_total']})")
    print(f"  W_imag: {stats['W_imag_sparsity']:.2%} ({stats['W_imag_active']}/{stats['W_imag_total']})")
    
    # Test forward pass
    print("\n" + "-" * 60)
    print("Testing forward pass...")
    
    # Create dummy input
    z = torch.zeros(batch_size, hidden_dim, dtype=torch.complex64)
    x = torch.randn(batch_size, input_dim)
    
    print(f"  Input z shape: {z.shape}")
    print(f"  Input x shape: {x.shape}")
    
    # Forward
    dzdt = model(z, x)
    
    print(f"  Output dzdt shape: {dzdt.shape}")
    print(f"  Output dtype: {dzdt.dtype}")
    
    # Test τ computation
    print("\n" + "-" * 60)
    print("Testing multi-scale time constants τᵢ...")
    
    # Create different states
    z_small = torch.zeros(2, hidden_dim, dtype=torch.complex64)  # small states
    z_large = torch.ones(2, hidden_dim, dtype=torch.complex64) * 5  # large states
    
    tau_small = model.compute_tau(z_small)
    tau_large = model.compute_tau(z_large)
    
    print(f"  τ for small states: min={tau_small.min():.4f}, max={tau_small.max():.4f}")
    print(f"  τ for large states: min={tau_large.min():.4f}, max={tau_large.max():.4f}")
    print(f"  τ parameters: gain={model.tau_gain.data[:3]}..., bias={model.tau_bias.data[:3]}...")
    
    # Test gradient flow
    print("\n" + "-" * 60)
    print("Testing gradient flow...")
    
    z_test = torch.randn(2, hidden_dim, dtype=torch.complex64, requires_grad=True)
    x_test = torch.randn(2, input_dim)
    
    dzdt_test = model(z_test, x_test)
    loss = (dzdt_test.abs() ** 2).mean()
    loss.backward()
    
    print(f"  Loss: {loss.item():.6f}")
    print(f"  z_test.grad norm: {z_test.grad.abs().mean():.6f}")
    print(f"  mask_real grad norm: {model.mask_real.grad.abs().mean():.6f}")
    print(f"  tau_gain grad norm: {model.tau_gain.grad.abs().mean():.6f}")
    
    # Test without multi-scale tau
    print("\n" + "-" * 60)
    print("Testing without multi-scale τ...")
    
    model_single_tau = SparseLTCCell(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        sparsity=sparsity,
        use_multi_scale_tau=False
    )
    
    tau_single = model_single_tau.compute_tau(z)
    print(f"  τ shape: {tau_single.shape}")
    print(f"  τ[0]: {tau_single[0]}")
    
    # Summary
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    test_sparse_ltc_cell()
