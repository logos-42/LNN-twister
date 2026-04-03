"""测试流形约束生长系统"""

import torch
import torch.nn as nn
from twistor_lnn.growable import GrowableTwistorLNN, GrowthConfig

print("=" * 60)
print("测试流形约束生长")
print("=" * 60)

config = GrowthConfig(
    min_hidden_dim=0,
    max_hidden_dim=2048,
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
    enable_mobius=True,
)

print(f"\n流形几何: {model.manifold_geometry is not None}")
print(f"权重初始化器: {model.weight_initializer is not None}")
print(f"生长规划器: {model.growth_planner is not None}")

riemannian_opt = model.create_riemannian_optimizer(lr=0.01)
print(f"黎曼优化器: {riemannian_opt is not None}")

for step in range(500):
    if model.hidden_dim > 0 and model.hidden_dim < 128:
        x = torch.randn(3, 2, 2)
        with torch.no_grad():
            model(x)

    result = model.growth_step()
    phase = model._get_current_developmental_phase()

    if step % 100 == 0 or step in [100, 300]:
        diag = model.get_diagnostics()
        active = len(
            [s for s in model.neuron_states if s.active and s.neuron_type == "hidden"]
        )
        conn_per_neuron = model._get_avg_connections_per_neuron()
        name = phase.name
        manifold_r = diag.get("manifold_radius", 0)
        twist = diag.get("twist_rate", 0)
        klein = diag.get("klein_mix", 0)
        manifold_dim = diag.get("manifold_dim", 0)
        mode = diag.get("mode", "N/A")
        print(
            f"Step {step:4d}: [{name:12s}] dim={diag['hidden_dim']:5d}, active={active:5d}, "
            f"conn={diag['connection_count']:6d}, conn/n={conn_per_neuron:.1f}, "
            f"manifold_r={manifold_r:.2f}, twist={twist:.2f}, klein={klein:.2f}, "
            f"manifold_dim={manifold_dim}, mode={mode}"
        )

print("\n" + "=" * 60)
print("测试黎曼优化器")
print("=" * 60)

model2 = GrowableTwistorLNN(
    input_dim=2,
    hidden_dim=4,
    output_dim=1,
    enable_mobius=True,
)

opt = model2.create_riemannian_optimizer(lr=0.01)

x = torch.randn(10, 4, 2)
y_target = torch.randn(10, 4, 1)

for epoch in range(50):
    opt.zero_grad()
    y_pred = model2(x)
    loss = nn.MSELoss()(y_pred, y_target)
    loss.backward()

    grad_norms = []
    for p in model2.parameters():
        if p.grad is not None:
            grad_norms.append(p.grad.norm().item())

    opt.step()

    if epoch % 10 == 0:
        max_grad = max(grad_norms) if grad_norms else 0
        print(f"Epoch {epoch:3d}: loss={loss.item():.4f}, max_grad_norm={max_grad:.4f}")

print("\n" + "=" * 60)
print("所有测试完成!")
print("=" * 60)
