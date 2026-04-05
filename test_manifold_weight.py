"""测试振幅+相位(流形约束)权重"""

import torch
from twistor_lnn.growable import GrowableTwistorLNN, GrowthConfig

print("测试流形约束权重...")

config = GrowthConfig(
    min_hidden_dim=0,
    max_hidden_dim=128,
    growth_interval=5,
    prune_interval=20,
    enable_developmental_schedule=True,
)

model = GrowableTwistorLNN(
    input_dim=4,
    hidden_dim=0,
    output_dim=2,
    growth_config=config,
    enable_growth=True,
    enable_mobius=True,
)

print(
    f"参数: manifold_theta={model.manifold_theta.shape}, W_amplitude={model.W_amplitude.shape}"
)
print(f"隐藏维度: {model.hidden_dim}")

# 增长到有几个神经元
for step in range(100):
    if model.hidden_dim > 0 and model.hidden_dim < 64:
        x = torch.randn(3, 4, 4)
        with torch.no_grad():
            y = model(x)
            if step % 25 == 0:
                print(f"Step {step}: dim={model.hidden_dim}, y.shape={y.shape}")

    model.growth_step()

print(f"\n最终: dim={model.hidden_dim}")

# 测试复数权重
if model.hidden_dim > 0:
    W = model.get_complex_weight()
    print(f"复数权重: {W.shape}, dtype={W.dtype}")
    print(f"  |W| mean={W.abs().mean():.4f}, std={W.abs().std():.4f}")
    print(f"  phase mean={W.angle().mean():.4f}, std={W.angle().std():.4f}")

    # 测试前向传播
    x = torch.randn(5, 4, 4)
    y = model(x)
    print(f"  前向传播: x={x.shape} → y={y.shape}")

    # 测试训练
    model.train()
    opt = model.create_riemannian_optimizer(lr=0.01)
    criterion = torch.nn.MSELoss()

    for epoch in range(30):
        opt.zero_grad()
        x = torch.randn(5, 4, 4)
        y_target = torch.randn(5, 4, 2)
        y_pred = model(x)
        loss = criterion(y_pred, y_target)
        loss.backward()
        opt.step()
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: loss={loss.item():.4f}")

print("\n测试完成!")
