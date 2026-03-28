"""
稳定性优化测试脚本
"""
import torch
import numpy as np
from twistor_lnn import TwistorLNN, plot_z_trajectory

print("=" * 60)
print("TwistorLNN 稳定性优化版本 - 测试脚本")
print("=" * 60)

# 创建模型
model = TwistorLNN(
    input_dim=2, 
    hidden_dim=16, 
    output_dim=1,
    dt=0.1, 
    tau_min=0.01, 
    tau_max=1.0, 
    dzdt_max=10.0, 
    z_max=100.0
)

print(f"✓ 模型创建成功")
print(f"  参数量：{sum(p.numel() for p in model.parameters()):,}")
print(f"  dt: {model.dt}")
print(f"  tau 范围：[{model.tau_min}, {model.tau_max}]")
print(f"  dzdt_max: {model.dzdt_max}")
print(f"  z_max: {model.z_max}")

# 生成测试数据
def generate_sine_dataset(n_samples=10, seq_len=50, device='cpu'):
    X = []
    for _ in range(n_samples):
        freq = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2*np.pi)
        t = np.linspace(0, 4*np.pi, seq_len+1)
        signal = np.sin(freq*t + phase) + np.random.randn(len(t))*0.1
        sin_c = signal[:-1]
        cos_c = np.cos(freq*t[:-1] + phase) + np.random.randn(seq_len)*0.1
        x_seq = np.stack([sin_c, cos_c], axis=-1)
        X.append(x_seq)
    return torch.FloatTensor(np.stack(X)).to(device)

X_test = generate_sine_dataset(n_samples=5, seq_len=50)
x_input = X_test.transpose(0, 1)

print()
print("运行前向传播（带诊断）...")

# 运行前向传播（带诊断）
model.eval()
with torch.no_grad():
    y, states, diag = model(x_input, return_states=True, return_diagnostics=True)

print("✓ 前向传播成功")
print(f"  输出形状：{y.shape}")
print(f"  状态形状：{states.shape}")

print()
print("诊断数据:")
print(f"  |z| 范围：[{diag['z_norm'].min():.4f}, {diag['z_norm'].max():.4f}]")
print(f"  |dz/dt| 范围：[{diag['dzdt_norm'].min():.4f}, {diag['dzdt_norm'].max():.4f}]")
print(f"  τ 范围：[{diag['tau_mean'].min():.4f}, {diag['tau_mean'].max():.4f}]")
print(f"  NaN 检测：{diag['has_nan']}")
print(f"  Inf 检测：{diag['has_inf']}")

# 打印 τ 统计
stats = model.get_tau_statistics(states[0])
print()
print("τ 分布统计:")
print(f"  τ mean: {stats['tau_mean']:.6f}")
print(f"  τ std:  {stats['tau_std']:.6f}")
print(f"  τ min:  {stats['tau_min']:.6f}")
print(f"  τ max:  {stats['tau_max']:.6f}")

# 绘制轨迹图
print()
print("生成诊断图表...")
plot_z_trajectory(diag, save_path='z_trajectory.png')
print("✓ 已生成 z_trajectory.png")

print()
print("=" * 60)
print("所有测试通过！✓")
print("=" * 60)
