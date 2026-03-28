"""
测试 RK4 积分器和动力学分析功能
"""
import torch
import numpy as np
from twistor_lnn_full import TwistorLNN, plot_phase_space, IntegrationConfig

print("=" * 60)
print("TwistorLNN - RK4 积分器和动力学分析测试")
print("=" * 60)

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

# 测试不同积分方法
methods = ['euler', 'rk4']

for method in methods:
    print(f"\n{'=' * 50}")
    print(f"测试积分方法：{method.upper()}")
    print('=' * 50)
    
    model = TwistorLNN(
        input_dim=2, 
        hidden_dim=16, 
        output_dim=1,
        dt=0.1, 
        tau_min=0.01, 
        tau_max=1.0, 
        dzdt_max=10.0, 
        z_max=100.0,
        integration_method=method,
    )
    
    print(f"✓ 模型创建成功 ({method})")
    print(f"  参数量：{sum(p.numel() for p in model.parameters()):,}")
    
    model.eval()
    with torch.no_grad():
        y, states, diag = model(x_input, return_states=True, return_diagnostics=True)
    
    print(f"✓ 前向传播成功")
    print(f"  输出形状：{y.shape}")
    print(f"  状态形状：{states[0].shape if isinstance(states, list) else states.shape}")
    print(f"  |z| 范围：[{diag['z_norm'].min():.4f}, {diag['z_norm'].max():.4f}]")
    print(f"  |dz/dt| 范围：[{diag['dzdt_norm'].min():.4f}, {diag['dzdt_norm'].max():.4f}]")
    print(f"  NaN 检测：{diag['has_nan']}")

# 测试不动点分析
print(f"\n{'=' * 50}")
print("测试不动点分析")
print('=' * 50)

model = TwistorLNN(
    input_dim=2, hidden_dim=16, output_dim=1,
    dt=0.1, integration_method='euler'
)

# 常数输入
x_const = torch.randn(1, 2) * 0.5

result = model.analyze_fixed_points(x_const, max_iter=500, tol=1e-6)

print(f"✓ 不动点分析完成")
print(f"  收敛：{result['converged']}")
print(f"  迭代次数：{result['iterations']}")
print(f"  最终 |dz/dt|: {result['dzdt_final']:.6f}")
print(f"  稳定：{result['is_stable']}")

if result['eigenvalues'] is not None:
    real_parts = [e.real for e in result['eigenvalues'][:5]]  # 前 5 个特征值
    print(f"  特征值实部 (前 5 个): {[f'{r:.4f}' for r in real_parts]}")

# 测试相空间可视化
print(f"\n{'=' * 50}")
print("生成相空间可视化...")
print('=' * 50)

model.eval()
with torch.no_grad():
    plot_phase_space(model, x_input[:10], n_steps=50, save_path='phase_space_test.png')

print("✓ 相空间图已生成")

print(f"\n{'=' * 60}")
print("所有测试完成！✓")
print('=' * 60)
