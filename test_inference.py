"""
Twistor-LNN 推理能力测试
测试推理速度、延迟、吞吐量、外推能力
"""
import torch
import time
import numpy as np
import importlib.util

# 从 twistor_lnn.py 导入
spec = importlib.util.spec_from_file_location("twistor_lnn_main", "twistor_lnn.py")
twistor_lnn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(twistor_lnn)

TwistorLNN = twistor_lnn.TwistorLNN
SelfTrainingTwistorLNN = twistor_lnn.SelfTrainingTwistorLNN

print("=" * 70)
print("Twistor-LNN 推理能力测试")
print("=" * 70)

# ============= 1. 推理速度测试 =============
print("\n1. 推理速度测试")
print("-" * 50)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TwistorLNN(input_dim=2, hidden_dim=32, output_dim=1, dt=0.05).to(device)
model.eval()

# 不同序列长度的推理测试
seq_lengths = [10, 30, 50, 100, 200]
batch_sizes = [1, 4, 8, 16]

print(f"设备：{device}")
print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")
print()

results = []

for seq_len in seq_lengths:
    x_test = torch.randn(seq_len, 1, 2).to(device)
    
    # 预热
    with torch.no_grad():
        _ = model(x_test)
    
    # 正式测试 (100 次平均)
    n_runs = 100
    start = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = model(x_test)
    end = time.time()
    
    avg_time = (end - start) / n_runs * 1000  # 毫秒
    results.append((seq_len, avg_time))
    
    print(f"   序列长度 {seq_len:3d}: {avg_time:7.3f} ms ({avg_time/seq_len*1000:6.1f} μs/步)")

# 绘制趋势
print()
print("推理时间趋势:")
for seq_len, avg_time in results:
    bar = '█' * int(avg_time / results[0][1] * 20)
    print(f"   {seq_len:3d}: {bar} {avg_time:.2f} ms")


# ============= 2. 批量推理吞吐量测试 =============
print("\n2. 批量推理吞吐量测试")
print("-" * 50)

seq_len = 50
throughputs = []

for batch_size in batch_sizes:
    x_test = torch.randn(seq_len, batch_size, 2).to(device)
    
    # 预热
    with torch.no_grad():
        _ = model(x_test)
    
    # 正式测试
    n_runs = 50
    start = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            y = model(x_test)
    end = time.time()
    
    # 计算吞吐量 (样本/秒)
    total_samples = batch_size * n_runs
    throughput = total_samples / (end - start)
    throughputs.append((batch_size, throughput))
    
    print(f"   Batch Size {batch_size:2d}: {throughput:7.1f} 样本/秒")


# ============= 3. 单步推理延迟测试 =============
print("\n3. 单步推理延迟测试 (实时性)")
print("-" * 50)

model.eval()
z = torch.zeros(1, 32, dtype=torch.complex64, device=device)

# 测试单步推理
n_steps = 1000
start = time.time()
with torch.no_grad():
    for _ in range(n_steps):
        x_t = torch.randn(1, 2).to(device)
        # 模拟单步推理 (内部实现)
        dzdt = model.compute_dzdt(z, x_t) if hasattr(model, 'compute_dzdt') else model.cell(z, x_t)
        z = z + 0.05 * dzdt
end = time.time()

single_step_latency = (end - start) / n_steps * 1000000  # 微秒
print(f"   单步推理延迟：{single_step_latency:.2f} μs")
print(f"   理论最大频率：{1/single_step_latency*1e6:.0f} Hz")

# 实时性评估
if single_step_latency < 1000:  # < 1ms
    print(f"   ✅ 满足实时控制要求 (< 1ms)")
elif single_step_latency < 10000:  # < 10ms
    print(f"   ⚠️ 满足一般交互要求 (< 10ms)")
else:
    print(f"   ❌ 不满足实时要求")


# ============= 4. 外推能力测试 =============
print("\n4. 外推能力测试 (泛化性)")
print("-" * 50)

# 训练一个简单模型
train_model = SelfTrainingTwistorLNN(input_dim=2, hidden_dim=32, output_dim=1)
train_model.performance_metrics['target_loss'] = 0.1

# 在短序列上训练
train_model.data_params['seq_len'] = 30
train_model.data_params['n_samples'] = 50

result = train_model.self_training_loop(
    n_iterations=3,
    epochs_per_iteration=20,
    task_type='sine',
    verbose=False,
)

# 测试不同序列长度的外推能力
print("   训练序列长度：30")
print("   测试不同长度的外推能力:")

test_lengths = [20, 30, 50, 70, 100]
extrapolation_results = []

for test_len in test_lengths:
    X_test, y_test = train_model.generate_data(task_type='sine', n_samples=10, seq_len=test_len)
    
    train_model.model.eval()
    with torch.no_grad():
        x_test = X_test.transpose(0, 1)
        y_test_t = y_test.transpose(0, 1)
        y_pred = train_model.model(x_test)
        mse = torch.nn.functional.mse_loss(y_pred, y_test_t).item()
    
    extrapolation_results.append((test_len, mse))
    ratio = test_len / 30
    print(f"   长度 {test_len:3d} (x{ratio:.1f}): MSE = {mse:.6f}")


# ============= 5. 内存占用测试 =============
print("\n5. 内存占用测试")
print("-" * 50)

def get_memory_usage(model, batch_size=1, seq_len=50):
    """估算内存占用"""
    # 参数内存
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # 激活内存 (估算)
    hidden_dim = model.hidden_dim
    activation_memory = seq_len * batch_size * hidden_dim * 8 * 4  # 复数 + 梯度
    
    total = param_memory + activation_memory
    return param_memory, activation_memory, total

param_mem, act_mem, total_mem = get_memory_usage(model)

print(f"   参数内存：{param_mem / 1024:.2f} KB")
print(f"   激活内存：{act_mem / 1024:.2f} KB (batch=1, seq_len=50)")
print(f"   总内存：  {total_mem / 1024:.2f} KB")


# ============= 6. 对比其他模型 =============
print("\n6. 与其他简单模型对比")
print("-" * 50)

# 创建对比模型
class SimpleRNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.rnn = torch.nn.RNN(input_dim, hidden_dim, batch_first=False)
        self.out = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.out(out)

class SimpleLSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, batch_first=False)
        self.out = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.out(out)

rnn_model = SimpleRNN(2, 32, 1).to(device)
lstm_model = SimpleLSTM(2, 32, 1).to(device)

seq_len = 50
x_test = torch.randn(seq_len, 1, 2).to(device)

# 推理时间对比
models_to_compare = [
    ('Twistor-LNN', model),
    ('RNN', rnn_model),
    ('LSTM', lstm_model),
]

print(f"   推理时间对比 (seq_len={seq_len}, 100 次平均):")
print()

compare_results = []
for name, m in models_to_compare:
    m.eval()
    n_runs = 100
    
    start = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = m(x_test)
    end = time.time()
    
    avg_time = (end - start) / n_runs * 1000
    compare_results.append((name, avg_time))
    
    params = sum(p.numel() for p in m.parameters())
    print(f"   {name:12s}: {avg_time:7.3f} ms (参数量：{params:,})")


# ============= 7. 总结 =============
print("\n" + "=" * 70)
print("Twistor-LNN 推理能力总结")
print("=" * 70)

print(f"""
推理性能:
  - 序列长度 50: {results[2][1]:.2f} ms ({results[2][1]/50*1000:.1f} μs/步)
  - 单步延迟：{single_step_latency:.2f} μs
  - 理论频率：{1/single_step_latency*1e6:.0f} Hz
  - 实时性：{'✅ 满足实时控制 (<1ms)' if single_step_latency < 1000 else '⚠️ 满足一般交互 (<10ms)'}

吞吐量:
  - Batch=1: {throughputs[0][1]:.1f} 样本/秒
  - Batch=16: {throughputs[3][1]:.1f} 样本/秒

外推能力:
  - 训练长度：30
  - 外推 100 (3.3x): MSE = {extrapolation_results[-1][1]:.6f}

内存占用:
  - 参数：{param_mem / 1024:.2f} KB
  - 总内存：{total_mem / 1024:.2f} KB

对比优势:
  - vs RNN: {'快' if compare_results[0][1] < compare_results[1][1] else '慢'} {abs(compare_results[0][1] - compare_results[1][1]):.2f} ms
  - vs LSTM: {'快' if compare_results[0][1] < compare_results[2][1] else '慢'} {abs(compare_results[0][1] - compare_results[2][1]):.2f} ms
""")

print("=" * 70)
