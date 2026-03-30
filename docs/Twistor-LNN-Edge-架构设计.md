# Twistor-LNN-Edge 架构设计文档

## 一、架构概述

Twistor-LNN-Edge 是基于 Twistor-LNN 的边缘优化版本，融合了 LFM2 的设计理念：

> **局部 ODE 动力学为主 + 少量关键 Attention 为辅**

核心公式：

```
dz/dt = (-z + W·tanh(z) + U·x) / τ(z)    # 局部 ODE（大部分时间）
+ [可选] GQA_Attention(z, z; τ)           # 全局交互（稀疏触发）
```

---

## 二、与 LFM2 的设计对应

| LFM2 特性 | Twistor-LNN-Edge 实现 |
|----------|---------------------|
| 短卷积（局部建模） | ODE 动力学 + tanh 非线性（状态依赖时间常数） |
| 稀疏 Attention | Grouped Query Attention (GQA) |
| GQA 分组查询 | `n_heads` 查询头共享 `n_kv_heads` 个 KV 头 |
| τ 阈值触发 | `tau_attention_threshold` 决定何时触发 Attention |

---

## 三、核心组件

### 1. ODE 动力学层

```python
class TwistorLNNwithGQA:
    def compute_dzdt(self, z, x):
        # dz/dt = (-z + W·tanh(z) + U·x) / τ(z)
        tau = self.compute_tau(z)  # 状态依赖时间常数
        dzdt = complex(dz_real / tau, dz_imag / tau)
        return dzdt
```

### 2. GQA Attention 层

```python
class GroupedQueryAttention:
    def __init__(self, dim, n_heads, n_kv_heads):
        # n_heads 个 Q 头，但只有 n_kv_heads 个 KV 头
        # 每 n_heads/n_kv_heads 个 Q 共享一个 KV
```

### 3. τ 阈值触发机制

```python
def should_trigger_attention(self, tau):
    tau_mean = tau.mean(dim=-1)
    return tau_mean < self.tau_attention_threshold
```

触发条件：
- τ < 阈值（状态变化快，需要全局信息）
- 每隔固定步数（`attention_interval`）

---

## 四、数据流

```
输入序列 x: (T, B, input_dim)
    │
    ▼
初始化 z = 0: (B, hidden_dim), complex
    │
    ▼
for t in range(T):
    │
    ├──► 1. ODE 动力学
    │    dzdt = compute_dzdt(z, x_t)
    │    z = z + dt * dzdt
    │    z = clamp(z)
    │
    ├──► 2. 可选：GQA Attention
    │    if should_trigger_attention(tau):
    │        attn_out = GQA(z)
    │        z = z + residual(attn_out)
    │
    └──► 3. 输出
         y_t = out(z.real)
```

---

## 五、关键参数

| 参数 | 说明 | 推荐值 |
|-----|------|-------|
| `hidden_dim` | 隐藏维度 | 32-128 |
| `n_heads` | Attention 头数 | 4-8 |
| `n_kv_heads` | KV 头数（分组共享） | 1-2 |
| `attention_interval` | Attention 触发间隔 | 3-5 |
| `tau_attention_threshold` | τ 阈值 | 0.3 |
| `dt` | Euler 积分步长 | 0.1 |
| `tau_min`, `tau_max` | 时间常数范围 | [0.01, 1.0] |

---

## 六、与传统 Transformer 的对比

| 特性 | Transformer | Twistor-LNN-Edge |
|-----|------------|-----------------|
| 全局建模 | 每一层 Full Attention | 稀疏触发 GQA |
| 局部建模 | FFN | ODE 动力学 |
| 时间常数 | 固定 | 状态依赖 τ(z) |
| 推理方式 | 并行 | 序贯（适合 Agent） |
| 适用场景 | 云端大模型 | 端侧实时推理 |

---

## 七、Agent 接口

```python
# 单步演化（用于强化学习/Agent）
z_new, output = model.step(z, x)

# 参数说明
# z: 当前状态 (B, hidden_dim), complex
# x: 当前输入/观测 (B, input_dim)
# z_new: 下一状态
# output: 动作/预测 (B, output_dim)
```

---

## 八、设计哲学总结

> **不是"全局 Attention everywhere"，而是"局部 ODE 为主 + 必要时 Attention"**

这与 LFM2 的理念一致：
- 90%：便宜计算（ODE/Conv）
- 10%：昂贵但关键（Attention）

通过 τ 阈值动态决定何时需要全局信息，实现计算效率与推理能力的平衡。

---

## 九、进一步优化方向

1. **蒸馏增强**：从更大模型蒸馏推理模式
2. **多尺度 τ**：不同神经元使用不同时间尺度
3. **稀疏路由**：学习动态决定哪些层用 Attention
4. **记忆机制**：结合外部 memory 实现长程依赖
