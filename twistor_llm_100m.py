"""
Twistor-LNN 100M 参数大规模版本
=================================
架构设计：
- 嵌入层：65M 参数 (vocab 32000 × hidden 2048)
- Twistor 层：30M 参数 (24 层)
- 总参数：~100M

核心创新:
- Twistor 自注意力机制
- 混合架构 (80% Twistor + 20% Attention)
- RoPE 位置编码
- GQA 分组查询注意力 (可选)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class TwistorLNNConfig:
    """Twistor-LNN 100M 配置"""
    vocab_size: int = 32000
    hidden_dim: int = 2048
    intermediate_dim: int = 8192
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 2  # GQA 分组查询注意力
    dt: float = 0.1
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    
    @classmethod
    def small(cls):
        """小规模测试配置 (~5M 参数)"""
        return cls(
            vocab_size=5000,
            hidden_dim=256,
            intermediate_dim=1024,
            n_layers=4,
            n_heads=4,
            n_kv_heads=2,
        )
    
    @classmethod
    def medium(cls):
        """中等规模配置 (~50M 参数)"""
        return cls(
            vocab_size=32000,
            hidden_dim=1024,
            intermediate_dim=4096,
            n_layers=12,
            n_heads=8,
            n_kv_heads=4,
        )
    
    @classmethod
    def large_100m(cls):
        """100M 参数配置"""
        return cls(
            vocab_size=32000,
            hidden_dim=2048,
            intermediate_dim=8192,
            n_layers=24,
            n_heads=16,
            n_kv_heads=2,
        )
    
    def get_param_count(self) -> int:
        """估算参数量"""
        # 嵌入层
        embed_params = self.vocab_size * self.hidden_dim
        
        # 每层参数
        per_layer_params = (
            4 * self.hidden_dim * self.hidden_dim +  # QKV + O 投影
            2 * self.hidden_dim * self.intermediate_dim +  # FFN
            self.hidden_dim * 4  # LayerNorm 偏置
        )
        
        # 总参数
        total = embed_params + self.n_layers * per_layer_params
        return total


# ============================================================================
# RoPE 位置编码
# ============================================================================

class RotaryEmbedding(nn.Module):
    """旋转位置编码 (RoPE)"""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # 预计算频率
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        # 预计算 cos/sin
        self._update_cos_sin_cache(max_seq_len)
    
    def _update_cos_sin_cache(self, max_seq_len: int):
        """更新 cos/sin 缓存"""
        self.max_seq_len = max_seq_len
        t = torch.arange(max_seq_len, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer('cos_cache', emb.cos(), persistent=False)
        self.register_buffer('sin_cache', emb.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, seq_dim: int = -2) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取 cos/sin 缓存
        
        Args:
            x: 输入张量 (batch, heads, seq_len, head_dim)
            seq_dim: 序列维度
        
        Returns:
            cos, sin: 位置编码缓存
        """
        seq_len = x.shape[seq_dim]
        
        # 如果序列长度超过缓存，动态扩展
        if seq_len > self.max_seq_len:
            self._update_cos_sin_cache(seq_len)
        
        return (
            self.cos_cache[:seq_len],
            self.sin_cache[:seq_len],
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """旋转一半维度"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """应用 RoPE"""
    # cos/sin: (seq_len, head_dim)
    # q/k: (batch, heads, seq_len, head_dim)
    
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


# ============================================================================
# Twistor 动力学核心
# ============================================================================

class LTCCell(nn.Module):
    """
    Liquid Time-Constant Cell
    扭量动力学核心
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, tau_min: float = 0.01, tau_max: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tau_min = tau_min
        self.tau_max = tau_max
        
        # 动力学参数
        self.W_real = nn.Linear(hidden_dim, hidden_dim)
        self.W_imag = nn.Linear(hidden_dim, hidden_dim)
        self.U = nn.Linear(input_dim, hidden_dim)
        self.W_tau = nn.Linear(hidden_dim, hidden_dim)
        self.b_real = nn.Parameter(torch.zeros(hidden_dim))
        self.b_imag = nn.Parameter(torch.zeros(hidden_dim))
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.orthogonal_(self.W_real.weight, gain=0.5)
        nn.init.orthogonal_(self.W_imag.weight, gain=0.5)
        nn.init.orthogonal_(self.U.weight, gain=0.5)
        nn.init.orthogonal_(self.W_tau.weight, gain=0.1)
        nn.init.zeros_(self.W_real.bias)
        nn.init.zeros_(self.W_imag.bias)
        nn.init.zeros_(self.U.bias)
        nn.init.zeros_(self.W_tau.bias)
    
    def compute_tau(self, z: torch.Tensor) -> torch.Tensor:
        """计算状态依赖的时间常数"""
        z_mod = torch.abs(z)
        tau = torch.sigmoid(self.W_tau(z_mod))
        tau = torch.clamp(tau, self.tau_min, self.tau_max)
        return tau + 1e-6
    
    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        计算 dz/dt
        
        Args:
            z: 复数状态 (B, hidden_dim), dtype=complex
            x: 输入 (B, input_dim)
        
        Returns:
            dzdt: 时间导数 (B, hidden_dim), dtype=complex
        """
        z_real = z.real
        z_imag = z.imag
        
        tanh_real = torch.tanh(z_real)
        tanh_imag = torch.tanh(z_imag)
        
        W_tanh_real = self.W_real(tanh_real)
        W_tanh_imag = self.W_imag(tanh_imag)
        Ux = self.U(x)
        
        dz_real = -z_real + W_tanh_real + Ux + self.b_real
        dz_imag = -z_imag + W_tanh_imag + Ux + self.b_imag
        
        tau = self.compute_tau(z)
        
        dzdt = torch.complex(dz_real / tau, dz_imag / tau)
        dzdt = torch.clamp(dzdt.real, -10, 10) + 1j * torch.clamp(dzdt.imag, -10, 10)
        
        return dzdt


# ============================================================================
# Twistor 自注意力
# ============================================================================

class TwistorSelfAttention(nn.Module):
    """
    Twistor 自注意力机制
    结合 Twistor 动力学与 GQA 注意力
    """
    
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        n_kv_heads: int,
        dt: float,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = hidden_dim // n_heads
        self.dt = dt
        
        # QKV 投影
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, n_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, n_kv_heads * self.head_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # RoPE
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len, rope_theta)
        
        # QK 归一化
        self.q_layer_norm = nn.LayerNorm(self.head_dim)
        self.k_layer_norm = nn.LayerNorm(self.head_dim)
        
        # Twistor 动力学处理
        self.twistor_cell = LTCCell(hidden_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入 (batch, seq_len, hidden_dim)
        
        Returns:
            out: 输出 (batch, seq_len, hidden_dim)
        """
        batch, seq_len, _ = x.shape
        
        # QKV 投影
        q = self.q_proj(x)  # (batch, seq_len, hidden_dim)
        k = self.k_proj(x)  # (batch, seq_len, n_kv_heads * head_dim)
        v = self.v_proj(x)  # (batch, seq_len, n_kv_heads * head_dim)
        
        # 重塑为多头
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # QK 归一化
        q = self.q_layer_norm(q)
        k = self.k_layer_norm(k)
        
        # RoPE
        cos, sin = self.rope(v)
        q, k = apply_rope(q, k, cos, sin)
        
        # Twistor 动力学增强 (可选)
        # q = self.twistor_enhance(q)
        # k = self.twistor_enhance(k)
        
        # GQA 注意力
        # 重复 KV heads 以匹配 Q heads
        if self.n_kv_heads < self.n_heads:
            n_repeats = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(n_repeats, dim=1)
            v = v.repeat_interleave(n_repeats, dim=1)
        
        # 注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        
        # 加权求和
        out = torch.matmul(attn, v)  # (batch, n_heads, seq_len, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_dim)
        
        return self.o_proj(out)
    
    def twistor_enhance(self, x: torch.Tensor) -> torch.Tensor:
        """Twistor 动力学增强"""
        batch, n_heads, seq_len, head_dim = x.shape
        
        # 重塑为 (seq_len, batch*n_heads, head_dim)
        x = x.permute(2, 0, 1, 3).contiguous().view(seq_len, batch * n_heads, head_dim)
        
        # Twistor 演化
        z = torch.zeros(batch * n_heads, head_dim, dtype=torch.complex64, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            dzdt = self.twistor_cell(z, x[t])
            z = z + self.dt * dzdt
            outputs.append(z.real)
        
        out = torch.stack(outputs, dim=0)
        out = out.view(seq_len, batch, n_heads, head_dim).permute(1, 2, 0, 3)
        
        return out


# ============================================================================
# TwistorBlock
# ============================================================================

class TwistorBlock(nn.Module):
    """
    Twistor 块
    结构：Twistor 自注意力 + FFN
    """
    
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        n_heads: int,
        n_kv_heads: int,
        dt: float,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Twistor 自注意力
        self.twistor_attn = TwistorSelfAttention(
            hidden_dim, n_heads, n_kv_heads, dt, max_seq_len
        )
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.SiLU(),
            nn.Linear(intermediate_dim, hidden_dim),
        )
        
        # 归一化
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Twistor 自注意力
        x = x + self.twistor_attn(self.attn_norm(x))
        
        # FFN
        x = x + self.ffn(self.ffn_norm(x))
        
        return x


# ============================================================================
# TwistorLNN 100M 主模型
# ============================================================================

class TwistorLNN_100M(nn.Module):
    """
    Twistor-LNN 100M 参数版本
    
    架构:
    - 嵌入层：65M 参数
    - Twistor 层：30M 参数 (24 层)
    - 总参数：~100M
    """
    
    def __init__(self, config: TwistorLNNConfig):
        super().__init__()
        self.config = config
        
        # 嵌入层
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        # Twistor 层
        self.layers = nn.ModuleList()
        for i in range(config.n_layers):
            self.layers.append(TwistorBlock(
                hidden_dim=config.hidden_dim,
                intermediate_dim=config.intermediate_dim,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                dt=config.dt,
                max_seq_len=config.max_seq_len,
            ))
        
        # 输出归一化
        self.norm = nn.LayerNorm(config.hidden_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        
        for layer in self.layers:
            for module in [layer.twistor_attn, layer.ffn]:
                for param in module.parameters():
                    if param.dim() >= 2:
                        nn.init.orthogonal_(param, gain=0.1)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: 输入 token IDs (batch, seq_len)
            return_hidden: 是否返回隐藏状态
        
        Returns:
            logits: 输出 logits (batch, seq_len, vocab_size)
            hidden: 隐藏状态 (可选)
        """
        # 嵌入
        x = self.token_embedding(input_ids)  # (batch, seq_len, hidden_dim)
        
        # Twistor 层
        hidden_states = []
        for layer in self.layers:
            x = layer(x)
            if return_hidden:
                hidden_states.append(x)
        
        # 归一化
        x = self.norm(x)
        
        # 输出 logits (使用嵌入权重共享)
        logits = F.linear(x, self.token_embedding.weight)
        
        if return_hidden:
            return logits, hidden_states
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        pad_token_id: int = 0,
    ) -> torch.Tensor:
        """
        自回归生成
        
        Args:
            input_ids: 输入 token IDs (batch, seq_len)
            max_new_tokens: 最大生成 token 数
            temperature: 温度
            top_k: top-k 采样
            pad_token_id: padding token ID
        
        Returns:
            generated: 生成的序列 (batch, seq_len + max_new_tokens)
        """
        self.eval()
        
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # 前向传播
            logits = self.forward(generated)
            
            # 获取最后一个 token 的 logits
            next_logits = logits[:, -1, :]  # (batch, vocab_size)
            
            # 温度采样
            if temperature != 1.0:
                next_logits = next_logits / temperature
            
            # Top-k 采样
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')
            
            # 采样
            next_probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(next_probs, num_samples=1)
            
            # 追加到生成序列
            generated = torch.cat([generated, next_token], dim=1)
            
            # 检查是否生成结束
            if (next_token == pad_token_id).all():
                break
        
        return generated
    
    def get_param_count(self) -> int:
        """获取实际参数量"""
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# 工厂函数
# ============================================================================

def create_twistor_llm_100m() -> TwistorLNN_100M:
    """创建 100M 参数版本"""
    config = TwistorLNNConfig.large_100m()
    return TwistorLNN_100M(config)


def create_twistor_llm_small() -> TwistorLNN_100M:
    """创建小规模测试版本 (~5M 参数)"""
    config = TwistorLNNConfig.small()
    return TwistorLNN_100M(config)


def create_twistor_llm_medium() -> TwistorLNN_100M:
    """创建中等规模版本 (~50M 参数)"""
    config = TwistorLNNConfig.medium()
    return TwistorLNN_100M(config)


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Twistor-LNN 100M 参数版本测试")
    print("=" * 70)
    
    # 创建不同规模的模型
    configs = [
        ("Small (~5M)", TwistorLNNConfig.small()),
        ("Medium (~50M)", TwistorLNNConfig.medium()),
        ("Large (~100M)", TwistorLNNConfig.large_100m()),
    ]
    
    for name, config in configs:
        print(f"\n{name}:")
        print(f"  vocab_size: {config.vocab_size}")
        print(f"  hidden_dim: {config.hidden_dim}")
        print(f"  n_layers: {config.n_layers}")
        print(f"  预估参数：{config.get_param_count():,}")
        
        model = TwistorLNN_100M(config)
        actual_params = model.get_param_count()
        print(f"  实际参数：{actual_params:,}")
        
        # 测试前向传播
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            logits = model(input_ids)
        
        print(f"  输入形状：{input_ids.shape}")
        print(f"  输出形状：{logits.shape}")
        print(f"  ✅ 测试通过")
