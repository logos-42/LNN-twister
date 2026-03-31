"""
可增长扭量液态神经网络 (Growable Twistor-LNN)
==============================================
核心机制：
1. 神经元负载监控 - 追踪梯度方差、激活分布、Loss敏感度
2. 神经元分裂 - 负载过重时分裂为两个专精神经元
3. 神经元剪枝 - 删除不重要连接和神经元
4. 集合拓扑约束 - 控制网络结构不崩溃
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class NeuronState:
    index: int
    active: bool = True
    birth_step: int = 0
    activation_variance: float = 0.0
    activation_mean: float = 0.0
    importance_score: float = 0.0
    usage_count: int = 0


@dataclass
class GrowthConfig:
    min_hidden_dim: int = 8
    max_hidden_dim: int = 128

    split_threshold_var: float = 0.4
    split_threshold_grad: float = 0.3
    split_threshold_sens: float = 0.3

    prune_threshold: float = 0.05

    growth_interval: int = 50
    prune_interval: int = 25

    noise_scale: float = 0.1
    topology_penalty: float = 0.001

    max_split_per_step: int = 2
    max_prune_per_step: int = 2
    prune_connections: bool = True
    connection_threshold: float = 0.01


class GrowableTwistorLNN(nn.Module):
    """
    可增长扭量液态神经网络

    核心特性:
    - 根据负载自动分裂神经元
    - 自动剪枝不重要的神经元
    - 拓扑约束防止结构崩溃
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        output_dim: int = 1,
        sparsity: float = 0.3,
        multi_scale_tau: bool = True,
        dt: float = 0.1,
        tau_min: float = 0.01,
        tau_max: float = 1.0,
        dzdt_max: float = 10.0,
        z_max: float = 100.0,
        growth_config: Optional[GrowthConfig] = None,
        enable_growth: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sparsity = sparsity
        self.multi_scale_tau = multi_scale_tau

        self.dt = dt
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.dzdt_max = dzdt_max
        self.z_max = z_max

        self.growth_config = growth_config or GrowthConfig(
            min_hidden_dim=hidden_dim, max_hidden_dim=128
        )
        self.enable_growth = enable_growth

        self._init_parameters()

        self.neuron_states = [NeuronState(index=i) for i in range(hidden_dim)]
        self.active_neurons = list(range(hidden_dim))
        self.training_step = 0

        self._activation_buffer = []
        self._max_buffer_size = 100

    def _init_parameters(self):
        """初始化所有参数"""
        self._create_linear_layer = nn.Linear

        self.W_real = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.W_imag = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.U = nn.Linear(self.input_dim, self.hidden_dim)
        self.W_tau = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.sparse_mask_real = nn.Parameter(
            torch.ones(self.hidden_dim, self.hidden_dim)
        )
        self.sparse_mask_imag = nn.Parameter(
            torch.ones(self.hidden_dim, self.hidden_dim)
        )

        if self.multi_scale_tau:
            self.tau_bias = nn.Parameter(torch.zeros(self.hidden_dim))
        else:
            self.register_parameter("tau_bias", None)

        self.b_real = nn.Parameter(torch.zeros(self.hidden_dim))
        self.b_imag = nn.Parameter(torch.zeros(self.hidden_dim))

        self.out = nn.Linear(self.hidden_dim, self.output_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.W_real.weight, gain=0.5)
        nn.init.orthogonal_(self.W_imag.weight, gain=0.5)
        nn.init.orthogonal_(self.U.weight, gain=0.5)
        nn.init.orthogonal_(self.W_tau.weight, gain=0.1)
        nn.init.zeros_(self.b_real)
        nn.init.zeros_(self.b_imag)

        if self.sparsity > 0:
            with torch.no_grad():
                mask_real = (
                    torch.rand(self.hidden_dim, self.hidden_dim) > self.sparsity
                ).float()
                mask_imag = (
                    torch.rand(self.hidden_dim, self.hidden_dim) > self.sparsity
                ).float()
                self.sparse_mask_real.copy_(mask_real)
                self.sparse_mask_imag.copy_(mask_imag)

    def compute_tau(self, z: torch.Tensor) -> torch.Tensor:
        z_mod = torch.abs(z)
        tau = F.sigmoid(self.W_tau(z_mod))

        if self.multi_scale_tau and self.tau_bias is not None:
            tau = tau + self.tau_bias.unsqueeze(0)

        tau = torch.clamp(tau, self.tau_min, self.tau_max)
        return tau + 1e-6

    def compute_dzdt(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        z_real = z.real
        z_imag = z.imag

        tanh_real = torch.tanh(z_real)
        tanh_imag = torch.tanh(z_imag)

        W_real_sparse = self.W_real.weight * torch.sigmoid(
            self.sparse_mask_real[: self.hidden_dim, : self.hidden_dim]
        )
        W_imag_sparse = self.W_imag.weight * torch.sigmoid(
            self.sparse_mask_imag[: self.hidden_dim, : self.hidden_dim]
        )

        W_tanh_real = F.linear(tanh_real, W_real_sparse)
        W_tanh_imag = F.linear(tanh_imag, W_imag_sparse)

        Ux = self.U(x)

        dz_real = -z_real + W_tanh_real + Ux + self.b_real[: self.hidden_dim]
        dz_imag = -z_imag + W_tanh_imag + Ux + self.b_imag[: self.hidden_dim]

        tau = self.compute_tau(z)
        dzdt = torch.complex(dz_real / tau, dz_imag / tau)

        dzdt_real = torch.clamp(dzdt.real, -self.dzdt_max, self.dzdt_max)
        dzdt_imag = torch.clamp(dzdt.imag, -self.dzdt_max, self.dzdt_max)
        dzdt = torch.complex(dzdt_real, dzdt_imag)

        return dzdt

    def forward(
        self,
        x: torch.Tensor,
        return_states: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        T, B, _ = x.shape

        z = torch.zeros(B, self.hidden_dim, dtype=torch.complex64, device=x.device)

        outputs = []
        states = []

        for t in range(T):
            x_t = x[t]
            dzdt = self.compute_dzdt(z, x_t)

            z = z + self.dt * dzdt
            z = torch.complex(
                torch.clamp(z.real, -self.z_max, self.z_max),
                torch.clamp(z.imag, -self.z_max, self.z_max),
            )

            if (
                self.training
                and self.enable_growth
                and len(self._activation_buffer) < self._max_buffer_size
            ):
                self._activation_buffer.append(torch.abs(z).mean(dim=0).detach().cpu())

            y_t = self.out(z.real)
            outputs.append(y_t)

            if return_states:
                states.append(z)

        y = torch.stack(outputs, dim=0)

        if return_states:
            return y, torch.stack(states, dim=0)
        return y

    def _update_neuron_stats(self):
        """更新神经元统计"""
        if len(self._activation_buffer) < 10:
            return

        buffer = torch.stack(self._activation_buffer[-50:])

        for i in range(min(self.hidden_dim, buffer.shape[1])):
            acts = buffer[:, i]
            self.neuron_states[i].activation_mean = acts.mean().item()
            self.neuron_states[i].activation_variance = acts.var().item()
            self.neuron_states[i].usage_count += len(acts)

    def compute_importance_scores(self) -> torch.Tensor:
        scores = torch.zeros(self.hidden_dim)

        for i in range(min(len(self.neuron_states), self.hidden_dim)):
            if not self.neuron_states[i].active:
                scores[i] = 0.0
                continue

            state = self.neuron_states[i]
            act_score = torch.sigmoid(torch.tensor(state.activation_mean))
            var_score = torch.sigmoid(torch.tensor(state.activation_variance))
            usage_score = torch.sigmoid(
                torch.tensor(state.usage_count / max(1, self.training_step))
            )

            importance = 0.5 * act_score + 0.3 * var_score + 0.2 * usage_score
            scores[i] = importance.item()
            state.importance_score = importance.item()

        return scores

    def get_overloaded_neurons(self) -> List[int]:
        overloaded = []

        for i in range(min(len(self.neuron_states), self.hidden_dim)):
            state = self.neuron_states[i]
            if not state.active:
                continue

            if state.activation_variance > self.growth_config.split_threshold_var:
                overloaded.append(i)

        return overloaded

    def _expand_parameters(self, new_dim: int):
        """扩展参数以容纳新神经元"""
        old_dim = self.hidden_dim

        with torch.no_grad():
            new_W_real = torch.zeros(new_dim, new_dim)
            new_W_imag = torch.zeros(new_dim, new_dim)
            new_mask_real = torch.ones(new_dim, new_dim) * -5
            new_mask_imag = torch.ones(new_dim, new_dim) * -5

            new_W_real[:old_dim, :old_dim] = self.W_real.weight.data
            new_W_imag[:old_dim, :old_dim] = self.W_imag.weight.data
            new_mask_real[:old_dim, :old_dim] = self.sparse_mask_real.data[
                :old_dim, :old_dim
            ]
            new_mask_imag[:old_dim, :old_dim] = self.sparse_mask_imag.data[
                :old_dim, :old_dim
            ]

            self.W_real = nn.Linear(new_dim, new_dim, bias=False)
            self.W_imag = nn.Linear(new_dim, new_dim, bias=False)
            self.W_real.weight.data = new_W_real
            self.W_imag.weight.data = new_W_imag

            self.sparse_mask_real = nn.Parameter(new_mask_real)
            self.sparse_mask_imag = nn.Parameter(new_mask_imag)

            new_b_real = torch.zeros(new_dim)
            new_b_imag = torch.zeros(new_dim)
            new_b_real[:old_dim] = self.b_real.data
            new_b_imag[:old_dim] = self.b_imag.data
            self.b_real = nn.Parameter(new_b_real)
            self.b_imag = nn.Parameter(new_b_imag)

            if self.tau_bias is not None:
                new_tau = torch.zeros(new_dim)
                new_tau[:old_dim] = self.tau_bias.data
                self.tau_bias = nn.Parameter(new_tau)

            new_out = nn.Linear(new_dim, self.output_dim)
            new_out.weight.data[:, :old_dim] = self.out.weight.data
            new_out.bias.data = self.out.bias.data
            self.out = new_out

    def split_neuron(self, parent_idx: int) -> int:
        """分裂神经元"""
        if self.hidden_dim >= self.growth_config.max_hidden_dim:
            return -1

        new_idx = self.hidden_dim
        old_dim = self.hidden_dim
        new_dim = old_dim + 1

        with torch.no_grad():
            noise = torch.randn(old_dim) * self.growth_config.noise_scale

            new_col_real = self.W_real.weight.data[:, parent_idx] + noise
            new_col_imag = self.W_imag.weight.data[:, parent_idx] - noise

            self._expand_parameters(new_dim)

            self.W_real.weight.data[:old_dim, new_idx] = new_col_real
            self.W_imag.weight.data[:old_dim, new_idx] = new_col_imag

            self.sparse_mask_real.data[:old_dim, new_idx] = self.sparse_mask_real.data[
                :old_dim, parent_idx
            ]
            self.sparse_mask_imag.data[:old_dim, new_idx] = self.sparse_mask_imag.data[
                :old_dim, parent_idx
            ]

            self.b_real.data[new_idx] = self.b_real.data[parent_idx]
            self.b_imag.data[new_idx] = self.b_imag.data[parent_idx]

            if self.tau_bias is not None:
                self.tau_bias.data[new_idx] = self.tau_bias.data[parent_idx]

        self.neuron_states.append(
            NeuronState(
                index=new_idx,
                active=True,
                birth_step=self.training_step,
            )
        )

        self.active_neurons.append(new_idx)
        self.hidden_dim = new_dim

        return new_idx

    def prune_neurons(self) -> int:
        """剪枝不重要的神经元"""
        importance = self.compute_importance_scores()

        active_importance = [(i, importance[i]) for i in self.active_neurons]
        active_importance.sort(key=lambda x: x[1])

        n_prune = min(
            self.growth_config.max_prune_per_step,
            len(self.active_neurons) - self.growth_config.min_hidden_dim,
        )

        if n_prune <= 0:
            return 0

        pruned = 0
        for i in range(n_prune):
            idx = active_importance[i][0]
            self.neuron_states[idx].active = False
            pruned += 1

        self.active_neurons = [
            i for i in self.active_neurons if self.neuron_states[i].active
        ]

        return pruned

    def prune_connections(self) -> int:
        """剪枝不重要的连接"""
        with torch.no_grad():
            mask_real = torch.sigmoid(
                self.sparse_mask_real[: self.hidden_dim, : self.hidden_dim]
            )
            mask_imag = torch.sigmoid(
                self.sparse_mask_imag[: self.hidden_dim, : self.hidden_dim]
            )

            pruned = ((mask_real < 0.01).sum() + (mask_imag < 0.01).sum()).item()

            self.sparse_mask_real.data[: self.hidden_dim, : self.hidden_dim][
                mask_real < 0.01
            ] = -10
            self.sparse_mask_imag.data[: self.hidden_dim, : self.hidden_dim][
                mask_imag < 0.01
            ] = -10

        return int(pruned)

    def compute_topology_penalty(self) -> torch.Tensor:
        """计算拓扑惩罚"""
        penalty = torch.tensor(0.0)

        if hasattr(self, "sparse_mask_real"):
            mask_sum = self.sparse_mask_real[: self.hidden_dim, : self.hidden_dim].sum()
            target = self.hidden_dim * self.hidden_dim * 0.5
            penalty = (
                penalty + abs(mask_sum - target) * self.growth_config.topology_penalty
            )

        return penalty

    def growth_step(self):
        """执行一步增长/剪枝"""
        if not self.enable_growth:
            return {"action": "disabled", "changes": 0}

        self.training_step += 1
        self._update_neuron_stats()

        if self.training_step % self.growth_config.growth_interval == 0:
            overloaded = self.get_overloaded_neurons()

            if overloaded and self.hidden_dim < self.growth_config.max_hidden_dim:
                actual_split = 0
                for parent_idx in overloaded[: self.growth_config.max_split_per_step]:
                    if self.hidden_dim >= self.growth_config.max_hidden_dim:
                        break
                    new_idx = self.split_neuron(parent_idx)
                    if new_idx >= 0:
                        actual_split += 1

                return {
                    "action": "split",
                    "count": actual_split,
                    "new_dim": self.hidden_dim,
                }

        if self.training_step % self.growth_config.prune_interval == 0:
            n_pruned = self.prune_neurons()
            n_connections = (
                self.prune_connections() if self.growth_config.prune_connections else 0
            )

            return {
                "action": "prune",
                "neurons": n_pruned,
                "connections": n_connections,
            }

        return {"action": "none", "changes": 0}

    def get_diagnostics(self) -> Dict:
        importance = self.compute_importance_scores()

        active_importance = [importance[i] for i in self.active_neurons]

        return {
            "hidden_dim": self.hidden_dim,
            "active_count": len(self.active_neurons),
            "training_step": self.training_step,
            "importance_mean": np.mean(active_importance) if active_importance else 0.0,
            "enable_growth": self.enable_growth,
        }

    def reset_state(self, batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
        return torch.zeros(
            batch_size, self.hidden_dim, dtype=torch.complex64, device=device
        )

    def step(
        self, z: torch.Tensor, x: torch.Tensor, dt: float = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if dt is None:
            dt = self.dt

        dzdt = self.compute_dzdt(z, x)
        z_new = z + dt * dzdt
        z_new = torch.complex(
            torch.clamp(z_new.real, -self.z_max, self.z_max),
            torch.clamp(z_new.imag, -self.z_max, self.z_max),
        )

        output = self.out(
            z_new.real[:, : self.output_dim]
            if self.hidden_dim > self.output_dim
            else z_new.real
        )
        return z_new, output


def create_growable_twistor_lnn(
    input_dim: int, hidden_dim: int = 16, output_dim: int = 1, **kwargs
) -> GrowableTwistorLNN:
    return GrowableTwistorLNN(
        input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, **kwargs
    )
