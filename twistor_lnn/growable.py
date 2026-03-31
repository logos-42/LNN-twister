"""
可增长扭量液态神经网络 (Growable Twistor-LNN)
==============================================
完全模仿NEAT (NeuroEvolution of Augmenting Topologies)机制：
1. Add Node Mutation - 分裂神经元，禁用原连接，插入新节点
2. Add Connection Mutation - 添加新连接
3. Disable Connection - 禁用连接(剪枝)
4. 从最小结构开始(0隐藏节点)
5. 扭量复数状态空间
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ConnectionGene:
    """连接基因 - NEAT风格的基因表示"""

    in_node: int
    out_node: int
    weight: float
    enabled: bool = True
    innovation: int = 0


@dataclass
class NeuronState:
    index: int
    neuron_type: str = "hidden"  # input, output, hidden
    active: bool = True
    birth_step: int = 0
    activation_variance: float = 0.0
    activation_mean: float = 0.0
    importance_score: float = 0.0
    usage_count: int = 0


@dataclass
class GrowthConfig:
    min_hidden_dim: int = 0
    max_hidden_dim: int = 128

    prob_add_connection: float = 0.05
    prob_add_node: float = 0.03
    prob_disable_connection: float = 0.1

    prune_threshold: float = 0.05
    connection_threshold: float = 0.01

    growth_interval: int = 50
    prune_interval: int = 25

    topology_penalty: float = 0.001


class GrowableTwistorLNN(nn.Module):
    """
    可增长扭量液态神经网络 - NEAT风格

    核心机制:
    - Add Node Mutation: 分裂神经元，禁用原连接
    - Add Connection Mutation: 添加新连接
    - Disable Connection: 剪枝不重要连接
    - 从最小结构开始，逐步增长
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 0,
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
        enable_mobius: bool = False,
        enable_resonance: bool = False,
        mobius_strength: float = 0.1,
        resonance_strength: float = 0.1,
        learn_manifold_dim: bool = True,
        sparse_resonance: bool = True,
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
            min_hidden_dim=0, max_hidden_dim=128
        )
        self.enable_growth = enable_growth

        self._next_innovation = 1
        self.connection_genes: List[ConnectionGene] = []

        self.mobius = None
        self.resonance = None
        self._resonance_mode = "additive"

        self._init_parameters()

        if enable_mobius or enable_resonance:
            self._init_mobius_resonance(
                enable_mobius=enable_mobius,
                enable_resonance=enable_resonance,
                mobius_strength=mobius_strength,
                resonance_strength=resonance_strength,
                learn_manifold_dim=learn_manifold_dim,
                sparse_resonance=sparse_resonance,
            )

        self.neuron_states: List[NeuronState] = []
        for i in range(input_dim):
            self.neuron_states.append(NeuronState(index=i, neuron_type="input"))
        for i in range(output_dim):
            self.neuron_states.append(
                NeuronState(index=input_dim + i, neuron_type="output")
            )
        for i in range(hidden_dim):
            self.neuron_states.append(
                NeuronState(index=input_dim + output_dim + i, neuron_type="hidden")
            )

        self.active_neurons = list(range(input_dim + output_dim))
        if hidden_dim > 0:
            self.active_neurons.extend(
                range(input_dim + output_dim, input_dim + output_dim + hidden_dim)
            )

        self.training_step = 0
        self._activation_buffer = []
        self._max_buffer_size = 100

    def _init_parameters(self):
        self.W_real = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.W_imag = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.U = nn.Linear(self.input_dim, self.hidden_dim)
        self.W_tau = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.sparse_mask_real = nn.Parameter(
            torch.ones(max(1, self.hidden_dim), max(1, self.hidden_dim))
        )
        self.sparse_mask_imag = nn.Parameter(
            torch.ones(max(1, self.hidden_dim), max(1, self.hidden_dim))
        )

        if self.multi_scale_tau:
            self.tau_bias = nn.Parameter(torch.zeros(max(1, self.hidden_dim)))
        else:
            self.register_parameter("tau_bias", None)

        self.b_real = nn.Parameter(torch.zeros(max(1, self.hidden_dim)))
        self.b_imag = nn.Parameter(torch.zeros(max(1, self.hidden_dim)))

        self.out = nn.Linear(self.hidden_dim, self.output_dim)

        self._init_weights()

    def _init_weights(self):
        if self.hidden_dim > 0:
            nn.init.orthogonal_(self.W_real.weight, gain=0.5)
            nn.init.orthogonal_(self.W_imag.weight, gain=0.5)
        nn.init.orthogonal_(self.U.weight, gain=0.5)
        if self.hidden_dim > 0:
            nn.init.orthogonal_(self.W_tau.weight, gain=0.1)
        nn.init.zeros_(self.b_real)
        nn.init.zeros_(self.b_imag)

        if self.sparsity > 0 and self.hidden_dim > 0:
            with torch.no_grad():
                mask_real = (
                    torch.rand(self.hidden_dim, self.hidden_dim) > self.sparsity
                ).float()
                mask_imag = (
                    torch.rand(self.hidden_dim, self.hidden_dim) > self.sparsity
                ).float()
                self.sparse_mask_real.copy_(mask_real)
                self.sparse_mask_imag.copy_(mask_imag)

    def _init_mobius_resonance(
        self,
        enable_mobius: bool = True,
        enable_resonance: bool = True,
        mobius_strength: float = 0.1,
        resonance_strength: float = 0.1,
        learn_manifold_dim: bool = True,
        sparse_resonance: bool = True,
    ):
        """初始化莫比乌斯约束和共振注意力"""
        if enable_mobius:
            from .mobius import MobiusConstraint

            max_h = self.growth_config.max_hidden_dim
            self.mobius = MobiusConstraint(
                max_dim=max(max_h * 4, 512),
                constraint_strength=mobius_strength,
                enable_learning=learn_manifold_dim,
                device=str(self.U.weight.device),
            )

        if enable_resonance:
            from .resonance import TwistorResonance

            self.resonance = TwistorResonance(
                hidden_dim=max(1, self.hidden_dim),
                resonance_strength=resonance_strength,
                sparse_mode=sparse_resonance,
                device=str(self.U.weight.device),
            )

    def compute_tau(self, z: torch.Tensor) -> torch.Tensor:
        if self.hidden_dim == 0:
            return torch.ones_like(z.real) * self.tau_min

        z_mod = torch.abs(z)

        if self.W_tau.weight.shape[0] == 0:
            return torch.ones_like(z_mod) * self.tau_min

        tau = F.sigmoid(self.W_tau(z_mod))

        if self.multi_scale_tau and self.tau_bias is not None:
            tau = tau + self.tau_bias.unsqueeze(0)

        tau = torch.clamp(tau, self.tau_min, self.tau_max)
        return tau + 1e-6

    def compute_dzdt(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.hidden_dim == 0:
            return torch.zeros_like(z)

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

        if self.resonance is not None and self.hidden_dim > 0:
            topo_weights = None
            if self.mobius is not None:
                topo_weights = self.mobius.topology_weight_matrix(self.hidden_dim)
            dzdt_resonance = self.resonance(
                z, topology_weights=topo_weights, mode=self._resonance_mode
            )
            dzdt = dzdt + dzdt_resonance

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

            if self.mobius is not None and self.hidden_dim > 0:
                z = self.mobius.project_state(z)

            z = torch.complex(
                torch.clamp(z.real, -self.z_max, self.z_max),
                torch.clamp(z.imag, -self.z_max, self.z_max),
            )

            if (
                self.training
                and self.enable_growth
                and self.hidden_dim > 0
                and len(self._activation_buffer) < self._max_buffer_size
            ):
                self._activation_buffer.append(torch.abs(z).mean(dim=0).detach().cpu())

            y_t = (
                self.out(z.real)
                if self.hidden_dim > 0
                else torch.zeros(B, self.output_dim, device=x.device)
            )
            outputs.append(y_t)

            if return_states:
                states.append(z)

        y = torch.stack(outputs, dim=0)

        if return_states:
            return y, torch.stack(states, dim=0)
        return y

    def _update_neuron_stats(self):
        if self.hidden_dim == 0 or len(self._activation_buffer) < 10:
            return

        buffer = torch.stack(self._activation_buffer[-50:])

        input_offset = self.input_dim
        output_offset = self.input_dim + self.output_dim

        for i in range(self.hidden_dim):
            state_idx = output_offset + i
            if state_idx >= len(self.neuron_states):
                continue
            acts = buffer[:, i]
            self.neuron_states[state_idx].activation_mean = acts.mean().item()
            self.neuron_states[state_idx].activation_variance = acts.var().item()
            self.neuron_states[state_idx].usage_count += len(acts)

    def compute_importance_scores(self) -> torch.Tensor:
        scores = torch.zeros(self.hidden_dim)

        input_offset = self.input_dim
        output_offset = self.input_dim + self.output_dim

        for i in range(self.hidden_dim):
            state_idx = output_offset + i
            if state_idx >= len(self.neuron_states):
                continue
            state = self.neuron_states[state_idx]
            if not state.active:
                scores[i] = 0.0
                continue

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

        input_offset = self.input_dim
        output_offset = self.input_dim + self.output_dim

        for i in range(self.hidden_dim):
            state_idx = output_offset + i
            if state_idx >= len(self.neuron_states):
                continue
            state = self.neuron_states[state_idx]
            if not state.active:
                continue

            if state.activation_variance > 0.3:
                overloaded.append(i)

        return overloaded

    def _get_next_innovation(self) -> int:
        """获取下一个创新编号"""
        self._next_innovation += 1
        return self._next_innovation - 1

    def _expand_parameters(self, new_dim: int):
        old_dim = self.hidden_dim
        if old_dim == 0:
            old_dim = 1

        with torch.no_grad():
            new_W_real = torch.zeros(new_dim, new_dim)
            new_W_imag = torch.zeros(new_dim, new_dim)
            new_W_tau = torch.zeros(new_dim, new_dim)
            new_mask_real = torch.ones(new_dim, new_dim) * -5
            new_mask_imag = torch.ones(new_dim, new_dim) * -5

            if self.hidden_dim > 0:
                new_W_real[:old_dim, :old_dim] = self.W_real.weight.data
                new_W_imag[:old_dim, :old_dim] = self.W_imag.weight.data
                new_W_tau[:old_dim, :old_dim] = self.W_tau.weight.data
                new_mask_real[:old_dim, :old_dim] = self.sparse_mask_real.data[
                    :old_dim, :old_dim
                ]
                new_mask_imag[:old_dim, :old_dim] = self.sparse_mask_imag.data[
                    :old_dim, :old_dim
                ]

            self.W_real = nn.Linear(new_dim, new_dim, bias=False)
            self.W_imag = nn.Linear(new_dim, new_dim, bias=False)
            self.W_tau = nn.Linear(new_dim, new_dim)
            if new_dim > 0:
                self.W_real.weight.data = new_W_real
                self.W_imag.weight.data = new_W_imag
                self.W_tau.weight.data = new_W_tau

            self.sparse_mask_real = nn.Parameter(new_mask_real)
            self.sparse_mask_imag = nn.Parameter(new_mask_imag)

            new_b_real = torch.zeros(new_dim)
            new_b_imag = torch.zeros(new_dim)
            if self.hidden_dim > 0:
                new_b_real[:old_dim] = self.b_real.data[:old_dim]
                new_b_imag[:old_dim] = self.b_imag.data[:old_dim]
            self.b_real = nn.Parameter(new_b_real)
            self.b_imag = nn.Parameter(new_b_imag)

            if self.tau_bias is not None:
                new_tau = torch.zeros(new_dim)
                if self.hidden_dim > 0:
                    new_tau[:old_dim] = self.tau_bias.data[:old_dim]
                self.tau_bias = nn.Parameter(new_tau)

            new_out = nn.Linear(new_dim, self.output_dim)
            if self.hidden_dim > 0:
                new_out.weight.data[:, :old_dim] = self.out.weight.data[:, :old_dim]
                new_out.bias.data = self.out.bias.data
            self.out = new_out

    def add_connection(self, in_node: int, out_node: int) -> bool:
        """添加新连接 - NEAT Add Connection Mutation"""
        if in_node < 0 or out_node < 0:
            return False
        if in_node >= self.input_dim + self.output_dim + self.hidden_dim:
            return False
        if out_node >= self.input_dim + self.output_dim + self.hidden_dim:
            return False

        hidden_in_offset = self.input_dim + self.output_dim
        hidden_out_offset = self.input_dim + self.output_dim

        for gene in self.connection_genes:
            if gene.in_node == in_node and gene.out_node == out_node:
                return False

        if in_node >= hidden_in_offset and out_node >= hidden_in_offset:
            in_idx = in_node - hidden_in_offset
            out_idx = out_node - hidden_out_offset
            if (
                in_idx >= 0
                and in_idx < self.hidden_dim
                and out_idx >= 0
                and out_idx < self.hidden_dim
            ):
                pass
        elif in_node < self.input_dim and out_node >= hidden_in_offset:
            pass
        elif (
            in_node >= hidden_in_offset and out_node >= self.input_dim + self.output_dim
        ):
            pass
        else:
            return False

        weight = torch.randn(1).item() * 0.5

        gene = ConnectionGene(
            in_node=in_node,
            out_node=out_node,
            weight=weight,
            enabled=True,
            innovation=self._get_next_innovation(),
        )
        self.connection_genes.append(gene)

        self._apply_connection_gene(gene)

        return True

    def _apply_connection_gene(self, gene: ConnectionGene):
        """应用连接基因到权重矩阵"""
        hidden_in_offset = self.input_dim + self.output_dim
        hidden_out_offset = self.input_dim + self.output_dim

        if gene.in_node >= hidden_in_offset and gene.out_node >= hidden_in_offset:
            in_idx = gene.in_node - hidden_in_offset
            out_idx = gene.out_node - hidden_in_offset

            if (
                in_idx >= 0
                and in_idx < self.hidden_dim
                and out_idx >= 0
                and out_idx < self.hidden_dim
            ):
                self.W_real.weight.data[out_idx, in_idx] = gene.weight
                self.W_imag.weight.data[out_idx, in_idx] = -gene.weight

                self.sparse_mask_real.data[out_idx, in_idx] = 2.0
                self.sparse_mask_imag.data[out_idx, in_idx] = 2.0

        elif gene.in_node < self.input_dim and gene.out_node >= hidden_in_offset:
            out_idx = gene.out_node - hidden_in_offset
            in_idx = gene.in_node

            if (
                out_idx >= 0
                and out_idx < self.hidden_dim
                and in_idx >= 0
                and in_idx < self.input_dim
            ):
                self.U.weight.data[out_idx, in_idx] = gene.weight

    def split_neuron(self, parent_idx: int) -> int:
        """分裂神经元 - NEAT Add Node Mutation

        NEAT机制:
        1. 找到父神经元的输入连接
        2. 禁用原连接
        3. 在原连接位置插入新节点
        4. in->new 权重=1.0, new->out 权重=原权重

        扭量扩展:
        - 两个子神经元分别继承父神经元的实部和虚部特性
        - 保持复数状态表示
        """
        if self.hidden_dim >= self.growth_config.max_hidden_dim:
            return -1

        if parent_idx < 0 or parent_idx >= self.hidden_dim:
            return -1

        new_idx = self.hidden_dim
        old_dim = self.hidden_dim
        new_dim = old_dim + 1

        input_offset = self.input_dim + self.output_dim
        parent_node = input_offset + parent_idx

        parent_connections = []
        for i, gene in enumerate(self.connection_genes):
            if gene.out_node == parent_node and gene.enabled:
                parent_connections.append((i, gene))

        if len(parent_connections) == 0:
            return -1

        gene_idx, parent_gene = parent_connections[0]

        parent_gene.enabled = False

        new_node = input_offset + new_idx

        in_to_new = ConnectionGene(
            in_node=parent_gene.in_node,
            out_node=new_node,
            weight=1.0,
            enabled=True,
            innovation=self._get_next_innovation(),
        )

        new_to_out = ConnectionGene(
            in_node=new_node,
            out_node=parent_node,
            weight=parent_gene.weight,
            enabled=True,
            innovation=self._get_next_innovation(),
        )

        self.connection_genes.append(in_to_new)
        self.connection_genes.append(new_to_out)

        with torch.no_grad():
            self._expand_parameters(new_dim)

            in_idx = parent_gene.in_node - input_offset
            if in_idx >= 0 and in_idx < old_dim:
                self.W_real.weight.data[new_idx, in_idx] = 1.0
                self.W_imag.weight.data[new_idx, in_idx] = 1.0

                self.sparse_mask_real.data[new_idx, in_idx] = 2.0
                self.sparse_mask_imag.data[new_idx, in_idx] = 2.0

                out_idx = parent_idx
                self.W_real.weight.data[out_idx, new_idx] = parent_gene.weight
                self.W_imag.weight.data[out_idx, new_idx] = -parent_gene.weight

                self.sparse_mask_real.data[out_idx, new_idx] = 2.0
                self.sparse_mask_imag.data[out_idx, new_idx] = 2.0
            else:
                in_input_idx = parent_gene.in_node
                if in_input_idx < self.input_dim:
                    self.U.weight.data[new_idx, in_input_idx] = 1.0
                    self.out.weight.data[:, new_idx] = torch.tensor(
                        [parent_gene.weight] * self.output_dim
                    )

            self.b_real.data[new_idx] = (
                self.b_real.data[parent_idx] + torch.randn(1).item() * 0.1
            )
            self.b_imag.data[new_idx] = (
                self.b_imag.data[parent_idx] + torch.randn(1).item() * 0.1
            )

            if self.tau_bias is not None:
                self.tau_bias.data[new_idx] = self.tau_bias.data[parent_idx]

        self.neuron_states.append(
            NeuronState(
                index=new_node,
                neuron_type="hidden",
                active=True,
                birth_step=self.training_step,
            )
        )

        self.active_neurons.append(new_node)
        self.hidden_dim = new_dim

        return new_idx

    def add_random_connection(self) -> bool:
        """随机添加连接"""
        if self.hidden_dim == 0:
            return False

        input_offset = self.input_dim + self.output_dim
        output_offset = self.input_dim + self.output_dim

        candidates = []

        for in_node in range(self.input_dim):
            for out_node in range(output_offset, output_offset + self.hidden_dim):
                exists = any(
                    g.in_node == in_node and g.out_node == out_node and g.enabled
                    for g in self.connection_genes
                )
                if not exists:
                    candidates.append((in_node, out_node))

        for in_node in range(input_offset, input_offset + self.hidden_dim):
            for out_node in range(input_offset, input_offset + self.hidden_dim):
                if in_node != out_node:
                    exists = any(
                        g.in_node == in_node and g.out_node == out_node and g.enabled
                        for g in self.connection_genes
                    )
                    if not exists:
                        candidates.append((in_node, out_node))

        for in_node in range(input_offset, input_offset + self.hidden_dim):
            for out_node in range(self.input_dim, self.input_dim + self.output_dim):
                exists = any(
                    g.in_node == in_node and g.out_node == out_node and g.enabled
                    for g in self.connection_genes
                )
                if not exists:
                    candidates.append((in_node, out_node))

        if not candidates:
            return False

        in_node, out_node = candidates[torch.randint(len(candidates), (1,)).item()]
        return self.add_connection(in_node, out_node)

    def disable_random_connection(self) -> bool:
        """随机禁用连接 - NEAT Disable Connection Mutation"""
        enabled_connections = [
            i for i, g in enumerate(self.connection_genes) if g.enabled
        ]
        if not enabled_connections:
            return False

        idx = torch.randint(len(enabled_connections), (1,)).item()
        gene_idx = enabled_connections[idx]
        self.connection_genes[gene_idx].enabled = False

        gene = self.connection_genes[gene_idx]

        hidden_in_offset = self.input_dim + self.output_dim
        hidden_out_offset = self.input_dim + self.output_dim

        if gene.in_node >= hidden_in_offset and gene.out_node >= hidden_in_offset:
            in_idx = gene.in_node - hidden_in_offset
            out_idx = gene.out_node - hidden_in_offset

            if (
                in_idx >= 0
                and in_idx < self.hidden_dim
                and out_idx >= 0
                and out_idx < self.hidden_dim
            ):
                self.sparse_mask_real.data[out_idx, in_idx] = -10
                self.sparse_mask_imag.data[out_idx, in_idx] = -10

        elif gene.in_node < self.input_dim and gene.out_node >= hidden_in_offset:
            out_idx = gene.out_node - hidden_in_offset
            if out_idx >= 0 and out_idx < self.hidden_dim:
                pass

        return True

    def prune_neurons(self) -> int:
        """剪枝不重要的神经元"""
        if self.hidden_dim == 0:
            return 0

        importance = self.compute_importance_scores()

        active_importance = [(i, importance[i]) for i in range(self.hidden_dim)]
        active_importance.sort(key=lambda x: x[1])

        n_prune = min(
            2,
            self.hidden_dim - self.growth_config.min_hidden_dim,
        )

        if n_prune <= 0:
            return 0

        input_offset = self.input_dim + self.output_dim
        pruned = 0
        for i in range(n_prune):
            idx = active_importance[i][0]
            state_idx = input_offset + idx
            if state_idx < len(self.neuron_states):
                self.neuron_states[state_idx].active = False
                pruned += 1

        return pruned

    def prune_connections(self) -> int:
        """剪枝不重要的连接"""
        if self.hidden_dim == 0:
            return 0

        with torch.no_grad():
            mask_real = torch.sigmoid(
                self.sparse_mask_real[: self.hidden_dim, : self.hidden_dim]
            )
            mask_imag = torch.sigmoid(
                self.sparse_mask_imag[: self.hidden_dim, : self.hidden_dim]
            )

            pruned_real = (
                (mask_real < self.growth_config.connection_threshold).sum().item()
            )
            pruned_imag = (
                (mask_imag < self.growth_config.connection_threshold).sum().item()
            )

            self.sparse_mask_real.data[: self.hidden_dim, : self.hidden_dim][
                mask_real < self.growth_config.connection_threshold
            ] = -10
            self.sparse_mask_imag.data[: self.hidden_dim, : self.hidden_dim][
                mask_imag < self.growth_config.connection_threshold
            ] = -10

        return int(pruned_real + pruned_imag)

    def compute_topology_penalty(self) -> torch.Tensor:
        """计算拓扑惩罚"""
        penalty = torch.tensor(
            0.0, device=self.b_real.device if hasattr(self, "b_real") else "cpu"
        )

        if hasattr(self, "sparse_mask_real") and self.hidden_dim > 0:
            mask_sum = self.sparse_mask_real[: self.hidden_dim, : self.hidden_dim].sum()
            target = self.hidden_dim * self.hidden_dim * 0.5
            penalty = (
                penalty + abs(mask_sum - target) * self.growth_config.topology_penalty
            )

        return penalty

    def add_first_neuron(self) -> int:
        """添加第一个神经元 - 从最小结构开始的关键步骤"""
        if self.hidden_dim != 0:
            return -1

        new_dim = 1

        with torch.no_grad():
            new_W_real = torch.zeros(1, 1)
            new_W_imag = torch.zeros(1, 1)
            new_mask_real = torch.ones(1, 1) * 2.0
            new_mask_imag = torch.ones(1, 1) * 2.0

            self.W_real = nn.Linear(1, 1, bias=False)
            self.W_imag = nn.Linear(1, 1, bias=False)
            self.W_real.weight.data = new_W_real
            self.W_imag.weight.data = new_W_imag

            self.sparse_mask_real = nn.Parameter(new_mask_real)
            self.sparse_mask_imag = nn.Parameter(new_mask_imag)

            new_b_real = torch.zeros(1)
            new_b_imag = torch.zeros(1)
            self.b_real = nn.Parameter(new_b_real)
            self.b_imag = nn.Parameter(new_b_imag)

            if self.tau_bias is not None:
                self.tau_bias = nn.Parameter(torch.zeros(1))

            new_out = nn.Linear(1, self.output_dim)
            self.out = new_out

        input_offset = self.input_dim + self.output_dim

        U_weight = torch.randn(1, self.input_dim) * 0.5
        self.U = nn.Linear(self.input_dim, 1)
        self.U.weight.data = U_weight

        out_weight = torch.randn(self.output_dim, 1) * 0.5
        self.out = nn.Linear(1, self.output_dim)
        self.out.weight.data = out_weight
        self.out.bias.data = torch.zeros(self.output_dim)

        for in_node in range(self.input_dim):
            for out_node in range(self.input_dim, self.input_dim + self.output_dim):
                gene = ConnectionGene(
                    in_node=in_node,
                    out_node=out_node,
                    weight=out_weight[out_node - self.input_dim, 0].item(),
                    enabled=True,
                    innovation=self._get_next_innovation(),
                )
                self.connection_genes.append(gene)

                out_idx = out_node - self.input_dim
                self.out.weight.data[out_idx, 0] = gene.weight

        for in_node in range(self.input_dim):
            gene = ConnectionGene(
                in_node=in_node,
                out_node=input_offset,
                weight=U_weight[0, in_node].item(),
                enabled=True,
                innovation=self._get_next_innovation(),
            )
            self.connection_genes.append(gene)

        self.neuron_states.append(
            NeuronState(
                index=input_offset,
                neuron_type="hidden",
                active=True,
                birth_step=self.training_step,
            )
        )

        self.active_neurons.append(input_offset)
        self.hidden_dim = new_dim

        return 0

    def growth_step(self):
        """执行一步增长/剪枝 - NEAT风格的突变操作"""
        if not self.enable_growth:
            return {"action": "disabled", "changes": 0}

        self.training_step += 1
        self._update_neuron_stats()

        if self.training_step % self.growth_config.growth_interval == 0:
            action_taken = None
            changes = 0

            if self.hidden_dim == 0 and self.training_step > 50:
                new_idx = self.add_first_neuron()
                if new_idx >= 0:
                    return {
                        "action": "init",
                        "count": 1,
                        "new_dim": self.hidden_dim,
                    }

            if self.hidden_dim > 0:
                rand_val = torch.rand(1).item()

                if rand_val < self.growth_config.prob_add_node:
                    overloaded = self.get_overloaded_neurons()
                    if overloaded:
                        parent = overloaded[torch.randint(len(overloaded), (1,)).item()]
                        new_idx = self.split_neuron(parent)
                        if new_idx >= 0:
                            action_taken = "split"
                            changes = 1

                elif self.hidden_dim < self.growth_config.max_hidden_dim:
                    if self.add_random_connection():
                        action_taken = "add_connection"
                        changes = 1
                    elif torch.rand(1).item() < 0.5:
                        self._update_neuron_stats()
                        overloaded = self.get_overloaded_neurons()
                        if overloaded:
                            parent = overloaded[
                                torch.randint(len(overloaded), (1,)).item()
                            ]
                            new_idx = self.split_neuron(parent)
                            if new_idx >= 0:
                                action_taken = "split"
                                changes = 1

            if action_taken:
                if self.mobius is not None:
                    self.mobius.on_dimension_change(self.hidden_dim)
                return {
                    "action": action_taken,
                    "count": changes,
                    "new_dim": self.hidden_dim,
                }

        if self.training_step % self.growth_config.prune_interval == 0:
            n_pruned = self.prune_neurons()
            n_connections = (
                self.prune_connections()
                if self.growth_config.connection_threshold > 0
                else 0
            )

            if n_pruned > 0 or n_connections > 0:
                return {
                    "action": "prune",
                    "neurons": n_pruned,
                    "connections": n_connections,
                }

        return {"action": "none", "changes": 0}

    def get_diagnostics(self) -> Dict:
        importance = self.compute_importance_scores()

        active_importance = (
            importance[importance > 0].tolist() if self.hidden_dim > 0 else []
        )

        return {
            "hidden_dim": self.hidden_dim,
            "active_count": len([s for s in self.neuron_states if s.active]),
            "training_step": self.training_step,
            "connection_count": len([g for g in self.connection_genes if g.enabled]),
            "importance_mean": np.mean(active_importance) if active_importance else 0.0,
            "enable_growth": self.enable_growth,
        }

    def reset_state(self, batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
        return torch.zeros(
            batch_size, max(1, self.hidden_dim), dtype=torch.complex64, device=device
        )

    def step(
        self, z: torch.Tensor, x: torch.Tensor, dt: float = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if dt is None:
            dt = self.dt

        if self.hidden_dim == 0:
            output = self.out.weight.sum() * torch.ones(
                x.shape[0], self.output_dim, device=x.device
            )
            z_empty = torch.zeros(x.shape[0], 1, dtype=torch.complex64, device=x.device)
            return z_empty, output

        dzdt = self.compute_dzdt(z, x)
        z_new = z + dt * dzdt

        if self.mobius is not None:
            z_new = self.mobius.project_state(z_new)

        z_new = torch.complex(
            torch.clamp(z_new.real, -self.z_max, self.z_max),
            torch.clamp(z_new.imag, -self.z_max, self.z_max),
        )

        output = (
            self.out(z_new.real[:, : self.output_dim])
            if self.output_dim <= z_new.shape[1]
            else self.out(z_new.real)
        )
        return z_new, output

    def get_mobius_info(self) -> Optional[Dict]:
        """获取莫比乌斯流形当前状态"""
        if self.mobius is None:
            return None
        return self.mobius.get_manifold_info(self.hidden_dim)


def create_growable_twistor_lnn(
    input_dim: int, hidden_dim: int = 0, output_dim: int = 1, **kwargs
) -> GrowableTwistorLNN:
    return GrowableTwistorLNN(
        input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, **kwargs
    )
