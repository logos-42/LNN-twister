"""
Twistor-inspired Liquid Neural Network (Complex-valued LNN) - Stability Optimized
==================================================================================
Implements continuous-time dynamics: dz/dt = (-z + W*tanh(z) + U*x + b) / tau(z)

Stability Features:
- Complex-valued hidden state z (torch.complex)
- State-dependent time constant tau(z) with clamping
- dz/dt normalization to prevent explosion
- Gradient clipping during training
- L2 regularization on z
- Tunable dt parameter
- NaN/Inf detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional


class TwistorLNN(nn.Module):
    """
    Twistor-inspired Liquid Neural Network with complex-valued states.
    Stability-optimized version.

    The dynamics follow: dz/dt = (-z + W*tanh(z) + U*x + b) / tau(z)
    where:
        - z ∈ ℂⁿ is complex hidden state
        - W is recurrent weight matrix (separate for real/imag)
        - U is input weight matrix
        - b is bias term (separate for real/imag)
        - tau(z) = clamp(sigmoid(W_tau * |z|), tau_min, tau_max) is state-dependent time constant
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
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sparsity = sparsity
        self.multi_scale_tau = multi_scale_tau

        # Stability parameters
        self.dt = dt  # Time step (tunable)
        self.tau_min = tau_min  # Minimum time constant
        self.tau_max = tau_max  # Maximum time constant
        self.dzdt_max = dzdt_max  # Maximum |dz/dt|
        self.z_max = z_max  # Maximum |z|

        # Weight matrices - SEPARATE for real and imag parts
        self.W_real = nn.Linear(hidden_dim, hidden_dim)
        self.W_imag = nn.Linear(hidden_dim, hidden_dim)
        self.U = nn.Linear(input_dim, hidden_dim)
        self.W_tau = nn.Linear(hidden_dim, hidden_dim)

        # Sparse connectivity masks
        self.sparse_mask_real = nn.Parameter(torch.ones(hidden_dim, hidden_dim))
        self.sparse_mask_imag = nn.Parameter(torch.ones(hidden_dim, hidden_dim))

        # Multi-scale tau bias
        if multi_scale_tau:
            self.tau_bias = nn.Parameter(torch.zeros(hidden_dim))
        else:
            self.tau_bias = None

        # Bias terms
        self.b_real = nn.Parameter(torch.zeros(hidden_dim))
        self.b_imag = nn.Parameter(torch.zeros(hidden_dim))

        # Output projection
        self.out = nn.Linear(hidden_dim, output_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with orthogonal initialization for stability."""
        nn.init.orthogonal_(self.W_real.weight, gain=0.5)
        nn.init.orthogonal_(self.W_imag.weight, gain=0.5)
        nn.init.orthogonal_(self.U.weight, gain=0.5)
        nn.init.orthogonal_(self.W_tau.weight, gain=0.1)
        nn.init.zeros_(self.W_real.bias)
        nn.init.zeros_(self.W_imag.bias)
        nn.init.zeros_(self.U.bias)
        nn.init.zeros_(self.W_tau.bias)
        nn.init.zeros_(self.b_real)
        nn.init.zeros_(self.b_imag)

        # Initialize sparse masks
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

        # Initialize multi-scale tau bias
        if self.multi_scale_tau and self.tau_bias is not None:
            nn.init.zeros_(self.tau_bias)

    def compute_tau(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute state-dependent time constant with clamping.

        tau_i(z) = clamp(sigmoid(W_tau(|z|)_i + tau_bias_i), tau_min, tau_max)

        Args:
            z: Complex state (B, hidden_dim), dtype=complex

        Returns:
            tau: Clamped time constant (B, hidden_dim), in [tau_min, tau_max]
        """
        z_mod = torch.abs(z)  # (B, hidden_dim)
        tau = F.sigmoid(self.W_tau(z_mod))

        # Add per-neuron bias for multi-scale time constants
        if self.multi_scale_tau and self.tau_bias is not None:
            tau = tau + self.tau_bias.unsqueeze(0)

        # Clamp tau to [tau_min, tau_max] for stability
        tau = torch.clamp(tau, self.tau_min, self.tau_max)

        return tau + 1e-6  # epsilon for numerical stability

    def compute_dzdt(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the time derivative dz/dt with stability normalization.

        Dynamics: dz/dt = normalize((-z + W*tanh(z) + Ux + b) / tau(z))

        Args:
            z: Complex hidden state (B, hidden_dim), dtype=complex
            x: Input (B, input_dim)

        Returns:
            dzdt: Normalized time derivative (B, hidden_dim), dtype=complex
        """
        z_real = z.real
        z_imag = z.imag

        # Apply tanh to real and imag parts separately
        tanh_real = torch.tanh(z_real)
        tanh_imag = torch.tanh(z_imag)

        # Compute numerator with sparse masks
        if self.sparsity > 0:
            W_real_sparse = self.W_real.weight * torch.sigmoid(self.sparse_mask_real)
            W_imag_sparse = self.W_imag.weight * torch.sigmoid(self.sparse_mask_imag)
            W_tanh_real = F.linear(tanh_real, W_real_sparse, self.W_real.bias)
            W_tanh_imag = F.linear(tanh_imag, W_imag_sparse, self.W_imag.bias)
        else:
            W_tanh_real = self.W_real(tanh_real)
            W_tanh_imag = self.W_imag(tanh_imag)

        # Input affects both real and imag parts
        Ux = self.U(x)

        # Compute derivatives
        dz_real = -z_real + W_tanh_real + Ux + self.b_real
        dz_imag = -z_imag + W_tanh_imag + Ux + self.b_imag

        # Compute clamped time constant
        tau = self.compute_tau(z)

        # Divide by tau
        dzdt = torch.complex(dz_real / tau, dz_imag / tau)

        # Normalize dz/dt to prevent explosion
        # Clip real and imag parts separately (clamp doesn't support complex)
        dzdt_real = torch.clamp(dzdt.real, -self.dzdt_max, self.dzdt_max)
        dzdt_imag = torch.clamp(dzdt.imag, -self.dzdt_max, self.dzdt_max)
        dzdt_clipped = torch.complex(dzdt_real, dzdt_imag)

        # Additional normalization: if mean |dz/dt| > threshold, scale down
        dzdt_norm = torch.abs(dzdt_clipped)
        mean_norm = dzdt_norm.mean()
        if mean_norm > self.dzdt_max / 2:
            scale = (self.dzdt_max / 2) / (mean_norm + 1e-6)
            dzdt_clipped = dzdt_clipped * scale

        return dzdt_clipped

    def check_numerical_stability(
        self, z: torch.Tensor, dzdt: torch.Tensor
    ) -> Dict[str, bool]:
        """
        Check for numerical instability (NaN/Inf).

        Args:
            z: Current state
            dzdt: Time derivative

        Returns:
            Dictionary with stability flags
        """
        return {
            "z_nan": torch.isnan(z).any().item(),
            "z_inf": torch.isinf(z).any().item(),
            "dzdt_nan": torch.isnan(dzdt).any().item(),
            "dzdt_inf": torch.isinf(dzdt).any().item(),
        }

    def forward(
        self,
        x: torch.Tensor,
        return_states: bool = False,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass with Euler integration and stability monitoring.

        Args:
            x: Input sequence (T, B, input_dim)
            return_states: If True, return all hidden states
            return_diagnostics: If True, return stability diagnostics

        Returns:
            y: Output sequence (T, B, output_dim)
            states: All hidden states (T, B, hidden_dim) if return_states=True
            diagnostics: Stability info if return_diagnostics=True
        """
        T, B, _ = x.shape

        # Initialize complex hidden state to zero
        z = torch.zeros(B, self.hidden_dim, dtype=torch.complex64, device=x.device)

        outputs = []
        states = []
        dzdts = []
        taus = []
        diagnostics = {
            "z_norm": [],
            "dzdt_norm": [],
            "tau_mean": [],
            "tau_std": [],
            "has_nan": False,
            "has_inf": False,
        }

        # Time loop: Euler integration
        for t in range(T):
            x_t = x[t]  # (B, input_dim)

            # Compute time derivative
            dzdt = self.compute_dzdt(z, x_t)

            # Check for numerical issues
            stability_check = self.check_numerical_stability(z, dzdt)
            if stability_check["z_nan"] or stability_check["z_inf"]:
                diagnostics["has_nan"] = stability_check["z_nan"]
                diagnostics["has_inf"] = stability_check["z_inf"]
                print(f"  Warning: Numerical instability at t={t}: {stability_check}")

            # Record diagnostics
            if return_diagnostics:
                diagnostics["z_norm"].append(torch.abs(z).mean().item())
                diagnostics["dzdt_norm"].append(torch.abs(dzdt).mean().item())

            # Euler step: z(t+dt) = z(t) + dt * dz/dt
            z = z + self.dt * dzdt

            # Clamp z to prevent explosion (separate for real/imag)
            z_real_clamped = torch.clamp(z.real, -self.z_max, self.z_max)
            z_imag_clamped = torch.clamp(z.imag, -self.z_max, self.z_max)
            z = torch.complex(z_real_clamped, z_imag_clamped)

            # Compute tau for diagnostics
            if return_diagnostics:
                tau_t = self.compute_tau(z)
                diagnostics["tau_mean"].append(tau_t.mean().item())
                diagnostics["tau_std"].append(tau_t.std().item())
                dzdts.append(dzdt)
                taus.append(tau_t)

            # Output from real part only
            y_t = self.out(z.real)

            outputs.append(y_t)
            if return_states:
                states.append(z)

        # Stack outputs
        y = torch.stack(outputs, dim=0)  # (T, B, output_dim)

        result = [y]

        if return_states:
            states = torch.stack(states, dim=0)  # (T, B, hidden_dim)
            result.append(states)

        if return_diagnostics:
            diagnostics["z_norm"] = np.array(diagnostics["z_norm"])
            diagnostics["dzdt_norm"] = np.array(diagnostics["dzdt_norm"])
            diagnostics["tau_mean"] = np.array(diagnostics["tau_mean"])
            diagnostics["tau_std"] = np.array(diagnostics["tau_std"])
            if dzdts:
                diagnostics["dzdts"] = torch.stack(dzdts, dim=0)
            if taus:
                diagnostics["taus"] = torch.stack(taus, dim=0)
            result.append(diagnostics)

        return tuple(result) if len(result) > 1 else result[0]

    def get_tau_statistics(self, z: torch.Tensor) -> Dict[str, float]:
        """
        Get statistics of time constant tau for a given state.

        Args:
            z: Complex state

        Returns:
            Dictionary with tau statistics
        """
        tau = self.compute_tau(z)
        return {
            "tau_mean": tau.mean().item(),
            "tau_std": tau.std().item(),
            "tau_min": tau.min().item(),
            "tau_max": tau.max().item(),
        }

    # ============================================================
    # Tensor Decoder: z → v ⊗ v (外积生成二阶张量)
    # ============================================================
    def decode_tensor(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode complex state to second-order tensor via outer product.

        z → v (real part) → v ⊗ v

        Args:
            z: Complex state (B, hidden_dim)

        Returns:
            tensor: Second-order tensor (B, hidden_dim, hidden_dim)
        """
        v = z.real  # (B, hidden_dim)
        tensor = torch.einsum("bi,bj->bij", v, v)  # (B, hidden_dim, hidden_dim)
        return tensor

    def decode_tensor_flat(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode to flattened tensor for output projection.

        Args:
            z: Complex state (B, hidden_dim)

        Returns:
            tensor_flat: Flattened tensor (B, hidden_dim * hidden_dim)
        """
        tensor = self.decode_tensor(z)
        return tensor.view(tensor.size(0), -1)

    # ============================================================
    # RK4 Integrator: 更精确的数值积分
    # ============================================================
    def rk4_step(
        self, z: torch.Tensor, x: torch.Tensor, dt: float = None
    ) -> torch.Tensor:
        """
        Runge-Kutta 4th order integration.

        More accurate than Euler, better for complex dynamics.

        Args:
            z: Current complex state (B, hidden_dim)
            x: Input (B, input_dim)
            dt: Time step (defaults to self.dt)

        Returns:
            z_new: Next state (B, hidden_dim)
        """
        if dt is None:
            dt = self.dt

        k1 = self.compute_dzdt(z, x)
        k2 = self.compute_dzdt(z + 0.5 * dt * k1, x)
        k3 = self.compute_dzdt(z + 0.5 * dt * k2, x)
        k4 = self.compute_dzdt(z + dt * k3, x)

        z_new = z + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return z_new

    def forward_rk4(
        self, x: torch.Tensor, return_states: bool = False, dt: float = None
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass with RK4 integration instead of Euler.

        Args:
            x: Input sequence (T, B, input_dim)
            return_states: If True, return all hidden states
            dt: Time step (defaults to self.dt)

        Returns:
            y: Output sequence (T, B, output_dim)
            states: All hidden states if return_states=True
        """
        if dt is None:
            dt = self.dt

        T, B, _ = x.shape
        z = torch.zeros(B, self.hidden_dim, dtype=torch.complex64, device=x.device)

        outputs = []
        states = []

        for t in range(T):
            x_t = x[t]
            z = self.rk4_step(z, x_t, dt)
            z = torch.clamp(z, -self.z_max, self.z_max)

            y_t = self.out(z.real)
            outputs.append(y_t)
            if return_states:
                states.append(z)

        y = torch.stack(outputs, dim=0)

        if return_states:
            states = torch.stack(states, dim=0)
            return y, states

        return y

    # ============================================================
    # Agent Interface: 单步演化用于强化学习
    # ============================================================
    def step(
        self, z: torch.Tensor, x: torch.Tensor, dt: float = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single step evolution for agent/RL use.

        Args:
            z: Current complex state (B, hidden_dim), complex
            x: Input/observation (B, input_dim), real
            dt: Time step (defaults to self.dt)

        Returns:
            z_new: Next state (B, hidden_dim), complex
            output: Action/prediction (B, output_dim), real
        """
        if dt is None:
            dt = self.dt

        dzdt = self.compute_dzdt(z, x)
        z_new = z + dt * dzdt
        z_new = torch.clamp(z_new, -self.z_max, self.z_max)

        output = self.out(z_new.real)
        return z_new, output

    def step_rk4(
        self, z: torch.Tensor, x: torch.Tensor, dt: float = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single step with RK4 integration for agent.

        Args:
            z: Current complex state (B, hidden_dim), complex
            x: Input/observation (B, input_dim), real
            dt: Time step (defaults to self.dt)

        Returns:
            z_new: Next state (B, hidden_dim), complex
            output: Action/prediction (B, output_dim), real
        """
        if dt is None:
            dt = self.dt

        z_new = self.rk4_step(z, x, dt)
        z_new = torch.clamp(z_new, -self.z_max, self.z_max)

        output = self.out(z_new.real)
        return z_new, output

    def reset_state(self, batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
        """
        Reset hidden state to zero.

        Args:
            batch_size: Number of parallel environments
            device: Device to create state on

        Returns:
            z: Zero state (batch_size, hidden_dim), complex
        """
        return torch.zeros(
            batch_size, self.hidden_dim, dtype=torch.complex64, device=device
        )


# ============================================================
# Coupled Twistor-LNN: 多空间耦合 (h + z)
# ============================================================
class CoupledTwistorLNN(nn.Module):
    """
    Multi-space coupled Twistor-LNN.

    Two state spaces:
    - h: Behavior space (real, standard LNN)
    - z: Structure space (complex, twistor-inspired)

    Coupled dynamics:
    - dh/dt = f(h, z, x)
    - dz/dt = g(z, h)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        output_dim: int = 1,
        coupling_strength: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.coupling_strength = coupling_strength
        self.dt = 0.1

        # Behavior space (real LNN)
        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        self.U_h = nn.Linear(input_dim, hidden_dim)
        self.W_tau_h = nn.Linear(hidden_dim, hidden_dim)
        self.b_h = nn.Parameter(torch.zeros(hidden_dim))

        # Structure space (complex Twistor)
        self.W_z_real = nn.Linear(hidden_dim, hidden_dim)
        self.W_z_imag = nn.Linear(hidden_dim, hidden_dim)
        self.U_z = nn.Linear(input_dim, hidden_dim)
        self.W_tau_z = nn.Linear(hidden_dim, hidden_dim)
        self.b_z_real = nn.Parameter(torch.zeros(hidden_dim))
        self.b_z_imag = nn.Parameter(torch.zeros(hidden_dim))

        # Coupling: h → z and z → h
        self.h_to_z_coupling = nn.Linear(hidden_dim, hidden_dim)
        self.z_to_h_coupling = nn.Linear(hidden_dim, hidden_dim)

        # Output decoder (uses both h and z)
        self.out_h = nn.Linear(hidden_dim, output_dim)
        self.out_z = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.W_h.weight, gain=0.5)
        nn.init.orthogonal_(self.W_z_real.weight, gain=0.5)
        nn.init.orthogonal_(self.W_z_imag.weight, gain=0.5)
        nn.init.orthogonal_(self.U_h.weight, gain=0.5)
        nn.init.orthogonal_(self.U_z.weight, gain=0.5)
        nn.init.orthogonal_(self.W_tau_h.weight, gain=0.1)
        nn.init.orthogonal_(self.W_tau_z.weight, gain=0.1)
        nn.init.zeros_(self.W_h.bias)
        nn.init.zeros_(self.W_z_real.bias)
        nn.init.zeros_(self.W_z_imag.bias)
        nn.init.zeros_(self.U_h.bias)
        nn.init.zeros_(self.U_z.bias)
        nn.init.zeros_(self.W_tau_h.bias)
        nn.init.zeros_(self.W_tau_z.bias)

    def compute_dhdt(
        self, h: torch.Tensor, z: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """Compute behavior space derivative."""
        tau_h = torch.sigmoid(self.W_tau_h(h)) + 1e-6
        coupling_from_z = self.z_to_h_coupling(z.real) * self.coupling_strength
        dh = (
            -h + torch.tanh(self.W_h(h)) + self.U_h(x) + self.b_h + coupling_from_z
        ) / tau_h
        return dh

    def compute_dzdt(
        self, z: torch.Tensor, h: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """Compute structure space derivative."""
        z_real = z.real
        z_imag = z.imag

        tau_z = torch.sigmoid(self.W_tau_z(torch.abs(z))) + 1e-6

        coupling_from_h = self.h_to_z_coupling(h) * self.coupling_strength

        dz_real = (
            -z_real
            + torch.tanh(self.W_z_real(z_real))
            + self.U_z(x)
            + self.b_z_real
            + coupling_from_h
        )
        dz_imag = (
            -z_imag
            + torch.tanh(self.W_z_imag(z_imag))
            + self.U_z(x)
            + self.b_z_imag
            + coupling_from_h
        )

        dzdt = torch.complex(dz_real / tau_z, dz_imag / tau_z)
        return dzdt

    def forward(
        self, x: torch.Tensor, return_states: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass with coupled dynamics.

        Args:
            x: Input sequence (T, B, input_dim)
            return_states: If True, return both h and z states

        Returns:
            y: Output sequence (T, B, output_dim)
            h_states: Behavior states if return_states=True
            z_states: Structure states if return_states=True
        """
        T, B, _ = x.shape

        h = torch.zeros(B, self.hidden_dim, device=x.device)
        z = torch.zeros(B, self.hidden_dim, dtype=torch.complex64, device=x.device)

        outputs = []
        h_states = []
        z_states = []

        for t in range(T):
            x_t = x[t]

            dhdt = self.compute_dhdt(h, z, x_t)
            dzdt = self.compute_dzdt(z, h, x_t)

            h = h + self.dt * dhdt
            z = z + self.dt * dzdt

            y_t = self.out_h(h) + self.out_z(z.real)
            outputs.append(y_t)

            if return_states:
                h_states.append(h)
                z_states.append(z)

        y = torch.stack(outputs, dim=0)

        if return_states:
            h_states = torch.stack(h_states, dim=0)
            z_states = torch.stack(z_states, dim=0)
            return y, h_states, z_states

        return y

    def step(
        self, h: torch.Tensor, z: torch.Tensor, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single step for agent.

        Args:
            h: Current behavior state (B, hidden_dim)
            z: Current structure state (B, hidden_dim), complex
            x: Input/observation (B, input_dim)

        Returns:
            h_new: Next behavior state
            z_new: Next structure state
            output: Action/prediction
        """
        dhdt = self.compute_dhdt(h, z, x)
        dzdt = self.compute_dzdt(z, h, x)

        h_new = h + self.dt * dhdt
        z_new = z + self.dt * dzdt

        output = self.out_h(h_new) + self.out_z(z_new.real)

        return h_new, z_new, output

    def reset_state(
        self, batch_size: int = 1, device: str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reset both states.

        Args:
            batch_size: Number of parallel environments
            device: Device to create states on

        Returns:
            h: Zero behavior state
            z: Zero structure state
        """
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        z = torch.zeros(
            batch_size, self.hidden_dim, dtype=torch.complex64, device=device
        )
        return h, z


# ============================================================
# Twistor Agent: 智能体封装
# ============================================================
class TwistorAgent:
    """
    Agent wrapper for Twistor-LNN.

    Usage:
        agent = TwistorAgent(obs_dim=4, action_dim=2, hidden_dim=32)
        obs = env.reset()
        agent.reset()

        for step in range(max_steps):
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            if done:
                agent.reset()
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 32,
        use_rk4: bool = False,
        dt: float = 0.1,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.use_rk4 = use_rk4
        self.dt = dt

        self.model = TwistorLNN(
            input_dim=obs_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )
        self.z = None

    def reset(self, batch_size: int = 1, device: str = "cpu"):
        """Reset agent state."""
        self.z = self.model.reset_state(batch_size, device)
        return self.z

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Get action from observation.

        Args:
            obs: Observation (obs_dim,) or (batch, obs_dim)
            deterministic: If True, return argmax (for discrete)

        Returns:
            action: Action (action_dim,) or (batch, action_dim)
        """
        if self.z is None:
            self.reset()

        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            batch_mode = False
        else:
            batch_mode = True

        with torch.no_grad():
            if self.use_rk4:
                self.z, action = self.model.step_rk4(self.z, obs, self.dt)
            else:
                self.z, action = self.model.step(self.z, obs, self.dt)

        if not batch_mode:
            action = action.squeeze(0)

        return action.cpu().numpy()

    def update(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Update state without getting action (for inference).

        Args:
            obs: Observation

        Returns:
            output: Model output
        """
        if self.z is None:
            self.reset(obs.size(0), str(obs.device))

        with torch.no_grad():
            self.z, output = self.model.step(self.z, obs, self.dt)

        return output


def generate_sine_dataset(
    n_samples: int = 1000,
    seq_len: int = 50,
    input_dim: int = 2,
    noise_std: float = 0.1,
    device: str = "cpu",
):
    """
    Generate synthetic sine wave prediction dataset.
    """
    X = []
    y = []

    for _ in range(n_samples):
        freq = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2 * np.pi)
        t = np.linspace(0, 4 * np.pi, seq_len + 1)
        signal = np.sin(freq * t + phase)
        signal += np.random.randn(len(t)) * noise_std

        sin_component = signal[:-1]
        cos_component = (
            np.cos(freq * t[:-1] + phase) + np.random.randn(seq_len) * noise_std
        )

        x_seq = np.stack([sin_component, cos_component], axis=-1)
        y_seq = signal[1:].reshape(-1, 1)

        X.append(x_seq)
        y.append(y_seq)

    X = torch.FloatTensor(np.stack(X)).to(device)
    y = torch.FloatTensor(np.stack(y)).to(device)

    return X, y


def plot_training_results(history: Dict[str, list]):
    """Plot training curves."""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history["train_mse"], label="Train MSE")
    plt.plot(history["val_mse"], label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Mean Squared Error")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("Training curves saved to 'training_curves.png'")
    plt.close()


def plot_predictions(
    model: TwistorLNN,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    device: str,
    n_samples: int = 5,
):
    """Plot sample predictions."""
    model.eval()

    with torch.no_grad():
        x_test = X_test[:n_samples].transpose(0, 1)
        y_pred = model(x_test).transpose(0, 1)
        y_true = y_test[:n_samples]

    plt.figure(figsize=(14, 8))

    for i in range(n_samples):
        plt.subplot(n_samples, 1, i + 1)
        plt.plot(
            y_true[i].cpu().numpy().flatten(),
            "o-",
            label="True",
            alpha=0.7,
            markersize=4,
        )
        plt.plot(
            y_pred[i].cpu().numpy().flatten(),
            "s-",
            label="Predicted",
            alpha=0.7,
            markersize=4,
        )
        plt.ylabel("Amplitude")
        plt.title(f"Sample {i + 1}")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)

    plt.xlabel("Time Step")
    plt.tight_layout()
    plt.savefig("predictions.png", dpi=150)
    print("Sample predictions saved to 'predictions.png'")
    plt.close()


def plot_z_trajectory(diagnostics: Dict, save_path: str = "z_trajectory.png"):
    """
    Plot z trajectory and tau distribution from diagnostics.

    Args:
        diagnostics: Dictionary from forward pass with return_diagnostics=True
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: |z| over time
    ax1 = axes[0, 0]
    ax1.plot(diagnostics["z_norm"], "b-", linewidth=2)
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("|z| (mean)")
    ax1.set_title("State Norm Over Time")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=diagnostics["z_norm"].mean(), color="r", linestyle="--", label="Mean")
    ax1.legend()

    # Plot 2: |dz/dt| over time
    ax2 = axes[0, 1]
    ax2.plot(diagnostics["dzdt_norm"], "g-", linewidth=2)
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("|dz/dt| (mean)")
    ax2.set_title("Time Derivative Norm Over Time")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(
        y=diagnostics["dzdt_norm"].mean(), color="r", linestyle="--", label="Mean"
    )
    ax2.legend()

    # Plot 3: tau mean over time
    ax3 = axes[1, 0]
    ax3.plot(diagnostics["tau_mean"], "m-", linewidth=2)
    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("τ (mean)")
    ax3.set_title("Time Constant Mean Over Time")
    ax3.grid(True, alpha=0.3)
    ax3.axhline(
        y=diagnostics["tau_mean"].mean(), color="r", linestyle="--", label="Mean"
    )
    ax3.legend()

    # Plot 4: tau distribution (histogram)
    ax4 = axes[1, 1]
    if "taus" in diagnostics:
        all_taus = diagnostics["taus"].flatten().cpu().numpy()
        ax4.hist(all_taus, bins=50, edgecolor="black", alpha=0.7)
        ax4.set_xlabel("τ value")
        ax4.set_ylabel("Frequency")
        ax4.set_title("Time Constant Distribution")
        ax4.grid(True, alpha=0.3)
        ax4.axvline(
            x=np.mean(all_taus),
            color="r",
            linestyle="--",
            label=f"Mean: {np.mean(all_taus):.4f}",
        )
        ax4.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Z trajectory saved to '{save_path}'")
    plt.close()


def print_tau_diagnostics(model: TwistorLNN, device: str = "cpu"):
    """
    Print tau distribution statistics for debugging.

    Args:
        model: Trained TwistorLNN model
        device: Device to run on
    """
    model.eval()

    # Generate random state
    z = torch.randn(1, model.hidden_dim, dtype=torch.complex64, device=device)

    stats = model.get_tau_statistics(z)

    print("\n" + "=" * 50)
    print("Tau Distribution Statistics:")
    print("=" * 50)
    print(f"  τ mean:  {stats['tau_mean']:.6f}")
    print(f"  τ std:   {stats['tau_std']:.6f}")
    print(f"  τ min:   {stats['tau_min']:.6f}")
    print(f"  τ max:   {stats['tau_max']:.6f}")
    print(f"  τ range: [{model.tau_min:.4f}, {model.tau_max:.4f}] (configured)")
    print("=" * 50)


def train_twistor_lnn(
    n_epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-2,
    hidden_dim: int = 16,
    stability_weight: float = 0.01,
    l2_weight: float = 0.001,
    sparsity: float = 0.3,
    multi_scale_tau: bool = True,
    dt: float = 0.1,
    tau_min: float = 0.01,
    tau_max: float = 1.0,
    dzdt_max: float = 10.0,
    z_max: float = 100.0,
    grad_clip: float = 1.0,
    device: str = "cpu",
    plot_diagnostics: bool = True,
):
    """
    Train the Twistor LNN on sine wave prediction with stability optimizations.

    Args:
        n_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        hidden_dim: Hidden dimension
        stability_weight: Weight for ||dz/dt||^2 regularization
        l2_weight: Weight for L2 regularization on z
        sparsity: Sparsity level for recurrent weights
        multi_scale_tau: Use per-neuron tau bias
        dt: Time step for Euler integration
        tau_min: Minimum time constant
        tau_max: Maximum time constant
        dzdt_max: Maximum |dz/dt|
        z_max: Maximum |z|
        grad_clip: Gradient clipping threshold
        device: Device to train on
        plot_diagnostics: If True, plot z trajectory and tau distribution

    Returns:
        model: Trained model
        history: Training history
    """
    print("=" * 60)
    print("Twistor-inspired Liquid Neural Network Training")
    print("(Stability Optimized Version)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Time step dt: {dt}")
    print(f"Tau range: [{tau_min}, {tau_max}]")
    print(f"Max |dz/dt|: {dzdt_max}")
    print(f"Max |z|: {z_max}")
    print(f"Gradient clip: {grad_clip}")
    print(f"Stability weight: {stability_weight}")
    print(f"L2 weight: {l2_weight}")
    print()

    # Generate dataset
    print("Generating synthetic sine wave dataset...")
    X_train, y_train = generate_sine_dataset(n_samples=1000, seq_len=50, device=device)
    X_val, y_val = generate_sine_dataset(n_samples=200, seq_len=50, device=device)
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    print(f"Sequence length: {X_train.shape[1]}, Input dim: {X_train.shape[2]}")
    print()

    # Initialize model with stability parameters
    model = TwistorLNN(
        input_dim=X_train.shape[2],
        hidden_dim=hidden_dim,
        output_dim=1,
        sparsity=sparsity,
        multi_scale_tau=multi_scale_tau,
        dt=dt,
        tau_min=tau_min,
        tau_max=tau_max,
        dzdt_max=dzdt_max,
        z_max=z_max,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Print initial tau statistics
    print_tau_diagnostics(model, device)
    print()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5
    )

    # Training loop
    n_batches = len(X_train) // batch_size
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_mse": [],
        "val_mse": [],
        "stability_loss": [],
        "l2_loss": [],
    }

    print("Starting training...")
    print("-" * 60)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_stability = 0.0
        epoch_l2 = 0.0

        # Shuffle data
        perm = torch.randperm(len(X_train), device=device)
        X_train = X_train[perm]
        y_train = y_train[perm]

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            x_batch = X_train[start_idx:end_idx].transpose(0, 1)
            y_batch = y_train[start_idx:end_idx].transpose(0, 1)

            optimizer.zero_grad()

            # Forward pass with states and diagnostics
            y_pred, states, diagnostics = model(
                x_batch, return_states=True, return_diagnostics=True
            )

            # Check for numerical issues
            if diagnostics["has_nan"] or diagnostics["has_inf"]:
                print(f"  Warning: Numerical instability detected at epoch {epoch + 1}")

            # MSE loss
            mse_loss = F.mse_loss(y_pred, y_batch)

            # Stability regularization: ||dz/dt||^2
            dzdt_norm_sq = 0.0
            for t in range(len(states) - 1):
                dzdt = states[t + 1] - states[t]
                dzdt_norm_sq += (dzdt.abs() ** 2).mean()
            stability_loss = dzdt_norm_sq / (len(states) - 1)

            # L2 regularization on z
            l2_loss = (states.abs() ** 2).mean()

            # Total loss
            loss = mse_loss + stability_weight * stability_loss + l2_weight * l2_loss

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()

            epoch_loss += loss.item()
            epoch_mse += mse_loss.item()
            epoch_stability += stability_loss.item()
            epoch_l2 += l2_loss.item()

        # Average losses
        avg_train_loss = epoch_loss / n_batches
        avg_train_mse = epoch_mse / n_batches
        avg_stability = epoch_stability / n_batches
        avg_l2 = epoch_l2 / n_batches

        history["train_loss"].append(avg_train_loss)
        history["train_mse"].append(avg_train_mse)
        history["stability_loss"].append(avg_stability)
        history["l2_loss"].append(avg_l2)

        # Validation
        model.eval()
        with torch.no_grad():
            x_val = X_val.transpose(0, 1)
            y_val_t = y_val.transpose(0, 1)
            y_val_pred = model(x_val)
            val_mse = F.mse_loss(y_val_pred, y_val_t).item()
            history["val_loss"].append(val_mse)
            history["val_mse"].append(val_mse)

        # Update learning rate
        scheduler.step(avg_train_loss)

        # Print progress
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:4d}/{n_epochs}: "
                f"Train Loss = {avg_train_loss:.6f}, "
                f"Train MSE = {avg_train_mse:.6f}, "
                f"Val MSE = {val_mse:.6f}, "
                f"Stab = {avg_stability:.6f}, "
                f"L2 = {avg_l2:.6f}, "
                f"LR = {optimizer.param_groups[0]['lr']:.6f}"
            )

    print("-" * 60)
    print(f"Training complete! Final Val MSE: {history['val_mse'][-1]:.6f}")
    print()

    # Print final tau statistics
    print_tau_diagnostics(model, device)

    # Plot results
    plot_training_results(history)
    plot_predictions(model, X_val, y_val, device)

    # Plot z trajectory and tau distribution
    if plot_diagnostics:
        print("\nGenerating diagnostics plots...")
        model.eval()
        with torch.no_grad():
            x_test = X_val[:1].transpose(0, 1)
            _, _, diagnostics = model(
                x_test, return_states=True, return_diagnostics=True
            )
        plot_z_trajectory(diagnostics)

    return model, history


if __name__ == "__main__":
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Train model with stability optimizations (reduced epochs for testing)
    model, history = train_twistor_lnn(
        n_epochs=50,  # Reduced for faster testing
        batch_size=32,
        lr=1e-2,
        hidden_dim=16,
        stability_weight=0.01,
        l2_weight=0.001,
        sparsity=0.3,
        multi_scale_tau=True,
        dt=0.1,
        tau_min=0.01,
        tau_max=1.0,
        dzdt_max=10.0,
        z_max=100.0,
        grad_clip=1.0,
        device=device,
        plot_diagnostics=True,
    )

    # Save model
    torch.save(model.state_dict(), "twistor_lnn_stable.pth")
    print("Model saved to 'twistor_lnn_stable.pth'")

    print()
    print("=" * 60)
    print("Training Summary:")
    print(f"  Initial Train Loss: {history['train_loss'][0]:.6f}")
    print(f"  Final Train Loss: {history['train_loss'][-1]:.6f}")
    print(f"  Initial Val MSE: {history['val_mse'][0]:.6f}")
    print(f"  Final Val MSE: {history['val_mse'][-1]:.6f}")
    print(
        f"  Convergence: {'Yes' if history['train_loss'][-1] < history['train_loss'][0] * 0.5 else 'Partial'}"
    )
    print("=" * 60)
