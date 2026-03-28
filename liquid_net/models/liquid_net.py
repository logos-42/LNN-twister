"""
Twistor-inspired Liquid Neural Network - Main network module.
Stability-optimized version.

Core dynamics: dz/dt = (-z + W*tanh(z) + U*x + b) / tau(z)

Stability features:
- Clamped tau (tau_min, tau_max)
- Normalized dz/dt
- Clamped z state
- Tunable dt
- NaN/Inf detection
- L2 regularization support
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict
from .ltc_cell import LTCCell


class TwistorLNN(nn.Module):
    """
    Twistor-inspired Liquid Neural Network with complex-valued states.
    Stability-optimized version.

    The network uses Euler integration to evolve the complex hidden state
    over time, following the dynamics:
        dz/dt = (-z + W*tanh(z) + U*x + b) / tau(z)

    Key features:
        - Complex-valued hidden state z (torch.complex)
        - State-dependent time constant tau(z) = clamp(sigmoid(W_tau * |z|))
        - Bias terms b for both real and imag parts
        - Input U*x affects both real and imag parts
        - Output from real part only
        - Euler integration with tunable dt
        - Stability monitoring and diagnostics
    """

    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 16, 
        output_dim: int = 1, 
        dt: float = 0.1,
        tau_min: float = 0.01,
        tau_max: float = 1.0,
        dzdt_max: float = 10.0,
        z_max: float = 100.0,
    ):
        """
        Initialize Twistor LNN.

        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden state (default: 16)
            output_dim: Dimension of output (default: 1)
            dt: Time step for Euler integration (default: 0.1)
            tau_min: Minimum time constant (default: 0.01)
            tau_max: Maximum time constant (default: 1.0)
            dzdt_max: Maximum |dz/dt| (default: 10.0)
            z_max: Maximum |z| state (default: 100.0)
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dt = dt
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.dzdt_max = dzdt_max
        self.z_max = z_max

        # Core dynamics cell
        self.cell = LTCCell(
            input_dim, 
            hidden_dim, 
            tau_min=tau_min, 
            tau_max=tau_max,
            dzdt_max=dzdt_max,
        )

        # Output projection (real part only)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(
        self, 
        x: torch.Tensor, 
        return_states: bool = False,
        return_diagnostics: bool = False
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
        diagnostics = {
            'z_norm': [],
            'dzdt_norm': [],
            'tau_mean': [],
            'tau_std': [],
            'has_nan': False,
            'has_inf': False,
        }

        # Time loop: Euler integration
        for t in range(T):
            x_t = x[t]

            # Compute time derivative
            dzdt = self.cell(z, x_t)

            # Check for numerical issues
            stability_check = self.cell.check_stability(z, dzdt)
            if stability_check['z_nan'] or stability_check['z_inf']:
                diagnostics['has_nan'] = stability_check['z_nan']
                diagnostics['has_inf'] = stability_check['z_inf']

            # Record diagnostics
            if return_diagnostics:
                diagnostics['z_norm'].append(torch.abs(z).mean().item())
                diagnostics['dzdt_norm'].append(torch.abs(dzdt).mean().item())

            # Euler step: z(t+dt) = z(t) + dt * dz/dt
            z = z + self.dt * dzdt

            # Clamp z to prevent explosion
            z = torch.clamp(z, -self.z_max, self.z_max)

            # Record tau diagnostics
            if return_diagnostics:
                tau_t = self.cell.compute_tau(z)
                diagnostics['tau_mean'].append(tau_t.mean().item())
                diagnostics['tau_std'].append(tau_t.std().item())

            # Output from real part only
            y_t = self.out(z.real)

            outputs.append(y_t)
            if return_states:
                states.append(z)

        # Stack outputs
        y = torch.stack(outputs, dim=0)

        result = [y]
        
        if return_states:
            states = torch.stack(states, dim=0)
            result.append(states)
        
        if return_diagnostics:
            diagnostics['z_norm'] = torch.tensor(diagnostics['z_norm'])
            diagnostics['dzdt_norm'] = torch.tensor(diagnostics['dzdt_norm'])
            diagnostics['tau_mean'] = torch.tensor(diagnostics['tau_mean'])
            diagnostics['tau_std'] = torch.tensor(diagnostics['tau_std'])
            result.append(diagnostics)

        return tuple(result) if len(result) > 1 else result[0]

    def get_tau_statistics(self, z: torch.Tensor) -> Dict[str, float]:
        """Get statistics of time constant tau for a given state."""
        tau = self.cell.compute_tau(z)
        return {
            'tau_mean': tau.mean().item(),
            'tau_std': tau.std().item(),
            'tau_min': tau.min().item(),
            'tau_max': tau.max().item(),
        }
