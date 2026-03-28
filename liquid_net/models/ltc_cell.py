"""
LTC Cell - Core dynamics for Twistor-inspired Liquid Neural Network.

Implements the continuous-time dynamics:
    dz/dt = (-z + W*tanh(z) + Ux) / tau(z)

where z is complex-valued and tau(z) is state-dependent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LTCCell(nn.Module):
    """
    Liquid Time-Constant Cell with complex-valued states.

    This is the core dynamical system component that computes
    the time derivative dz/dt for the Twistor-inspired LNN.

    Mathematical formulation:
        dz/dt = (-z + W*tanh(z) + Ux) / tau(z)

    where:
        - z is complex hidden state
        - W is recurrent weight matrix
        - U is input weight matrix
        - tau(z) = sigmoid(W_tau(z.real)) + epsilon is state-dependent time constant
    """

    def __init__(self, input_dim: int, hidden_dim: int = 16):
        """
        Initialize LTC Cell.

        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden state (default: 16)
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Weight matrices - SEPARATE for real and imag parts (as required)
        # W_real: recurrent weight for real part (hidden -> hidden)
        self.W_real = nn.Linear(hidden_dim, hidden_dim)
        # W_imag: recurrent weight for imag part (hidden -> hidden)
        self.W_imag = nn.Linear(hidden_dim, hidden_dim)
        # U: input weight (input -> hidden) - shared
        self.U = nn.Linear(input_dim, hidden_dim)
        # W_tau: for computing state-dependent time constant
        self.W_tau = nn.Linear(hidden_dim, hidden_dim)

        # Initialize weights with small values for stability
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

    def compute_tau(self, z_real: torch.Tensor) -> torch.Tensor:
        """
        Compute state-dependent time constant.

        tau(z) = sigmoid(W_tau(z.real)) + epsilon

        Args:
            z_real: Real part of complex state (B, hidden_dim)

        Returns:
            tau: Time constant (B, hidden_dim), always positive
        """
        return F.sigmoid(self.W_tau(z_real)) + 1e-6

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the time derivative dz/dt.

        Dynamics: dz/dt = (-z + W*tanh(z) + Ux) / tau(z)

        Real part: dz_real/dt = (-z_real + W_real*tanh(z_real) + U*x) / tau
        Imag part: dz_imag/dt = (-z_imag + W_imag*tanh(z_imag)) / tau

        Args:
            z: Complex hidden state (B, hidden_dim), dtype=complex
            x: Input (B, input_dim)

        Returns:
            dzdt: Time derivative (B, hidden_dim), dtype=complex
        """
        # Extract real and imaginary parts
        z_real = z.real
        z_imag = z.imag

        # Apply tanh to real and imag parts separately
        tanh_real = torch.tanh(z_real)
        tanh_imag = torch.tanh(z_imag)

        # Compute numerator: -z + W*tanh(z) + Ux
        # Use SEPARATE weight matrices for real and imag (as required)
        W_tanh_real = self.W_real(tanh_real)  # (B, hidden_dim)
        W_tanh_imag = self.W_imag(tanh_imag)  # (B, hidden_dim)
        Ux = self.U(x)  # (B, hidden_dim)

        # Compute real and imag derivatives separately
        dz_real = -z_real + W_tanh_real + Ux
        dz_imag = -z_imag + W_tanh_imag

        # Compute state-dependent time constant (from real part only)
        tau = self.compute_tau(z_real)

        # Divide by tau and recombine
        dzdt = torch.complex(dz_real / tau, dz_imag / tau)

        return dzdt
