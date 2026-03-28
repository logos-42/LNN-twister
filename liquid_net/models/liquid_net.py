"""
Twistor-inspired Liquid Neural Network - Main network module.

Implements the full network with Euler integration over time.
"""

import torch
import torch.nn as nn
from .ltc_cell import LTCCell


class TwistorLNN(nn.Module):
    """
    Twistor-inspired Liquid Neural Network with complex-valued states.
    
    The network uses Euler integration to evolve the complex hidden state
    over time, following the dynamics:
        dz/dt = (-z + W*tanh(z) + Ux) / tau(z)
    
    Key features:
        - Complex-valued hidden state z (torch.complex)
        - State-dependent time constant tau(z)
        - Output from real part only
        - Euler integration for time evolution
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 16, output_dim: int = 1, dt: float = 0.1):
        """
        Initialize Twistor LNN.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden state (default: 16)
            output_dim: Dimension of output (default: 1)
            dt: Time step for Euler integration (default: 0.1)
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dt = dt
        
        # Core dynamics cell
        self.cell = LTCCell(input_dim, hidden_dim)
        
        # Output projection (real part only)
        self.out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor, return_states: bool = False) -> torch.Tensor:
        """
        Forward pass with Euler integration over time.
        
        Args:
            x: Input sequence (T, B, input_dim)
            return_states: If True, return all hidden states
        
        Returns:
            y: Output sequence (T, B, output_dim)
            states: All hidden states (T, B, hidden_dim) if return_states=True
        """
        T, B, _ = x.shape
        
        # Initialize complex hidden state to zero
        z = torch.zeros(B, self.hidden_dim, dtype=torch.complex64, device=x.device)
        
        outputs = []
        states = []
        
        # Time loop: Euler integration
        for t in range(T):
            # Get input at time t
            x_t = x[t]  # (B, input_dim)
            
            # Compute time derivative
            dzdt = self.cell(z, x_t)
            
            # Euler step: z(t+dt) = z(t) + dt * dz/dt
            z = z + self.dt * dzdt
            
            # Output from real part only
            y_t = self.out(z.real)  # (B, output_dim)
            
            outputs.append(y_t)
            if return_states:
                states.append(z)
        
        # Stack outputs: (T, B, output_dim)
        y = torch.stack(outputs, dim=0)
        
        if return_states:
            states = torch.stack(states, dim=0)  # (T, B, hidden_dim)
            return y, states
        
        return y
