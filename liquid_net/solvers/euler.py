"""
Euler solver for Twistor-inspired Liquid Neural Network.

Implements simple Euler integration for continuous-time dynamics.
"""

import torch


def euler_step(z: torch.Tensor, dzdt: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
    """
    Perform a single Euler integration step.
    
    Euler method: z(t+dt) = z(t) + dt * dz/dt
    
    Args:
        z: Current state (can be complex)
        dzdt: Time derivative at current state
        dt: Time step size (default: 0.1)
    
    Returns:
        z_new: Updated state after one Euler step
    """
    return z + dt * dzdt
