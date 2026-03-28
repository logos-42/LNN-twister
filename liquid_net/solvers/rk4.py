"""
RK4 Integrator for Twistor-inspired Liquid Neural Network.

Runge-Kutta 4th order integrator for complex-valued ODEs.

RK4 formula:
    k1 = f(t, z)
    k2 = f(t + dt/2, z + dt*k1/2)
    k3 = f(t + dt/2, z + dt*k2/2)
    k4 = f(t + dt, z + dt*k3)
    z(t+dt) = z(t) + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
"""

import torch
from typing import Callable, List


class RK4Integrator:
    """
    Runge-Kutta 4th order integrator for complex-valued ODEs.
    """
    
    def __init__(self, dt: float = 0.1):
        """
        Initialize RK4 integrator.
        
        Args:
            dt: Time step size
        """
        self.dt = dt
    
    def step(self, dzdt_func: Callable, z: torch.Tensor, 
             x: torch.Tensor, t: int = 0) -> torch.Tensor:
        """
        Perform a single RK4 integration step.
        
        Args:
            dzdt_func: Function computing dz/dt given (z, x)
            z: Current state (B, hidden_dim), dtype=complex
            x: Input (B, input_dim)
            t: Time step index
            
        Returns:
            z_new: Updated state after one RK4 step
        """
        dt = self.dt
        
        # k1 = f(t, z)
        k1 = dzdt_func(z, x)
        
        # k2 = f(t + dt/2, z + dt*k1/2)
        z_mid = z + dt * k1 / 2
        k2 = dzdt_func(z_mid, x)
        
        # k3 = f(t + dt/2, z + dt*k2/2)
        z_mid = z + dt * k2 / 2
        k3 = dzdt_func(z_mid, x)
        
        # k4 = f(t + dt, z + dt*k3)
        z_end = z + dt * k3
        k4 = dzdt_func(z_end, x)
        
        # z(t+dt) = z(t) + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        z_new = z + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        
        return z_new
    
    def integrate(self, dzdt_func: Callable, z0: torch.Tensor, 
                  x_seq: torch.Tensor) -> List[torch.Tensor]:
        """
        Integrate over a sequence of inputs.
        
        Args:
            dzdt_func: Function computing dz/dt given (z, x)
            z0: Initial state (B, hidden_dim)
            x_seq: Input sequence (T, B, input_dim)
            
        Returns:
            List of states at each time step
        """
        T, B, _ = x_seq.shape
        z = z0
        states = [z]
        
        for t in range(T):
            z = self.step(dzdt_func, z, x_seq[t], t)
            states.append(z)
        
        return states
