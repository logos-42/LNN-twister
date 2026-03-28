"""
ODE Solver wrapper for Twistor-inspired Liquid Neural Network.

This module provides a wrapper for torchdiffeq if available,
otherwise falls back to Euler integration.
"""

import torch
from typing import Callable, Optional

try:
    from torchdiffeq import odeint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    print("Warning: torchdiffeq not available. Using Euler integration.")


class ODESolver:
    """
    ODE Solver wrapper that supports both torchdiffeq and Euler methods.
    """
    
    def __init__(self, method: str = 'euler', dt: float = 0.1):
        """
        Initialize ODE Solver.
        
        Args:
            method: Integration method ('euler', 'dopri5', 'rk4', etc.)
            dt: Time step for Euler method (default: 0.1)
        """
        self.method = method
        self.dt = dt
        
        if method != 'euler' and not TORCHDIFFEQ_AVAILABLE:
            print(f"Warning: torchdiffeq not available. Falling back to Euler method.")
            self.method = 'euler'
    
    def solve(
        self,
        func: Callable,
        y0: torch.Tensor,
        t: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Solve ODE using specified method.
        
        Args:
            func: Function computing dy/dt
            y0: Initial state
            t: Time points to evaluate
            **kwargs: Additional arguments for torchdiffeq
        
        Returns:
            Solution at time points
        """
        if self.method == 'euler':
            return self._euler_solve(func, y0, t)
        else:
            if TORCHDIFFEQ_AVAILABLE:
                return odeint(func, y0, t, method=self.method, **kwargs)
            else:
                return self._euler_solve(func, y0, t)
    
    def _euler_solve(
        self,
        func: Callable,
        y0: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Solve ODE using Euler method.
        
        Args:
            func: Function computing dy/dt
            y0: Initial state
            t: Time points to evaluate
        
        Returns:
            Solution at time points
        """
        solutions = [y0]
        y = y0
        
        for i in range(len(t) - 1):
            dt = t[i + 1] - t[i]
            dydt = func(t[i], y)
            y = y + dt * dydt
            solutions.append(y)
        
        return torch.stack(solutions, dim=0)
