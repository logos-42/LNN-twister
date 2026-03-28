"""
ODE Solver wrapper for Twistor-inspired Liquid Neural Network.

Supports multiple integration methods:
- Euler (basic, fast)
- RK4 (4th order Runge-Kutta)
- torchdiffeq solvers (dopri5, rk45, etc.)
"""

import torch
from typing import Callable, Optional

try:
    from torchdiffeq import odeint, odeint_adjoint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    print("Warning: torchdiffeq not available. Install with: pip install torchdiffeq")


class ODESolver:
    """
    ODE Solver wrapper supporting multiple integration methods.
    """

    def __init__(self, method: str = 'euler', dt: float = 0.1,
                 rtol: float = 1e-5, atol: float = 1e-5):
        """
        Initialize ODE Solver.

        Args:
            method: Integration method ('euler', 'rk4', 'dopri5', 'rk45', 'dopri8')
            dt: Time step for discrete methods
            rtol: Relative tolerance (for adaptive solvers)
            atol: Absolute tolerance (for adaptive solvers)
        """
        self.method = method
        self.dt = dt
        self.rtol = rtol
        self.atol = atol

        if method not in ['euler', 'rk4'] and not TORCHDIFFEQ_AVAILABLE:
            print(f"Warning: torchdiffeq not available. Falling back to Euler method.")
            self.method = 'euler'

    def solve(self, func: Callable, y0: torch.Tensor, 
              t: torch.Tensor, x_interp: Optional[torch.Tensor] = None,
              **kwargs) -> torch.Tensor:
        """
        Solve ODE using specified method.

        Args:
            func: Function computing dy/dt = f(t, y, x)
            y0: Initial state
            t: Time points to evaluate
            x_interp: Interpolated inputs at time points (optional)
            **kwargs: Additional arguments for torchdiffeq

        Returns:
            Solution at time points
        """
        if self.method == 'euler':
            return self._euler_solve(func, y0, t, x_interp)
        elif self.method == 'rk4':
            return self._rk4_solve(func, y0, t, x_interp)
        else:
            if TORCHDIFFEQ_AVAILABLE:
                return odeint(func, y0, t, method=self.method, 
                             rtol=self.rtol, atol=self.atol, **kwargs)
            else:
                return self._euler_solve(func, y0, t, x_interp)

    def _euler_solve(self, func: Callable, y0: torch.Tensor, 
                     t: torch.Tensor, x_interp: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Solve ODE using Euler method."""
        solutions = [y0]
        y = y0

        for i in range(len(t) - 1):
            dt = t[i + 1] - t[i]
            x_t = x_interp[i] if x_interp is not None else None
            if x_t is not None:
                dydt = func(t[i], y, x_t)
            else:
                dydt = func(t[i], y)
            y = y + dt * dydt
            solutions.append(y)

        return torch.stack(solutions, dim=0)

    def _rk4_solve(self, func: Callable, y0: torch.Tensor, 
                   t: torch.Tensor, x_interp: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Solve ODE using RK4 method."""
        dt = self.dt
        solutions = [y0]
        y = y0

        for i in range(len(t) - 1):
            x_t = x_interp[i] if x_interp is not None else None
            
            if x_t is not None:
                k1 = func(t[i], y, x_t)
                k2 = func(t[i] + dt/2, y + dt*k1/2, x_t)
                k3 = func(t[i] + dt/2, y + dt*k2/2, x_t)
                k4 = func(t[i] + dt, y + dt*k3, x_t)
            else:
                k1 = func(t[i], y)
                k2 = func(t[i] + dt/2, y + dt*k1/2)
                k3 = func(t[i] + dt/2, y + dt*k2/2)
                k4 = func(t[i] + dt, y + dt*k3)
            
            y = y + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            solutions.append(y)

        return torch.stack(solutions, dim=0)


class AdjointODESolver:
    """
    Adjoint ODE solver for memory-efficient backpropagation.
    Uses torchdiffeq's odeint_adjoint if available.
    """
    
    def __init__(self, method: str = 'dopri5', rtol: float = 1e-5, atol: float = 1e-5):
        """
        Initialize Adjoint ODE Solver.
        
        Args:
            method: Integration method
            rtol: Relative tolerance
            atol: Absolute tolerance
        """
        self.method = method
        self.rtol = rtol
        self.atol = atol
    
    def solve(self, func: Callable, y0: torch.Tensor, 
              t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Solve ODE using adjoint method.
        
        Args:
            func: Function computing dy/dt
            y0: Initial state
            t: Time points to evaluate
            
        Returns:
            Solution at time points
        """
        if not TORCHDIFFEQ_AVAILABLE:
            raise ImportError("torchdiffeq required for adjoint ODE solving")
        
        return odeint_adjoint(func, y0, t, method=self.method,
                             rtol=self.rtol, atol=self.atol, **kwargs)
