"""
Solvers module for Twistor-inspired Liquid Neural Network.

Provides numerical integration methods:
- Euler (basic, fast)
- RK4 (4th order Runge-Kutta, more accurate)
- ODE solvers via torchdiffeq (most accurate, adaptive)
"""

from .euler import euler_step
from .rk4 import RK4Integrator
from .ode_solver import ODESolver, AdjointODESolver, TORCHDIFFEQ_AVAILABLE

__all__ = [
    'euler_step',
    'RK4Integrator',
    'ODESolver',
    'AdjointODESolver',
    'TORCHDIFFEQ_AVAILABLE',
]
