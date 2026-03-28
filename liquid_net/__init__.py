"""
Twistor-inspired Liquid Neural Network (Complex-valued LNN)
============================================================
Implements continuous-time dynamics: dz/dt = (-z + W*tanh(z) + U*x + b) / tau(z)

Features:
- Complex-valued hidden state z (torch.complex)
- State-dependent time constant tau(z) with clamping
- Multiple integration methods: Euler, RK4, torchdiffeq
- Stability optimizations (dz/dt normalization, gradient clipping)
- Dynamics analysis tools (fixed points, phase space, bifurcation)
"""

from .models.ltc_cell import LTCCell
from .models.liquid_net import TwistorLNN
from .solvers.euler import euler_step
from .solvers.rk4 import RK4Integrator
from .solvers.ode_solver import ODESolver, AdjointODESolver, TORCHDIFFEQ_AVAILABLE
from .analysis.dynamics import DynamicsAnalyzer, plot_bifurcation_diagram

__version__ = '2.0.0'
__all__ = [
    # Core models
    'LTCCell',
    'TwistorLNN',
    
    # Solvers
    'euler_step',
    'RK4Integrator',
    'ODESolver',
    'AdjointODESolver',
    'TORCHDIFFEQ_AVAILABLE',
    
    # Analysis
    'DynamicsAnalyzer',
    'plot_bifurcation_diagram',
]
