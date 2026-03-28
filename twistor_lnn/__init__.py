"""
Twistor-LNN: 扭量驱动的液态神经网络
===================================

A Twistor-inspired Liquid Neural Network implementation with:
- Complex-valued hidden states
- State-dependent time constants
- Sparse connectivity
- Multi-scale dynamics
- Multiple integration methods (Euler, RK4, ODE solver)
- Agent interfaces
- Fixed point analysis
- Phase space visualization

Quick Start:
    from twistor_lnn import TwistorLNN, TwistorAgent

    model = TwistorLNN(input_dim=4, hidden_dim=32, output_dim=2)
    agent = TwistorAgent(obs_dim=4, action_dim=2)

Analysis:
    from twistor_lnn.analysis import FixedPointFinder, StabilityAnalyzer
    from twistor_lnn.visualization import plot_phase_space_2d

Visualization:
    from twistor_lnn.visualization import (
        plot_phase_space_2d,
        plot_vector_field,
        plot_complex_plane,
    )
"""

from .core import TwistorLNN
from .coupled import CoupledTwistorLNN, StackedCoupledLNN
from .agent import TwistorAgent, TwistorAgentWithPolicy, MultiAgent
from .decoder import TwistorDecoder, TensorTwistorDecoder, create_decoder

# Import from liquid_net.solvers (maintained implementation)
try:
    from liquid_net.solvers import (
        euler_step,
        RK4Integrator,
        ODESolver,
        AdjointODESolver,
        TORCHDIFFEQ_AVAILABLE,
    )

    # Keep custom implementations as fallbacks/alternatives
    from .integrators import (
        heun_step,
        dopri5_step,
        Integrator,
        create_integrator,
    )
except ImportError:
    # Fallback to built-in if liquid_net not available
    from .integrators import (
        euler_step,
        rk4_step,
        heun_step,
        dopri5_step,
        Integrator,
        create_integrator,
    )

    RK4Integrator = None
    ODESolver = None
    AdjointODESolver = None
    TORCHDIFFEQ_AVAILABLE = False
from .ode_solver import TwistorODE, ODEDynamics, odeint_wrapper, create_ode_solver
from .analysis import (
    FixedPointFinder,
    StabilityAnalyzer,
    BifurcationAnalyzer,
    analyze_model,
)
from .visualization import (
    plot_phase_space_2d,
    plot_phase_space_3d,
    plot_vector_field,
    plot_tau_evolution,
    plot_complex_plane,
    plot_stability_analysis,
    plot_training_diagnostics,
)

__version__ = "1.1.0"

__all__ = [
    # Core
    "TwistorLNN",
    # Coupled
    "CoupledTwistorLNN",
    "StackedCoupledLNN",
    # Agent
    "TwistorAgent",
    "TwistorAgentWithPolicy",
    "MultiAgent",
    # Decoder
    "TwistorDecoder",
    "TensorTwistorDecoder",
    "create_decoder",
    # Integrators (from liquid_net.solvers)
    "euler_step",
    "RK4Integrator",
    "ODESolver",
    "AdjointODESolver",
    "TORCHDIFFEQ_AVAILABLE",
    # Additional Integrators
    "heun_step",
    "dopri5_step",
    "create_integrator",
    # ODE Solver (twistor-specific)
    "TwistorODE",
    "ODEDynamics",
    "odeint_wrapper",
    "create_ode_solver",
    # Analysis
    "FixedPointFinder",
    "StabilityAnalyzer",
    "BifurcationAnalyzer",
    "analyze_model",
    # Visualization
    "plot_phase_space_2d",
    "plot_phase_space_3d",
    "plot_vector_field",
    "plot_tau_evolution",
    "plot_complex_plane",
    "plot_stability_analysis",
    "plot_training_diagnostics",
]
