"""
Twistor-LNN: 扭量驱动的液态神经网络
===================================

A Twistor-inspired Liquid Neural Network implementation with:
- Complex-valued hidden states
- State-dependent time constants
- Sparse connectivity
- Multi-scale dynamics
- Multiple integration methods
- Agent interfaces

Quick Start:
    from twistor_lnn import TwistorLNN, TwistorAgent

    model = TwistorLNN(input_dim=4, hidden_dim=32, output_dim=2)
    agent = TwistorAgent(obs_dim=4, action_dim=2)
"""

from .core import TwistorLNN
from .coupled import CoupledTwistorLNN, StackedCoupledLNN
from .agent import TwistorAgent, TwistorAgentWithPolicy, MultiAgent
from .decoder import TwistorDecoder, TensorTwistorDecoder, create_decoder
from .integrators import (
    euler_step,
    rk4_step,
    heun_step,
    dopri5_step,
    Integrator,
    create_integrator,
)

__version__ = "1.0.0"

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
    # Integrators
    "euler_step",
    "rk4_step",
    "heun_step",
    "dopri5_step",
    "Integrator",
    "create_integrator",
]
