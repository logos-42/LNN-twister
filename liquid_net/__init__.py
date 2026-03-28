"""
Twistor-inspired Liquid Neural Network (Complex-valued LNN)
============================================================
Implements continuous-time dynamics: dz/dt = (-z + W*tanh(z) + Ux) / tau(z)

Key features:
- Complex-valued hidden state z (torch.complex)
- State-dependent time constant tau(z)
- Euler integration for time evolution
- Stability regularization via ||dz/dt||^2
"""

from .models.ltc_cell import LTCCell
from .models.liquid_net import TwistorLNN
from .solvers.euler import euler_step
from .training.loss import twistor_loss
from .training.train import train_twistor_lnn, generate_sine_dataset

__all__ = [
    'LTCCell',
    'TwistorLNN',
    'euler_step',
    'twistor_loss',
    'train_twistor_lnn',
    'generate_sine_dataset'
]
