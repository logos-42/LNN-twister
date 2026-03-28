"""
Twistor-LNN: 扭量驱动的液态神经网络
===================================

完整的 Twistor-inspired Liquid Neural Network 实现，包含：
- 复数值隐藏状态
- 状态依赖的时间常数
- 稀疏连接
- 多尺度动力学
- 多种积分方法
- 智能体接口
- 不动点分析
- 相空间可视化

快速开始:
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
    RK4Integrator,
    ODESolver,
    AdjointODESolver,
    TORCHDIFFEQ_AVAILABLE,
    rk4_step,
    heun_step,
    dopri5_step,
    create_integrator,
)
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
from .datasets import (
    generate_lorenz_dataset,
    generate_mackey_glass_dataset,
    generate_van_der_pol_dataset,
    generate_sine_dataset,
    create_dataset,
)
from .training import (
    train_model,
    train_on_task,
    plot_training_results,
    plot_predictions,
)

__version__ = "1.3.0"

__all__ = [
    # ========== Core ==========
    "TwistorLNN",
    # ========== Coupled Models ==========
    "CoupledTwistorLNN",
    "StackedCoupledLNN",
    # ========== Agent ==========
    "TwistorAgent",
    "TwistorAgentWithPolicy",
    "MultiAgent",
    # ========== Decoder ==========
    "TwistorDecoder",
    "TensorTwistorDecoder",
    "create_decoder",
    # ========== Integrators ==========
    "euler_step",
    "rk4_step",
    "RK4Integrator",
    "heun_step",
    "dopri5_step",
    "ODESolver",
    "AdjointODESolver",
    "TORCHDIFFEQ_AVAILABLE",
    "create_integrator",
    # ========== ODE Solver (Twistor-specific) ==========
    "TwistorODE",
    "ODEDynamics",
    "odeint_wrapper",
    "create_ode_solver",
    # ========== Analysis ==========
    "FixedPointFinder",
    "StabilityAnalyzer",
    "BifurcationAnalyzer",
    "analyze_model",
    # ========== Visualization ==========
    "plot_phase_space_2d",
    "plot_phase_space_3d",
    "plot_vector_field",
    "plot_tau_evolution",
    "plot_complex_plane",
    "plot_stability_analysis",
    "plot_training_diagnostics",
    # ========== Datasets ==========
    "generate_lorenz_dataset",
    "generate_mackey_glass_dataset",
    "generate_van_der_pol_dataset",
    "generate_sine_dataset",
    "create_dataset",
    # ========== Training ==========
    "train_model",
    "train_on_task",
    "plot_training_results",
    "plot_predictions",
]
