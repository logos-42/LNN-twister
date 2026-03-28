"""
Twistor-LNN Integrators Module
===============================
Numerical integration methods for ODE dynamics.

Methods:
- Euler: First-order, simple but less accurate
- RK4: Fourth-order Runge-Kutta, more accurate
- Heun: Second-order, balance between speed and accuracy
"""

import torch
from typing import Callable, Optional


def euler_step(
    dzdt_fn: Callable, z: torch.Tensor, x: torch.Tensor, dt: float, *args, **kwargs
) -> torch.Tensor:
    """
    Euler integration (first-order).

    z(t+dt) = z(t) + dt * dz/dt

    Args:
        dzdt_fn: Function computing dz/dt
        z: Current state
        x: Input
        dt: Time step
        *args, **kwargs: Additional arguments for dzdt_fn

    Returns:
        z_next: Next state
    """
    dzdt = dzdt_fn(z, x, *args, **kwargs)
    return z + dt * dzdt


def rk4_step(
    dzdt_fn: Callable, z: torch.Tensor, x: torch.Tensor, dt: float, *args, **kwargs
) -> torch.Tensor:
    """
    Runge-Kutta 4th order integration.

    More accurate than Euler, better for complex dynamics.

    Args:
        dzdt_fn: Function computing dz/dt
        z: Current state
        x: Input
        dt: Time step
        *args, **kwargs: Additional arguments for dzdt_fn

    Returns:
        z_next: Next state
    """
    k1 = dzdt_fn(z, x, *args, **kwargs)
    k2 = dzdt_fn(z + 0.5 * dt * k1, x, *args, **kwargs)
    k3 = dzdt_fn(z + 0.5 * dt * k2, x, *args, **kwargs)
    k4 = dzdt_fn(z + dt * k3, x, *args, **kwargs)

    return z + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_adaptive_step(
    dzdt_fn: Callable,
    z: torch.Tensor,
    x: torch.Tensor,
    dt: float,
    tolerance: float = 1e-6,
    *args,
    **kwargs,
) -> torch.Tensor:
    """
    Adaptive RK4 with step size control.

    Uses RK4 and RK2 to estimate error and adjust step size.

    Args:
        dzdt_fn: Function computing dz/dt
        z: Current state
        x: Input
        dt: Time step
        tolerance: Error tolerance

    Returns:
        z_next: Next state
    """
    k1 = dzdt_fn(z, x, *args, **kwargs)
    k2 = dzdt_fn(z + 0.5 * dt * k1, x, *args, **kwargs)
    k3 = dzdt_fn(z + 0.5 * dt * k2, x, *args, **kwargs)
    k4 = dzdt_fn(z + dt * k3, x, *args, **kwargs)

    z_rk4 = z + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    k2_simple = dzdt_fn(z + dt * k1, x, *args, **kwargs)
    z_rk2 = z + dt * k2_simple

    error = torch.abs(z_rk4 - z_rk2).max()

    if error > tolerance:
        dt_new = dt * (tolerance / error).clamp(0.1, 2.0)
        return rk4_adaptive_step(dzdt_fn, z, x, dt_new, tolerance, *args, **kwargs)

    return z_rk4


def heun_step(
    dzdt_fn: Callable, z: torch.Tensor, x: torch.Tensor, dt: float, *args, **kwargs
) -> torch.Tensor:
    """
    Heun's method (second-order).

    Balance between Euler and RK4.

    Args:
        dzdt_fn: Function computing dz/dt
        z: Current state
        x: Input
        dt: Time step

    Returns:
        z_next: Next state
    """
    k1 = dzdt_fn(z, x, *args, **kwargs)
    k2 = dzdt_fn(z + dt * k1, x, *args, **kwargs)

    return z + (dt / 2) * (k1 + k2)


def dopri5_step(
    dzdt_fn: Callable, z: torch.Tensor, x: torch.Tensor, dt: float, *args, **kwargs
) -> torch.Tensor:
    """
    Dormand-Prince (DOPRI5) - 5th order method.

    Commonly used in ODE solvers (e.g., dopri5 in scipy).

    Args:
        dzdt_fn: Function computing dz/dt
        z: Current state
        x: Input
        dt: Time step

    Returns:
        z_next: Next state
    """
    c = torch.tensor([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1])
    a = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0],
            [1 / 5, 0, 0, 0, 0, 0],
            [3 / 40, 9 / 40, 0, 0, 0, 0],
            [44 / 45, -56 / 15, 32 / 9, 0, 0, 0],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0],
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0],
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
        ]
    )
    b = torch.tensor([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])

    k = []
    k.append(dzdt_fn(z, x, *args, **kwargs))

    for i in range(1, 7):
        z_temp = z + dt * sum(a[i, j] * k[j] for j in range(i))
        k.append(dzdt_fn(z_temp, x, *args, **kwargs))

    return z + dt * sum(b[i] * k[i] for i in range(7))


class Integrator:
    """
    Integration method wrapper.
    """

    METHODS = {
        "euler": euler_step,
        "rk4": rk4_step,
        "heun": heun_step,
        "dopri5": dopri5_step,
    }

    def __init__(self, method: str = "rk4"):
        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}")
        self.method = self.METHODS[method]
        self.method_name = method

    def step(
        self,
        dzdt_fn: Callable,
        z: torch.Tensor,
        x: torch.Tensor,
        dt: float,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Single integration step."""
        return self.method(dzdt_fn, z, x, dt, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.step(*args, **kwargs)


def create_integrator(method: str = "rk4", **kwargs) -> Integrator:
    """
    Factory function to create integrator.

    Args:
        method: 'euler', 'rk4', 'heun', or 'dopri5'

    Returns:
        integrator: Integrator instance
    """
    return Integrator(method)
