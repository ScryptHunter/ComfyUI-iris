"""Exact Iris upstream flow sigma grid, independent of Comfy scheduler tables."""

from __future__ import annotations

import math

import torch


def iris_sigma_grid(steps: int, shift: float) -> torch.Tensor:
    """Return descending shifted FP64 sigmas including an exact terminal zero.

    This is the FlowDPMSolver.time_grid construction from the vendored Iris
    reference: ``1-linspace(1,0.001,steps+1)``, shift, then reverse.
    """
    if int(steps) < 1:
        raise ValueError("steps must be positive")
    if not float(shift) > 0:
        raise ValueError("shift must be positive")
    sigma = 1.0 - torch.linspace(1.0, 0.001, int(steps) + 1, dtype=torch.float64)
    shifted = float(shift) * sigma / (1.0 + (float(shift) - 1.0) * sigma)
    result = shifted.flip(0)
    result[-1] = 0.0
    return result


def img2img_sigma_subset(sigmas: torch.Tensor, strength: float) -> torch.Tensor:
    """Select the denoising suffix of a complete Iris sigma grid.

    Positive strength selects approximately ``round(strength * steps)`` active
    intervals, clamped to one interval, by skipping the earliest schedule
    entries. Strength zero returns the one-element ``[0]`` schedule; Comfy's
    DPM++ 2M SDE sampler treats this as a no-op while still returning the input
    pixel latent unchanged.
    """
    if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or sigmas.numel() < 2:
        raise ValueError("Img2Img expects the complete one-dimensional Iris SIGMAS schedule")
    strength = float(strength)
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("Img2Img strength must be finite and in [0,1]")
    if not torch.isfinite(sigmas).all():
        raise ValueError("Iris SIGMAS contains NaN or infinity")
    grid = sigmas.detach().to(device="cpu", dtype=torch.float64)
    if not torch.all(grid[:-1] > grid[1:]) or abs(float(grid[-1])) > 1e-8:
        raise ValueError("Img2Img requires strictly descending Iris SIGMAS ending at zero")
    if float(grid[0]) <= 0.0:
        raise ValueError("The full Iris sigma schedule must start above zero")
    if strength == 0.0:
        return sigmas.new_zeros((1,))

    total_steps = sigmas.numel() - 1
    active_steps = max(1, min(total_steps, math.floor(strength * total_steps + 0.5)))
    return sigmas[-(active_steps + 1):].clone()
