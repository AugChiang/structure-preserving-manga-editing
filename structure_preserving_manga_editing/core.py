"""Model-independent implementation of Algorithm 1."""

from __future__ import annotations

from typing import Protocol

import torch


class FlowScheduler(Protocol):
    """Minimal scheduler interface needed to map a step to flow time."""

    sigmas: torch.Tensor
    step_index: int | None

    def _init_step_index(self, timestep: torch.Tensor) -> None: ...


def compute_reconstruction_target(
    input_latents: torch.Tensor,
    empty_prompt_velocity: torch.Tensor,
    time_to_terminal: torch.Tensor | float,
) -> torch.Tensor:
    """Compute the one-step empty-prompt reconstruction target.

    This is Algorithm 1, line 6:
    ``target = input_latents - time_to_terminal * empty_prompt_velocity``.
    Diffusers parameterizes its flow scheduler with decreasing sigma, so its
    signed ``time_to_terminal`` is ``terminal_sigma - current_sigma``.
    """
    return input_latents - time_to_terminal * empty_prompt_velocity


def correction_is_active(step_index: int, correction_steps: int) -> bool:
    """Return whether Algorithm 1 applies at this denoising step."""
    if step_index < 0:
        raise ValueError("step_index must be non-negative")
    if correction_steps < 0:
        raise ValueError("correction_steps must be non-negative")
    return step_index < correction_steps


def apply_trajectory_correction(
    latents: torch.Tensor,
    input_latents: torch.Tensor,
    empty_prompt_velocity: torch.Tensor,
    scheduler: FlowScheduler,
    timestep: torch.Tensor,
    correction_strength: float,
) -> torch.Tensor:
    """Blend the current state with the reconstruction target."""
    if not 0.0 <= correction_strength <= 1.0:
        raise ValueError("correction_strength must be between 0 and 1")
    if latents.shape != input_latents.shape or latents.shape != empty_prompt_velocity.shape:
        raise ValueError("latent, input, and velocity tensors must have identical shapes")

    if scheduler.step_index is None:
        scheduler._init_step_index(timestep)

    current_sigma = scheduler.sigmas[scheduler.step_index]
    terminal_sigma = scheduler.sigmas[-1]
    time_to_terminal = (terminal_sigma - current_sigma).to(
        device=latents.device,
        dtype=latents.dtype,
    )
    target = compute_reconstruction_target(
        input_latents,
        empty_prompt_velocity,
        time_to_terminal,
    )
    return torch.lerp(latents, target, correction_strength)
