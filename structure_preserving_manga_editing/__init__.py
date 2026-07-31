"""Inference-time trajectory correction for structure-preserving manga editing."""

from .core import (
    apply_trajectory_correction,
    compute_reconstruction_target,
    correction_is_active,
)
from .tasks import TASKS, TaskConfig

__all__ = [
    "TASKS",
    "TaskConfig",
    "apply_trajectory_correction",
    "compute_reconstruction_target",
    "correction_is_active",
]
