from __future__ import annotations

import unittest

import torch

from structure_preserving_manga_editing.core import (
    apply_trajectory_correction,
    compute_reconstruction_target,
    correction_is_active,
)


class FakeScheduler:
    def __init__(self) -> None:
        self.sigmas = torch.tensor([1.0, 0.5, 0.0])
        self.step_index = None

    def _init_step_index(self, timestep: torch.Tensor) -> None:
        self.step_index = int(timestep.item())


class CoreTests(unittest.TestCase):
    def test_reconstruction_target_uses_empty_prompt_velocity(self) -> None:
        source = torch.tensor([2.0])
        empty_prompt_velocity = torch.tensor([3.0])
        target = compute_reconstruction_target(
            source,
            empty_prompt_velocity,
            -0.5,
        )
        torch.testing.assert_close(target, torch.tensor([3.5]))

    def test_zero_strength_keeps_current_state(self) -> None:
        scheduler = FakeScheduler()
        current = torch.tensor([1.0])
        corrected = apply_trajectory_correction(
            current,
            torch.tensor([2.0]),
            torch.tensor([3.0]),
            scheduler,
            torch.tensor(0),
            0.0,
        )
        torch.testing.assert_close(corrected, current)

    def test_unit_strength_reaches_target(self) -> None:
        scheduler = FakeScheduler()
        corrected = apply_trajectory_correction(
            torch.tensor([1.0]),
            torch.tensor([2.0]),
            torch.tensor([3.0]),
            scheduler,
            torch.tensor(1),
            1.0,
        )
        torch.testing.assert_close(corrected, torch.tensor([3.5]))

    def test_steps_after_configured_prefix_are_not_corrected(self) -> None:
        self.assertTrue(correction_is_active(2, 3))
        self.assertFalse(correction_is_active(3, 3))
        self.assertFalse(correction_is_active(9, 3))

    def test_shape_mismatch_is_rejected(self) -> None:
        scheduler = FakeScheduler()
        with self.assertRaises(ValueError):
            apply_trajectory_correction(
                torch.ones(1),
                torch.ones(2),
                torch.ones(1),
                scheduler,
                torch.tensor(0),
                0.01,
            )


if __name__ == "__main__":
    unittest.main()
