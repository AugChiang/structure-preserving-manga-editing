from __future__ import annotations

import unittest

import torch

from structure_preserving_manga_editing.qwen_image_edit import _pad_prompt_pair


class PadPromptPairTests(unittest.TestCase):
    def test_missing_masks_are_treated_as_all_valid(self) -> None:
        first = torch.ones(1, 2, 4)
        second = torch.ones(1, 3, 4)
        first, first_mask, second, second_mask = _pad_prompt_pair(first, None, second, None)
        self.assertEqual(first.shape, (1, 3, 4))
        torch.testing.assert_close(first_mask, torch.tensor([[1, 1, 0]]))
        torch.testing.assert_close(second_mask, torch.tensor([[1, 1, 1]]))


if __name__ == "__main__":
    unittest.main()
