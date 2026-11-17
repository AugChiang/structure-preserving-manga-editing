from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch
from PIL import Image

from structure_preserving_manga_editing.cli import build_parser, resolve_config
from structure_preserving_manga_editing.qwen_image21 import (
    QWEN_IMAGE21_NUM_INFERENCE_STEPS,
    QWEN_IMAGE21_TRUE_CFG_SCALE,
    StructurePreservingQwenImage21Pipeline,
    _pad_prompt,
)
from structure_preserving_manga_editing.tasks import QWEN_IMAGE21_TRANSFORMER_GGUF

CONTEXT_DIM = 32
IMAGE_SLOTS = 4  # a 64x64 input is a 4x4 latent grid: 16 tokens, 4 slots of 2x2


class FakeTokenizer:
    def encode(self, text):
        return [7]


class FakeProcessor:
    tokenizer = FakeTokenizer()

    def apply_chat_template(self, *args, **kwargs):
        return [[0, 0, 0]]


def fake_encode_prompt(lengths: dict[str, int]):
    """Stand-in for the Qwen3-VL text encoder: image slots first, then text."""

    calls = []

    def encode_prompt(prompt, image=None, device=None, **kwargs):
        calls.append(prompt)
        length = IMAGE_SLOTS + lengths[prompt]
        generator = torch.Generator().manual_seed(length)
        embeds = torch.randn(1, length, CONTEXT_DIM, generator=generator)
        slots = torch.zeros(1, length, dtype=torch.bool)
        slots[:, :IMAGE_SLOTS] = True
        return embeds, None, slots

    return encode_prompt, calls


def tiny_pipeline() -> StructurePreservingQwenImage21Pipeline:
    from diffusers import (
        AutoencoderKLQwenImage21,
        FlowMatchEulerDiscreteScheduler,
        QwenImage21Transformer2DModel,
    )

    torch.manual_seed(0)
    transformer = QwenImage21Transformer2DModel(
        num_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        context_in_dim=CONTEXT_DIM,
        axes_dims_rope=(4, 6, 6),
    ).eval()
    vae = AutoencoderKLQwenImage21(
        base_dim=8,
        decoder_base_dim=8,
        dim_mult=[1, 1, 1, 1, 1],
        num_res_blocks=1,
    ).eval()
    return StructurePreservingQwenImage21Pipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True),
        vae=vae,
        text_encoder=None,
        processor=FakeProcessor(),
        transformer=transformer,
    )


class PadPromptTests(unittest.TestCase):
    def test_pads_embeddings_and_both_masks(self) -> None:
        embeds = torch.ones(1, 3, 2)
        slots = torch.tensor([[True, False, False]])
        padded, mask, padded_slots = _pad_prompt(embeds, None, slots, 5)
        self.assertEqual(padded.shape, (1, 5, 2))
        self.assertTrue(torch.all(padded[:, 3:] == 0))
        self.assertEqual(mask.tolist(), [[1, 1, 1, 0, 0]])
        self.assertEqual(padded_slots.tolist(), [[True, False, False, False, False]])

    def test_no_padding_needed(self) -> None:
        embeds = torch.ones(1, 3, 2)
        slots = torch.zeros(1, 3, dtype=torch.bool)
        padded, mask, _ = _pad_prompt(embeds, None, slots, 3)
        self.assertIs(padded, embeds)
        self.assertEqual(mask.tolist(), [[1, 1, 1]])


class TinyPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = Image.new("RGB", (64, 64), color=(200, 200, 200))

    def run_edit(self, correction_steps: int, correction_strength: float = 0.5, use_kv_cache: bool = False):
        pipeline = tiny_pipeline()
        encode_prompt, prompts = fake_encode_prompt({"edit": 6, "": 2})
        calls = []
        original = pipeline.transformer.forward

        def spy(**kwargs):
            calls.append(kwargs)
            return original(**kwargs)

        with (
            mock.patch.object(pipeline, "encode_prompt", side_effect=encode_prompt),
            mock.patch.object(pipeline.transformer, "forward", side_effect=spy),
        ):
            image = pipeline.edit(
                self.image,
                "edit",
                torch.Generator().manual_seed(0),
                num_inference_steps=3,
                correction_steps=correction_steps,
                correction_strength=correction_strength,
                use_kv_cache=use_kv_cache,
                resolution=64,
            )
        return image, calls, prompts

    def test_both_prompts_share_one_batch(self) -> None:
        image, calls, prompts = self.run_edit(correction_steps=2)
        self.assertEqual(image.size, (64, 64))
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(prompts, ["edit", ""])
        self.assertEqual(len(calls), 3)
        first = calls[0]
        # Condition tokens first, then target tokens, for both branches.
        self.assertEqual(first["hidden_states"].shape, (2, 32, 64))
        self.assertEqual(first["encoder_hidden_states"].shape, (2, IMAGE_SLOTS + 6, CONTEXT_DIM))
        # The empty prompt is padded and masked; the edit prompt is fully valid.
        self.assertEqual(first["encoder_hidden_states_mask"].sum(dim=1).tolist(), [IMAGE_SLOTS + 2, IMAGE_SLOTS + 6])
        # Image slots, then one slot per 2x2 group of target tokens.
        self.assertEqual(first["img_mask"].shape, (2, IMAGE_SLOTS + 6 + 4))
        self.assertEqual(first["img_shapes"], [[(1, 4, 4), (1, 4, 4)]] * 2)

    def test_zero_strength_matches_uncorrected_sampling(self) -> None:
        corrected, _, _ = self.run_edit(correction_steps=2, correction_strength=0.0)
        plain, _, _ = self.run_edit(correction_steps=0)
        np.testing.assert_array_equal(np.asarray(corrected), np.asarray(plain))

    def test_correction_changes_the_result(self) -> None:
        corrected, _, _ = self.run_edit(correction_steps=2, correction_strength=0.5)
        plain, _, _ = self.run_edit(correction_steps=0)
        difference = np.abs(np.asarray(corrected, dtype=int) - np.asarray(plain, dtype=int))
        self.assertGreater(difference.max(), 0)

    def test_kv_cache_prefills_then_decodes(self) -> None:
        image, calls, _ = self.run_edit(correction_steps=2, use_kv_cache=True)
        self.assertEqual(image.size, (64, 64))
        self.assertEqual([c["kv_cache_mode"] for c in calls], ["extract", "cached", "cached"])

    def test_invalid_arguments(self) -> None:
        pipeline = tiny_pipeline()
        with self.assertRaises(ValueError):
            pipeline.edit(self.image, "edit", torch.Generator(), correction_steps=-1)
        with self.assertRaises(ValueError):
            pipeline.edit(self.image, "edit", torch.Generator(), correction_strength=2.0)


class CliTests(unittest.TestCase):
    def test_defaults(self) -> None:
        args = build_parser("qwen-image-2.1").parse_args(["--task", "text-removal", "--input", "input.png"])
        config = resolve_config("qwen-image-2.1", args)
        self.assertEqual(config["model_id"], "Qwen/Qwen-Image-2.1")
        self.assertEqual(config["num_inference_steps"], QWEN_IMAGE21_NUM_INFERENCE_STEPS)
        self.assertEqual(config["transformer_gguf"], QWEN_IMAGE21_TRANSFORMER_GGUF)
        self.assertEqual(config["prompt"], "Remove text. Keep everything else unchanged.")
        self.assertFalse(config["kv_cache"])
        self.assertEqual(QWEN_IMAGE21_TRUE_CFG_SCALE, 5.0)

    def test_overrides(self) -> None:
        args = build_parser("qwen-image-2.1").parse_args(
            ["--task", "screentone-synthesis", "--input", "input.png", "--transformer-gguf", "q5.gguf", "--kv-cache"]
        )
        config = resolve_config("qwen-image-2.1", args)
        self.assertEqual(config["transformer_gguf"], "q5.gguf")
        self.assertTrue(config["kv_cache"])

    def test_other_backends_have_no_default_gguf(self) -> None:
        args = build_parser("qwen-image-edit").parse_args(["--task", "text-removal", "--input", "input.png"])
        config = resolve_config("qwen-image-edit", args)
        self.assertIsNone(config["transformer_gguf"])
        self.assertNotIn("kv_cache", config)


if __name__ == "__main__":
    unittest.main()
