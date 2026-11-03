from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

from structure_preserving_manga_editing import flux3_klein
from structure_preserving_manga_editing.flux3_klein import (
    FLUX3_TEXT_EMBED_DIM,
    PrecomputedPromptEncoder,
    PromptEmbeddings,
    Qwen3VLPromptEncoderPlaceholder,
    StructurePreservingFlux3KleinPipeline,
    flow_map_targets,
    pack_tokens,
    patchify_latents,
    prepare_image_ids,
    prepare_text_ids,
    save_prompt_embeddings,
    unpack_tokens,
    unpatchify_latents,
)


def embeddings(length: int, seed: int = 0) -> torch.Tensor:
    return torch.randn(1, length, FLUX3_TEXT_EMBED_DIM, generator=torch.Generator().manual_seed(seed))


class DictPromptEncoder:
    def __init__(self, table: dict[str, torch.Tensor]) -> None:
        self.table = table
        self.calls: list[str] = []

    def encode(self, prompt: str) -> PromptEmbeddings:
        self.calls.append(prompt)
        return PromptEmbeddings(self.table[prompt])


def tiny_pipeline(encoder) -> StructurePreservingFlux3KleinPipeline:
    from diffusers import AutoencoderKLFlux2, Flux3Transformer2DModel

    torch.manual_seed(0)
    transformer = Flux3Transformer2DModel(
        in_channels=128,
        cond_in_channels=128,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=FLUX3_TEXT_EMBED_DIM,
        axes_dims_rope=(4, 4, 4, 4),
        flow_map=True,
    ).eval()
    vae = AutoencoderKLFlux2(
        block_out_channels=(32, 32, 32, 32),
        layers_per_block=1,
        latent_channels=32,
        norm_num_groups=32,
    ).eval()
    return StructurePreservingFlux3KleinPipeline(transformer, vae, encoder)


class LayoutTests(unittest.TestCase):
    def test_patchify_round_trip(self) -> None:
        latents = torch.randn(2, 32, 8, 6)
        patched = patchify_latents(latents)
        self.assertEqual(patched.shape, (2, 128, 4, 3))
        torch.testing.assert_close(unpatchify_latents(patched), latents)

    def test_pack_round_trip(self) -> None:
        latents = torch.randn(1, 128, 4, 3)
        tokens = pack_tokens(latents)
        self.assertEqual(tokens.shape, (1, 12, 128))
        torch.testing.assert_close(unpack_tokens(tokens, 4, 3), latents)

    def test_ids(self) -> None:
        text = prepare_text_ids(3)
        torch.testing.assert_close(text[:, 3], torch.tensor([0.0, 1.0, 2.0]))
        self.assertTrue(torch.all(text[:, :3] == 0))
        image = prepare_image_ids(2, 2, l_coord=-1)
        torch.testing.assert_close(image[-1], torch.tensor([0.0, 0.5, 0.5, -1.0]))

    def test_flow_map_targets_end_at_zero(self) -> None:
        timesteps = torch.tensor([1000.0, 500.0])
        self.assertEqual(flow_map_targets(timesteps, 0), (torch.tensor(1.0), torch.tensor(0.5)))
        self.assertEqual(flow_map_targets(timesteps, 1), (torch.tensor(0.5), torch.tensor(0.0)))


class PromptEncoderTests(unittest.TestCase):
    def test_embedding_shape_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            PromptEmbeddings(torch.zeros(1, 80, 2560))
        with self.assertRaises(ValueError):
            PromptEmbeddings(torch.zeros(80, FLUX3_TEXT_EMBED_DIM))

    def test_qwen_placeholder_is_not_implemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            Qwen3VLPromptEncoderPlaceholder()

    def test_precomputed_round_trip(self) -> None:
        table = {"Remove text.": embeddings(80, 1), "": embeddings(80, 2)}
        # Windows may keep the memory-mapped file open; do not fail on cleanup.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            path = Path(directory) / "embeds.safetensors"
            save_prompt_embeddings(path, table)
            encoder = PrecomputedPromptEncoder(path)
        self.assertEqual(sorted(encoder.prompts), ["", "Remove text."])
        torch.testing.assert_close(encoder.encode("").embeds, table[""])
        with self.assertRaises(KeyError):
            encoder.encode("missing")


class TinyPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = Image.new("RGB", (64, 64), color=(200, 200, 200))

    def run_edit(self, empty_length: int, correction_steps: int, correction_strength: float = 0.5):
        encoder = DictPromptEncoder({"edit": embeddings(80, 1), "": embeddings(empty_length, 2)})
        pipeline = tiny_pipeline(encoder)
        calls = []
        original = pipeline.transformer.forward

        def spy(**kwargs):
            calls.append(kwargs)
            return original(**kwargs)

        with mock.patch.object(pipeline.transformer, "forward", side_effect=spy):
            image = pipeline.edit(
                self.image,
                "edit",
                torch.Generator().manual_seed(0),
                num_inference_steps=3,
                correction_steps=correction_steps,
                correction_strength=correction_strength,
            )
        return image, calls, encoder

    def test_batched_correction_uses_terminal_target_for_empty_branch(self) -> None:
        image, calls, encoder = self.run_edit(empty_length=80, correction_steps=2)
        self.assertEqual(image.size, (64, 64))
        self.assertEqual(encoder.calls, ["edit", ""])
        self.assertEqual([c["hidden_states"].shape[0] for c in calls], [2, 2, 1])
        first = calls[0]["target_timestep"]
        self.assertEqual(first[0].item(), 0.0)
        self.assertGreater(first[1].item(), 0.0)
        self.assertEqual(calls[-1]["target_timestep"].item(), 0.0)

    def test_unequal_lengths_fall_back_to_sequential_calls(self) -> None:
        _, calls, _ = self.run_edit(empty_length=160, correction_steps=1)
        self.assertEqual([c["encoder_hidden_states"].shape[1] for c in calls], [160, 80, 80, 80])
        self.assertEqual(calls[0]["target_timestep"].item(), 0.0)

    def test_zero_strength_matches_uncorrected_sampling(self) -> None:
        corrected, _, _ = self.run_edit(empty_length=80, correction_steps=2, correction_strength=0.0)
        plain, calls, encoder = self.run_edit(empty_length=80, correction_steps=0)
        self.assertEqual(encoder.calls, ["edit"])
        self.assertEqual(len(calls), 3)
        # Batched and single transformer calls may differ by one rounding step.
        difference = np.abs(np.asarray(corrected, dtype=int) - np.asarray(plain, dtype=int))
        self.assertLessEqual(difference.max(), 1)

    def test_correction_changes_the_result(self) -> None:
        corrected, _, _ = self.run_edit(empty_length=80, correction_steps=2, correction_strength=0.5)
        plain, _, _ = self.run_edit(empty_length=80, correction_steps=0)
        difference = np.abs(np.asarray(corrected, dtype=int) - np.asarray(plain, dtype=int))
        self.assertGreater(difference.max(), 1)


class ModuleConstantsTests(unittest.TestCase):
    def test_checkpoint_defaults(self) -> None:
        self.assertEqual(flux3_klein.FLUX3_KLEIN_MODEL_ID, "black-forest-labs/flux-3-image-klein")
        self.assertEqual(flux3_klein.FLUX3_KLEIN_NUM_INFERENCE_STEPS, 16)
        self.assertEqual(flux3_klein.FLUX3_KLEIN_SHIFT, 6.93)


if __name__ == "__main__":
    unittest.main()
