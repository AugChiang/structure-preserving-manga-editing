# Copyright 2026 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified to implement the trajectory correction described
# in Algorithm 1 of "Inference-time Trajectory Optimization for
# Structure-Preserving Manga Image Editing." The sampling, latent packing and
# batch-norm (de)normalization follow the Diffusers FLUX.3 modular pipeline
# (diffusers 0.42.1, `diffusers/modular_pipelines/flux3`).

"""FLUX 3 Image Klein integration for inference-time trajectory correction.

FLUX 3 Image Klein is a step-distilled *flow-map* model: every transformer call
receives the current flow time ``t`` and the time ``t_target`` the step jumps
to, and predicts the average velocity over ``[t, t_target]``. Algorithm 1 needs
a one-step reconstruction from the current time to the terminal time, so the
empty-prompt branch is evaluated with ``t_target = 0`` (the terminal time),
which is exactly the jump that ``compute_reconstruction_target`` extrapolates.
The edit branch keeps the regular ``t_target = t_next``.

Text encoding
-------------
The official text encoder of FLUX 3 Image Klein is Qwen3-VL-4B-Instruct, and
the official prompt upsampler is a Qwen3.5-9B fine-tune. By project policy,
models from the Qwen series are NOT downloaded or loaded by this repository.
The prompt encoder is therefore an interface (:class:`Flux3PromptEncoder`):

* :class:`Qwen3VLPromptEncoderPlaceholder` marks where the official encoder
  goes and raises ``NotImplementedError`` (see its TODO comments);
* :class:`PrecomputedPromptEncoder` loads prompt embeddings that were computed
  elsewhere, so the transformer + VAE path can run without any Qwen weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from .core import apply_trajectory_correction, correction_is_active
from .tasks import FLUX3_KLEIN_MODEL_ID  # noqa: F401  (re-exported)

# Values from the checkpoint's `config.json` (`sampler` and `text_encoder`).
FLUX3_KLEIN_NUM_INFERENCE_STEPS = 16
FLUX3_KLEIN_SHIFT = 6.93
FLUX3_TEXT_HIDDEN_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
FLUX3_TEXT_PAD_MULTIPLE = 80
FLUX3_TEXT_EMBED_DIM = 20480  # 8 stacked Qwen3-VL-4B hidden states x 2560

# Recommended quantized transformers for a 12 GB GPU (e.g. RTX 4070, Ada).
# GGUF loads through Diffusers on any CUDA GPU. BFL's NVFP4 variant needs a
# Blackwell GPU and does not run natively on Ada.
FLUX3_KLEIN_GGUF_Q8 = (
    "https://huggingface.co/unsloth/flux-3-image-klein-GGUF/blob/main/flux-3-image-klein-Q8_0.gguf"
)
FLUX3_KLEIN_GGUF_Q4 = (
    "https://huggingface.co/unsloth/flux-3-image-klein-GGUF/blob/main/flux-3-image-klein-Q4_K_M.gguf"
)


@dataclass(frozen=True)
class PromptEmbeddings:
    """Stacked text-encoder hidden states for one prompt, shape ``(1, L, 20480)``."""

    embeds: torch.Tensor

    def __post_init__(self) -> None:
        if self.embeds.ndim != 3 or self.embeds.shape[0] != 1:
            raise ValueError("prompt embeddings must have shape (1, sequence_length, dim)")
        if self.embeds.shape[-1] != FLUX3_TEXT_EMBED_DIM:
            raise ValueError(
                f"prompt embeddings must have {FLUX3_TEXT_EMBED_DIM} channels, got {self.embeds.shape[-1]}"
            )


class Flux3PromptEncoder(Protocol):
    """Turns a prompt into FLUX 3 context tokens.

    Implementations must reproduce the Diffusers ``Flux3TextEncoderStep``:
    chat template, right padding to a multiple of 80 tokens (FLUX 3 attends to
    the padding tokens, so they must be real encoder outputs, not zeros), and
    the hidden states of layers 4, 8, ..., 32 stacked along the channel axis.
    """

    def encode(self, prompt: str) -> PromptEmbeddings: ...


class Qwen3VLPromptEncoderPlaceholder:
    """Placeholder for the official Qwen3-VL-4B-Instruct prompt encoder.

    Not implemented on purpose: models from the Qwen series must not be
    downloaded by this project.
    """

    # TODO(qwen-policy): The official FLUX 3 Image Klein text encoder is
    #   Qwen3-VL-4B-Instruct, shipped in `black-forest-labs/flux-3-image-klein`
    #   under `text_encoder/` (Qwen3VLForConditionalGeneration + Qwen3VLProcessor,
    #   ~8.9 GB in bf16). It is NOT downloaded here because of the project's
    #   no-Qwen-model policy. If the policy changes, implement `encode` as
    #   `Flux3TextEncoderStep._get_qwen3_vl_prompt_embeds` in diffusers 0.42.1:
    #   chat template, right padding to a multiple of FLUX3_TEXT_PAD_MULTIPLE,
    #   explicit position ids (padding positions = 1), and the hidden states of
    #   FLUX3_TEXT_HIDDEN_LAYERS stacked to (1, L, 20480).
    # TODO(qwen-policy): On a 12 GB GPU (RTX 4070), the bf16 encoder plus the
    #   transformer does not fit at once. Encode both prompts first, then free
    #   the encoder (or load it in NF4 via bitsandbytes, ~2.5 GB) before
    #   denoising.
    # TODO(qwen-policy): Encode the edit prompt and the empty prompt with the
    #   same padded length so that both can share one batched transformer call.
    # TODO(qwen-policy): BFL also recommends a prompt upsampler
    #   (`black-forest-labs/flux-3-image-klein-pup`, a Qwen3.5-9B fine-tune).
    #   It is also excluded by the policy. The experiments use the paper's
    #   exact task prompts without upsampling anyway.
    # TODO(qwen-policy): Replacing the encoder with a non-Qwen model is not a
    #   drop-in fix: the transformer's `context_embedder` was trained on
    #   Qwen3-VL-4B hidden states and would need retraining or an adapter.

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "The FLUX 3 Image Klein text encoder (Qwen3-VL-4B-Instruct) is not integrated: "
            "Qwen-series models are excluded by project policy. Pass precomputed prompt "
            "embeddings with --prompt-embeds instead. See the TODO comments in "
            "structure_preserving_manga_editing/flux3_klein.py."
        )

    def encode(self, prompt: str) -> PromptEmbeddings:  # pragma: no cover - unreachable
        raise NotImplementedError


class PrecomputedPromptEncoder:
    """Prompt encoder backed by a safetensors file of precomputed embeddings.

    The file maps each prompt string to its ``(1, L, 20480)`` embedding tensor.
    Safetensors keys must be strings without special meaning, so prompts are
    stored under ``prompt_<index>`` keys and the prompt texts are kept in the
    file metadata under the same keys. Use :func:`save_prompt_embeddings` to
    write a compatible file.
    """

    def __init__(self, path: str | Path, device: str | torch.device = "cpu") -> None:
        from safetensors import safe_open

        self._embeddings: dict[str, torch.Tensor] = {}
        with safe_open(str(path), framework="pt", device=str(device)) as handle:
            metadata = handle.metadata() or {}
            for key in handle.keys():
                if key not in metadata:
                    raise ValueError(f"{path}: tensor {key!r} has no prompt text in the metadata")
                self._embeddings[metadata[key]] = handle.get_tensor(key)

    @property
    def prompts(self) -> list[str]:
        return list(self._embeddings)

    def encode(self, prompt: str) -> PromptEmbeddings:
        if prompt not in self._embeddings:
            raise KeyError(f"no precomputed embeddings for prompt {prompt!r}")
        return PromptEmbeddings(self._embeddings[prompt])


def save_prompt_embeddings(path: str | Path, embeddings: dict[str, torch.Tensor]) -> None:
    """Write ``{prompt: (1, L, 20480) tensor}`` in the PrecomputedPromptEncoder format."""
    from safetensors.torch import save_file

    tensors, metadata = {}, {}
    for index, (prompt, tensor) in enumerate(embeddings.items()):
        PromptEmbeddings(tensor)  # validate the shape
        key = f"prompt_{index}"
        tensors[key] = tensor.detach().contiguous().cpu()
        metadata[key] = prompt
    save_file(tensors, str(path), metadata=metadata)


def prepare_text_ids(sequence_length: int) -> torch.Tensor:
    """4D RoPE ids ``(t, h, w, l)`` of the text tokens; text advances along ``l``."""
    ids = torch.zeros(sequence_length, 4, dtype=torch.float32)
    ids[:, 3] = torch.arange(sequence_length, dtype=torch.float32)
    return ids


def prepare_image_ids(
    latent_height: int,
    latent_width: int,
    image_rope_scale: float = 0.5,
    l_coord: int = 0,
) -> torch.Tensor:
    """4D RoPE ids of an image token grid (``l=0`` output, ``l=-1`` first reference)."""
    ids = torch.cartesian_prod(
        torch.arange(1),
        torch.arange(latent_height),
        torch.arange(latent_width),
        torch.tensor([l_coord]),
    ).to(torch.float32)
    ids[:, 1:3] *= image_rope_scale
    return ids


def patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """``(B, C, H, W) -> (B, 4C, H/2, W/2)``, as the FLUX.2/3 VAE expects."""
    batch, channels, height, width = latents.shape
    latents = latents.view(batch, channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(batch, channels * 4, height // 2, width // 2)


def unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`patchify_latents`."""
    batch, channels, height, width = latents.shape
    latents = latents.reshape(batch, channels // 4, 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(batch, channels // 4, height * 2, width * 2)


def pack_tokens(latents: torch.Tensor) -> torch.Tensor:
    """``(B, C, H, W) -> (B, H*W, C)``."""
    return latents.flatten(2).transpose(1, 2)


def unpack_tokens(tokens: torch.Tensor, latent_height: int, latent_width: int) -> torch.Tensor:
    """``(B, H*W, C) -> (B, C, H, W)``."""
    batch, _, channels = tokens.shape
    return tokens.view(batch, latent_height, latent_width, channels).permute(0, 3, 1, 2)


def flow_map_targets(timesteps: torch.Tensor, step_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(t, t_next)`` in ``[0, 1]`` for a step of the scheduler's timesteps."""
    current = timesteps[step_index] / 1000
    if step_index + 1 < len(timesteps):
        following = timesteps[step_index + 1] / 1000
    else:
        following = torch.zeros_like(current)
    return current, following


class StructurePreservingFlux3KleinPipeline:
    """A one-image FLUX 3 Image Klein editing pipeline implementing Algorithm 1."""

    def __init__(
        self,
        transformer,
        vae,
        prompt_encoder: Flux3PromptEncoder,
        scheduler=None,
        image_processor=None,
        dtype: torch.dtype | None = None,
    ):
        from diffusers import FlowMatchEulerDiscreteScheduler
        from diffusers.modular_pipelines.flux3.image_processor import Flux3ImageProcessor

        if not transformer.config.flow_map:
            raise ValueError("this pipeline expects the step-distilled flow-map FLUX 3 Image Klein transformer")
        self.transformer = transformer
        self.vae = vae
        self.prompt_encoder = prompt_encoder
        # Compute dtype. Not `transformer.dtype`: GGUF weights are stored as uint8.
        self.dtype = dtype or transformer.dtype
        if not self.dtype.is_floating_point:
            raise ValueError("pass the compute dtype explicitly for a quantized transformer")
        self.scheduler = scheduler or FlowMatchEulerDiscreteScheduler(shift=FLUX3_KLEIN_SHIFT)
        self.vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        self.image_processor = image_processor or Flux3ImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.device = torch.device("cpu")

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        prompt_encoder: Flux3PromptEncoder,
        torch_dtype: torch.dtype = torch.bfloat16,
        transformer_gguf: str | None = None,
    ) -> "StructurePreservingFlux3KleinPipeline":
        """Load only the BFL transformer and VAE; the text encoder is never fetched."""
        from diffusers import AutoencoderKLFlux2, Flux3Transformer2DModel

        if transformer_gguf:
            from diffusers import GGUFQuantizationConfig

            transformer = Flux3Transformer2DModel.from_single_file(
                transformer_gguf,
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch_dtype),
                config=model_id,
                subfolder="transformer",
                torch_dtype=torch_dtype,
            )
        else:
            transformer = Flux3Transformer2DModel.from_pretrained(
                model_id, subfolder="transformer", torch_dtype=torch_dtype
            )
        # The checkpoint and the Diffusers pipeline both run the VAE in float32.
        vae = AutoencoderKLFlux2.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        return cls(transformer.eval(), vae.eval(), prompt_encoder, dtype=torch_dtype)

    def to(self, device: str | torch.device) -> "StructurePreservingFlux3KleinPipeline":
        self.device = torch.device(device)
        self.transformer.to(self.device)
        self.vae.to(self.device)
        return self

    def enable_low_vram(self, device: str | torch.device) -> "StructurePreservingFlux3KleinPipeline":
        """Stream transformer weights from CPU memory leaf by leaf; keep the small VAE on the GPU."""
        self.device = torch.device(device)
        self.transformer.enable_group_offload(
            onload_device=self.device,
            offload_device="cpu",
            offload_type="leaf_level",
            use_stream=True,
            low_cpu_mem_usage=True,
        )
        self.vae.to(self.device)
        return self

    def _vae_bn_stats(self, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        bn = self.vae.bn
        mean = bn.running_mean.view(1, -1, 1, 1).to(self.device, dtype)
        std = torch.sqrt(bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(self.device, dtype)
        return mean, std

    @torch.no_grad()
    def encode_image(self, input_image) -> tuple[torch.Tensor, tuple[int, int, int, int] | None]:
        """Encode a PIL image to normalized, patchified latents ``(1, 128, H/16, W/16)``."""
        image, postprocess_kwargs = self.image_processor.preprocess(input_image, resize_mode="fill")
        image = image.to(device=self.device, dtype=self.vae.dtype)
        latents = self.vae.encode(image).latent_dist.mode()
        latents = patchify_latents(latents)
        mean, std = self._vae_bn_stats(latents.dtype)
        return (latents - mean) / std, postprocess_kwargs["reference_crop"]

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor, reference_crop=None):
        mean, std = self._vae_bn_stats(torch.float32)
        latents = latents.to(torch.float32) * std + mean
        image = self.vae.decode(unpatchify_latents(latents).to(self.vae.dtype), return_dict=False)[0]
        return self.image_processor.postprocess(image, output_type="pil", reference_crop=reference_crop)[0]

    def _transformer(self, latents, prompt_embeds, timestep, target_timestep, image_ids, text_ids, cond_tokens, cond_ids):
        batch = latents.shape[0]
        return self.transformer(
            hidden_states=latents,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep.expand(batch).to(latents.dtype),
            target_timestep=target_timestep.to(latents.dtype),
            img_ids=image_ids,
            txt_ids=text_ids,
            image_cond_hidden_states=cond_tokens.expand(batch, -1, -1),
            image_cond_ids=cond_ids,
            return_dict=False,
        )[0]

    @torch.no_grad()
    def edit(
        self,
        input_image,
        prompt: str,
        generator: torch.Generator,
        num_inference_steps: int = FLUX3_KLEIN_NUM_INFERENCE_STEPS,
        correction_steps: int = 3,
        correction_strength: float = 0.01,
        prompt_embeds: PromptEmbeddings | None = None,
        empty_prompt_embeds: PromptEmbeddings | None = None,
    ):
        if correction_steps < 0:
            raise ValueError("correction_steps must be non-negative")
        if not 0.0 <= correction_strength <= 1.0:
            raise ValueError("correction_strength must be between 0 and 1")

        dtype = self.dtype
        if prompt_embeds is None:
            prompt_embeds = self.prompt_encoder.encode(prompt)
        if empty_prompt_embeds is None and correction_steps > 0:
            empty_prompt_embeds = self.prompt_encoder.encode("")
        edit_embeds = prompt_embeds.embeds.to(self.device, dtype)
        edit_text_ids = prepare_text_ids(edit_embeds.shape[1]).to(self.device)

        # Reference (input) image: clean conditioning tokens at l = -1.
        source, reference_crop = self.encode_image(input_image)
        latent_height, latent_width = source.shape[-2:]
        cond_tokens = pack_tokens(source).to(dtype)
        cond_ids = prepare_image_ids(latent_height, latent_width, l_coord=-1).to(self.device)
        image_ids = prepare_image_ids(latent_height, latent_width).to(self.device)

        # The generated image starts from noise on the same grid as the input,
        # so the source latents are the Algorithm 1 reconstruction anchor.
        noise = torch.randn(
            source.shape, generator=generator, device=generator.device, dtype=torch.float32
        ).to(self.device)
        latents = pack_tokens(noise).to(dtype)
        source_latents = cond_tokens.detach()

        # Pairing with the empty prompt in one batch needs equal text lengths.
        # Otherwise both branches run sequentially, which also lowers peak VRAM.
        if correction_steps > 0:
            empty_embeds = empty_prompt_embeds.embeds.to(self.device, dtype)
            empty_text_ids = prepare_text_ids(empty_embeds.shape[1]).to(self.device)
            can_batch = empty_embeds.shape[1] == edit_embeds.shape[1]
            if can_batch:
                dual_embeds = torch.cat([empty_embeds, edit_embeds], dim=0)

        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        self.scheduler.set_timesteps(sigmas=sigmas, device=self.device)
        self.scheduler.set_begin_index(0)
        timesteps = self.scheduler.timesteps
        terminal = torch.zeros((), device=self.device)

        for step_index, timestep in enumerate(timesteps):
            current, following = flow_map_targets(timesteps, step_index)
            if correction_is_active(step_index, correction_steps):
                # Empty-prompt branch jumps straight to the terminal time (t_target = 0).
                if can_batch:
                    velocity = self._transformer(
                        torch.cat([latents, latents]),
                        dual_embeds,
                        current,
                        torch.stack([terminal, following]),
                        image_ids,
                        edit_text_ids,
                        cond_tokens,
                        cond_ids,
                    )
                    empty_velocity, edit_velocity = velocity.chunk(2)
                else:
                    empty_velocity = self._transformer(
                        latents, empty_embeds, current, terminal.reshape(1),
                        image_ids, empty_text_ids, cond_tokens, cond_ids,
                    )
                    edit_velocity = self._transformer(
                        latents, edit_embeds, current, following.reshape(1),
                        image_ids, edit_text_ids, cond_tokens, cond_ids,
                    )
                latents = apply_trajectory_correction(
                    latents,
                    source_latents,
                    empty_velocity,
                    self.scheduler,
                    timestep,
                    correction_strength,
                )
            else:
                edit_velocity = self._transformer(
                    latents, edit_embeds, current, following.reshape(1),
                    image_ids, edit_text_ids, cond_tokens, cond_ids,
                )
            latents = self.scheduler.step(edit_velocity, timestep, latents, return_dict=False)[0]

        latents = unpack_tokens(latents, latent_height, latent_width)
        return self.decode_latents(latents, reference_crop)

