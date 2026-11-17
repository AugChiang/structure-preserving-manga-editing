# Copyright 2025 Qwen-Image Team and The HuggingFace Team. All rights reserved.
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
# Structure-Preserving Manga Image Editing." The preprocessing, latent layout
# and sampling follow the Diffusers Qwen-Image 2.1 pipeline (diffusers 0.42.1,
# `diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py`).

"""Qwen-Image 2.1 integration for inference-time trajectory correction.

Qwen-Image 2.1 differs from Qwen Image Edit 2509/2511 in ways that matter here:

- latents are fed to the transformer unpatched, one token per 16x16 pixels;
- condition-image tokens come *before* the target tokens, so the prediction for
  the target is the last ``N`` tokens of the output;
- the transformer needs an ``img_mask`` marking the image slots of the text
  encoder's sequence (each slot is a 2x2 group of latent tokens), followed by
  one slot per 2x2 group of target tokens; and
- guidance is plain true CFG (no norm rescale), with 35 steps and scale 5.0
  recommended for the checkpoints used here.

The empty-prompt branch serves both as the CFG negative and as the Algorithm 1
reconstruction velocity. Diffusers encodes ``""`` as ``" "``, which is also the
negative prompt the chatrlabs model card recommends.
"""

from __future__ import annotations

import numpy as np
import torch
from diffusers import QwenImage21Pipeline
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import (
    calculate_dimensions,
    calculate_shift,
    retrieve_timesteps,
)

from .core import apply_trajectory_correction, correction_is_active

QWEN_IMAGE21_NUM_INFERENCE_STEPS = 35
QWEN_IMAGE21_TRUE_CFG_SCALE = 5.0
QWEN_IMAGE21_RESOLUTION = 1024


def _pad_prompt(
    embeds: torch.Tensor,
    mask: torch.Tensor | None,
    image_pad_mask: torch.Tensor,
    length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad one encoded prompt (embeddings, valid mask, image-slot mask) to ``length``."""
    # encode_prompt returns no mask when every token is valid.
    if mask is None:
        mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device)
    amount = length - embeds.shape[1]
    if amount:
        embeds = torch.cat([embeds, embeds.new_zeros(embeds.shape[0], amount, embeds.shape[2])], dim=1)
        mask = torch.cat([mask, mask.new_zeros(mask.shape[0], amount)], dim=1)
        image_pad_mask = torch.cat(
            [image_pad_mask, image_pad_mask.new_zeros(image_pad_mask.shape[0], amount)], dim=1
        )
    return embeds, mask, image_pad_mask


class StructurePreservingQwenImage21Pipeline(QwenImage21Pipeline):
    """A one-image Qwen-Image 2.1 editing pipeline implementing Algorithm 1."""

    def _velocity(self, latent_input, timestep, embeds, mask, img_mask, img_shapes, cache, step_index):
        if cache is None:
            mode = None
        else:
            mode = "extract" if step_index == 0 else "cached"
        batch = embeds.shape[0]
        output = self.transformer(
            hidden_states=latent_input.expand(batch, -1, -1),
            timestep=timestep.expand(batch).to(latent_input.dtype) / 1000,
            encoder_hidden_states=embeds,
            encoder_hidden_states_mask=mask,
            img_shapes=img_shapes * batch,
            img_mask=img_mask,
            attention_kwargs=self.attention_kwargs,
            kv_cache=cache,
            kv_cache_mode=mode,
            return_dict=False,
        )[0]
        return output

    @torch.no_grad()
    def edit(
        self,
        input_image,
        prompt: str,
        generator: torch.Generator,
        num_inference_steps: int = QWEN_IMAGE21_NUM_INFERENCE_STEPS,
        true_cfg_scale: float = QWEN_IMAGE21_TRUE_CFG_SCALE,
        correction_steps: int = 3,
        correction_strength: float = 0.01,
        use_kv_cache: bool = False,
        resolution: int = QWEN_IMAGE21_RESOLUTION,
    ):
        if correction_steps < 0:
            raise ValueError("correction_steps must be non-negative")
        if not 0.0 <= correction_strength <= 1.0:
            raise ValueError("correction_strength must be between 0 and 1")

        device = self._execution_device
        self._attention_kwargs = {}
        self._current_timestep = None
        self._interrupt = False

        # One size for the condition image and the output: the trajectory
        # correction needs identical input and output latent grids. A square
        # input is processed at 1024 by 1024 pixels.
        width, height, _ = calculate_dimensions(resolution * resolution, input_image.width / input_image.height)
        multiple_of = self.vae_scale_factor * 2
        width, height = width // multiple_of * multiple_of, height // multiple_of * multiple_of
        # The 2.1 VAE reads RGBA; the text encoder composites the alpha over white.
        rgba = input_image.convert("RGBA")
        condition_image = self.image_processor.resize(rgba, width=width, height=height)
        vae_image = self.image_processor.preprocess(rgba, width=width, height=height).unsqueeze(2)

        prompt_embeds, prompt_mask, prompt_slots = self.encode_prompt(
            prompt=prompt, image=[condition_image], device=device
        )
        empty_embeds, empty_mask, empty_slots = self.encode_prompt(
            prompt="", image=[condition_image], device=device
        )
        length = max(prompt_embeds.shape[1], empty_embeds.shape[1])
        prompt_embeds, prompt_mask, prompt_slots = _pad_prompt(prompt_embeds, prompt_mask, prompt_slots, length)
        empty_embeds, empty_mask, empty_slots = _pad_prompt(empty_embeds, empty_mask, empty_slots, length)

        latents, image_latents = self.prepare_latents(
            [vae_image],
            1,
            self.transformer.config.in_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            None,
        )
        if image_latents is None or image_latents.shape != latents.shape:
            raise ValueError("the input and output latent grids must have identical shapes")
        source_latents = image_latents.detach()
        target_tokens = latents.shape[1]

        grid = (1, height // self.vae_scale_factor, width // self.vae_scale_factor)
        img_shapes = [[grid, grid]]  # condition image first, target last

        def with_target_slots(slots: torch.Tensor) -> torch.Tensor:
            return torch.cat([slots, slots.new_ones(slots.shape[0], target_tokens // 4)], dim=1)

        prompt_slots, empty_slots = with_target_slots(prompt_slots), with_target_slots(empty_slots)

        # The transformer reads the token layout from row 0 of `img_mask`, so
        # both prompts share one batch only if their image slots line up. They
        # do whenever the image precedes the prompt text, as in the 2.1 template.
        can_batch = torch.equal(prompt_slots, empty_slots)
        if can_batch:
            branches = [
                (
                    torch.cat([empty_embeds, prompt_embeds]),
                    torch.cat([empty_mask, prompt_mask]),
                    torch.cat([empty_slots, prompt_slots]),
                )
            ]
        else:
            branches = [(empty_embeds, empty_mask, empty_slots), (prompt_embeds, prompt_mask, prompt_slots)]
        use_kv_cache = use_kv_cache and self.transformer.config.causal_condition
        num_blocks = len(self.transformer.transformer_blocks)
        caches = [QwenImage21KVCache(num_blocks) if use_kv_cache else None for _ in branches]

        sigmas = self.config.sample_sigmas
        if sigmas is None:
            sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        mu = calculate_shift(
            target_tokens,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, _ = retrieve_timesteps(self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)

        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for step_index, timestep in enumerate(timesteps):
                self._current_timestep = timestep
                latent_input = torch.cat([image_latents, latents], dim=1)
                velocity = torch.cat(
                    [
                        self._velocity(latent_input, timestep, *branch, img_shapes, cache, step_index)
                        for branch, cache in zip(branches, caches)
                    ]
                )[:, -target_tokens:]
                empty_velocity, edit_velocity = velocity.chunk(2)
                guided_velocity = empty_velocity + true_cfg_scale * (edit_velocity - empty_velocity)

                if correction_is_active(step_index, correction_steps):
                    latents = apply_trajectory_correction(
                        latents,
                        source_latents,
                        empty_velocity,
                        self.scheduler,
                        timestep,
                        correction_strength,
                    )

                latents = self.scheduler.step(guided_velocity, timestep, latents, return_dict=False)[0]
                progress_bar.update()

        del caches
        self._current_timestep = None
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor).to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents * latents_std + latents_mean
        image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
        image = self.image_processor.postprocess(image, output_type="pil")
        self.maybe_free_model_hooks()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return image[0].convert("RGB")
