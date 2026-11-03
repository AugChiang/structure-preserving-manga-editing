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
# Structure-Preserving Manga Image Editing."

"""Qwen Image Edit integration for inference-time trajectory correction."""

from __future__ import annotations

import numpy as np
import torch
from diffusers import QwenImageEditPlusPipeline
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
    CONDITION_IMAGE_SIZE,
    VAE_IMAGE_SIZE,
    calculate_dimensions,
    calculate_shift,
    retrieve_timesteps,
)

from .core import apply_trajectory_correction, correction_is_active


def _pad_prompt_pair(
    first_embeds: torch.Tensor,
    first_mask: torch.Tensor | None,
    second_embeds: torch.Tensor,
    second_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target_length = max(first_embeds.shape[1], second_embeds.shape[1])

    def pad(embeds: torch.Tensor, mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        # encode_prompt returns no mask when every token is valid.
        if mask is None:
            mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device)
        amount = target_length - embeds.shape[1]
        if amount == 0:
            return embeds, mask
        embeds = torch.cat(
            [embeds, embeds.new_zeros(embeds.shape[0], amount, embeds.shape[2])],
            dim=1,
        )
        mask = torch.cat([mask, mask.new_zeros(mask.shape[0], amount)], dim=1)
        return embeds, mask

    first_embeds, first_mask = pad(first_embeds, first_mask)
    second_embeds, second_mask = pad(second_embeds, second_mask)
    return first_embeds, first_mask, second_embeds, second_mask


class StructurePreservingQwenImageEditPipeline(QwenImageEditPlusPipeline):
    """A one-image Qwen Image Edit pipeline implementing Algorithm 1."""

    @torch.no_grad()
    def edit(
        self,
        input_image,
        prompt: str,
        generator: torch.Generator,
        num_inference_steps: int = 40,
        guidance_scale: float = 1.0,
        true_cfg_scale: float = 4.0,
        correction_steps: int = 3,
        correction_strength: float = 0.01,
    ):
        if correction_steps < 0:
            raise ValueError("correction_steps must be non-negative")
        if not 0.0 <= correction_strength <= 1.0:
            raise ValueError("correction_strength must be between 0 and 1")

        device = self._execution_device
        self._guidance_scale = guidance_scale
        self._attention_kwargs = {}
        self._current_timestep = None
        self._interrupt = False

        # Keep the input aspect ratio at about one megapixel. The output and
        # input latent grids must match for the trajectory correction, and a
        # square input is processed at 1024 by 1024 pixels as in the paper.
        aspect_ratio = input_image.width / input_image.height
        condition_width, condition_height = calculate_dimensions(CONDITION_IMAGE_SIZE, aspect_ratio)
        vae_width, vae_height = calculate_dimensions(VAE_IMAGE_SIZE, aspect_ratio)
        width, height = vae_width, vae_height
        condition_image = self.image_processor.resize(
            input_image,
            condition_height,
            condition_width,
        )
        vae_image = self.image_processor.preprocess(
            input_image,
            vae_height,
            vae_width,
        ).unsqueeze(2)

        prompt_embeds, prompt_mask = self.encode_prompt(
            image=[condition_image],
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=1024,
        )
        empty_embeds, empty_mask = self.encode_prompt(
            image=[condition_image],
            prompt="",
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=1024,
        )
        empty_embeds, empty_mask, prompt_embeds, prompt_mask = _pad_prompt_pair(
            empty_embeds,
            empty_mask,
            prompt_embeds,
            prompt_mask,
        )

        num_channels_latents = self.transformer.config.in_channels // 4
        vae_image = vae_image.to(device=device, dtype=prompt_embeds.dtype)
        encoded_input = self._encode_vae_image(vae_image, generator)
        latents, image_latents = self.prepare_latents(
            [encoded_input],
            1,
            num_channels_latents,
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

        img_shapes = [
            [
                (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2),
                (
                    1,
                    vae_height // self.vae_scale_factor // 2,
                    vae_width // self.vae_scale_factor // 2,
                ),
            ]
        ]
        dual_img_shapes = img_shapes + img_shapes

        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, _ = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )

        if self.transformer.config.guidance_embeds:
            guidance = torch.full(
                [2],
                guidance_scale,
                device=device,
                dtype=torch.float32,
            )
        else:
            guidance = None

        dual_prompt_embeds = torch.cat([empty_embeds, prompt_embeds], dim=0)
        dual_prompt_mask = torch.cat([empty_mask, prompt_mask], dim=0)
        dual_image_latents = torch.cat([image_latents, image_latents], dim=0)

        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for step_index, timestep in enumerate(timesteps):
                model_latents = torch.cat([latents, latents], dim=0)
                model_latents = torch.cat([model_latents, dual_image_latents], dim=1)
                model_timestep = timestep.expand(2).to(model_latents.dtype)

                velocity = self.transformer(
                    hidden_states=model_latents,
                    timestep=model_timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=dual_prompt_mask,
                    encoder_hidden_states=dual_prompt_embeds,
                    img_shapes=dual_img_shapes,
                    attention_kwargs=self.attention_kwargs,
                    return_dict=False,
                )[0]
                empty_velocity, edit_velocity = velocity[:, : latents.size(1)].chunk(2)

                guided_velocity = empty_velocity + true_cfg_scale * (
                    edit_velocity - empty_velocity
                )
                edit_norm = torch.norm(edit_velocity, dim=-1, keepdim=True)
                guided_norm = torch.norm(guided_velocity, dim=-1, keepdim=True).clamp_min(1e-12)
                guided_velocity = guided_velocity * (edit_norm / guided_norm)

                if correction_is_active(step_index, correction_steps):
                    latents = apply_trajectory_correction(
                        latents,
                        source_latents,
                        empty_velocity,
                        self.scheduler,
                        timestep,
                        correction_strength,
                    )

                latents = self.scheduler.step(
                    guided_velocity,
                    timestep,
                    latents,
                    return_dict=False,
                )[0]
                progress_bar.update()

        self._current_timestep = None
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            1.0
            / torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents / latents_std + latents_mean
        image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
        image = self.image_processor.postprocess(image, output_type="pil")
        self.maybe_free_model_hooks()
        return image[0]
