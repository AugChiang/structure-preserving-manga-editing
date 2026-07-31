# Copyright 2025 Black Forest Labs and The HuggingFace Team. All rights reserved.
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

"""FLUX.1 Kontext integration for inference-time trajectory correction."""

from __future__ import annotations

import numpy as np
import torch
from diffusers import FluxKontextPipeline
from diffusers.pipelines.flux.pipeline_flux_kontext import (
    calculate_shift,
    retrieve_timesteps,
)

from .core import apply_trajectory_correction, correction_is_active


class StructurePreservingFluxKontextPipeline(FluxKontextPipeline):
    """A one-image FLUX.1 Kontext pipeline implementing Algorithm 1."""

    @torch.no_grad()
    def edit(
        self,
        input_image,
        prompt: str,
        generator: torch.Generator,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        correction_steps: int = 3,
        correction_strength: float = 0.01,
    ):
        if correction_steps < 0:
            raise ValueError("correction_steps must be non-negative")
        if not 0.0 <= correction_strength <= 1.0:
            raise ValueError("correction_strength must be between 0 and 1")

        height = width = 1024
        device = self._execution_device
        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = {}
        self._current_timestep = None
        self._interrupt = False

        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )
        empty_prompt_embeds, empty_pooled_prompt_embeds, _ = self.encode_prompt(
            prompt="",
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )

        image_tensor = self.image_processor.resize(input_image, height, width)
        image_tensor = self.image_processor.preprocess(image_tensor, height, width)

        num_channels_latents = self.transformer.config.in_channels // 4
        latents, image_latents, latent_ids, image_ids = self.prepare_latents(
            image_tensor,
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
        if image_ids is not None:
            latent_ids = torch.cat([latent_ids, image_ids], dim=0)

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
            guidance_single = torch.full(
                [1],
                guidance_scale,
                device=device,
                dtype=torch.float32,
            )
            guidance_dual = guidance_single.expand(2)
        else:
            guidance_single = None
            guidance_dual = None

        dual_prompt_embeds = torch.cat([empty_prompt_embeds, prompt_embeds], dim=0)
        dual_pooled_embeds = torch.cat(
            [empty_pooled_prompt_embeds, pooled_prompt_embeds],
            dim=0,
        )
        dual_image_latents = torch.cat([image_latents, image_latents], dim=0)

        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for step_index, timestep in enumerate(timesteps):
                self._current_timestep = timestep
                correct_this_step = correction_is_active(
                    step_index,
                    correction_steps,
                )
                if correct_this_step:
                    model_latents = torch.cat([latents, latents], dim=0)
                    model_latents = torch.cat([model_latents, dual_image_latents], dim=1)
                    step_prompt_embeds = dual_prompt_embeds
                    step_pooled_embeds = dual_pooled_embeds
                    step_guidance = guidance_dual
                else:
                    model_latents = torch.cat([latents, image_latents], dim=1)
                    step_prompt_embeds = prompt_embeds
                    step_pooled_embeds = pooled_prompt_embeds
                    step_guidance = guidance_single

                model_timestep = timestep.expand(model_latents.shape[0]).to(model_latents.dtype)
                velocity = self.transformer(
                    hidden_states=model_latents,
                    timestep=model_timestep / 1000,
                    guidance=step_guidance,
                    pooled_projections=step_pooled_embeds,
                    encoder_hidden_states=step_prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]
                velocity = velocity[:, : latents.size(1)]

                if correct_this_step:
                    empty_velocity, edit_velocity = velocity.chunk(2)
                    latents = apply_trajectory_correction(
                        latents,
                        source_latents,
                        empty_velocity,
                        self.scheduler,
                        timestep,
                        correction_strength,
                    )
                else:
                    edit_velocity = velocity

                latents = self.scheduler.step(
                    edit_velocity,
                    timestep,
                    latents,
                    return_dict=False,
                )[0]
                progress_bar.update()

        self._current_timestep = None
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type="pil")
        self.maybe_free_model_hooks()
        return image[0]
