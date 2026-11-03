"""Task prompts and default model checkpoints reported in the paper."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskConfig:
    prompt: str
    qwen_model_id: str


TASKS = {
    "text-removal": TaskConfig(
        prompt="Remove text. Keep everything else unchanged.",
        qwen_model_id="Qwen/Qwen-Image-Edit-2509",
    ),
    "bar-removal": TaskConfig(
            prompt="Remove semi-transparent black bars. Restore the screentones, halftones, lines, and shading underneath. Keep everything else unchanged.",
            qwen_model_id="Qwen/Qwen-Image-Edit-2509",
        ),
    "screentone-synthesis": TaskConfig(
        prompt="Convert the line art into a finished Japanese black and white monochrome manga image.",
        qwen_model_id="Qwen/Qwen-Image-Edit-2511",
    ),
}

FLUX_MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"
FLUX3_KLEIN_MODEL_ID = "black-forest-labs/flux-3-image-klein"
