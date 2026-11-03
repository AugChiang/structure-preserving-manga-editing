"""Task prompts and default model checkpoints reported in the paper."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskConfig:
    prompt: str
    qwen_model_id: str
    # FLUX 3 Klein guesses what a terse prompt covers, so it may get a more
    # explicit one. The other backends always use the paper's prompt.
    flux3_prompt: str | None = None


TASKS = {
    "text-removal": TaskConfig(
        prompt="Remove text. Keep everything else unchanged.",
        qwen_model_id="Qwen/Qwen-Image-Edit-2509",
        flux3_prompt=(
            "Remove all text and lettering, including speech balloon dialogue and sound "
            "effects. Keep the empty speech balloons, characters, hatching, effect lines, "
            "panel borders and screentones unchanged."
        ),
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
