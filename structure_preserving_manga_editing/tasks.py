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

# Qwen-Image 2.1: the text encoder (Qwen3-VL-8B), processor and VAE come from
# the base repository; the transformer is replaced by a quantized fine-tune.
QWEN_IMAGE21_MODEL_ID = "Qwen/Qwen-Image-2.1"
_CHATRLABS_QWEN_IMAGE21 = (
    "https://huggingface.co/chatrlabs/Qwen-Image-2.1-Uncensored-Omni-Quantized/blob/main/"
    "qwen-image-2.1-step90000-{}.gguf"
)
QWEN_IMAGE21_GGUF_Q8 = _CHATRLABS_QWEN_IMAGE21.format("Q8_0")
QWEN_IMAGE21_GGUF_Q5 = _CHATRLABS_QWEN_IMAGE21.format("Q5_K_M")
QWEN_IMAGE21_GGUF_Q4 = _CHATRLABS_QWEN_IMAGE21.format("Q4_K_M")
QWEN_IMAGE21_TRANSFORMER_GGUF = QWEN_IMAGE21_GGUF_Q8
