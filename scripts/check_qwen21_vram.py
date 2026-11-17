"""Check that the Qwen-Image 2.1 pipeline fits in GPU memory.

Runs one 1024x1024 text-removal edit (35 steps, 3 corrected) on a blank page
with the full low-VRAM pipeline (NF4 Qwen3-VL-8B text encoder, streamed GGUF
transformer, VAE) and reports load time, edit time, peak VRAM and the peak of
the CUDA cache.

Examples (from the repository root):

    python scripts/check_qwen21_vram.py --variant q8
    python scripts/check_qwen21_vram.py --variant q5 --kv-cache
    python scripts/check_qwen21_vram.py --variant q8 --steps 4   # quick check
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from structure_preserving_manga_editing.cli import load_pipeline  # noqa: E402
from structure_preserving_manga_editing.qwen_image21 import (  # noqa: E402
    QWEN_IMAGE21_NUM_INFERENCE_STEPS,
    StructurePreservingQwenImage21Pipeline,
)
from structure_preserving_manga_editing.tasks import (  # noqa: E402
    QWEN_IMAGE21_GGUF_Q4,
    QWEN_IMAGE21_GGUF_Q5,
    QWEN_IMAGE21_GGUF_Q8,
    QWEN_IMAGE21_MODEL_ID,
)

TRANSFORMERS = {"q8": QWEN_IMAGE21_GGUF_Q8, "q5": QWEN_IMAGE21_GGUF_Q5, "q4": QWEN_IMAGE21_GGUF_Q4}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=sorted(TRANSFORMERS), default="q8")
    parser.add_argument("--kv-cache", action="store_true", help="enable the prefix KV cache")
    parser.add_argument("--steps", type=int, default=QWEN_IMAGE21_NUM_INFERENCE_STEPS)
    parser.add_argument("--vram-cap-gib", type=float, help="limit this process's GPU memory (simulation)")
    parser.add_argument("--output", default="outputs/check_qwen21_vram.png")
    args = parser.parse_args()

    from diffusers import QwenImage21Transformer2DModel

    if args.vram_cap_gib:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(args.vram_cap_gib * 2**30 / total, 0)

    config = {
        "model_id": QWEN_IMAGE21_MODEL_ID,
        "transformer_gguf": TRANSFORMERS[args.variant],
        "low_vram": True,
        "device": "cuda",
    }
    start = time.time()
    pipeline = load_pipeline(
        StructurePreservingQwenImage21Pipeline,
        QwenImage21Transformer2DModel,
        "text_encoder",
        config,
        torch.bfloat16,
    )
    load_seconds = time.time() - start

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    result = pipeline.edit(
        Image.new("RGB", (1024, 1024), "white"),
        "Remove text. Keep everything else unchanged.",
        torch.Generator("cuda").manual_seed(0),
        num_inference_steps=args.steps,
        use_kv_cache=args.kv_cache,
    )
    torch.cuda.synchronize()
    print(
        f"{torch.cuda.get_device_name(0)} | variant={args.variant} kv_cache={args.kv_cache} "
        f"steps={args.steps} | load {load_seconds:.0f}s | edit {time.time() - start:.1f}s | "
        f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB | "
        f"peak cache {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB"
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)


if __name__ == "__main__":
    main()
