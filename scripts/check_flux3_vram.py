"""Check that the FLUX 3 Image Klein pipeline fits in GPU memory.

Runs one 1024x1024 edit (16 steps, 3 corrected) with *random* prompt
embeddings, because the Qwen3-VL text encoder is intentionally not integrated.
The output image is therefore meaningless; this only verifies loading, the
quantized transformer, the correction loop, speed and peak VRAM.

Examples (from the repository root, in the spme-flux3 environment):

    python scripts/check_flux3_vram.py --variant q8
    python scripts/check_flux3_vram.py --variant bf16 --low-vram
    # simulate a 12 GB card on a larger GPU:
    python scripts/check_flux3_vram.py --variant q8 --vram-cap-gib 11.5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from structure_preserving_manga_editing.flux3_klein import (  # noqa: E402
    FLUX3_KLEIN_GGUF_Q4,
    FLUX3_KLEIN_GGUF_Q8,
    FLUX3_KLEIN_MODEL_ID,
    FLUX3_TEXT_EMBED_DIM,
    PromptEmbeddings,
    StructurePreservingFlux3KleinPipeline,
)

TRANSFORMERS = {"q8": FLUX3_KLEIN_GGUF_Q8, "q4": FLUX3_KLEIN_GGUF_Q4, "bf16": None}


class RandomPromptEncoder:
    """Stand-in for the excluded Qwen3-VL encoder: random 80-token embeddings."""

    def encode(self, prompt: str) -> PromptEmbeddings:
        generator = torch.Generator().manual_seed(len(prompt))
        return PromptEmbeddings(torch.randn(1, 80, FLUX3_TEXT_EMBED_DIM, generator=generator) * 0.1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=sorted(TRANSFORMERS), default="q8")
    parser.add_argument("--low-vram", action="store_true", help="stream the transformer from CPU memory")
    parser.add_argument("--vram-cap-gib", type=float, help="limit this process's GPU memory (simulation)")
    parser.add_argument("--output", default="outputs/check_flux3_vram.png")
    args = parser.parse_args()

    if args.vram_cap_gib:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(args.vram_cap_gib * 2**30 / total, 0)

    start = time.time()
    pipeline = StructurePreservingFlux3KleinPipeline.from_pretrained(
        FLUX3_KLEIN_MODEL_ID,
        RandomPromptEncoder(),
        torch_dtype=torch.bfloat16,
        transformer_gguf=TRANSFORMERS[args.variant],
    )
    if args.low_vram:
        pipeline.enable_low_vram("cuda")
    else:
        pipeline.to("cuda")
    load_seconds = time.time() - start

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    result = pipeline.edit(
        Image.new("RGB", (1024, 1024), "white"),
        "Remove text. Keep everything else unchanged.",
        torch.Generator("cuda").manual_seed(0),
    )
    torch.cuda.synchronize()
    print(
        f"{torch.cuda.get_device_name(0)} | variant={args.variant} low_vram={args.low_vram} | "
        f"load {load_seconds:.0f}s | edit {time.time() - start:.1f}s | "
        f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)


if __name__ == "__main__":
    main()
