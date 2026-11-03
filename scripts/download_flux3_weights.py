"""Download the FLUX 3 Image Klein weights this project uses.

Fetches BFL's text encoder (Qwen3-VL-4B-Instruct, `text_encoder/`, 8.9 GB),
transformer and VAE, and the unsloth GGUF transformer(s). The prompt upsampler
(Qwen3.5-9B) is not used and never fetched.

Works the same on Windows (PowerShell / Anaconda Prompt) and Linux:

    python scripts/download_flux3_weights.py              # transformer + VAE + Q8_0
    python scripts/download_flux3_weights.py --gguf Q8_0 Q4_K_M
    python scripts/download_flux3_weights.py --no-bf16    # skip the 11 GB BF16 transformer
"""

from __future__ import annotations

import argparse

from huggingface_hub import hf_hub_download, snapshot_download

MODEL_ID = "black-forest-labs/flux-3-image-klein"
GGUF_REPO = "unsloth/flux-3-image-klein-GGUF"
BASE_PATTERNS = ["text_encoder/*", "vae/*", "transformer/config.json", "config.json", "LICENSE.md", "NOTICE.md", "README.md"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gguf", nargs="*", default=["Q8_0"], help="GGUF quant levels, e.g. Q8_0 Q4_K_M")
    parser.add_argument(
        "--no-bf16",
        action="store_true",
        help="skip the BF16 transformer weights (11 GB); the transformer config is still fetched",
    )
    args = parser.parse_args()

    patterns = list(BASE_PATTERNS)
    if not args.no_bf16:
        patterns.append("transformer/*")
    path = snapshot_download(MODEL_ID, allow_patterns=patterns)
    print(f"{MODEL_ID}: {path}")
    for level in args.gguf:
        print(hf_hub_download(GGUF_REPO, f"flux-3-image-klein-{level}.gguf"))


if __name__ == "__main__":
    main()
