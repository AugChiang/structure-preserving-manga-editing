"""Download the Qwen-Image 2.1 weights this project uses.

Fetches the text encoder (Qwen3-VL-8B, `text_encoder/`, 17.5 GB), processor,
VAE and scheduler of Qwen/Qwen-Image-2.1 and the chatrlabs GGUF transformer(s).
Only the config of the base BF16 transformer (14 GB) is fetched; its weights
are replaced by the GGUF.

Works the same on Windows (PowerShell / Anaconda Prompt) and Linux:

    python scripts/download_qwen21_weights.py              # base components + Q8_0
    python scripts/download_qwen21_weights.py --gguf Q8_0 Q5_K_M
"""

from __future__ import annotations

import argparse

from huggingface_hub import hf_hub_download, snapshot_download

MODEL_ID = "Qwen/Qwen-Image-2.1"
GGUF_REPO = "chatrlabs/Qwen-Image-2.1-Uncensored-Omni-Quantized"
BASE_PATTERNS = [
    "model_index.json",
    "processor/*",
    "scheduler/*",
    "text_encoder/*",
    "vae/*",
    "transformer/config.json",
    "LICENSE",
    "README.md",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gguf", nargs="*", default=["Q8_0"], help="GGUF quant levels, e.g. Q8_0 Q5_K_M Q4_K_M")
    args = parser.parse_args()

    print(f"{MODEL_ID}: {snapshot_download(MODEL_ID, allow_patterns=BASE_PATTERNS)}")
    for level in args.gguf:
        print(hf_hub_download(GGUF_REPO, f"qwen-image-2.1-step90000-{level}.gguf"))
    print(hf_hub_download(GGUF_REPO, "LICENSE"))


if __name__ == "__main__":
    main()
