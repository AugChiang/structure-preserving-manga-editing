# Project notes for Claude

## Hard rule: no Qwen or other Chinese-origin models

Do NOT download, load, or add automatic downloads of any Qwen-series model or
any other model from China. When a pipeline needs one, leave a placeholder and
`TODO(qwen-policy)` comments instead. This applies to the FLUX 3 Image Klein
text encoder (Qwen3-VL-4B-Instruct, in the repo's `text_encoder/` folder) and
its prompt upsampler (`flux-3-image-klein-pup`, Qwen3.5-9B). Download
FLUX 3 weights with allow-patterns that exclude `text_encoder/*`.

The pre-existing Qwen Image Edit backend (`qwen_image_edit.py`,
`demo_qwen_image_edit.py`) is the paper's original code. Leave it in place,
but do not run it or download its weights.

## Target hardware

Deployment and test target: RTX 4070, 12 GB VRAM (Ada, sm_89), on
**Windows 11** with **Miniconda**. NVFP4 is out (Blackwell only). Prefer GGUF
transformers. Write shell instructions that work in PowerShell: no bash line
continuations (`\`) and no `python -c` scripts that span several lines. Put
the logic in a script under `scripts/` instead.

## Environments

- `spme-flux3` (`conda env create -f environment-flux3.yml`): FLUX 3 work, diffusers 0.42.1.
  Miniconda has no `mamba`. Use `conda`, which uses the libmamba solver.
- `requirements.txt`: the pinned paper environment (diffusers 0.37); FLUX 3 does not run there.

Tests: `python -m unittest discover -s tests` (CPU only, no weights needed).

## Current work

FLUX 3 Image Klein integration: status and next steps are in
[docs/HANDOVER-flux3-klein.md](docs/HANDOVER-flux3-klein.md).
