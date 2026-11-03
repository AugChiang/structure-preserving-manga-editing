# Project notes for Claude

## Hard rule: Qwen and other Chinese-origin models only on the RTX 4070 PC

The FLUX 3 Image Klein pipeline loads its official text encoder,
Qwen3-VL-4B-Instruct (the repo's `text_encoder/` folder), as part of the
pipeline. The RTX 4070 PC is allowed to download and run it.

On any other machine, do NOT download, load, or add automatic downloads of
any Qwen-series model or any other model from China. There, do not run the
FLUX 3 demo, `scripts/download_flux3_weights.py` or
`scripts/check_flux3_vram.py`; CPU unit tests are fine. The prompt upsampler
(`flux-3-image-klein-pup`, Qwen3.5-9B) is not used anywhere.

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

- FLUX 3 work needs diffusers >= 0.42. The Miniconda environment on the RTX 4070
  PC is already set up, so do not recreate it. `environment-flux3.yml` records
  the versions tested on the development PC.
- `requirements.txt`: the pinned paper environment (diffusers 0.37); FLUX 3 does not run there.

Tests: `python -m unittest discover -s tests` (CPU only, no weights needed).

## Current work

FLUX 3 Image Klein integration: status and next steps are in
[docs/HANDOVER-flux3-klein.md](docs/HANDOVER-flux3-klein.md).
