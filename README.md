# Inference-time Trajectory Optimization for Structure-Preserving Manga Image Editing

[![arXiv](https://img.shields.io/badge/arXiv-2603.27790-b31b1b.svg)](https://arxiv.org/abs/2603.27790)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Official implementation of [**Inference-time Trajectory Optimization for
Structure-Preserving Manga Image Editing**](https://arxiv.org/abs/2603.27790).

## Overview

Pretrained image-editing models can unintentionally alter the structure and
appearance of manga images. This repository implements a training-free sampler
correction that encourages an edit trajectory to reconstruct its input under
an empty prompt. The method:

- changes only the inference trajectory and does not update model parameters;
- adds the empty-prompt prediction as a batched model input during early steps;
- supports FLUX.1 Kontext and Qwen Image Edit; and
- is demonstrated on text removal and screentone synthesis.

No manga images, model weights, or font files are distributed in this
repository.

## Installation

Python 3.11 and a CUDA-capable PyTorch installation are recommended. From the
repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The tested package versions are pinned in `requirements.txt`.

## Model Access

Authenticate with Hugging Face before running the demos:

```bash
hf auth login
```

The default checkpoints are:

- [FLUX.1 Kontext [dev]](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
  for both tasks;
- [Qwen Image Edit 2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)
  for text removal; and
- [Qwen Image Edit 2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
  for screentone synthesis.

FLUX.1 Kontext [dev] is gated and requires accepting its access conditions.
Its model weights are subject to the FLUX.1 [dev] Non-Commercial License.
Qwen Image Edit 2509 and 2511 are provided under Apache License 2.0. Always
review the current model cards and terms before use.

## One-image Demo

Run all commands from the repository root. Input and output paths must be
relative to this directory. Inputs are converted to RGB and resized to
1024-by-1024 pixels.

Text removal with FLUX.1 Kontext:

```bash
python demo_flux_kontext.py \
  --task text-removal \
  --input path/to/input.png
```

Screentone synthesis with FLUX.1 Kontext:

```bash
python demo_flux_kontext.py \
  --task screentone-synthesis \
  --input path/to/line_art.png
```

Text removal with Qwen Image Edit 2509:

```bash
python demo_qwen_image_edit.py \
  --task text-removal \
  --input path/to/input.png
```

Screentone synthesis with Qwen Image Edit 2511:

```bash
python demo_qwen_image_edit.py \
  --task screentone-synthesis \
  --input path/to/line_art.png
```

Default outputs are written under `outputs/`. The task selects the exact
experimental prompt, Qwen checkpoint, and number of inference steps. Both
demos also accept:

```text
--output RELATIVE_PATH
--seed INTEGER
--device DEVICE
--model-id MODEL_ID
--correction-steps INTEGER
--correction-strength FLOAT
```

Use `--dry-run` to validate and print the resolved configuration without
loading a model:

```bash
python demo_flux_kontext.py \
  --task text-removal \
  --input path/to/input.png \
  --dry-run
```

The default prompts are:

```text
Text removal:
Remove text. Keep everything else unchanged.

Screentone synthesis:
Convert the line art into a finished Japanese black and white monochrome manga image.
```

### Running on a 12 GB GPU

The full-precision checkpoints need far more than 12 GB of VRAM. Two optional
flags reduce memory use: `--transformer-gguf` replaces the transformer with a
GGUF-quantized one, and `--low-vram` loads the text encoder in 4-bit NF4,
keeps the transformer weights in CPU memory and streams them to the GPU layer
by layer, and offloads the other models while they are idle. The transformer
therefore needs CPU RAM rather than VRAM. Install the extra dependencies
first:

```bash
python -m pip install gguf bitsandbytes
```

FLUX.1 Kontext with a Q4_K_S transformer (6.8 GB) fits comfortably:

```bash
python demo_flux_kontext.py \
  --task text-removal \
  --input path/to/input.png \
  --transformer-gguf https://huggingface.co/QuantStack/FLUX.1-Kontext-dev-GGUF/blob/main/flux1-kontext-dev-Q4_K_S.gguf \
  --low-vram
```

For Qwen Image Edit, pick the GGUF that matches the task's checkpoint. Q4_K_M
needs about 13 GB of free CPU RAM; use Q3_K_S if RAM is short:

```bash
python demo_qwen_image_edit.py \
  --task text-removal \
  --input path/to/input.png \
  --transformer-gguf https://huggingface.co/QuantStack/Qwen-Image-Edit-2509-GGUF/blob/main/Qwen-Image-Edit-2509-Q4_K_M.gguf \
  --low-vram

python demo_qwen_image_edit.py \
  --task screentone-synthesis \
  --input path/to/line_art.png \
  --transformer-gguf https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF/blob/main/qwen-image-edit-2511-Q4_K_M.gguf \
  --low-vram
```

Quantized results may differ from those reported in the paper.

## FLUX 3 Image Klein (experimental)

[FLUX 3 Image Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein)
is a 5.6B step-distilled editing model (16 steps, no guidance). It is a
flow-map model: each transformer call also receives the time the step jumps
to. During the correction steps, the empty-prompt branch jumps straight to the
terminal time, which is the one-step reconstruction Algorithm 1 uses. The edit
branch keeps the regular next-step target.

**Text encoder not included.** The official text encoder is
Qwen3-VL-4B-Instruct, and the recommended prompt upsampler is a Qwen3.5-9B
fine-tune. This project does not download or load Qwen-series models, so
`Qwen3VLPromptEncoderPlaceholder` in
`structure_preserving_manga_editing/flux3_klein.py` raises
`NotImplementedError` and holds `TODO(qwen-policy)` notes. Only the BFL
transformer and VAE are fetched. A real edit therefore needs prompt embeddings
computed elsewhere and passed with `--prompt-embeds` (write the file with
`save_prompt_embeddings`). The file must contain both the task prompt and the
empty prompt `""`. Without it, the demo stops before loading any model.

FLUX 3 needs a newer Diffusers than the pinned paper environment, so it uses
its own conda environment:

```bash
conda env create -f environment-flux3.yml
conda activate spme-flux3
python scripts/download_flux3_weights.py
```

The download script fetches only the BFL transformer, the VAE and the Q8_0
GGUF, never the Qwen text encoder. It and the commands below also work on
Windows 11 (Anaconda Prompt). In PowerShell, put each command on one line
instead of using `\` line continuations.

The FLUX 3 Image Klein weights are released under the FLUX Kommunity License
(non-commercial use, with a commercial-output exception for Qualifying Users).
Accept the access conditions on the model page first.

### Quantized transformers for a 12 GB GPU (RTX 4070)

| Transformer | Size | Runs on RTX 4070 (Ada) | Notes |
| --- | --- | --- | --- |
| BF16 (official) | 11.1 GB | with `--low-vram` only | streamed from CPU RAM |
| [unsloth Q8_0 GGUF](https://huggingface.co/unsloth/flux-3-image-klein-GGUF) | 6.1 GB | yes | recommended default |
| [unsloth Q4_K_M GGUF](https://huggingface.co/unsloth/flux-3-image-klein-GGUF) | 3.7 GB | yes | more headroom, lower quality |
| BFL `variants/fp8r` | 6.3 GB | hardware yes | BFL's own format; no Diffusers loader |
| BFL `variants/nvfp4` | 5.3 GB | no | needs a Blackwell GPU |

Other GGUF builds exist (for example `leejet/flux-3-image-klein-GGUF`), as do
Nunchaku SVDQuant and ComfyUI INT8 builds. Only the unsloth GGUFs were tested
with this code.

```bash
python demo_flux3_klein.py \
  --task text-removal \
  --input path/to/input.png \
  --prompt-embeds path/to/flux3_prompt_embeds.safetensors \
  --transformer-gguf https://huggingface.co/unsloth/flux-3-image-klein-GGUF/blob/main/flux-3-image-klein-Q8_0.gguf
```

Add `--low-vram` to stream the transformer from CPU memory instead. The
default of 3 correction steps comes from the 28-step FLUX.1 Kontext setting.
It has not been tuned for the 16-step FLUX 3 schedule.

## Dataset Reproduction

The one-image demos above operate on user-provided images. This section is a
separate, optional procedure for reconstructing the two experimental datasets
from Manga109-s. Dataset preparation is not required to run a demo.

### Prepare Manga109-s

Download [Manga109-s](https://manga109.github.io/manga109-project-website/en/)
manually and arrange it as follows:

```text
data/Manga109s/
  annotations/
    TITLE.xml
  images/
    TITLE/
      000.jpg
      ...
```

Manga109-s is not included here and remains subject to its own terms. In
particular, do not commit derived manga crops to this repository.

All preprocessing commands use sorted input order. Synthetic text placement
and font selection use seed 0 by default.

### Text-removal dataset

1. Extract candidate crops that do not overlap Manga109-s `text` boxes:

   ```bash
   python preprocessing/text_removal/extract_clean_patches.py
   ```

2. Inspect candidates and mark crops that still contain text:

   ```bash
   python preprocessing/text_removal/annotate_text_presence.py --start-unlabeled
   ```

   In the browser, Up marks “has text,” Down marks “no text,” Left and Right
   navigate, and `S` saves.

3. Copy the accepted text-free crops:

   ```bash
   python preprocessing/text_removal/select_no_text.py
   ```

4. Download the 24 fonts listed in
   `preprocessing/text_removal/fonts.txt`:

   ```bash
   python preprocessing/text_removal/download_fonts.py
   ```

5. Render synthetic vertical text and its exact pixel mask:

   ```bash
   python preprocessing/text_removal/render_synthetic_text.py \
     --save-mask \
     --mask-dir data/derived/text_removal/masks
   ```

Manual inspection produced 3,204 accepted crops in the paper. Reproducing the
exact count requires applying the same human acceptance criterion.

### Screentone-synthesis dataset

The six experiment titles are listed in
`preprocessing/screentone_synthesis/titles.txt`.

1. Extract body bounding boxes from those titles:

   ```bash
   python preprocessing/screentone_synthesis/extract_body_patches.py
   ```

2. Convert the finished crops to structural line images:

   ```bash
   python preprocessing/screentone_synthesis/extract_line_art.py
   ```

The crop settings produce 1,355 pairs. The second command downloads
[p1atdev/MangaLineExtraction-hf](https://huggingface.co/p1atdev/MangaLineExtraction-hf)
through Transformers and executes its remote model code. Review that code and
its MIT License before setting `trust_remote_code=True`.

## Testing

Run the lightweight CPU test suite with:

```bash
python -m unittest discover -s tests -v
```

Validate all four task configurations with any local image:

```bash
python demo_flux_kontext.py --task text-removal --input path/to/input.png --dry-run
python demo_flux_kontext.py --task screentone-synthesis --input path/to/input.png --dry-run
python demo_qwen_image_edit.py --task text-removal --input path/to/input.png --dry-run
python demo_qwen_image_edit.py --task screentone-synthesis --input path/to/input.png --dry-run
```

The FLUX 3 tests (`tests/test_flux3_klein.py`) need the `spme-flux3`
environment. They run a tiny, randomly initialized FLUX 3 transformer and VAE
on the CPU, so no weights are downloaded.

## Citation

If this work is useful in your research, please cite:

```bibtex
@misc{furuta2026inference,
  title         = {Inference-time Trajectory Optimization for Structure-Preserving Manga Image Editing},
  author        = {Ryosuke Furuta},
  year          = {2026},
  eprint        = {2603.27790},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2603.27790}
}
```

## Third-party Components

This repository uses Diffusers and externally downloaded model and font
resources. Their licenses are independent of this repository's license. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for source links and
applicable terms.

## License

The original code in this repository is released under the
[Apache License 2.0](LICENSE). This license does not apply to pretrained model
weights, Manga109-s, downloaded fonts, or other third-party materials.

## Acknowledgements

The initial implementation was developed based on the
[FlowChef codebase](https://github.com/FlowChef/FlowChef). We thank the
FlowChef authors for making their work available.
