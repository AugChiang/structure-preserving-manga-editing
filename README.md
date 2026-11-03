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

## Demo

Run all commands from the repository root. Input and output paths must be
relative to this directory. `--input` accepts a single image or a folder, in
which case every `.png`, `.jpg`, `.jpeg`, `.webp` and `.bmp` image directly
inside it is edited with one model load. Inputs are converted to RGB and
processed at about one megapixel with their aspect ratio preserved; a square
input is processed at 1024-by-1024 pixels as in the paper. Each result is
resized back to the size of its input.

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

Results are written to the `outputs/` folder under the same file names as
their inputs. The task selects the exact experimental prompt, Qwen checkpoint,
and number of inference steps. Both demos also accept:

```text
--output RELATIVE_FOLDER
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

The pipeline loads the official text encoder, Qwen3-VL-4B-Instruct
(`text_encoder/`, 8.9 GB), together with the transformer and the VAE. Each
batch encodes its task prompt and the empty prompt `""` once. The recommended
prompt upsampler, a Qwen3.5-9B fine-tune, is not integrated; the paper's exact
task prompts are used without upsampling.

FLUX 3 needs a newer Diffusers than the pinned paper environment, so it uses
its own conda environment:

```bash
conda env create -f environment-flux3.yml
conda activate spme-flux3
python scripts/download_flux3_weights.py
```

The download script fetches the BFL text encoder, transformer and VAE and the
Q8_0 GGUF. It and the commands below also work on
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
  --input path/to/folder \
  --transformer-gguf https://huggingface.co/unsloth/flux-3-image-klein-GGUF/blob/main/flux-3-image-klein-Q8_0.gguf \
  --low-vram
```

`--low-vram` streams the text encoder and the transformer from CPU memory,
which needs about 16 GB of free system RAM with the Q8_0 transformer. Without
it, all models sit on the GPU, which does not fit in 12 GB. On an RTX 4070
12 GB under Windows 11, this command edited a 900-by-1300 and a 1024-by-1024
page in about 130 s in total, including model loading, with either the Q8_0
or the Q4_K_M transformer. The peak allocation was 4.1 GiB and the CUDA cache
peaked at 7.3 GiB. Like the other backends, FLUX 3 processes each
input at about one megapixel and resizes the result back to the input size.

For text removal, FLUX 3 uses a more explicit prompt than the paper's
(`flux3_prompt` in `tasks.py`). With the short prompt "Remove text. Keep
everything else unchanged.", it guesses what counts as text and removed only
some lettering, or whole balloons, depending on the seed. The other backends
keep the paper's prompts. Results still vary with `--seed`: on the test pages
the explicit prompt removed all lettering in every run, but sometimes also
erased hatching or a balloon, so try a couple of seeds.

The correction defaults (3 steps, strength 0.01) are kept from FLUX.1 Kontext.
On the 16-step FLUX 3 schedule they have no visible effect. Strength 0.1
cancels the edit, and values in between change which elements get edited
rather than keeping the structure.

## Dataset Reproduction

The demos above operate on user-provided images. This section is a
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
