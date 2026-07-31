# Inference-time Trajectory Optimization for Manga Image Editing

[![arXiv](https://img.shields.io/badge/arXiv-2603.27790-b31b1b.svg)](https://arxiv.org/abs/2603.27790)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Official implementation of **Inference-time Trajectory Optimization for Manga
Image Editing** by [Ryosuke Furuta](https://arxiv.org/abs/2603.27790) (Mantra
Inc. and The University of Tokyo).

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

## Method

For each of the first \(M\) denoising steps, the sampler evaluates the edit
prompt and an empty prompt in one batch. It forms the reconstruction target

\[
Z^*_{t_i} = X_{\mathrm{in}} - (t_N-t_i)
v_\theta(Z_{t_i}, t_i, \emptyset, X_{\mathrm{in}})
\]

and corrects the current latent with

\[
Z_{t_i} \leftarrow (1-\alpha)Z_{t_i}+\alpha Z^*_{t_i}.
\]

The default experimental setting is \(M=3\) and \(\alpha=0.01\). See
[Algorithm 1 in the paper](https://arxiv.org/abs/2603.27790) for details.

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

## Citation

If this work is useful in your research, please cite:

```bibtex
@misc{furuta2026inference,
  title         = {Inference-time Trajectory Optimization for Manga Image Editing},
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
