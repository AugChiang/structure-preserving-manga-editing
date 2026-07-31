# Third-Party Notices

The Apache License 2.0 in this repository covers only the original project
code and the project's modifications to permissively licensed source. It does
not change the terms of dependencies, downloaded models, datasets, or fonts.
No model weights, Manga109-s images, or font binaries are distributed here.

## Hugging Face Diffusers

The model-specific sampler integrations in
`structure_preserving_manga_editing/flux_kontext.py` and
`structure_preserving_manga_editing/qwen_image_edit.py` are modified from the
corresponding Hugging Face Diffusers pipelines.

- Source: <https://github.com/huggingface/diffusers>
- License: Apache License 2.0
- License text: <https://github.com/huggingface/diffusers/blob/main/LICENSE>
- FLUX pipeline copyright: Copyright 2025 Black Forest Labs and The
  HuggingFace Team. All rights reserved.
- Qwen Image pipeline copyright: Copyright 2025 Qwen-Image Team and The
  HuggingFace Team. All rights reserved.

The required copyright and license notices are retained in the modified files.

## Pretrained Models

The following checkpoints are downloaded separately at runtime:

- `black-forest-labs/FLUX.1-Kontext-dev`
  - Model card: <https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev>
  - Terms: FLUX.1 [dev] Non-Commercial License
- `Qwen/Qwen-Image-Edit-2509`
  - Model card: <https://huggingface.co/Qwen/Qwen-Image-Edit-2509>
  - License: Apache License 2.0
- `Qwen/Qwen-Image-Edit-2511`
  - Model card: <https://huggingface.co/Qwen/Qwen-Image-Edit-2511>
  - License: Apache License 2.0

Users are responsible for reviewing and complying with the current terms
published with each checkpoint.

## Manga Line Extraction

The optional screentone preprocessing script downloads and executes
`p1atdev/MangaLineExtraction-hf` with Transformers remote-code loading.

- Source and model card:
  <https://huggingface.co/p1atdev/MangaLineExtraction-hf>
- License: MIT

The external model source and weights are not included in this repository.

## Fonts

`preprocessing/text_removal/download_fonts.py` downloads fonts from:

- Google Fonts: <https://github.com/google/fonts>
- Noto CJK: <https://github.com/notofonts/noto-cjk>

The selected Google Fonts are provided under either the SIL Open Font License
1.1 or Apache License 2.0, as identified by their upstream directories. Noto
CJK fonts are provided under the SIL Open Font License 1.1. Font binaries and
their license files are not distributed in this repository; users must review
the upstream license files when downloading them.

## Manga109-s

The preprocessing scripts expect users to obtain Manga109-s separately.

- Dataset website and terms:
  <https://manga109.github.io/manga109-project-website/en/>

Manga109-s and crops derived from it are not included in this repository.
