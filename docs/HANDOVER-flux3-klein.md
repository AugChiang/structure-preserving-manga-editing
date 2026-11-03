# Hand-over: FLUX 3 Image Klein integration

Written 2026-11-03 on the development PC (RTX 6000 Ada 48 GB). Work continues
on an RTX 4070 12 GB PC.

## Status

Done:

- `structure_preserving_manga_editing/flux3_klein.py` runs the paper's
  Algorithm 1 on FLUX 3 Image Klein, using only the transformer and VAE.
  FLUX 3 Klein is a flow-map model, so each transformer call also gets the
  time the step jumps to. In the correction steps, the empty-prompt branch
  jumps straight to `t = 0`. The edit branch jumps to the next step.
- CLI backend `flux3-klein` (`demo_flux3_klein.py`) with the options
  `--prompt-embeds`, `--transformer-gguf` and `--low-vram`.
- 32 CPU tests pass in `spme-flux3`, including an end-to-end run on a tiny
  random FLUX 3 model.
- Real-weight runs with random prompt embeddings, with the 48 GB GPU capped at
  11.5 GiB to simulate a 12 GB card:

| Transformer | Peak VRAM | Edit time (16 steps) |
| --- | --- | --- |
| unsloth Q8_0 GGUF | 9.88 GiB | 20 s |
| unsloth Q4_K_M GGUF | 7.54 GiB | 21 s |
| BF16 + `--low-vram` | 4.10 GiB | 47 s |
| BF16 on the GPU | out of memory | - |

  Edit times came from the RTX 6000 Ada and will be slower on a 4070.

Blocked by policy (see `CLAUDE.md`):

- **Text encoder.** The official encoder is Qwen3-VL-4B-Instruct.
  `Qwen3VLPromptEncoderPlaceholder` raises `NotImplementedError` and holds
  `TODO(qwen-policy)` notes. Real edits need precomputed embeddings
  (`--prompt-embeds`, written with `save_prompt_embeddings`) that contain the
  task prompt and the empty prompt `""`. A non-Qwen encoder is not a drop-in
  replacement, because the transformer was trained on Qwen3-VL-4B hidden
  states.
- **Prompt upsampler.** It is a Qwen3.5-9B fine-tune, so it is left out. The
  paper uses its exact task prompts anyway.

## Setting up the RTX 4070 PC

```bash
mamba env create -f environment-flux3.yml
conda activate spme-flux3
hf auth login            # accept the FLUX 3 Image Klein terms on the model page first
python -m unittest discover -s tests
```

Download the weights without the Qwen encoder. The first run downloads them
too, but this makes the exclusion explicit:

```bash
python -c "
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download('black-forest-labs/flux-3-image-klein',
    allow_patterns=['transformer/*', 'vae/*', 'config.json', 'LICENSE.md', 'NOTICE.md', 'README.md'])
hf_hub_download('unsloth/flux-3-image-klein-GGUF', 'flux-3-image-klein-Q8_0.gguf')
"
```

Disk needed: about 11.5 GB for the transformer and VAE, plus 6.1 GB for Q8_0.
Add 3.7 GB if you also get Q4_K_M.

## Next steps

1. On the 4070, run `python scripts/check_flux3_vram.py --variant q8` and
   `--variant q4`, with no cap. Compare the results with the table above.
   Leave room for the desktop, which uses some VRAM.
2. Decide how to get prompt embeddings. This is the user's decision under the
   policy. The options are a policy-approved encoder, embeddings computed
   elsewhere, or leaving FLUX 3 as is.
3. With real embeddings, tune `--correction-steps` and `--correction-strength`
   for the 16-step schedule. The current defaults, 3 and 0.01, come from the
   28-step FLUX.1 Kontext setting.

## Estimated memory if a Qwen encoder were ever allowed

These numbers are from the config only and were not measured. Running the
encoder first and freeing it before denoising keeps the peak at the
transformer's 9.9 GiB with Q8, whatever the encoder's precision. An NF4
encoder (about 2.5 to 3 GiB) kept on the GPU with the Q8 transformer would
need about 12.4 GiB, which does not fit. With Q4_K_M it would need about
10 to 10.5 GiB, which fits but is tight. The task prompts are fixed, so their
embeddings can be computed once and cached.

## Quantized FLUX 3 Klein models (surveyed 2026-11-03)

- unsloth/flux-3-image-klein-GGUF: Q2 to Q8 and BF16. Tested and working with
  diffusers `from_single_file`.
- leejet/flux-3-image-klein-GGUF: not tested.
- BFL `variants/fp8r`: BFL's own format, with no diffusers loader.
- BFL `variants/nvfp4`: Blackwell GPUs only.
- unsloth/flux-3-image-klein-FP8, tonera/flux-3-image-klein-nunchaku
  (SVDQuant), and the `-comfy` INT8 files: not tested.
