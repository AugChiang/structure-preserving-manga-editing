# Hand-over: FLUX 3 Image Klein integration

Written 2026-11-03 on the development PC (Linux, RTX 6000 Ada 48 GB). Work
continues on an RTX 4070 12 GB PC running **Windows 11 with Miniconda**.
Nothing has been run on Windows yet.

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

## On the RTX 4070 PC (Windows 11, Miniconda)

The Miniconda environment on the 4070 PC is already set up; do not recreate
it. The FLUX 3 code needs diffusers 0.42 or newer, so check once:
`python -c "import diffusers; print(diffusers.__version__)"`.
`environment-flux3.yml` lists the versions tested on the development PC.

Before measuring memory, set **NVIDIA Control Panel → Manage 3D settings →
CUDA - Sysmem Fallback Policy → Prefer No Sysmem Fallback**. By default, the
Windows driver silently spills into system RAM when VRAM runs out. A run that
does not fit in 12 GB then gets very slow instead of failing, which hides the
problem.

With the environment active, each command is a single line:

```powershell
hf auth login
python -m unittest discover -s tests
python scripts/download_flux3_weights.py --gguf Q8_0 Q4_K_M
python scripts/check_flux3_vram.py --variant q8
```

Accept the FLUX 3 Image Klein terms on the model page before you run
`hf auth login`. `download_flux3_weights.py` never fetches `text_encoder/*`
(Qwen). Add `--no-bf16` to skip the 11 GB BF16 transformer if you only use
GGUF. Disk needed: about 11.5 GB for the BF16 transformer and VAE, 6.1 GB for
Q8_0 and 3.7 GB for Q4_K_M.

On Windows, the Hugging Face cache warns that it cannot use symlinks unless
Developer Mode is on. This is harmless; files are copied instead.

Notes for Windows:

- `--low-vram` (group offload) already pins memory one piece at a time
  (`low_cpu_mem_usage=True`), because Windows cannot page-lock a whole large
  model at once. It still needs about 12 GB of free system RAM for the BF16
  transformer.
- The desktop (DWM) uses some VRAM, often 0.5 to 1 GB, so close other GPU
  apps before you measure.
- The CLI only accepts relative paths. On Windows, `test-img\page.png` works.
  `C:\...` is rejected.

## Next steps

1. On the 4070, with sysmem fallback turned off, run
   `python scripts/check_flux3_vram.py --variant q8`, then `--variant q4`, and
   then `--variant bf16 --low-vram`, all with no cap. Compare the results with
   the table above. If a run is much slower than about 2x the times listed,
   the driver is probably spilling into system RAM.
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
