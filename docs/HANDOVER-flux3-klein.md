# Hand-over: FLUX 3 Image Klein integration

Written 2026-11-03 on the development PC (Linux, RTX 6000 Ada 48 GB). Work
continues on an RTX 4070 12 GB PC running **Windows 11 with Miniconda**.

## Update 2026-11-03: RTX 4070 PC

- Qwen is allowed on the 4070 PC, so the pipeline is now complete:
  `StructurePreservingFlux3KleinPipeline.from_pretrained` loads the
  Qwen3-VL-4B text encoder (`Qwen3VLPromptEncoder`, the Diffusers recipe)
  with the transformer and VAE. The precomputed-embeddings interface
  (`--prompt-embeds`) and the placeholder are gone. Embeddings are cached per
  prompt, so a batch encodes once. With `--low-vram`, the encoder's language
  model is streamed from CPU memory (1.5 GiB peak, matching an fp32 CPU run
  within bf16 rounding).
- `edit()` empties the CUDA cache between the text, denoising and decoding
  phases. Without it, the cache of streamed blocks reached 12.05 GiB and the
  Windows driver spilled into system RAM; with it, the cache peaks at 7.3 GiB.
- The CLI takes a folder of images. FLUX 3 inputs are scaled to about one
  megapixel, and results are resized back to the input size.
- Runs in the existing `manga-edit` environment, upgraded to the
  `environment-flux3.yml` versions except torch (2.9.1+cu128). 33 tests pass.
- CLI runs with `--low-vram` on two pages (900x1300 and 1024x1024), Windows
  desktop running:

| Transformer | Peak allocated | Peak reserved | Total time (2 images, with loading) |
| --- | --- | --- | --- |
| unsloth Q8_0 GGUF | 4.10 GiB | 7.31 GiB | 129 s |
| unsloth Q4_K_M GGUF | 4.10 GiB | 7.31 GiB | 128 s |

  With the paper's prompt, text removal was only partial; see the prompt
  sweep below.

## Update 2026-11-04: text-removal prompt and correction sweep

Q8_0, `--low-vram`, both test pages, about 55 s per page. Outputs are in
`outputs/flux3-sweep/<run>/` (not tracked).

- **Paper prompt** ("Remove text. Keep everything else unchanged."): the
  model guessed what "text" covers. In different seeds it kept the balloon
  dialogue, kept "BOOM", erased a whole balloon but left its tail, or erased
  the bottom panel border. Turning correction off gave the same images, so
  correction was not the cause.
- **Long prompt** ("Erase all lettering ... fill with white paper ..."): all
  text removed in both seeds, but it also erased the eyes, mouth and hatching.
- **Chosen prompt** (`flux3_prompt` in `tasks.py`): "Remove all text and
  lettering, including speech balloon dialogue and sound effects. Keep the
  empty speech balloons, characters, hatching, effect lines, panel borders and
  screentones unchanged." It removed all lettering in all 3 seeds on both pages.
  The portrait was clean in seeds 0 and 2; seed 1 also erased the balloon. On
  the square page, the hatching over the head was erased in every seed (and
  the balloon tail in seed 2).
- **Correction strength** (chosen prompt, seed 0): 0.01 has no visible effect;
  0.05 and 0.07 changed which elements were edited (balloon lost, or
  "THE END?" / "BOOM" left) instead of preserving structure; 3 x 0.1 cancelled
  the edit. With 16 steps the model commits early, so the correction acts like
  a re-roll. Defaults stay at 3 / 0.01 for all backends.
- Qwen Image Edit and FLUX.1 Kontext still get the paper's prompts.

## Status (development PC)

Done:

- `structure_preserving_manga_editing/flux3_klein.py` runs the paper's
  Algorithm 1 on FLUX 3 Image Klein, using only the transformer and VAE.
  FLUX 3 Klein is a flow-map model, so each transformer call also gets the
  time the step jumps to. In the correction steps, the empty-prompt branch
  jumps straight to `t = 0`. The edit branch jumps to the next step.
- CLI backend `flux3-klein` (`demo_flux3_klein.py`) with the options
  `--transformer-gguf` and `--low-vram`.
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

Not integrated:

- **Prompt upsampler.** It is a Qwen3.5-9B fine-tune and is left out. The
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
`hf auth login`. `download_flux3_weights.py` also fetches the 8.9 GB
`text_encoder/` (Qwen3-VL-4B). Add `--no-bf16` to skip the 11 GB BF16 transformer if you only use
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
2. Done: the pipeline uses the official Qwen3-VL-4B encoder on the 4070 PC.
3. Correction tuning was tried (see the 2026-11-04 update): no strength both
   preserves structure and keeps the edit. Open ideas: apply the correction
   only after step 1 or 2, once the edit is committed; add FLUX 3 prompts for
   bar removal and screentone synthesis; test on real Manga109 pages, since
   the test pages are simple drawings.

## Quantized FLUX 3 Klein models (surveyed 2026-11-03)

- unsloth/flux-3-image-klein-GGUF: Q2 to Q8 and BF16. Tested and working with
  diffusers `from_single_file`.
- leejet/flux-3-image-klein-GGUF: not tested.
- BFL `variants/fp8r`: BFL's own format, with no diffusers loader.
- BFL `variants/nvfp4`: Blackwell GPUs only.
- unsloth/flux-3-image-klein-FP8, tonera/flux-3-image-klein-nunchaku
  (SVDQuant), and the `-comfy` INT8 files: not tested.
