"""Shared command-line interface for the demos."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image

from .tasks import FLUX3_KLEIN_MODEL_ID, FLUX_MODEL_ID, TASKS

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise argparse.ArgumentTypeError("paths must be relative to the project directory")
    return path


def build_parser(backend: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Run the proposed {backend} demo on an image or a folder of images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument(
        "--input",
        required=True,
        type=_relative_path,
        help="Input image, or folder of images, relative to the repository root.",
    )
    parser.add_argument(
        "--output",
        type=_relative_path,
        default=Path("outputs"),
        help="Output folder. Each result keeps the name of its input image.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", help="Default to: cuda",default="cuda")
    parser.add_argument("--model-id")
    parser.add_argument("--correction-steps", help="Number of correction steps, default to: 3.", type=int, default=3)
    parser.add_argument("--correction-strength", help="Correction strength, default to: 0.01.", type=float, default=0.01)
    parser.add_argument(
        "--transformer-gguf",
        help="Local path or Hugging Face URL of a GGUF-quantized transformer "
        "that replaces the transformer of --model-id.",
    )
    parser.add_argument(
        "--low-vram",
        action="store_true",
        help="Stream the transformer from CPU memory. FLUX 3 also streams its text "
        "encoder; the other backends load it in 4-bit NF4 and offload the other "
        "models while they are idle.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the resolved configuration without loading a model.",
    )
    return parser


def resolve_config(backend: str, args: argparse.Namespace) -> dict:
    task = TASKS[args.task]
    input_path = args.input
    output_path = args.output or Path("outputs")
    model_id = args.model_id
    if model_id is None:
        model_id = {
            "flux-kontext": FLUX_MODEL_ID,
            "flux3-klein": FLUX3_KLEIN_MODEL_ID,
        }.get(backend, task.qwen_model_id)

    if input_path.is_absolute() or output_path.is_absolute():
        raise ValueError("input and output paths must be relative")
    if args.correction_steps < 0:
        raise ValueError("correction_steps must be non-negative")
    if not 0.0 <= args.correction_strength <= 1.0:
        raise ValueError("correction_strength must be between 0 and 1")
    prompt = task.prompt
    if backend == "flux3-klein" and task.flux3_prompt:
        prompt = task.flux3_prompt

    return {
        "backend": backend,
        "task": args.task,
        "prompt": prompt,
        "input": str(input_path),
        "output": str(output_path),
        "model_id": model_id,
        "seed": args.seed,
        "device": args.device,
        "correction_steps": args.correction_steps,
        "correction_strength": args.correction_strength,
        "num_inference_steps": {"flux-kontext": 28, "flux3-klein": 16}.get(backend, 40),
        "transformer_gguf": getattr(args, "transformer_gguf", None),
        "low_vram": getattr(args, "low_vram", False),
    }


def load_pipeline(pipeline_class, transformer_class, text_encoder_name: str, config: dict, dtype):
    """Load a pipeline, optionally with a GGUF transformer and low-VRAM settings."""
    kwargs = {"torch_dtype": dtype}
    if config["transformer_gguf"]:
        from diffusers import GGUFQuantizationConfig

        kwargs["transformer"] = transformer_class.from_single_file(
            config["transformer_gguf"],
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
            config=config["model_id"],
            subfolder="transformer",
            torch_dtype=dtype,
        )
    if config["low_vram"]:
        from diffusers import PipelineQuantizationConfig

        kwargs["quantization_config"] = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": dtype,
            },
            components_to_quantize=[text_encoder_name],
        )

    pipeline = pipeline_class.from_pretrained(config["model_id"], **kwargs)
    if config["low_vram"]:
        enable_low_vram_offload(pipeline, config["device"])
    else:
        pipeline.to(config["device"])
    return pipeline


def enable_low_vram_offload(pipeline, device: str) -> None:
    """Stream the transformer layer by layer and offload the other models when idle.

    Whole-model offloading leaves too little headroom on 12 GB GPUs, so the
    transformer weights stay in CPU memory and are prefetched leaf by leaf.
    Weights are pinned on the fly because Windows cannot page-lock a whole
    20B transformer at once.
    """
    import torch
    from accelerate import cpu_offload_with_hook

    pipeline.transformer.enable_group_offload(
        onload_device=device,
        offload_device="cpu",
        offload_type="leaf_level",
        use_stream=True,
        low_cpu_mem_usage=True,
    )
    hook = None
    for name in pipeline.model_cpu_offload_seq.split("->"):
        component = getattr(pipeline, name, None)
        if name != "transformer" and isinstance(component, torch.nn.Module):
            _, hook = cpu_offload_with_hook(component, device, prev_module_hook=hook)


def collect_jobs(input_path: Path, output_dir: Path) -> list[tuple[Path, Path]]:
    """Pair each input image with its output path before loading large model checkpoints."""
    if input_path.is_dir():
        images = sorted(
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not images:
            raise FileNotFoundError(f"no images found in folder: {input_path}")
    elif input_path.is_file():
        images = [input_path]
    else:
        raise FileNotFoundError(f"input image or folder not found: {input_path}")

    jobs = [(image, output_dir / image.name) for image in images]
    for image, output in jobs:
        if output.resolve() == image.resolve():
            raise ValueError(f"output would overwrite its input: {image}")
    return jobs


def run_cli(backend: str) -> None:
    args = build_parser(backend).parse_args()
    config = resolve_config(backend, args)
    jobs = collect_jobs(Path(config["input"]), Path(config["output"]))

    if args.dry_run:
        config["jobs"] = [[str(image), str(output)] for image, output in jobs]
        print(json.dumps(config, indent=2, sort_keys=True))
        return

    import numpy as np
    import torch

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    device = config["device"]
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    if backend == "flux3-klein":
        from .flux3_klein import StructurePreservingFlux3KleinPipeline

        pipeline = StructurePreservingFlux3KleinPipeline.from_pretrained(
            config["model_id"],
            torch_dtype=dtype,
            transformer_gguf=config["transformer_gguf"],
        )
        if config["low_vram"]:
            pipeline.enable_low_vram(device)
        else:
            pipeline.to(device)
    elif backend == "flux-kontext":
        from diffusers import FluxTransformer2DModel

        from .flux_kontext import StructurePreservingFluxKontextPipeline

        pipeline = load_pipeline(
            StructurePreservingFluxKontextPipeline,
            FluxTransformer2DModel,
            "text_encoder_2",
            config,
            dtype,
        )
    else:
        from diffusers import QwenImageTransformer2DModel

        from .qwen_image_edit import StructurePreservingQwenImageEditPipeline

        pipeline = load_pipeline(
            StructurePreservingQwenImageEditPipeline,
            QwenImageTransformer2DModel,
            "text_encoder",
            config,
            dtype,
        )

    for index, (input_path, output_path) in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {input_path}")
        image = Image.open(input_path).convert("RGB")
        # Reseed per image so that each result does not depend on the folder contents.
        generator = torch.Generator(device=device).manual_seed(config["seed"])
        model_input = image
        if backend == "flux3-klein":
            # The FLUX 3 pipeline encodes images at their given size, so scale
            # to about one megapixel here as the other pipelines do internally.
            aspect_ratio = image.width / image.height
            width = round((1024 * 1024 * aspect_ratio) ** 0.5 / 16) * 16
            height = round(width / aspect_ratio / 16) * 16
            model_input = image.resize((width, height), Image.Resampling.LANCZOS)
        result = pipeline.edit(
            model_input,
            config["prompt"],
            generator,
            num_inference_steps=config["num_inference_steps"],
            correction_steps=config["correction_steps"],
            correction_strength=config["correction_strength"],
        )
        # The model runs near one megapixel; restore the input resolution.
        if result.size != image.size:
            result = result.resize(image.size, Image.Resampling.LANCZOS)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)
        print(f"Saved result to {output_path}")
