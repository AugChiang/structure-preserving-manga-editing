"""Shared command-line interface for the two one-image demos."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image

from .tasks import FLUX_MODEL_ID, TASKS


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise argparse.ArgumentTypeError("paths must be relative to the project directory")
    return path


def build_parser(backend: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Run the proposed one-image {backend} demo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument(
        "--input",
        required=True,
        type=_relative_path,
        help="Input image path relative to the repository root.",
    )
    parser.add_argument("--output", type=_relative_path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-id")
    parser.add_argument("--correction-steps", type=int, default=3)
    parser.add_argument("--correction-strength", type=float, default=0.01)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the resolved configuration without loading a model.",
    )
    return parser


def resolve_config(backend: str, args: argparse.Namespace) -> dict:
    task = TASKS[args.task]
    input_path = args.input
    output_path = args.output or Path("outputs") / f"{backend}_{args.task}.png"
    model_id = args.model_id
    if model_id is None:
        model_id = FLUX_MODEL_ID if backend == "flux-kontext" else task.qwen_model_id

    if input_path.is_absolute() or output_path.is_absolute():
        raise ValueError("input and output paths must be relative")
    if args.correction_steps < 0:
        raise ValueError("correction_steps must be non-negative")
    if not 0.0 <= args.correction_strength <= 1.0:
        raise ValueError("correction_strength must be between 0 and 1")

    return {
        "backend": backend,
        "task": args.task,
        "prompt": task.prompt,
        "input": str(input_path),
        "output": str(output_path),
        "model_id": model_id,
        "seed": args.seed,
        "device": args.device,
        "correction_steps": args.correction_steps,
        "correction_strength": args.correction_strength,
        "num_inference_steps": 28 if backend == "flux-kontext" else 40,
    }


def validate_input_file(input_path: Path) -> None:
    """Reject missing demo inputs before loading large model checkpoints."""
    if not input_path.is_file():
        raise FileNotFoundError(f"input image not found: {input_path}")


def run_cli(backend: str) -> None:
    args = build_parser(backend).parse_args()
    config = resolve_config(backend, args)
    input_path = Path(config["input"])
    validate_input_file(input_path)

    if args.dry_run:
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
    generator = torch.Generator(device=device).manual_seed(config["seed"])
    image = Image.open(input_path).convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)

    if backend == "flux-kontext":
        from .flux_kontext import StructurePreservingFluxKontextPipeline

        pipeline = StructurePreservingFluxKontextPipeline.from_pretrained(
            config["model_id"],
            torch_dtype=dtype,
        )
        result = pipeline.to(device).edit(
            image,
            config["prompt"],
            generator,
            num_inference_steps=config["num_inference_steps"],
            correction_steps=config["correction_steps"],
            correction_strength=config["correction_strength"],
        )
    else:
        from .qwen_image_edit import StructurePreservingQwenImageEditPipeline

        pipeline = StructurePreservingQwenImageEditPipeline.from_pretrained(
            config["model_id"],
            torch_dtype=dtype,
        )
        result = pipeline.to(device).edit(
            image,
            config["prompt"],
            generator,
            num_inference_steps=config["num_inference_steps"],
            correction_steps=config["correction_steps"],
            correction_strength=config["correction_strength"],
        )

    output_path = Path(config["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    print(f"Saved result to {output_path}")
