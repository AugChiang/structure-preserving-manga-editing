"""Convert character crops to structural line images with a Hub model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise argparse.ArgumentTypeError("paths must be relative")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=relative_path,
        default=Path("data/derived/screentone_synthesis/finished"),
    )
    parser.add_argument(
        "--output-dir",
        type=relative_path,
        default=Path("data/derived/screentone_synthesis/line_art"),
    )
    parser.add_argument("--model-id", default="p1atdev/MangaLineExtraction-hf")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    processor = AutoImageProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    ).to(args.device)
    model.eval()

    image_paths = sorted(
        path
        for path in args.input_dir.rglob("*")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    for image_path in image_paths:
        relative = image_path.relative_to(args.input_dir).with_suffix(".png")
        output_path = args.output_dir / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(image_path) as image:
            inputs = processor(image.convert("RGB"), return_tensors="pt")
        pixel_values = inputs.pixel_values.to(args.device)
        with torch.no_grad():
            result = model(pixel_values)
        array = result.pixel_values[0].detach().cpu().numpy().astype(np.uint8)
        Image.fromarray(array, mode="L").save(output_path)
        print(output_path)


if __name__ == "__main__":
    main()
