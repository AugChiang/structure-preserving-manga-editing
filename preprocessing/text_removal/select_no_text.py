"""Copy manually accepted text-free crops into the rendering input tree."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise argparse.ArgumentTypeError("paths must be relative")
    return path


def copy_selected(
    annotation_file: Path,
    candidates_dir: Path,
    output_dir: Path,
) -> int:
    annotations = json.loads(annotation_file.read_text(encoding="utf-8"))
    accepted = sorted(
        relative
        for relative, label in annotations.get("annotations", {}).items()
        if label == 0
    )
    count = 0
    for relative in accepted:
        source = candidates_dir / relative
        destination = output_dir / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation-file",
        type=relative_path,
        default=Path("data/work/text_removal/text_presence_annotations.json"),
    )
    parser.add_argument(
        "--candidates-dir",
        type=relative_path,
        default=Path("data/work/text_removal/candidates"),
    )
    parser.add_argument(
        "--output-dir",
        type=relative_path,
        default=Path("data/work/text_removal/no_text"),
    )
    args = parser.parse_args()
    count = copy_selected(args.annotation_file, args.candidates_dir, args.output_dir)
    print(f"Copied {count} text-free crops")


if __name__ == "__main__":
    main()
