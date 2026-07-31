"""Extract image crops that do not overlap Manga109s text annotations."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Rect:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def width(self) -> int:
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        return self.ymax - self.ymin

    @property
    def area(self) -> int:
        return self.width * self.height

    def intersects(self, other: "Rect") -> bool:
        return not (
            self.xmax <= other.xmin
            or other.xmax <= self.xmin
            or self.ymax <= other.ymin
            or other.ymax <= self.ymin
        )


def parse_text_annotations(xml_path: Path) -> tuple[str, dict[int, list[Rect]]]:
    root = ET.parse(xml_path).getroot()
    title = root.get("title") or xml_path.stem
    pages: dict[int, list[Rect]] = {}
    pages_element = root.find("pages")
    if pages_element is None:
        return title, pages

    for page in pages_element.findall("page"):
        page_index = int(page.get("index", "0"))
        pages[page_index] = [
            Rect(
                int(element.get("xmin", "0")),
                int(element.get("ymin", "0")),
                int(element.get("xmax", "0")),
                int(element.get("ymax", "0")),
            )
            for element in page.findall("text")
        ]
    return title, pages


def expand_regions(regions: list[Rect], margin: int, width: int, height: int) -> list[Rect]:
    return [
        Rect(
            max(0, region.xmin - margin),
            max(0, region.ymin - margin),
            min(width, region.xmax + margin),
            min(height, region.ymax + margin),
        )
        for region in regions
    ]


def candidate_squares(
    width: int,
    height: int,
    blocked: list[Rect],
    min_size: int,
    step: int,
) -> list[Rect]:
    max_size = min(width, height)
    sizes = {size for size in (1024, 896, 768, 640, 512) if min_size <= size <= max_size}
    size = max_size
    while size >= min_size:
        sizes.add(size)
        size = int(size * 0.75)

    candidates: list[Rect] = []
    for size in sorted(sizes, reverse=True):
        stride = max(step, size // 4)
        x_positions = list(range(0, width - size + 1, stride))
        y_positions = list(range(0, height - size + 1, stride))
        if x_positions[-1] != width - size:
            x_positions.append(width - size)
        if y_positions[-1] != height - size:
            y_positions.append(height - size)
        for y in y_positions:
            for x in x_positions:
                candidate = Rect(x, y, x + size, y + size)
                if not any(candidate.intersects(region) for region in blocked):
                    candidates.append(candidate)
    return candidates


def overlap_ratio(first: Rect, second: Rect) -> float:
    x1, y1 = max(first.xmin, second.xmin), max(first.ymin, second.ymin)
    x2, y2 = min(first.xmax, second.xmax), min(first.ymax, second.ymax)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1) / first.area


def select_candidates(candidates: list[Rect], limit: int) -> list[Rect]:
    selected: list[Rect] = []
    for candidate in sorted(candidates, key=lambda item: item.area, reverse=True):
        if all(overlap_ratio(candidate, other) <= 0.3 for other in selected):
            selected.append(candidate)
        if len(selected) == limit:
            break
    return selected


def is_blank(image: Image.Image) -> bool:
    array = np.asarray(image.convert("L"), dtype=np.float32)
    if float(array.std()) < 10.0:
        return True
    gradient_x = np.abs(np.diff(array, axis=1)).mean()
    gradient_y = np.abs(np.diff(array, axis=0)).mean()
    return float((gradient_x + gradient_y) / 2.0) < 5.0


def extract_book(
    xml_path: Path,
    images_root: Path,
    output_root: Path,
    min_size: int,
    output_size: int,
    max_per_page: int,
    margin: int,
    step: int,
) -> int:
    title, pages = parse_text_annotations(xml_path)
    title_output = output_root / title
    title_output.mkdir(parents=True, exist_ok=True)
    count = 0

    for page_index, text_regions in sorted(pages.items()):
        image_path = images_root / title / f"{page_index:03d}.jpg"
        if not image_path.is_file():
            continue
        with Image.open(image_path) as page:
            page = page.convert("RGB")
            blocked = expand_regions(text_regions, margin, *page.size)
            candidates = select_candidates(
                candidate_squares(*page.size, blocked, min_size, step),
                max_per_page * 2,
            )
            saved = 0
            for candidate in candidates:
                crop = page.crop(
                    (candidate.xmin, candidate.ymin, candidate.xmax, candidate.ymax)
                )
                if is_blank(crop):
                    continue
                crop = crop.resize((output_size, output_size), Image.Resampling.LANCZOS)
                name = (
                    f"{title}_{page_index:03d}_p{saved:02d}_"
                    f"{candidate.xmin}_{candidate.ymin}_"
                    f"{candidate.width}x{candidate.height}.png"
                )
                crop.save(title_output / name)
                saved += 1
                count += 1
                if saved == max_per_page:
                    break
    return count


def relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise argparse.ArgumentTypeError("paths must be relative")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations-dir",
        type=relative_path,
        default=Path("data/Manga109s/annotations"),
    )
    parser.add_argument(
        "--images-dir",
        type=relative_path,
        default=Path("data/Manga109s/images"),
    )
    parser.add_argument(
        "--titles-file",
        type=relative_path,
        default=Path("preprocessing/text_removal/manga109s_titles.txt"),
    )
    parser.add_argument(
        "--output-dir",
        type=relative_path,
        default=Path("data/work/text_removal/candidates"),
    )
    parser.add_argument("--min-size", type=int, default=512)
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--max-patches-per-page", type=int, default=10)
    parser.add_argument("--margin", type=int, default=10)
    parser.add_argument("--step", type=int, default=64)
    args = parser.parse_args()

    titles = {
        line.strip()
        for line in args.titles_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for xml_path in sorted(args.annotations_dir.glob("*.xml")):
        if xml_path.stem not in titles:
            continue
        count = extract_book(
            xml_path,
            args.images_dir,
            args.output_dir,
            args.min_size,
            args.output_size,
            args.max_patches_per_page,
            args.margin,
            args.step,
        )
        total += count
        print(f"{xml_path.stem}: {count}")
    print(f"Total candidate crops: {total}")


if __name__ == "__main__":
    main()
