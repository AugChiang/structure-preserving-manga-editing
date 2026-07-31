"""
Extract body bounding box patches from Manga109 dataset.

This script:
1. Reads the list of titles to process from a text file
2. Parses annotation XML files to extract body bounding boxes
3. Crops the corresponding regions from manga images
4. Saves patches organized by title
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from PIL import Image
from tqdm import tqdm


@dataclass
class BodyAnnotation:
    """Body bounding box annotation."""
    id: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    character: str

    @property
    def width(self) -> int:
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        return self.ymax - self.ymin

    @property
    def area(self) -> int:
        return self.width * self.height


def relative_path(value: str) -> str:
    """Reject machine-specific input and output paths."""
    if Path(value).is_absolute():
        raise argparse.ArgumentTypeError("paths must be relative")
    return value


def load_titles(titles_file: str) -> List[str]:
    """Load list of titles from a text file."""
    titles = []
    with open(titles_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                titles.append(line)
    return titles


def parse_annotations(xml_path: str) -> Tuple[str, Dict[int, List[BodyAnnotation]]]:
    """
    Parse annotation XML file and extract body bounding boxes.

    Returns:
        Tuple of (book_title, dict mapping page_index to list of BodyAnnotation)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    book_title = root.get('title')
    pages_data = {}

    pages_elem = root.find('pages')
    if pages_elem is None:
        return book_title, pages_data

    for page in pages_elem.findall('page'):
        page_index = int(page.get('index'))
        bodies = []

        # Parse body elements
        for body_elem in page.findall('body'):
            body = BodyAnnotation(
                id=body_elem.get('id', ''),
                xmin=int(body_elem.get('xmin')),
                ymin=int(body_elem.get('ymin')),
                xmax=int(body_elem.get('xmax')),
                ymax=int(body_elem.get('ymax')),
                character=body_elem.get('character', '')
            )
            bodies.append(body)

        if bodies:
            pages_data[page_index] = bodies

    return book_title, pages_data


def extract_patch(
    image: Image.Image,
    body: BodyAnnotation,
    min_size: int = 64,
    padding: int = 0
) -> Optional[Image.Image]:
    """
    Extract a patch from the image based on body bounding box.

    Args:
        image: Source PIL Image
        body: BodyAnnotation with bounding box coordinates
        min_size: Minimum size for both width and height
        padding: Optional padding around the bounding box

    Returns:
        Cropped PIL Image or None if too small
    """
    img_width, img_height = image.size

    # Apply padding
    xmin = max(0, body.xmin - padding)
    ymin = max(0, body.ymin - padding)
    xmax = min(img_width, body.xmax + padding)
    ymax = min(img_height, body.ymax + padding)

    width = xmax - xmin
    height = ymax - ymin

    # Skip if too small
    if width < min_size or height < min_size:
        return None

    # Crop the patch
    patch = image.crop((xmin, ymin, xmax, ymax))

    return patch


def process_title(
    title: str,
    annotations_dir: str,
    images_dir: str,
    output_dir: str,
    min_size: int = 64,
    padding: int = 0
) -> int:
    """
    Process a single manga title and extract body patches.

    Returns:
        Number of patches extracted
    """
    xml_path = os.path.join(annotations_dir, f"{title}.xml")
    images_path = os.path.join(images_dir, title)
    title_output_dir = os.path.join(output_dir, title)

    # Check if files exist
    if not os.path.exists(xml_path):
        print(f"Warning: Annotation file not found: {xml_path}")
        return 0

    if not os.path.exists(images_path):
        print(f"Warning: Images directory not found: {images_path}")
        return 0

    # Create output directory
    os.makedirs(title_output_dir, exist_ok=True)

    # Parse annotations
    book_title, pages_data = parse_annotations(xml_path)

    patch_count = 0

    # Process each page with body annotations
    for page_index, bodies in tqdm(pages_data.items(), desc=f"Processing {title}", leave=False):
        # Load image
        image_path = os.path.join(images_path, f"{page_index:03d}.jpg")
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue

        # Extract patches for each body
        for body_idx, body in enumerate(bodies):
            patch = extract_patch(image, body, min_size=min_size, padding=padding)

            if patch is None:
                continue

            # Generate filename
            # Format: {title}_{page:03d}_b{body_idx:02d}_{xmin}_{ymin}_{width}x{height}.png
            width, height = patch.size
            filename = f"{title}_{page_index:03d}_b{body_idx:02d}_{body.xmin}_{body.ymin}_{width}x{height}.png"
            output_path = os.path.join(title_output_dir, filename)

            # Save patch
            patch.save(output_path)
            patch_count += 1

        # Close image to free memory
        image.close()

    return patch_count


def main():
    parser = argparse.ArgumentParser(
        description='Extract body bounding box patches from Manga109 dataset'
    )
    parser.add_argument(
        '--manga109-dir',
        type=relative_path,
        default='data/Manga109s',
        help='Path to Manga109 dataset root directory'
    )
    parser.add_argument(
        '--annotations-dir',
        type=relative_path,
        default=None,
        help='Path to annotations directory (default: {manga109-dir}/annotations)'
    )
    parser.add_argument(
        '--images-dir',
        type=relative_path,
        default=None,
        help='Path to images directory (default: {manga109-dir}/images)'
    )
    parser.add_argument(
        '--titles-file',
        type=relative_path,
        default='preprocessing/screentone_synthesis/titles.txt',
        help='Path to text file containing list of titles to process'
    )
    parser.add_argument(
        '--output-dir',
        type=relative_path,
        default='data/derived/screentone_synthesis/finished',
        help='Output directory for extracted patches'
    )
    parser.add_argument(
        '--min-size',
        type=int,
        default=256,
        help='Minimum size for both width and height of patches (default: 256)'
    )
    parser.add_argument(
        '--padding',
        type=int,
        default=0,
        help='Padding around bounding boxes (default: 0)'
    )

    args = parser.parse_args()

    # Set default paths
    if args.annotations_dir is None:
        args.annotations_dir = os.path.join(args.manga109_dir, 'annotations')
    if args.images_dir is None:
        args.images_dir = os.path.join(args.manga109_dir, 'images')

    # Load titles
    titles = load_titles(args.titles_file)
    print(f"Loaded {len(titles)} titles to process:")
    for title in titles:
        print(f"  - {title}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each title
    total_patches = 0
    for title in tqdm(titles, desc="Processing titles"):
        patch_count = process_title(
            title=title,
            annotations_dir=args.annotations_dir,
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            min_size=args.min_size,
            padding=args.padding
        )
        print(f"  {title}: {patch_count} patches extracted")
        total_patches += patch_count

    print(f"\nTotal patches extracted: {total_patches}")
    print(f"Output saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
