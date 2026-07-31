"""
Render manga-style text on clean patches.

This script:
1. Takes manually verified text-free patches
2. Gets text from original Manga109 annotations (same page or same title)
3. Renders vertical Japanese text with manga-style effects
4. Supports two patterns: white outline/black text, black outline/white text
5. Random font size and font selection
"""

import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import re
from dataclasses import dataclass


@dataclass
class TextAnnotation:
    """Text annotation from Manga109."""
    text: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int


def relative_path(value: str) -> str:
    """Reject machine-specific input and output paths."""
    if Path(value).is_absolute():
        raise argparse.ArgumentTypeError("paths must be relative")
    return value


def parse_manga109_annotations(xml_path: str) -> Dict[int, List[TextAnnotation]]:
    """
    Parse Manga109 annotation XML and extract text per page.

    Args:
        xml_path: Path to the XML annotation file

    Returns:
        Dictionary mapping page index to list of TextAnnotation
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    pages_text = {}
    pages_elem = root.find('pages')

    if pages_elem is None:
        return pages_text

    for page in pages_elem.findall('page'):
        page_index = int(page.get('index'))
        texts = []

        for text_elem in page.findall('text'):
            text_content = text_elem.text
            if text_content and text_content.strip():
                texts.append(TextAnnotation(
                    text=text_content.strip(),
                    xmin=int(text_elem.get('xmin', 0)),
                    ymin=int(text_elem.get('ymin', 0)),
                    xmax=int(text_elem.get('xmax', 0)),
                    ymax=int(text_elem.get('ymax', 0))
                ))

        pages_text[page_index] = texts

    return pages_text


def get_all_texts_for_title(pages_text: Dict[int, List[TextAnnotation]]) -> List[str]:
    """Get all text strings from a title."""
    all_texts = []
    for texts in pages_text.values():
        for t in texts:
            if t.text and len(t.text) > 1:  # Filter very short texts
                all_texts.append(t.text)
    return all_texts


def parse_patch_filename(filename: str) -> Tuple[str, int]:
    """
    Parse patch filename to extract title and page index.

    Format: {title}_{page:03d}_p{patch_num:02d}_{x}_{y}_{width}x{height}.png

    Returns:
        Tuple of (title, page_index)
    """
    # Remove extension
    name = filename.replace('.png', '').replace('.jpg', '')

    # Split by underscore
    parts = name.split('_')

    # Find the page number (3 digits after title)
    # Title may contain underscores (e.g., "LoveHina_vol01")
    # Page number is always 3 digits

    for i, part in enumerate(parts):
        if len(part) == 3 and part.isdigit():
            # Found page number
            title = '_'.join(parts[:i])
            page_index = int(part)
            return title, page_index

    # Fallback: assume last pattern
    return parts[0], 0


def find_available_fonts(font_dir: str, verbose: bool = True) -> List[str]:
    """Find the 24 experiment fonts listed in fonts.txt."""
    manifest = Path(__file__).with_name("fonts.txt")
    expected_names = [
        line.strip()
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    available = {
        path.name: str(path)
        for path in sorted(Path(font_dir).rglob("*"))
        if path.suffix.lower() in {".ttf", ".otf", ".ttc"}
    }
    missing = [name for name in expected_names if name not in available]
    if missing:
        raise FileNotFoundError(
            f"Missing experiment fonts in {font_dir}: {', '.join(missing)}"
        )
    fonts = [available[name] for name in expected_names]
    if verbose:
        print(f"Found all {len(fonts)} experiment fonts in {font_dir}")
    return fonts


# Characters that need 90-degree rotation for vertical text
ROTATE_CHARS = set([
    # Long vowel marks (these MUST be rotated)
    'ー', 'ｰ', '−', '―', '‐', '-', '–', '—',
    # Wave dash
    '〜', '～',
    # Ellipsis and dots
    '…', '‥', '⋯',
    # Horizontal arrows (rotate to become vertical)
    '→', '←', '↔', '⇒', '⇐', '⇔',
    # Underscores and horizontal lines
    '_', '＿', '￣',
])

# Small kana that should be positioned to the upper-right in vertical text
SMALL_KANA = set([
    'ぁ', 'ぃ', 'ぅ', 'ぇ', 'ぉ', 'っ', 'ゃ', 'ゅ', 'ょ', 'ゎ',
    'ァ', 'ィ', 'ゥ', 'ェ', 'ォ', 'ッ', 'ャ', 'ュ', 'ョ', 'ヮ',
    'ヵ', 'ヶ',
])

# Punctuation that should be positioned to the upper-right in vertical text
VERTICAL_PUNCTUATION = set([
    '。', '、', '，', '．', '・', '：', '；',
])


def get_content_bbox(img: Image.Image) -> Tuple[int, int, int, int]:
    """
    Get the bounding box of non-transparent content in an RGBA image.
    Returns (left, top, right, bottom) or None if empty.
    """
    if img.mode == 'RGBA':
        alpha = img.split()[3]
    elif img.mode == 'L':
        alpha = img
    else:
        return (0, 0, img.width, img.height)

    bbox = alpha.getbbox()
    if bbox is None:
        return (0, 0, img.width, img.height)
    return bbox


def render_char_image(
    char: str,
    font: ImageFont.FreeTypeFont,
    fill_color: Tuple[int, int, int],
    stroke_color: Tuple[int, int, int],
    stroke_width: int,
    rotate: bool = False,
    target_cell_size: Optional[int] = None
) -> Tuple[Image.Image, Image.Image]:
    """
    Render a single character to an image with stroke effect.

    Args:
        char: Character to render
        font: Font to use
        fill_color: Fill color (RGB)
        stroke_color: Stroke color (RGB)
        stroke_width: Width of stroke
        rotate: Whether to rotate 90 degrees for vertical text
        target_cell_size: If provided, pad the result to this size (centered)

    Returns:
        Tuple of (character_image, mask_image) with transparency
    """
    # Get character size with padding for stroke
    try:
        bbox = font.getbbox(char)
        char_width = bbox[2] - bbox[0]
        char_height = bbox[3] - bbox[1]
        offset_x = -bbox[0]
        offset_y = -bbox[1]
    except:
        char_width = font.size
        char_height = font.size
        offset_x = 0
        offset_y = 0

    # Add extra padding to ensure content is not clipped after rotation
    padding = stroke_width + 4
    img_width = char_width + padding * 2
    img_height = char_height + padding * 2

    # Create transparent image for character
    char_img = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
    char_draw = ImageDraw.Draw(char_img)

    # Create mask image
    mask_img = Image.new('L', (img_width, img_height), 0)
    mask_draw = ImageDraw.Draw(mask_img)

    # Position to draw character (centered in the canvas)
    draw_x = padding + offset_x
    draw_y = padding + offset_y

    # Draw stroke (outline)
    if stroke_width > 0:
        for dx in range(-stroke_width, stroke_width + 1):
            for dy in range(-stroke_width, stroke_width + 1):
                if dx * dx + dy * dy <= stroke_width * stroke_width:
                    char_draw.text(
                        (draw_x + dx, draw_y + dy), char,
                        font=font, fill=stroke_color + (255,)
                    )
                    mask_draw.text(
                        (draw_x + dx, draw_y + dy), char,
                        font=font, fill=255
                    )

    # Draw main text
    char_draw.text((draw_x, draw_y), char, font=font, fill=fill_color + (255,))
    mask_draw.text((draw_x, draw_y), char, font=font, fill=255)

    # Rotate if needed
    if rotate:
        char_img = char_img.rotate(90, expand=True, resample=Image.BICUBIC)
        mask_img = mask_img.rotate(90, expand=True, resample=Image.BICUBIC)

    # Crop to actual content to ensure proper centering
    content_bbox = get_content_bbox(char_img)
    if content_bbox:
        # Add small margin around content
        margin = 1
        left = max(0, content_bbox[0] - margin)
        top = max(0, content_bbox[1] - margin)
        right = min(char_img.width, content_bbox[2] + margin)
        bottom = min(char_img.height, content_bbox[3] + margin)

        char_img = char_img.crop((left, top, right, bottom))
        mask_img = mask_img.crop((left, top, right, bottom))

    # If target cell size is specified, center the character in a cell-sized canvas
    if target_cell_size is not None and target_cell_size > 0:
        cw, ch = char_img.size
        if cw < target_cell_size or ch < target_cell_size:
            new_img = Image.new('RGBA', (target_cell_size, target_cell_size), (0, 0, 0, 0))
            new_mask = Image.new('L', (target_cell_size, target_cell_size), 0)

            paste_x = (target_cell_size - cw) // 2
            paste_y = (target_cell_size - ch) // 2

            new_img.paste(char_img, (paste_x, paste_y), char_img)
            new_mask.paste(mask_img, (paste_x, paste_y))

            char_img = new_img
            mask_img = new_mask

    return char_img, mask_img


def render_vertical_text(
    image: Image.Image,
    mask: Optional[Image.Image],
    text: str,
    position: Tuple[int, int],
    font: ImageFont.FreeTypeFont,
    fill_color: Tuple[int, int, int],
    stroke_color: Tuple[int, int, int],
    stroke_width: int = 3,
    cell_size: Optional[int] = None,
    max_height: Optional[int] = None,
    column_spacing: Optional[int] = None
) -> Tuple[int, int]:
    """
    Render vertical text with stroke effect, properly handling rotated characters.
    Supports multi-column layout when text exceeds max_height.

    Args:
        image: PIL Image to render on
        mask: Optional mask image (white on black)
        text: Text to render
        position: (x, y) position for top-right of first column
        font: Font to use
        fill_color: Text fill color (RGB)
        stroke_color: Stroke/outline color (RGB)
        stroke_width: Width of stroke
        cell_size: Size of each character cell (default: based on font)
        max_height: Maximum height before wrapping to new column
        column_spacing: Spacing between columns (default: cell_size * 1.2)

    Returns:
        (total_width, total_height) of rendered text
    """
    img_w, img_h = image.size

    # Calculate cell size based on font metrics
    if cell_size is None:
        try:
            # Use a reference character for consistent cell size
            ref_bbox = font.getbbox('あ')
            base_cell_size = max(ref_bbox[2] - ref_bbox[0], ref_bbox[3] - ref_bbox[1])
            # Add minimal spacing for stroke (just enough to prevent overlap)
            cell_size = base_cell_size + stroke_width + 1
        except:
            cell_size = font.size + stroke_width + 1

    # Set default max height and column spacing
    if max_height is None:
        max_height = img_h - position[1] - stroke_width - 5

    if column_spacing is None:
        column_spacing = int(cell_size * 1.3)

    # Starting position (top-right of first column in Japanese vertical text)
    start_x, start_y = position
    current_x = start_x
    current_y = start_y

    total_width = 0
    max_column_height = 0

    for char in text:
        # Skip newlines (could also be used to force new column)
        if char in '\n\r':
            # Move to next column
            current_x -= column_spacing
            current_y = start_y
            continue

        # Estimate character height before rendering (for column wrap check)
        # This is approximate, but helps prevent starting a character that won't fit
        try:
            char_bbox = font.getbbox(char)
            estimated_char_height = char_bbox[3] - char_bbox[1] + stroke_width + 1
        except:
            estimated_char_height = cell_size

        # Check if we need to wrap to a new column
        # Use estimated height to avoid starting a character that won't fit
        if current_y + estimated_char_height > start_y + max_height:
            current_x -= column_spacing
            current_y = start_y

        # Stop if we've run out of horizontal space
        if current_x - cell_size // 2 < 0:
            break

        # Check if character needs rotation
        needs_rotation = char in ROTATE_CHARS

        # Check if small kana or punctuation (these are rendered smaller)
        is_small_char = char in SMALL_KANA or char in VERTICAL_PUNCTUATION

        # Render character to image with target cell size for proper centering
        # For rotated characters and normal characters, center them in the cell
        char_img, mask_char_img = render_char_image(
            char, font, fill_color, stroke_color, stroke_width,
            rotate=needs_rotation,
            target_cell_size=cell_size if needs_rotation else None
        )

        char_w, char_h = char_img.size

        # Calculate paste position
        if needs_rotation:
            # Rotated characters are already centered in cell_size x cell_size canvas
            paste_x = current_x - cell_size // 2
            paste_y = current_y
        elif is_small_char:
            # Small kana and punctuation: position in upper-right of cell
            # These characters are naturally smaller, so offset them
            paste_x = current_x - char_w // 2 + cell_size // 4
            paste_y = current_y
        else:
            # Normal characters: center horizontally, align top
            # Use actual character height for better spacing
            paste_x = current_x - char_w // 2
            paste_y = current_y

        # Clamp to image bounds
        paste_x = max(0, min(paste_x, img_w - char_w))
        paste_y = max(0, min(paste_y, img_h - char_h))

        # Paste character onto main image
        try:
            image.paste(char_img, (paste_x, paste_y), char_img)
        except Exception:
            # Fallback without alpha if paste fails
            image.paste(char_img.convert('RGB'), (paste_x, paste_y))

        # Paste onto mask if provided
        if mask is not None:
            try:
                mask_rgb = Image.merge('RGB', (mask_char_img, mask_char_img, mask_char_img))
                mask.paste(mask_rgb, (paste_x, paste_y), char_img)
            except Exception:
                mask.paste(mask_rgb, (paste_x, paste_y))

        # Calculate actual height used by this character (including stroke)
        # Use the actual rendered height plus minimal spacing
        actual_char_height = char_h
        # Add minimal spacing to prevent overlap (just enough for stroke)
        # Use smaller spacing for better visual appearance
        spacing = max(1, stroke_width // 3)

        # Track dimensions
        max_column_height = max(max_column_height, current_y + actual_char_height - start_y)

        # Move to next character position (downward)
        # Use actual character height + minimal spacing to prevent overlap
        current_y += actual_char_height + spacing

    # Calculate total width (from rightmost to leftmost column)
    total_width = start_x - current_x + column_spacing

    return total_width, max_column_height


def render_text_on_image(
    image: Image.Image,
    text: str,
    font_path: str,
    font_size: int,
    style: str = 'white_on_black',
    create_mask: bool = False
) -> Tuple[Image.Image, Optional[Image.Image], dict]:
    """
    Render vertical text on an image.

    Args:
        image: PIL Image to render on
        text: Text to render
        font_path: Path to font file
        font_size: Font size
        style: 'white_on_black' or 'black_on_white'
        create_mask: If True, also create a mask image

    Returns:
        Tuple of (rendered_image, mask_image or None, metadata)
    """
    # Create a copy (convert to RGBA for alpha compositing)
    result = image.copy().convert('RGBA')

    # Create mask if requested
    mask = None
    if create_mask:
        mask = Image.new('RGB', image.size, (0, 0, 0))

    # Load font
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"Error loading font {font_path}: {e}")
        # Fallback to default
        font = ImageFont.load_default()

    # Set colors based on style
    if style == 'white_on_black':
        fill_color = (255, 255, 255)
        stroke_color = (0, 0, 0)
    else:  # black_on_white
        fill_color = (0, 0, 0)
        stroke_color = (255, 255, 255)

    # Calculate cell size from font metrics
    try:
        ref_bbox = font.getbbox('あ')
        cell_size = max(ref_bbox[2] - ref_bbox[0], ref_bbox[3] - ref_bbox[1])
    except:
        cell_size = font_size

    # Calculate text dimensions
    text_chars = [c for c in text if c not in '\n\r']
    if not text_chars:
        result_rgb = result.convert('RGB')
        return result_rgb, mask, {'error': 'empty text'}

    img_width, img_height = image.size

    # Randomize stroke width (between thin and thick outlines)
    min_stroke = max(1, font_size // 15)
    max_stroke = max(3, font_size // 6)
    stroke_width = random.randint(min_stroke, max_stroke)

    margin = stroke_width + 10

    # Calculate available height for text (leave margin at top and bottom)
    available_height = img_height - margin * 2

    # Calculate how many characters fit in one column
    chars_per_column = max(1, available_height // cell_size)

    # Calculate number of columns needed
    num_columns = (len(text_chars) + chars_per_column - 1) // chars_per_column

    # Column spacing
    column_spacing = int(cell_size * 1.3)

    # Calculate total text block width
    text_block_width = column_spacing * num_columns

    # Calculate position range for the starting point (top-right of text block)
    # In Japanese vertical text, we start from the RIGHT side
    min_x = text_block_width + margin
    max_x = img_width - cell_size // 2 - margin
    min_y = margin
    max_y = margin + 20  # Small variation in Y position

    if max_x <= min_x:
        # Not enough horizontal space, start from right edge
        x = img_width - cell_size // 2 - margin
    else:
        x = random.randint(min_x, max_x)

    if max_y <= min_y:
        y = margin
    else:
        y = random.randint(min_y, max_y)

    # Render text
    render_vertical_text(
        result, mask, text, (x, y), font,
        fill_color, stroke_color, stroke_width,
        cell_size=cell_size,
        max_height=available_height,
        column_spacing=column_spacing
    )

    # Convert back to RGB
    result_rgb = result.convert('RGB')

    metadata = {
        'text': text,
        'position': (x, y),
        'font_path': font_path,
        'font_size': font_size,
        'style': style,
        'stroke_width': stroke_width,
        'cell_size': cell_size,
        'num_columns': num_columns
    }

    return result_rgb, mask, metadata


def process_patches(
    input_dir: str,
    output_dir: str,
    annotation_dir: str,
    fonts: List[str],
    min_font_size: int = 20,
    max_font_size: int = 48,
    max_text_length: int = 20,
    save_mask: bool = False,
    mask_dir: Optional[str] = None
) -> Tuple[int, int]:
    """
    Process all patches and render text on them.

    Args:
        input_dir: Directory containing no-text patches
        output_dir: Directory to save rendered images
        annotation_dir: Directory containing Manga109 XML annotations
        fonts: List of font paths to use
        min_font_size: Minimum font size
        max_font_size: Maximum font size
        max_text_length: Maximum text length to render
        save_mask: If True, save mask images
        mask_dir: Directory to save mask images (defaults to output_dir + '_masks')

    Returns:
        Tuple of (processed_count, skipped_count)
    """
    # Cache for parsed annotations
    annotation_cache: Dict[str, Dict[int, List[TextAnnotation]]] = {}

    processed = 0
    skipped = 0

    input_path = Path(input_dir)

    # Determine mask directory
    if save_mask and mask_dir is None:
        mask_dir = output_dir.rstrip('/') + '_masks'

    # Process each subdirectory (title)
    for title_dir in sorted(input_path.iterdir()):
        if not title_dir.is_dir():
            continue

        title = title_dir.name

        # Load annotations for this title
        xml_path = os.path.join(annotation_dir, f'{title}.xml')
        if not os.path.exists(xml_path):
            print(f"Warning: Annotation not found for {title}")
            continue

        if title not in annotation_cache:
            annotation_cache[title] = parse_manga109_annotations(xml_path)

        pages_text = annotation_cache[title]
        all_texts = get_all_texts_for_title(pages_text)

        if not all_texts:
            print(f"Warning: No text found for {title}")
            continue

        # Create output directory
        output_title_dir = os.path.join(output_dir, title)
        os.makedirs(output_title_dir, exist_ok=True)

        # Create mask directory if needed
        if save_mask:
            mask_title_dir = os.path.join(mask_dir, title)
            os.makedirs(mask_title_dir, exist_ok=True)

        # Process each patch
        for patch_file in sorted(title_dir.glob('*.png')):
            try:
                # Parse filename to get page index
                _, page_idx = parse_patch_filename(patch_file.name)

                # Get text from same page, or fallback to any page
                page_texts = pages_text.get(page_idx, [])
                if page_texts:
                    text_candidates = [t.text for t in page_texts if len(t.text) > 1]
                else:
                    text_candidates = []

                if not text_candidates:
                    text_candidates = all_texts

                # Select random text
                text = random.choice(text_candidates)

                # Truncate if too long
                if len(text) > max_text_length:
                    text = text[:max_text_length]

                # Select random font
                font_path = random.choice(fonts)

                # Select random font size
                font_size = random.randint(min_font_size, max_font_size)

                # Select random style
                style = random.choice(['white_on_black', 'black_on_white'])

                # Load and process image
                image = Image.open(patch_file)

                result, mask, metadata = render_text_on_image(
                    image, text, font_path, font_size, style,
                    create_mask=save_mask
                )

                # Save result
                output_path = os.path.join(output_title_dir, patch_file.name)
                result.save(output_path)

                # Save mask if requested
                if save_mask and mask is not None:
                    mask_path = os.path.join(mask_title_dir, patch_file.name)
                    mask.save(mask_path)

                processed += 1

                if processed % 100 == 0:
                    print(f"Processed {processed} images...")

            except Exception as e:
                print(f"Error processing {patch_file}: {e}")
                skipped += 1

    return processed, skipped


def main():
    parser = argparse.ArgumentParser(
        description='Render manga-style text on clean patches'
    )
    parser.add_argument(
        '--input-dir',
        type=relative_path,
        default='data/work/text_removal/no_text',
        help='Directory containing no-text patches'
    )
    parser.add_argument(
        '--output-dir',
        type=relative_path,
        default='data/derived/text_removal/with_text',
        help='Directory to save rendered images'
    )
    parser.add_argument(
        '--annotation-dir',
        type=relative_path,
        default='data/Manga109s/annotations',
        help='Directory containing Manga109 XML annotations'
    )
    parser.add_argument(
        '--min-font-size',
        type=int,
        default=24,
        help='Minimum font size (default: 24)'
    )
    parser.add_argument(
        '--max-font-size',
        type=int,
        default=48,
        help='Maximum font size (default: 48)'
    )
    parser.add_argument(
        '--max-text-length',
        type=int,
        default=100,
        help='Maximum text length to render (default: 100)'
    )
    parser.add_argument(
        '--font-dir',
        type=relative_path,
        default='data/fonts',
        help='Directory containing the 24 experiment fonts'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed'
    )
    parser.add_argument(
        '--save-mask',
        action='store_true',
        help='Save mask images (white=text, black=background)'
    )
    parser.add_argument(
        '--mask-dir',
        type=relative_path,
        default=None,
        help='Directory to save mask images (default: output_dir + "_masks")'
    )
    parser.add_argument(
        '--list-fonts',
        action='store_true',
        help='List available fonts and exit'
    )

    args = parser.parse_args()

    # Find available fonts
    random.seed(args.seed)
    fonts = find_available_fonts(args.font_dir)

    if args.list_fonts:
        print("Available fonts:")
        for font in sorted(fonts):
            print(f"  {font}")
        print(f"\nTotal: {len(fonts)} fonts")
        return

    if not fonts:
        print("Error: No fonts found! Please install Japanese fonts or specify --font-dir")
        return

    # Determine mask directory
    mask_dir = args.mask_dir
    if args.save_mask and mask_dir is None:
        mask_dir = args.output_dir.rstrip('/') + '_masks'

    print("=" * 60)
    print("Manga Text Renderer")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    if args.save_mask:
        print(f"Mask directory: {mask_dir}")
    print(f"Annotation directory: {args.annotation_dir}")
    print(f"Font size range: {args.min_font_size} - {args.max_font_size}")
    print(f"Max text length: {args.max_text_length}")
    print(f"Save masks: {args.save_mask}")
    print(f"Available fonts: {len(fonts)}")
    print("=" * 60)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create mask directory if needed
    if args.save_mask:
        os.makedirs(mask_dir, exist_ok=True)

    # Process patches
    processed, skipped = process_patches(
        args.input_dir,
        args.output_dir,
        args.annotation_dir,
        fonts,
        args.min_font_size,
        args.max_font_size,
        args.max_text_length,
        save_mask=args.save_mask,
        mask_dir=mask_dir
    )

    print("=" * 60)
    print(f"Processing complete!")
    print(f"  Processed: {processed} images")
    print(f"  Skipped: {skipped} images")
    print(f"  Output: {args.output_dir}")
    if args.save_mask:
        print(f"  Masks: {mask_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
