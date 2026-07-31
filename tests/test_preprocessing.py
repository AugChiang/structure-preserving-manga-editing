from __future__ import annotations

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw

from preprocessing.screentone_synthesis.extract_body_patches import (
    extract_patch,
    parse_annotations,
)
from preprocessing.text_removal.extract_clean_patches import (
    Rect,
    candidate_squares,
    extract_book,
    parse_text_annotations,
)
from preprocessing.text_removal.download_fonts import SOURCES
from preprocessing.text_removal.select_no_text import copy_selected


class PreprocessingTests(unittest.TestCase):
    def test_manifests(self) -> None:
        font_names = {
            line.strip()
            for line in Path("preprocessing/text_removal/fonts.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        }
        titles = [
            line.strip()
            for line in Path("preprocessing/screentone_synthesis/titles.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.assertEqual(len(font_names), 24)
        self.assertEqual(font_names, set(SOURCES))
        self.assertEqual(len(titles), 6)

    def test_text_annotation_and_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            xml_path = Path(directory) / "Sample.xml"
            root = ET.Element("book", title="Sample")
            pages = ET.SubElement(root, "pages")
            page = ET.SubElement(pages, "page", index="0")
            ET.SubElement(
                page,
                "text",
                xmin="0",
                ymin="0",
                xmax="128",
                ymax="128",
            )
            ET.ElementTree(root).write(xml_path)
            title, parsed = parse_text_annotations(xml_path)
            self.assertEqual(title, "Sample")
            self.assertEqual(parsed[0], [Rect(0, 0, 128, 128)])
            candidates = candidate_squares(640, 640, parsed[0], 512, 64)
            self.assertTrue(all(not item.intersects(parsed[0][0]) for item in candidates))

            images_root = Path(directory) / "images"
            image_dir = images_root / "Sample"
            image_dir.mkdir(parents=True)
            image = Image.new("RGB", (640, 640), "white")
            draw = ImageDraw.Draw(image)
            for offset in range(0, 640, 16):
                draw.line((0, offset, 639, (offset + 320) % 640), fill="black", width=2)
            image.save(image_dir / "000.jpg")
            output_root = Path(directory) / "output"
            count = extract_book(
                xml_path,
                images_root,
                output_root,
                min_size=512,
                output_size=512,
                max_per_page=1,
                margin=0,
                step=64,
            )
            self.assertEqual(count, 1)
            self.assertEqual(len(list(output_root.rglob("*.png"))), 1)

    def test_manual_selection_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "candidates"
            source = candidates / "Sample" / "image.png"
            source.parent.mkdir(parents=True)
            Image.new("RGB", (16, 16), "white").save(source)
            annotation_file = root / "annotations.json"
            annotation_file.write_text(
                json.dumps({"annotations": {"Sample/image.png": 0}}),
                encoding="utf-8",
            )
            output = root / "selected"
            count = copy_selected(annotation_file, candidates, output)
            self.assertEqual(count, 1)
            self.assertTrue((output / "Sample" / "image.png").is_file())

    def test_body_annotation_and_crop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            xml_path = Path(directory) / "Sample.xml"
            root = ET.Element("book", title="Sample")
            pages = ET.SubElement(root, "pages")
            page = ET.SubElement(pages, "page", index="0")
            ET.SubElement(
                page,
                "body",
                id="0",
                xmin="8",
                ymin="12",
                xmax="72",
                ymax="92",
                character="sample",
            )
            ET.ElementTree(root).write(xml_path)
            title, parsed = parse_annotations(str(xml_path))
            self.assertEqual(title, "Sample")
            image = Image.new("RGB", (100, 120), "white")
            ImageDraw.Draw(image).rectangle((8, 12, 72, 92), fill="black")
            crop = extract_patch(image, parsed[0][0], min_size=32)
            self.assertEqual(crop.size, (64, 80))


if __name__ == "__main__":
    unittest.main()
