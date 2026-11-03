from __future__ import annotations

import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from structure_preserving_manga_editing.cli import (
    build_parser,
    resolve_config,
    collect_jobs,
)
from structure_preserving_manga_editing.tasks import TASKS


def arguments(task: str) -> argparse.Namespace:
    return argparse.Namespace(
        task=task,
        input=Path("input.png"),
        output=None,
        seed=0,
        device="cuda",
        model_id=None,
        correction_steps=3,
        correction_strength=0.01,
        dry_run=True,
    )


class TaskTests(unittest.TestCase):
    def test_prompts_are_exact(self) -> None:
        self.assertEqual(
            TASKS["text-removal"].prompt,
            "Remove text. Keep everything else unchanged.",
        )
        self.assertEqual(
            TASKS["screentone-synthesis"].prompt,
            "Convert the line art into a finished Japanese black and white monochrome manga image.",
        )

    def test_qwen_models_are_task_specific(self) -> None:
        text = resolve_config("qwen-image-edit", arguments("text-removal"))
        tone = resolve_config("qwen-image-edit", arguments("screentone-synthesis"))
        self.assertEqual(text["model_id"], "Qwen/Qwen-Image-Edit-2509")
        self.assertEqual(tone["model_id"], "Qwen/Qwen-Image-Edit-2511")
        self.assertEqual(text["num_inference_steps"], 40)

    def test_flux_defaults(self) -> None:
        config = resolve_config("flux-kontext", arguments("text-removal"))
        self.assertEqual(config["model_id"], "black-forest-labs/FLUX.1-Kontext-dev")
        self.assertEqual(config["num_inference_steps"], 28)
        self.assertEqual(config["output"], "outputs")

    def test_flux3_klein_defaults(self) -> None:
        config = resolve_config("flux3-klein", arguments("text-removal"))
        self.assertEqual(config["model_id"], "black-forest-labs/flux-3-image-klein")
        self.assertEqual(config["num_inference_steps"], 16)
        self.assertEqual(config["output"], "outputs")

    def test_only_flux3_uses_its_own_prompt(self) -> None:
        task = TASKS["text-removal"]
        for backend in ("flux-kontext", "qwen-image-edit"):
            self.assertEqual(resolve_config(backend, arguments("text-removal"))["prompt"], task.prompt)
        self.assertEqual(
            resolve_config("flux3-klein", arguments("text-removal"))["prompt"], task.flux3_prompt
        )
        # Tasks without a FLUX 3 prompt fall back to the paper's prompt.
        tone = resolve_config("flux3-klein", arguments("screentone-synthesis"))
        self.assertEqual(tone["prompt"], TASKS["screentone-synthesis"].prompt)

    def test_low_vram_options(self) -> None:
        default = resolve_config("qwen-image-edit", arguments("text-removal"))
        self.assertIsNone(default["transformer_gguf"])
        self.assertFalse(default["low_vram"])

        args = build_parser("qwen-image-edit").parse_args(
            [
                "--task",
                "text-removal",
                "--input",
                "input.png",
                "--transformer-gguf",
                "https://huggingface.co/a/b/blob/main/model.gguf",
                "--low-vram",
            ]
        )
        config = resolve_config("qwen-image-edit", args)
        self.assertEqual(config["transformer_gguf"], "https://huggingface.co/a/b/blob/main/model.gguf")
        self.assertTrue(config["low_vram"])

    def test_input_is_required(self) -> None:
        parser = build_parser("flux-kontext")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--task", "text-removal"])

    def test_absolute_paths_are_rejected(self) -> None:
        parser = build_parser("flux-kontext")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--task",
                    "text-removal",
                    "--input",
                    str(Path.cwd() / "invalid.png"),
                ]
            )

    def test_missing_input_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.png"
            with self.assertRaises(FileNotFoundError):
                collect_jobs(missing, Path("outputs"))

    def test_existing_input_file_keeps_its_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "input.png"
            image.touch()
            self.assertEqual(
                collect_jobs(image, Path("outputs")),
                [(image, Path("outputs") / "input.png")],
            )

    def test_folder_input_collects_sorted_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name in ("b.jpg", "a.PNG", "notes.txt"):
                (folder / name).touch()
            (folder / "nested").mkdir()
            self.assertEqual(
                collect_jobs(folder, Path("outputs")),
                [
                    (folder / "a.PNG", Path("outputs") / "a.PNG"),
                    (folder / "b.jpg", Path("outputs") / "b.jpg"),
                ],
            )

    def test_folder_without_images_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                collect_jobs(Path(directory), Path("outputs"))

    def test_output_cannot_overwrite_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "input.png"
            image.touch()
            with self.assertRaises(ValueError):
                collect_jobs(image, Path(directory))


if __name__ == "__main__":
    unittest.main()
