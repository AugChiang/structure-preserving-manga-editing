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
    validate_input_file,
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
        self.assertEqual(config["output"], "outputs/flux-kontext_text-removal.png")

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
                validate_input_file(missing)

    def test_existing_input_file_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "input.png"
            image.touch()
            validate_input_file(image)


if __name__ == "__main__":
    unittest.main()
