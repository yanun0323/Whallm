from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from deepseek_v4_ssd import cli

from deepseek_v4_ssd.cli import (
    _read_prompt,
    _select_approximation_mode,
    _token_sha256,
)


class CLITests(unittest.TestCase):
    def test_qwen_grouping_default_and_explicit_override_reach_runtime(self):
        for is_qwen, flags, expected in (
            (True, [], False), (False, [], False),
            (True, ["--no-qwen-grouped-experts"], False),
        ):
            with self.subTest(is_qwen=is_qwen, flags=flags):
                installed = SimpleNamespace(is_qwen=is_qwen, has_mtp=False)
                with patch("sys.argv", ["cli", "--model", "/unused", "--prompt", "hello", *flags]), \
                     patch.object(cli.InstalledModel, "open", return_value=installed), \
                     patch.object(cli.ModelRuntime, "open", side_effect=RuntimeError("config captured")) as opened:
                    with self.assertRaisesRegex(RuntimeError, "config captured"):
                        cli.main()
                    config = opened.call_args.args[1]
                    self.assertEqual(config.qwen_grouped_experts, expected)
                    self.assertFalse(config.mtp_enabled)


    def test_approximation_defaults_to_exact_and_explicit_modes_are_model_aware(self):
        self.assertEqual(
            _select_approximation_mode(
                None,
                is_qwen=False,
                dspark_enabled=False,
            ),
            "exact",
        )
        for is_qwen, dspark_enabled in ((True, True), (False, True)):
            with self.subTest(is_qwen=is_qwen, dspark_enabled=dspark_enabled):
                self.assertEqual(
                    _select_approximation_mode(
                        None,
                        is_qwen=is_qwen,
                        dspark_enabled=dspark_enabled,
                    ),
                    "exact",
                )
                with self.assertRaises(ValueError):
                    _select_approximation_mode(
                        "learned-route-drop-lowest-1",
                        is_qwen=is_qwen,
                        dspark_enabled=dspark_enabled,
                    )

        self.assertEqual(
            _select_approximation_mode(
                "exact",
                is_qwen=False,
                dspark_enabled=False,
            ),
            "exact",
        )

    def test_prompt_file_and_token_hash_are_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompt.txt"
            path.write_text("fixed prompt", encoding="utf-8")

            self.assertEqual(_read_prompt(None, str(path)), "fixed prompt")
        self.assertEqual(
            _token_sha256([1, 2, 3]),
            "8a6ae15122001229edb8866f56e342af12ae8187203c3e3b33931743e7c0c48d",
        )


if __name__ == "__main__":
    unittest.main()
