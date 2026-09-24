from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from deepseek_v4_ssd.mimo.install import copy_checkpoint, header, safe_path, verify_reconstruction
from deepseek_v4_ssd.mimo.codec import MiMoToolStreamParser
from deepseek_v4_ssd.model_support import get_support
from deepseek_v4_ssd.model import RuntimeConfig


def fixture(source):
    metadata = {"weight": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
                "scale": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]}}
    raw = json.dumps(metadata).encode()
    raw += b" " * (-len(raw) % 8)
    prefix = len(raw).to_bytes(8, "little") + raw
    data = prefix + bytes(range(8))
    (source / "test.safetensors").write_bytes(data)
    (source / "README.md").write_bytes(b"license and model card\n")
    companion = (source / "README.md").read_bytes()
    return {"headers": {"test.safetensors": {"path": "checkpoint/header", "size": len(prefix),
                                               "sha256": hashlib.sha256(prefix).hexdigest()}},
            "sourceFiles": {"test.safetensors": {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()},
                            "README.md": {"size": len(companion)}},
            "copies": [{"source": "test.safetensors", "sourceOffset": len(prefix), "destination": "experts/layer_00.bin", "offset": 0, "length": 4},
                       {"source": "test.safetensors", "sourceOffset": len(prefix) + 4, "destination": "checkpoint/common.bin", "offset": 0, "length": 4}],
            "wholeFiles": {"README.md": {"destination": "checkpoint/README.md", "size": len(companion),
                                         "blobID": hashlib.sha1(f"blob {len(companion)}\0".encode() + companion).hexdigest()}},
            "outputSizes": {"experts/layer_00.bin": 4, "checkpoint/common.bin": 4}}


class MiMoInstallTests(unittest.TestCase):
    def test_raw_repack_reconstructs_every_source_byte(self):
        with tempfile.TemporaryDirectory() as directory:
            source, root = Path(directory) / "source", Path(directory) / "installed"
            source.mkdir()
            root.mkdir()
            plan = fixture(source)
            original = (source / "test.safetensors").read_bytes()
            copy_checkpoint(source, root, plan)
            self.assertEqual((root / "experts/layer_00.bin").read_bytes(), bytes(range(4)))
            self.assertEqual((root / "checkpoint/common.bin").read_bytes(), bytes(range(4, 8)))
            self.assertTrue(verify_reconstruction(root, plan)["allSourceBytesPreserved"])
            self.assertEqual((source / "test.safetensors").read_bytes(), original)
            (root / "experts/layer_00.bin").write_bytes(b"oops")
            with self.assertRaisesRegex(ValueError, "reconstruction failed"):
                verify_reconstruction(root, plan)

    def test_copy_rejects_bad_source_hash_and_companion_blob(self):
        for target in ("shard", "companion"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                source, root = Path(directory) / "source", Path(directory) / "installed"
                source.mkdir()
                root.mkdir()
                plan = fixture(source)
                if target == "shard":
                    plan["sourceFiles"]["test.safetensors"]["sha256"] = "0" * 64
                else:
                    plan["wholeFiles"]["README.md"]["blobID"] = "0" * 40
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    copy_checkpoint(source, root, plan)

    def test_header_rejects_duplicate_metadata_and_uncovered_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.safetensors"
            fixture(path.parent)
            raw, tensors = header(path)
            self.assertEqual(set(tensors), {"weight", "scale"})
            path.write_bytes(path.read_bytes() + b"extra")
            with self.assertRaisesRegex(ValueError, "coverage"):
                header(path)
            duplicate = b'{"x":{},"x":{}}'
            duplicate += b" " * (-len(duplicate) % 8)
            path.write_bytes(len(duplicate).to_bytes(8, "little") + duplicate)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                header(path)

    def test_artifact_paths_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "installed"
            root.mkdir()
            (root / "link").symlink_to(root.parent)
            for name in ("../escape", "/etc/passwd", "a/../b", "a//b", "a\\b", "link/escape"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    safe_path(root, name)

    def test_retained_drafts_do_not_advertise_runtime_support(self):
        support = get_support("mimo-v2.6-flash-rl")
        support.validate_config(RuntimeConfig())
        self.assertTrue(support.uses_layer_major_prefill(RuntimeConfig(), 1024))
        self.assertFalse(support.uses_layer_major_prefill(RuntimeConfig(), 1023))
        self.assertFalse(support.uses_layer_major_prefill(RuntimeConfig(layer_major_prefill=False), 100000))
        self.assertFalse(support.uses_layer_major_prefill(RuntimeConfig(layer_major_prefill_threshold=4096), 1024))
        for config in (RuntimeConfig(mtp_enabled=True), RuntimeConfig(dspark_enabled=True),
                       RuntimeConfig(staged_expert_streaming=True)):
            with self.assertRaises(ValueError):
                support.validate_config(config)
        self.assertFalse(support.descriptor.supports("promptCache"))

    def test_thinking_stream_handles_opening_tag_at_every_boundary(self):
        text = '<think>check</think>OK<tool_call><function=weather><parameter=city>Paris</parameter></function></tool_call>'
        for split in range(len(text) + 1):
            parser = MiMoToolStreamParser("thinking")
            deltas = (*parser.feed(text[:split]), *parser.feed(text[split:]), *parser.finish())
            self.assertFalse(parser.failed)
            self.assertEqual("".join(d.reasoning_content for d in deltas), "check")
            self.assertEqual("".join(d.content for d in deltas), "OK")
            self.assertEqual("".join(d.arguments for d in deltas), '{"city":"Paris"}')
