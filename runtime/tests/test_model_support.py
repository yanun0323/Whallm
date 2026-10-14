from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from deepseek_v4_ssd import model_support, model_manager
from deepseek_v4_ssd.generation import GenerationOptions, ModelRuntime
from deepseek_v4_ssd.manifest import InstalledModel
from deepseek_v4_ssd.model import RuntimeConfig
from deepseek_v4_ssd.model_manager import MODEL_IDS
from deepseek_v4_ssd.model_support.base import ModelSupport
from deepseek_v4_ssd.model_support.catalog import BY_KIND, DESCRIPTORS, read_catalog, _catalog_path
from deepseek_v4_ssd.server import _options, _reasoning_settings, ServerDefaults


class _TinyModel(nn.Module):
    layers = ()

    def __call__(self, tokens, cache=None):
        return mx.broadcast_to(mx.array([0.0, 0.0, 10.0, -10.0]), (*tokens.shape, 4))

    def make_cache(self):
        return []


class _Resources:
    def __init__(self):
        self.closes = 0

    def close(self):
        self.closes += 1


class _FourthSupport(ModelSupport):
    def manifest_contract(self, raw):
        if raw["revision"] != self.descriptor.checkpoint_revision:
            raise ValueError("wrong fixture revision")
        return {
            "required": {"common.bin", "config.json", "experts/layer_00.bin"},
            "allowed": {"common.bin", "config.json", "experts/layer_00.bin"},
            "model_kind": self.descriptor.kind, "layer_count": 1,
            "expert_count": 1, "selected_expert_count": 1, "expert_blob_size": 4,
            "expert_regions": (("fixture", "F32", (1,), 0, 4),), "maximum_context": 32,
        }

    def load(self, installed, config, raw_config, weights, read_limiter):
        return _TinyModel(), _Resources()

    def prefill(self, model, tokens, cache, step_size, expert_cache, config):
        mx.eval(model(mx.array(tokens)[None], cache=cache))


class ModelSupportTests(unittest.TestCase):
    def test_builtin_metadata_and_runtime_agree(self):
        for descriptor in DESCRIPTORS:
            with self.subTest(kind=descriptor.kind):
                support = model_support.get_support(descriptor.kind)
                self.assertIs(support.descriptor, descriptor)
                self.assertIs(
                    model_support.support_for_manifest({"formatVersion": descriptor.manifest_version,
                                                       "modelKind": descriptor.kind}), support,
                )
                self.assertEqual(descriptor.api_model_id,
                                 MODEL_IDS[descriptor.kind])

    def test_unknown_and_crossed_manifest_identity_are_rejected(self):
        for raw in ({"formatVersion": 999}, {"formatVersion": 1, "modelKind": "deepseek-v4.1"},
                    {"formatVersion": 3, "modelKind": "arbitrary.python.module"}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                model_support.support_for_manifest(raw)
        with self.assertRaises(ValueError):
            model_support.get_support("arbitrary.python.module")

    def test_descriptor_rejects_duplicate_ids_and_unsafe_paths(self):
        raw = json.loads(_catalog_path().read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            for key, value in (("kind", raw["models"][0]["kind"]),
                               ("apiModelID", raw["models"][0]["apiModelID"]),
                               ("directoryName", "../escape"),
                               ("maximumContext", 0)):
                invalid = json.loads(json.dumps(raw))
                invalid["models"][1][key] = value
                path.write_text(json.dumps(invalid))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    read_catalog(path)

    def test_v41_rejects_unsupported_state_and_uses_exact_requests(self):
        support = model_support.get_support("deepseek-v4.1")
        self.assertTrue(support.uses_layer_major_prefill(RuntimeConfig(), 10_000))
        self.assertFalse(support.uses_layer_major_prefill(RuntimeConfig(v41_layer_major_prefill=False), 10_000))
        self.assertEqual(support.default_approximation(), "exact")
        for operation, arguments in ((support.clone_cache, ([],)),
                                     (support.snapshot_cache, ([],)),
                                     (support.restore_cache, ([], []))):
            with self.assertRaises(ValueError):
                operation(*arguments)
        support.validate_config(RuntimeConfig(dspark_enabled=True))
        for config in (RuntimeConfig(mtp_enabled=True),
                       RuntimeConfig(staged_expert_streaming=True)):
            with self.assertRaises(ValueError):
                support.validate_config(config)

    def test_model_request_policy_controls_sampling_and_reasoning(self):
        support = model_support.get_support("qwen3.8-flash-next")
        self.assertEqual(_reasoning_settings({"reasoning_effort": "medium"}, support=support),
                         ("thinking", "medium"))
        options = _options({}, ServerDefaults(), support=support,
                           approximation_default=support.default_approximation(), thinking_mode="thinking")
        self.assertEqual((options.temperature, options.top_p, options.top_k), (1.0, 0.95, 20))

    def test_fourth_package_loads_generates_recovers_and_closes_without_core_branches(self):
        descriptor = replace(DESCRIPTORS[0], kind="test-fourth", api_model_id="test-fourth",
                             checkpoint_model_id="fixture/fourth", checkpoint_revision="fixture",
                             manifest_version=1, features=frozenset({"promptCache"}))
        types = {**model_support.support_types(), descriptor.kind: _FourthSupport}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experts").mkdir()
            (root / "common.bin").write_bytes(bytes(4))
            (root / "experts/layer_00.bin").write_bytes(bytes(4))
            (root / "config.json").write_text("{}")
            tokenizer = Tokenizer(models.WordLevel({"<unk>": 0, "hi": 1, "ok": 2, "<eos>": 3}, unk_token="<unk>"))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>", eos_token="<eos>").save_pretrained(root / "tokenizer")
            tensor = {"name": "fixture", "dtype": "F32", "shape": [1], "offset": 0, "length": 4}
            (root / "manifest.json").write_text(json.dumps({
                "formatVersion": 1, "modelKind": descriptor.kind, "modelID": descriptor.checkpoint_model_id,
                "revision": "fixture", "commonTensors": [tensor], "expertRegions": [tensor],
                "files": [{"path": name, "size": (root / name).stat().st_size}
                          for name in ("common.bin", "config.json", "experts/layer_00.bin")],
            }))
            with (patch.object(model_support, "support_types", return_value=types),
                  patch.object(model_support, "BY_KIND", {**BY_KIND, descriptor.kind: descriptor}),
                  patch.dict(model_manager.MODEL_IDS, {descriptor.kind: descriptor.api_model_id}),
                  patch.dict(model_manager.MODEL_OWNERS, {descriptor.kind: descriptor.owner})):
                model_support.get_support.cache_clear()
                try:
                    config = RuntimeConfig(persistent_prompt_cache=False)
                    specs = model_manager.parse_model_catalog({"version": 1, "models": [{
                        "id": descriptor.api_model_id, "alias": None, "model_kind": descriptor.kind,
                        "path": str(root), "runtime": asdict(config), "warmup_prompt_path": None,
                        "defaults": {"max_tokens": 2, "temperature": 0, "top_p": 1, "top_k": 0},
                    }]})
                    manager = model_manager.ModelManager(specs)
                    self.assertEqual(manager.models()[0]["id"], descriptor.api_model_id)
                    with manager.request(descriptor.api_model_id) as request:
                        runtime = request.runtime
                        self.assertEqual(runtime.installed.model_kind, descriptor.kind)
                        pieces = list(runtime.stream("hi", GenerationOptions(max_tokens=2, temperature=0)))
                        self.assertEqual([piece.token for piece in pieces], [2, 2])
                        self.assertEqual("".join(piece.text for piece in pieces).strip(), "ok ok")
                        pending = runtime.stream("hi hi", GenerationOptions(max_tokens=3, temperature=0))
                        next(pending)
                        pending.close()
                    with manager.request(descriptor.api_model_id) as request:
                        resumed = list(request.runtime.stream("hi", GenerationOptions(max_tokens=1, temperature=0)))
                        self.assertEqual([piece.token for piece in resumed], [2])
                    manager.unload(descriptor.api_model_id)
                    runtime.close()
                    self.assertEqual(runtime.expert_cache.closes, 1)
                finally:
                    model_support.get_support.cache_clear()

    def test_cleanup_closes_remaining_resources_after_one_failure(self):
        support = model_support.get_support("deepseek-v4")
        good = _Resources()
        class Broken:
            def close(self):
                raise OSError("close failed")
        with self.assertRaises(OSError):
            support.close(SimpleNamespace(mtp_expert_cache=Broken(), ane_prefill=good), good)
        self.assertEqual(good.closes, 1)
