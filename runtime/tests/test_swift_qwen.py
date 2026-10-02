"""Independent Swift identity on the existing Qwen engine; no full-model inference.

The fixture is the unchanged public manifest at artifact commit
257cb509d72ecc07519be35a8b7558fde3b31b21 of
Yanun/Swift1.5-Qwen3.8-Flash-Next-Whallm-MXFP4.
"""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from deepseek_v4_ssd.generation import ModelRuntime, _prompt_cache_contract
from deepseek_v4_ssd.manifest import InstalledModel
from deepseek_v4_ssd.model import RuntimeConfig
from deepseek_v4_ssd.model_manager import ModelDefaults, parse_model_catalog
from deepseek_v4_ssd.model_support import get_support, support_for_manifest
from deepseek_v4_ssd.model_support.catalog import BY_KIND
from deepseek_v4_ssd.model_support.qwen import QwenSupport

SWIFT = "swift1.5-qwen3.8-flash-next"
FP8 = "qwen3.8-flash-next"
FIXTURE = Path(__file__).with_name("fixtures") / "swift_qwen_manifest.json"


class SwiftQwenTests(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads(FIXTURE.read_bytes())
        self.support = get_support(SWIFT)

    def test_published_manifest_identity_and_size(self):
        self.assertEqual(hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
                         "64ba0d5965085a6aedc0e81035b4d9fd7a6f5ad48581d2caf2034f39dc233b38")
        before = copy.deepcopy(self.raw)
        self.assertIs(support_for_manifest(self.raw), self.support)
        contract = self.support.manifest_contract(self.raw)
        self.assertEqual(contract["model_kind"], SWIFT)
        self.assertEqual(self.raw, before)
        self.assertEqual(sum(f["size"] for f in self.raw["files"]), 126_816_657_774)
        self.assertEqual(len(contract["required"]), 59)
        self.raw["modelKind"] = SWIFT
        self.assertEqual(self.support.manifest_contract(self.raw), contract)

    def test_wrong_source_revision_and_mtp_are_rejected(self):
        with self.assertRaises(ValueError):
            get_support(FP8).manifest_contract(self.raw)
        for key, value in (("modelID", "unknown/model"), ("revision", "0" * 40),
                           ("modelKind", "deepseek-v4.1"), ("mtp", None), ("formatVersion", 3)):
            raw = {**self.raw, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                support_for_manifest(raw).manifest_contract(raw)
        fp8 = BY_KIND[FP8]
        raw = {**self.raw, "modelID": fp8.checkpoint_model_id, "revision": fp8.checkpoint_revision}
        with self.assertRaises(ValueError):
            self.support.manifest_contract(raw)
        self.assertEqual(get_support(FP8).manifest_contract(raw)["model_kind"], FP8)

    def test_sparse_installed_files_open_as_swift_and_use_qwen_layout(self):
        # Sparse zero-filled files validate paths/sizes/tensor tables only, not
        # actual weight content, hashes, generation quality or performance.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_bytes(FIXTURE.read_bytes())
            for f in self.raw["files"]:
                target = root / f["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("wb") as out:
                    out.truncate(f["size"])
            installed = InstalledModel.open(root)
            self.assertEqual(installed.model_kind, SWIFT)
            self.assertTrue(installed.is_qwen)
            self.assertTrue(installed.has_mtp)
            self.assertEqual(installed.model_id, BY_KIND[SWIFT].checkpoint_model_id)
            self.assertIs(self.support.expert_layout, get_support(FP8).expert_layout)

    def test_reuses_loader_with_own_installed_weights_config_and_mtp(self):
        self.assertIsInstance(self.support, QwenSupport)
        self.assertIsNot(self.support, get_support(FP8))
        installed = object()
        config = RuntimeConfig(mtp_enabled=True, qwen_ngram_io="pread", qwen_sparse_sdpa=True)
        raw_config, weights, limiter = {}, {}, object()
        model, experts, mtp, mtp_experts = SimpleNamespace(), object(), object(), object()
        with patch("deepseek_v4_ssd.qwen4_exp.load", return_value=(model, experts)) as load, \
             patch("deepseek_v4_ssd.model._load_qwen_mtp", return_value=(mtp, mtp_experts)) as load_mtp:
            model.args = object()
            self.assertEqual(self.support.load(installed, config, raw_config, weights, limiter), (model, experts))
            load.assert_called_once_with(installed, config, raw_config, weights, limiter)
            load_mtp.assert_called_once_with(installed, model.args, config, limiter)
            self.assertIs(model.mtp, mtp)
            self.assertIs(model.mtp_expert_cache, mtp_experts)

    def test_qwen_options_and_sampling_remain_independently_configurable(self):
        config = RuntimeConfig(qwen_ngram_io="pread", qwen_sparse_sdpa=True,
                               qwen_quantized_kv=True, qwen_quantized_index=True,
                               qwen_prefill_read_experts=4, mtp_enabled=True)
        self.support.validate_config(config)
        with self.assertRaises(ValueError):
            get_support("deepseek-v4").validate_config(config)
        self.assertTrue(self.support.uses_layer_major_prefill(config, 128))
        self.assertFalse(self.support.uses_layer_major_prefill(config, 127))
        with self.assertRaises(ValueError):
            self.support.validate_config(replace(config, dspark_enabled=True))
        models = []
        for kind, cfg, temperature in ((FP8, RuntimeConfig(), 0.7), (SWIFT, config, 0.4)):
            models.append(dict(id=BY_KIND[kind].api_model_id, alias=None, path=f"/fixture/{kind}",
                               model_kind=kind, runtime=asdict(cfg), warmup_prompt_path=None,
                               defaults=asdict(ModelDefaults(128, temperature, 0.8, 20,
                                                             qwen_adaptive_sampling=False))))
        original, swift = parse_model_catalog(dict(version=1, models=models))
        self.assertNotEqual(original.id, swift.id)
        self.assertEqual(original.runtime.qwen_ngram_io, "mmap")
        self.assertEqual(swift.runtime.qwen_ngram_io, "pread")
        self.assertEqual(self.support.sampling_defaults(swift.defaults, "chat")["temperature"], 0.4)
        self.assertEqual(get_support(FP8).sampling_defaults(original.defaults, "chat")["temperature"], 0.7)

    def test_disk_cache_identity_and_state_do_not_cross_models(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RuntimeConfig(persistent_prompt_cache=True, prompt_cache_directory=directory)
            roots, contracts = [], []
            for kind in (FP8, SWIFT):
                d = BY_KIND[kind]
                installed = SimpleNamespace(root=Path(directory), model_id=d.checkpoint_model_id,
                                            revision=d.checkpoint_revision, format_version=2)
                runtime = ModelRuntime.__new__(ModelRuntime)
                runtime.support, runtime.config, runtime.installed = get_support(kind), config, installed
                roots.append(runtime._open_prompt_cache_directory())
                contracts.append(_prompt_cache_contract(installed, config))
            self.assertTrue(all(root.is_dir() for root in roots))
            self.assertNotEqual(*roots)
            self.assertNotEqual(*contracts)
            state = [[1, 2]]
            cloned = self.support.clone_cache(state)
            cloned[0].append(3)
            self.assertEqual(state, [[1, 2]])
