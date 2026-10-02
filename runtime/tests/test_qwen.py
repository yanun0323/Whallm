from __future__ import annotations

import json
import tempfile
import threading
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.cache import ArraysCache, CacheList, KVCache

from deepseek_v4_ssd.expert_cache import ExpertCache
from deepseek_v4_ssd.ane_prefill import (
    ANEPrefillController,
    ANEPrefillLinear,
    _ane_output_channels,
)
from deepseek_v4_ssd.generation import _qwen_layer_major_prefill
from deepseek_v4_ssd.cancellation import GenerationCancelled, cancellation_scope
from deepseek_v4_ssd.manifest import (
    QWEN_EXPERT_REGIONS,
    QWEN_MODEL_ID,
    QWEN_NGRAM_HEAD_OFFSETS,
    QWEN_NGRAM_HEAD_VOCAB_SIZES,
    QWEN_REVISION,
    InstalledModel,
    NGram,
    Tensor,
    _qwen_contract,
)
from deepseek_v4_ssd.qwen4_exp import (
    ModelArgs,
    MTPModel,
    NGramStore,
    PLELayer,
    QSAAttention,
    RMSNormGated,
    SparseMoE,
    generate_mtp_tokens,
    ngram_ids,
    qsa_causal_block_mask,
    mtp_prefill_pairs,
    rollback_mtp_cache,
)
from deepseek_v4_ssd.tool_codec import (
    QwenToolCodec,
    QwenToolStreamParser,
    ToolCall,
    ToolChoice,
)


class FakeTokenizer:
    def __init__(self):
        self.arguments = None

    def apply_chat_template(self, messages, **arguments):
        self.arguments = (messages, arguments)
        return "qwen prompt"


class FakeTargetCache:
    def __init__(self):
        self.offset = 0


class FakeMTPGenerationCache:
    def __init__(self):
        self.offset = 0

    def size(self):
        return self.offset

    def trim(self, count):
        self.offset -= count
        return count


class FakeGreedyTarget:
    def __init__(self):
        weight = mx.zeros((32, 1))
        self.model = SimpleNamespace(
            embed_tokens=SimpleNamespace(weight=weight),
        )
        self.lm_head = SimpleNamespace(weight=weight)
        self.maximum_input_tokens = 0

    def forward_with_hidden(self, input_ids, cache):
        tokens = np.asarray(input_ids, dtype=np.int32)
        self.maximum_input_tokens = max(self.maximum_input_tokens, tokens.shape[1])
        cache[0].offset += tokens.shape[1]
        logits = np.full((*tokens.shape, 32), -1_000.0, dtype=np.float32)
        for position, token in enumerate(tokens[0]):
            logits[0, position, (int(token) + 1) % 32] = 1_000.0
        hidden = tokens[..., None].astype(np.float32)
        return mx.array(logits), mx.array(hidden)


class FakeGreedyMTP:
    def __init__(self, reject_first_draft=False, reject_input_token=None):
        self.reject_first_draft = reject_first_draft
        self.reject_input_token = reject_input_token
        self.cache = None
        self.expert_cache = SimpleNamespace(slots=10)
        self.args = SimpleNamespace(num_experts_per_tok=10)
        self.maximum_input_tokens = 0

    def make_cache(self):
        self.cache = FakeMTPGenerationCache()
        return self.cache

    def __call__(
        self,
        target_hidden,
        next_token_ids,
        embedding_weight,
        lm_head_weight,
        cache,
    ):
        del target_hidden, embedding_weight, lm_head_weight
        tokens = np.asarray(next_token_ids, dtype=np.int32)
        self.maximum_input_tokens = max(self.maximum_input_tokens, tokens.shape[1])
        cache.offset += tokens.shape[1]
        logits = np.full((*tokens.shape, 32), -1_000.0, dtype=np.float32)
        for position, token in enumerate(tokens[0]):
            prediction = int(token) + 1
            if (
                self.reject_first_draft and int(token) == 3
            ) or int(token) == self.reject_input_token:
                prediction += 1
            logits[0, position, prediction % 32] = 1_000.0
        hidden = tokens[..., None].astype(np.float32)
        return mx.array(logits), mx.array(hidden)


class FakeSamplingTarget(FakeGreedyTarget):
    def __init__(self, supported_tokens=(0, 1)):
        super().__init__()
        self.supported_tokens = supported_tokens

    def forward_with_hidden(self, input_ids, cache):
        tokens = np.asarray(input_ids, dtype=np.int32)
        self.maximum_input_tokens = max(self.maximum_input_tokens, tokens.shape[1])
        cache[0].offset += tokens.shape[1]
        logits = np.full((*tokens.shape, 32), -np.inf, dtype=np.float32)
        logits[..., list(self.supported_tokens)] = 0
        hidden = tokens[..., None].astype(np.float32)
        return mx.array(logits), mx.array(hidden)


class FakeSamplingMTP(FakeGreedyMTP):
    def __init__(self, supported_tokens=(0, 1)):
        super().__init__()
        self.supported_tokens = supported_tokens

    def __call__(
        self,
        target_hidden,
        next_token_ids,
        embedding_weight,
        lm_head_weight,
        cache,
    ):
        del target_hidden, embedding_weight, lm_head_weight
        tokens = np.asarray(next_token_ids, dtype=np.int32)
        self.maximum_input_tokens = max(self.maximum_input_tokens, tokens.shape[1])
        cache.offset += tokens.shape[1]
        logits = np.full((*tokens.shape, 32), -np.inf, dtype=np.float32)
        logits[..., list(self.supported_tokens)] = 0
        hidden = tokens[..., None].astype(np.float32)
        return mx.array(logits), mx.array(hidden)


class FakeANEProjection:
    def __init__(self, weight, spatial, error=None):
        self.weight = np.asarray(weight, dtype=np.float16)
        self.input_channels = self.weight.shape[1]
        self.output_channels = self.weight.shape[0]
        self.spatial = spatial
        self.error = error
        self.closed = False

    def evaluate(self, value):
        if self.error is not None:
            raise RuntimeError(self.error)
        return value @ self.weight.T

    def close(self):
        self.closed = True


class QwenTests(unittest.TestCase):
    descriptor = NGram(
        "ngram.bin",
        "F8_E4M3",
        4,
        2,
        8,
        tuple(range(0, 160, 10)),
        tuple(range(11, 27)),
    )

    def test_ane_prefill_ratio_selects_aligned_output_channels(self):
        self.assertEqual(_ane_output_channels(0), 0)
        self.assertEqual(_ane_output_channels(0.25), 3_072)
        self.assertEqual(_ane_output_channels(0.5), 6_144)
        self.assertEqual(_ane_output_channels(1), 12_288)
        self.assertEqual(_ane_output_channels(0.333), 4_096)
        for ratio in (-0.1, 1.1, True):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                _ane_output_channels(ratio)

    def test_ane_prefill_splits_output_channels_without_reduction(self):
        for ane_channels in (2, 6):
            with self.subTest(ane_channels=ane_channels):
                linear = nn.Linear(4, 6, bias=False)
                linear.weight = mx.arange(24, dtype=mx.float32).reshape(6, 4) / 100
                projection = FakeANEProjection(
                    np.asarray(linear.weight[-ane_channels:]), spatial=3
                )
                controller = ANEPrefillController(True)
                controller.add(projection)
                controller.active = True
                split = ANEPrefillLinear(linear, projection, controller)
                value = mx.arange(12, dtype=mx.float32).reshape(1, 3, 4) / 10

                expected = linear(value)
                result = split(value)
                mx.eval(expected, result)

                np.testing.assert_allclose(
                    np.asarray(result), np.asarray(expected), rtol=0, atol=5e-4
                )
                self.assertEqual(controller.evaluations, 1)
                self.assertEqual(controller.fallbacks, 0)

    def test_ane_prefill_error_disables_ane_and_retries_original_projection(self):
        linear = nn.Linear(4, 6, bias=False)
        linear.weight = mx.arange(24, dtype=mx.float32).reshape(6, 4) / 100
        projection = FakeANEProjection(
            np.asarray(linear.weight[-2:]), spatial=3, error="interface changed"
        )
        controller = ANEPrefillController(True)
        controller.add(projection)
        controller.active = True
        split = ANEPrefillLinear(linear, projection, controller)
        value = mx.arange(12, dtype=mx.float32).reshape(1, 3, 4) / 10

        expected = linear(value)
        result = split(value)
        mx.eval(expected, result)

        np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))
        self.assertFalse(controller.active)
        self.assertEqual(controller.fallbacks, 1)
        self.assertEqual(controller.error, "interface changed")
        self.assertTrue(projection.closed)

    def test_qwen_manifest_accepts_pinned_mtp_sidecar(self):
        raw = {
            "formatVersion": 2,
            "modelKind": "qwen3.8-flash-next",
            "modelID": QWEN_MODEL_ID,
            "revision": QWEN_REVISION,
            "layerCount": 48,
            "expertCount": 512,
            "selectedExpertCount": 10,
            "expertBlobSize": 2_611_200,
            "maximumContext": 262_144,
            "expertQuantization": {
                "bits": 4,
                "conversionVersion": 2,
                "groupSize": 32,
                "mode": "mxfp4",
            },
            "ngram": {
                "file": "ngram.bin",
                "dtype": "F8_E4M3",
                "rowBytes": 160,
                "shardCount": 128,
                "shardRowCount": 2_500_012,
                "headOffsets": list(QWEN_NGRAM_HEAD_OFFSETS),
                "headVocabSizes": list(QWEN_NGRAM_HEAD_VOCAB_SIZES),
            },
            "files": [
                {"path": "ngram.bin", "size": 128 * 2_500_012 * 160},
                {"path": "mtp/common.bin", "size": 181_136_896},
                {"path": "mtp/experts/layer_00.bin", "size": 512 * 2_611_200},
            ],
            "mtp": {
                "layerCount": 1,
                "useDedicatedEmbeddings": False,
                "commonTensors": [
                    {
                        "name": f"mtp.fixture_{index}",
                        "dtype": "BF16",
                        "shape": [1],
                        "offset": index * 2,
                        "length": 2,
                    }
                    for index in range(29)
                ],
            },
        }

        contract = _qwen_contract(raw)

        self.assertIn("mtp/common.bin", contract["required"])
        self.assertIn("mtp/experts/layer_00.bin", contract["required"])

    def test_mtp_prefill_pairs_next_token_with_previous_hidden_state(self):
        hidden = mx.arange(5 * 4).reshape(1, 5, 4)
        tokens = mx.array([[10, 11, 12, 13, 14]])

        paired_hidden, paired_tokens = mtp_prefill_pairs(hidden, tokens)

        np.testing.assert_array_equal(np.asarray(paired_hidden), np.arange(16).reshape(1, 4, 4))
        np.testing.assert_array_equal(np.asarray(paired_tokens), [[11, 12, 13, 14]])

    def test_mtp_rollback_restores_both_qsa_caches(self):
        cache = CacheList(KVCache(), KVCache())
        cache[0].offset = 7
        cache[1].offset = 7

        rollback_mtp_cache(cache, 4)

        self.assertEqual(cache[0].offset, 4)
        self.assertEqual(cache[1].offset, 4)

    def test_mtp_graph_uses_qsa_and_two_cache_branches(self):
        args = ModelArgs(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=4,
            shared_expert_intermediate_size=4,
            indexer_n_heads=1,
            indexer_kv_heads=1,
            indexer_head_dim=4,
            hc_count=2,
            hc_lowrank=2,
            layer_types=("full_attention",),
            ple_layer_ids=(),
        )
        model = MTPModel(args, SimpleNamespace())

        self.assertEqual(model.layers[0].layer_type, "full_attention")
        self.assertEqual(len(model.make_cache().caches), 2)
        self.assertEqual(model.fc_hidden.weight.shape, (8, 8))
        self.assertIsNone(model.pre_fc_norm_hidden.group_size)

    def test_mtp_generation_commits_verified_drafts_and_bonus_token(self):
        target = FakeGreedyTarget()
        mtp = FakeGreedyMTP()
        target_cache = [FakeTargetCache()]
        rounds = []

        generated = list(
            generate_mtp_tokens(
                [1, 2],
                target,
                mtp,
                target_cache,
                max_tokens=7,
                prefill_step_size=1,
                record_round=lambda *values: rounds.append(values),
            )
        )

        self.assertEqual([token for token, _ in generated], [3, 4, 5, 6, 7, 8, 9])
        self.assertEqual([draft for _, draft in generated], [False, True, True, True, True, True, False])
        self.assertEqual(target_cache[0].offset, 8)
        self.assertEqual(target.maximum_input_tokens, 6)
        self.assertEqual(mtp.cache.offset, 7)
        self.assertEqual(mtp.maximum_input_tokens, 1)
        self.assertEqual(rounds[0][:2], (5, 5))
        self.assertFalse(rounds[0][-1])

    def test_mtp_sampling_accepts_an_identical_draft_distribution(self):
        mx.random.seed(7)
        target = FakeSamplingTarget()
        mtp = FakeSamplingMTP()
        target_cache = [FakeTargetCache()]
        rounds = []
        processor_contexts = []

        def record_context(tokens, logits):
            processor_contexts.append(np.asarray(tokens).tolist())
            return logits

        generated = list(
            generate_mtp_tokens(
                [1, 2],
                target,
                mtp,
                target_cache,
                max_tokens=3,
                prefill_step_size=2,
                temperature=0.7,
                top_p=0.8,
                top_k=2,
                logits_processors=[record_context],
                record_round=lambda *values: rounds.append(values),
            )
        )

        self.assertEqual([draft for _, draft in generated], [False, True, False])
        self.assertEqual(rounds[0][:2], (1, 1))
        self.assertEqual(processor_contexts[0], [2])
        self.assertIn([2, generated[0][0]], processor_contexts)

    def test_mtp_sampling_uses_the_corrected_target_distribution(self):
        mx.random.seed(7)
        target = FakeSamplingTarget((0,))
        mtp = FakeSamplingMTP((1,))
        rounds = []

        generated = list(
            generate_mtp_tokens(
                [1, 2],
                target,
                mtp,
                [FakeTargetCache()],
                max_tokens=3,
                prefill_step_size=2,
                temperature=1.0,
                record_round=lambda *values: rounds.append(values),
            )
        )

        self.assertEqual([token for token, _ in generated], [0, 0, 0])
        self.assertEqual([draft for _, draft in generated], [False, False, False])
        self.assertEqual(rounds[0][:2], (1, 0))
        self.assertTrue(rounds[0][-1])

    def test_mtp_generation_falls_back_after_zero_acceptance(self):
        target = FakeGreedyTarget()
        mtp = FakeGreedyMTP(reject_first_draft=True)
        target_cache = [FakeTargetCache()]
        rounds = []

        generated = list(
            generate_mtp_tokens(
                [1, 2],
                target,
                mtp,
                target_cache,
                max_tokens=7,
                prefill_step_size=2,
                record_round=lambda *values: rounds.append(values),
            )
        )

        self.assertEqual([token for token, _ in generated], [3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(target_cache[0].offset, 8)
        self.assertEqual(mtp.cache.offset, 2)
        self.assertEqual(rounds[0][:2], (5, 0))
        self.assertTrue(rounds[0][-1])

    def test_mtp_generation_replays_only_a_partially_accepted_prefix(self):
        target = FakeGreedyTarget()
        mtp = FakeGreedyMTP(reject_input_token=5)
        target_cache = [FakeTargetCache()]
        rounds = []

        generated = list(
            generate_mtp_tokens(
                [1, 2],
                target,
                mtp,
                target_cache,
                max_tokens=5,
                prefill_step_size=2,
                record_round=lambda *values: rounds.append(values),
            )
        )

        self.assertEqual([token for token, _ in generated], [3, 4, 5, 6, 7])
        self.assertEqual(target_cache[0].offset, 6)
        self.assertEqual(mtp.cache.offset, 4)
        self.assertEqual(rounds[0][:2], (3, 2))
        self.assertGreater(rounds[0][4], 0)

    def test_mtp_generation_accepts_layer_major_prefill_hidden_states(self):
        target = FakeGreedyTarget()
        mtp = FakeGreedyMTP()
        target_cache = [FakeTargetCache()]
        target_cache[0].offset = 1

        generated = list(
            generate_mtp_tokens(
                [1, 2],
                target,
                mtp,
                target_cache,
                max_tokens=3,
                prefill_step_size=2,
                prefilled_hidden=mx.array([[[1.0]]]),
            )
        )

        self.assertEqual([token for token, _ in generated], [3, 4, 5])
        self.assertEqual(target_cache[0].offset, 4)

    def test_mtp_layer_major_prefill_covers_whole_prompt_in_slot_chunks(self):
        target = FakeGreedyTarget()
        mtp = FakeGreedyMTP()
        mtp.expert_cache.slots = 20
        target_cache = [FakeTargetCache()]
        target_cache[0].offset = 6

        generated = list(
            generate_mtp_tokens(
                [1, 2, 3, 4, 5, 6, 7],
                target,
                mtp,
                target_cache,
                max_tokens=1,
                prefill_step_size=128,
                prefilled_hidden=mx.ones((1, 6, 1)),
            )
        )

        self.assertEqual([token for token, _ in generated], [8])
        # Without whole-layer buffers, chunks stay within slots // top-k.
        self.assertEqual(mtp.cache.offset, 6)
        self.assertEqual(mtp.maximum_input_tokens, 2)
        self.assertEqual(target_cache[0].offset, 7)

    def test_mtp_layer_major_prefill_uses_one_whole_layer_buffer(self):
        target = FakeGreedyTarget()
        mtp = FakeGreedyMTP()
        events = []

        @contextmanager
        def reuse_layer_buffers():
            events.append("reuse")
            yield
            events.append("release")

        @contextmanager
        def batched_layer(layer):
            events.append(("layer", layer))
            yield
            events.append(("done", layer, mtp.cache.offset))

        mtp.expert_cache = SimpleNamespace(
            slots=10, reuse_layer_buffers=reuse_layer_buffers, batched_layer=batched_layer)
        target_cache = [FakeTargetCache()]
        target_cache[0].offset = 6

        generated = list(
            generate_mtp_tokens(
                [1, 2, 3, 4, 5, 6, 7],
                target,
                mtp,
                target_cache,
                max_tokens=1,
                prefill_step_size=4,
                prefilled_hidden=mx.ones((1, 6, 1)),
            )
        )

        self.assertEqual([token for token, _ in generated], [8])
        self.assertEqual(events, ["reuse", ("layer", 0), ("done", 0, 6), "release"])
        self.assertEqual(mtp.maximum_input_tokens, 4)
        self.assertEqual(target_cache[0].offset, 7)

    def test_ngram_hash_keeps_cross_chunk_context(self):
        tokens = np.array([[4, 5, 6, 7]], dtype=np.int64)
        multipliers = np.array([1, 3, 5], dtype=np.int64)
        complete = ngram_ids(
            np.concatenate([np.array([[99, 99]]), tokens], axis=1),
            multipliers,
            self.descriptor,
            eos_token_id=99,
        )[:, -4:]
        first = ngram_ids(
            np.array([[99, 99, 4, 5]]),
            multipliers,
            self.descriptor,
            eos_token_id=99,
        )[:, -2:]
        second = ngram_ids(
            np.array([[4, 5, 6, 7]]),
            multipliers,
            self.descriptor,
            eos_token_id=99,
        )[:, -2:]

        np.testing.assert_array_equal(np.concatenate([first, second], axis=1), complete)

    def test_ngram_store_copies_only_requested_rows(self):
        descriptor = NGram(
            "ngram.bin", "F8_E4M3", 2, 2, 3, (0,) * 16, (3,) * 16
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ngram.bin"
            np.array(
                [
                    0x00, 0x30, 0x38, 0x3C, 0x40, 0x44,
                    0x48, 0x4C, 0xB8, 0xBC, 0xC0, 0xC4,
                ],
                dtype=np.uint8,
            ).tofile(path)
            store = NGramStore(path, descriptor, 0.5)

            result = store.lookup(np.array([[1, 4]]))
            mx.eval(result)

            np.testing.assert_array_equal(
                np.array(result.astype(mx.float32)),
                np.array([[[0.5, 0.75], [-0.5, -0.75]]], dtype=np.float32),
            )

    def test_ple_matches_across_prompt_chunks(self):
        descriptor = NGram(
            "ngram.bin", "F8_E4M3", 2, 1, 100, (0,) * 16, (100,) * 16
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ngram.bin"
            np.resize(
                np.array([0x30, 0x38, 0x3C, 0x40], dtype=np.uint8), 200
            ).tofile(path)
            args = ModelArgs(
                hidden_size=4,
                hc_count=2,
                hc_lowrank=2,
                ple_embed_dim=32,
                ple_conv_kernel_size=2,
                ngram_size=3,
                heads_per_ngram=8,
                eos_token_id=99,
            )
            layer = PLELayer(args, NGramStore(path, descriptor))
            tokens = mx.array([[1, 2, 3, 4]])
            hidden = mx.arange(32, dtype=mx.float32).reshape(1, 4, 8) / 100
            complete = layer(hidden, tokens, ArraysCache(size=4))
            cache = ArraysCache(size=4)
            chunked = mx.concatenate(
                [
                    layer(hidden[:, :2], tokens[:, :2], cache),
                    layer(hidden[:, 2:], tokens[:, 2:], cache),
                ],
                axis=1,
            )
            mx.eval(complete, chunked)

            np.testing.assert_allclose(
                np.array(chunked), np.array(complete), rtol=0, atol=1e-5
            )

    def test_qsa_causal_mask_selects_only_complete_blocks(self):
        np.testing.assert_array_equal(
            qsa_causal_block_mask(np.array([2, 3, 7]), 3),
            np.array(
                [
                    [False, False, False],
                    [True, False, False],
                    [True, True, False],
                ]
            ),
        )

    def test_gated_delta_norm_uses_sigmoid(self):
        norm = RMSNormGated(2, 1e-6)
        value = mx.array([[[3.0, 4.0]]])
        gate = mx.array([[[1.0, -1.0]]])
        result = norm(value, gate)
        mx.eval(result)

        gate_array = np.array(gate)
        expected = np.array(value / np.sqrt(12.5 + 1e-6)) * (
            1 / (1 + np.exp(-gate_array))
        )
        np.testing.assert_allclose(np.array(result), expected, rtol=0, atol=1e-6)

    def test_qsa_chunked_prefill_matches_one_pass(self):
        mx.random.seed(0)
        args = ModelArgs(
            hidden_size=16,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            indexer_n_heads=2,
            indexer_kv_heads=1,
            indexer_head_dim=8,
            indexer_budget=4,
            indexer_compress_ratio=2,
            partial_rotary_factor=0.5,
            max_position_embeddings=64,
        )
        hidden = mx.arange(96, dtype=mx.float32).reshape(1, 6, 16) / 100
        attention = QSAAttention(args)
        complete = attention(hidden, None)
        cache = CacheList(KVCache(), KVCache())
        chunked = mx.concatenate(
            [attention(hidden[:, :3], cache), attention(hidden[:, 3:], cache)],
            axis=1,
        )
        mx.eval(complete, chunked)

        np.testing.assert_allclose(
            np.array(chunked), np.array(complete), rtol=0, atol=1e-6
        )

    def test_qsa_does_not_repeat_kv_heads(self):
        mx.random.seed(0)
        args = ModelArgs(
            hidden_size=16,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            indexer_n_heads=2,
            indexer_kv_heads=1,
            indexer_head_dim=8,
            indexer_budget=4,
            indexer_compress_ratio=2,
            partial_rotary_factor=0.5,
            max_position_embeddings=64,
        )
        hidden = mx.arange(96, dtype=mx.float32).reshape(1, 6, 16) / 100
        attention = QSAAttention(args)
        with patch(
            "deepseek_v4_ssd.qwen4_exp.mx.repeat",
            side_effect=AssertionError("grouped KV must not repeat K/V heads"),
        ):
            result = attention(hidden, None)
            mx.eval(result)

    def test_qsa_decode_uses_both_caches(self):
        mx.random.seed(0)
        args = ModelArgs(
            hidden_size=16,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            indexer_n_heads=2,
            indexer_kv_heads=1,
            indexer_head_dim=8,
            indexer_budget=4,
            indexer_compress_ratio=2,
            partial_rotary_factor=0.5,
            max_position_embeddings=64,
        )
        hidden = mx.arange(96, dtype=mx.float32).reshape(1, 6, 16) / 100
        attention = QSAAttention(args)
        complete = attention(hidden, None)[:, -1:]
        cache = CacheList(KVCache(), KVCache())
        _ = attention(hidden[:, :5], cache)
        decoded = attention(hidden[:, 5:], cache)
        mx.eval(complete, decoded)

        np.testing.assert_allclose(
            np.array(decoded), np.array(complete), rtol=0, atol=3e-4
        )
        self.assertEqual(cache[0].offset, 6)
        self.assertEqual(cache[1].offset, 6)

    def test_qwen_codec_and_stream_parser_produce_same_call(self):
        tokenizer = FakeTokenizer()
        codec = QwenToolCodec(tokenizer)
        prompt = codec.encode(
            [{"role": "user", "content": "Weather?"}],
            "thinking",
            [{"type": "function", "function": {"name": "weather"}}],
            ToolChoice("required"),
            "medium",
        )
        self.assertEqual(prompt, "qwen prompt")
        self.assertTrue(tokenizer.arguments[1]["enable_thinking"])
        self.assertEqual(tokenizer.arguments[1]["reasoning_effort"], "medium")

        raw = (
            "plan</think>Summary\n\n<tool_call>\n<function=weather>\n"
            "<parameter=city>\nTaipei\n</parameter>\n"
            "<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>"
        )
        turn = codec.parse(raw, "thinking")
        parser = QwenToolStreamParser("thinking")
        deltas = [delta for character in raw for delta in parser.feed(character)]
        deltas.extend(parser.finish())
        call = ToolCall(
            next(delta.tool_name for delta in deltas if delta.tool_name),
            "".join(delta.arguments for delta in deltas if delta.tool_index == 0),
        )

        self.assertEqual(turn.reasoning_content, "plan")
        self.assertEqual(turn.content, "Summary")
        self.assertEqual(turn.tool_calls, (ToolCall("weather", '{"city":"Taipei","days":2}'),))
        self.assertTrue(parser.matches((call,)))
        self.assertTrue(parser.matches(turn.tool_calls))

    def test_qwen_stream_parser_multiple_calls_at_all_chunk_boundaries(self):
        command = "  printf '繁中'\n  "
        first = (
            "<tool_call><function=bash><parameter=command>"
            + json.dumps(command, ensure_ascii=False)
            + "</parameter></function></tool_call>"
        )
        second = (
            "<tool_call><function=read><parameter=path>README.md"
            "</parameter></function></tool_call>"
        )
        expected = (
            ToolCall("bash", json.dumps({"command": command}, ensure_ascii=False, separators=(",", ":"))),
            ToolCall("read", '{"path":"README.md"}'),
        )
        for mode in ("chat", "thinking"):
            prefix = "plan</think>Checking files.\n" if mode == "thinking" else "Checking files.\n"
            for separator in ("", "\n", "\r\n \t", "\u2003"):
                raw = prefix + first + separator + second + "\n"
                turn = QwenToolCodec(None).parse(raw, mode)
                self.assertEqual(turn.tool_calls, expected)
                chunkings = [[raw], list(raw)]
                chunkings.extend([raw[:split], raw[split:]] for split in range(len(raw) + 1))
                for case, chunks in enumerate(chunkings):
                    with self.subTest(mode=mode, separator=repr(separator), case=case):
                        parser = QwenToolStreamParser(mode)
                        deltas = [delta for chunk in chunks for delta in parser.feed(chunk)]
                        deltas.extend(parser.finish())
                        self.assertFalse(parser.failed)
                        self.assertTrue(parser.matches(turn.tool_calls))
                        self.assertEqual(parser.streamed_tool_count, 2)
                        for index, call in enumerate(expected):
                            self.assertEqual(
                                [d.tool_name for d in deltas if d.tool_index == index and d.tool_name],
                                [call.name],
                            )
                            self.assertEqual(
                                json.loads("".join(d.arguments for d in deltas if d.tool_index == index)),
                                json.loads(call.arguments),
                            )

    def test_qwen_stream_parser_emits_each_call_before_the_next(self):
        call = "<tool_call><function=bash><parameter=command>pwd</parameter></function></tool_call>"
        parser = QwenToolStreamParser("chat")
        first = parser.feed(call)
        self.assertEqual([d.tool_name for d in first if d.tool_name], ["bash"])
        self.assertEqual(parser.streamed_tool_count, 1)
        self.assertEqual(parser.feed("\n \t"), ())
        second = parser.feed(call)
        self.assertEqual([d.tool_name for d in second if d.tool_name], ["bash"])
        self.assertEqual([d.tool_index for d in second], [1, 1])
        parser.finish()
        self.assertTrue(parser.matches(QwenToolCodec(None).parse(call + "\n \t" + call, "chat").tool_calls))

    def test_qwen_stream_parser_rejects_invalid_suffix_after_separator(self):
        first = "<tool_call><function=bash><parameter=command>pwd</parameter></function></tool_call>"
        invalid_suffixes = (
            "unexpected text",
            "<tool_call><function=read>",
            "<tool_call><function=read></tool_call>",
            "<tool_call><function=read><parameter=path>a</parameter>"
            "<parameter=path>b</parameter></function></tool_call>",
        )
        for mode in ("chat", "thinking"):
            prefix = "plan</think>" if mode == "thinking" else ""
            for suffix in invalid_suffixes:
                with self.subTest(mode=mode, suffix=suffix):
                    codec = QwenToolCodec(None)
                    turn = codec.parse(prefix + first, mode)
                    parser = QwenToolStreamParser(mode)
                    parser.feed(prefix + first)
                    for character in "\n \t" + suffix:
                        parser.feed(character)
                    parser.finish()
                    self.assertTrue(parser.failed)
                    self.assertFalse(parser.matches(turn.tool_calls))
                    with self.assertRaises(ValueError):
                        codec.parse(prefix + first + "\n \t" + suffix, mode)

    def test_qwen_codec_adds_forced_function_instruction(self):
        tokenizer = FakeTokenizer()
        QwenToolCodec(tokenizer).encode(
            [{"role": "user", "content": "Weather?"}],
            "chat",
            [{"type": "function", "function": {"name": "weather"}}],
            ToolChoice("function", "weather"),
        )

        messages = tokenizer.arguments[0]
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn('Call the "weather" tool', messages[0]["content"])

    def test_qwen_codec_adds_tool_marker_escape_instruction(self):
        tokenizer = FakeTokenizer()
        QwenToolCodec(tokenizer).encode(
            [{"role": "user", "content": "Write the file."}],
            "chat",
            [{"type": "function", "function": {"name": "write_file"}}],
        )

        messages = tokenizer.arguments[0]
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("use &lt;", messages[0]["content"])
        self.assertIn("Use &amp;lt;", messages[0]["content"])

    def test_qwen_codec_escapes_reserved_markers_in_tool_definitions(self):
        tokenizer = FakeTokenizer()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "inspect",
                    "description": "Explain literal </tools> text.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "value": {
                                "type": "string",
                                "enum": ["<|im_end|>"],
                            }
                        },
                    },
                },
            }
        ]

        QwenToolCodec(tokenizer).encode(
            [{"role": "user", "content": "Inspect."}],
            "chat",
            tools,
        )

        prepared_tools = tokenizer.arguments[1]["tools"]
        self.assertEqual(
            prepared_tools[0]["function"]["description"],
            "Explain literal &lt;/tools> text.",
        )
        self.assertEqual(
            prepared_tools[0]["function"]["parameters"]["properties"]["value"][
                "enum"
            ],
            ["&lt;|im_end|>"],
        )
        self.assertEqual(
            tools[0]["function"]["description"],
            "Explain literal </tools> text.",
        )

    def test_qwen_codec_merges_leading_developer_messages(self):
        tokenizer = FakeTokenizer()
        QwenToolCodec(tokenizer).encode(
            [
                {"role": "developer", "content": "Codex instructions."},
                {"role": "developer", "content": "Workspace instructions."},
                {"role": "user", "content": "Reply with OK."},
            ],
            "chat",
        )

        self.assertEqual(
            tokenizer.arguments[0],
            [
                {
                    "role": "system",
                    "content": "Codex instructions.\n\nWorkspace instructions.",
                },
                {"role": "user", "content": "Reply with OK."},
            ],
        )

    def test_qwen_codec_moves_all_instruction_messages_to_start(self):
        tokenizer = FakeTokenizer()
        messages = [
            {"role": "user", "content": "First request."},
            {"role": "developer", "content": "Workspace instruction."},
            {"role": "assistant", "content": "First response."},
            {"role": "system", "content": "System instruction."},
            {"role": "user", "content": "Second request."},
        ]

        QwenToolCodec(tokenizer).encode(messages, "chat")

        self.assertEqual(
            tokenizer.arguments[0],
            [
                {
                    "role": "system",
                    "content": "Workspace instruction.\n\nSystem instruction.",
                },
                {"role": "user", "content": "First request."},
                {"role": "assistant", "content": "First response."},
                {"role": "user", "content": "Second request."},
            ],
        )
        self.assertEqual(messages[1]["role"], "developer")
        self.assertEqual(messages[3]["role"], "system")

    def test_qwen_codec_adds_user_anchor_when_history_has_no_user(self):
        tokenizer = FakeTokenizer()
        QwenToolCodec(tokenizer).encode(
            [
                {"role": "developer", "content": "Reply with OK."},
                {"role": "assistant", "content": "Prior response."},
            ],
            "chat",
        )

        self.assertEqual(
            tokenizer.arguments[0],
            [
                {"role": "system", "content": "Reply with OK."},
                {"role": "user", "content": ""},
                {"role": "assistant", "content": "Prior response."},
            ],
        )

    def test_qwen_codec_escapes_reserved_markers_in_messages(self):
        tokenizer = FakeTokenizer()
        messages = [
            {"role": "system", "content": "Literal <|im_start|> token."},
            {
                "role": "user",
                "content": "  <tool_response>\nliteral text\n</tool_response>  ",
            },
            {
                "role": "assistant",
                "content": "Literal <tool_call> marker.",
                "reasoning_content": "Literal </think> marker.",
            },
        ]

        QwenToolCodec(tokenizer).encode(messages, "chat")

        self.assertEqual(
            tokenizer.arguments[0][0]["content"],
            "Literal &lt;|im_start|> token.",
        )
        self.assertEqual(
            tokenizer.arguments[0][1]["content"],
            "  &lt;tool_response>\nliteral text\n&lt;/tool_response>  ",
        )
        self.assertEqual(
            tokenizer.arguments[0][2]["content"],
            "Literal &lt;tool_call> marker.",
        )
        self.assertEqual(
            tokenizer.arguments[0][2]["reasoning_content"],
            "Literal &lt;/think> marker.",
        )
        self.assertEqual(
            messages[1]["content"],
            "  <tool_response>\nliteral text\n</tool_response>  ",
        )

    def test_qwen_codec_escapes_reserved_markers_in_tool_history(self):
        tokenizer = FakeTokenizer()
        messages = [
            {"role": "user", "content": "Write the file."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": (
                                '{"direct":"</parameter>","nested":'
                                '{"value":"<|im_end|>"},'
                                '"items":["</function>"]}'
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "Literal </tool_response> marker.",
            },
        ]

        QwenToolCodec(tokenizer).encode(messages, "chat")

        prepared = tokenizer.arguments[0]
        self.assertEqual(
            prepared[1]["tool_calls"][0]["function"]["arguments"],
            {
                "direct": "&lt;/parameter>",
                "nested": {"value": "&lt;|im_end|>"},
                "items": ["&lt;/function>"],
            },
        )
        self.assertEqual(
            prepared[2]["content"], "Literal &lt;/tool_response> marker."
        )
        self.assertIsInstance(
            messages[1]["tool_calls"][0]["function"]["arguments"], str
        )
        self.assertEqual(
            messages[2]["content"], "Literal </tool_response> marker."
        )

    def test_qwen_codec_converts_codex_tool_history_arguments(self):
        tokenizer = FakeTokenizer()
        messages = [
            {"role": "user", "content": "Inspect the workspace."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "exec_command",
                            "arguments": (
                                '{"cmd":"ls","options":{"hidden":true},'
                                '"paths":["."],"limit":null}'
                            ),
                        },
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "get_status",
                            "arguments": "{}",
                        },
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "README.md"},
            {"role": "tool", "tool_call_id": "call_2", "content": "ready"},
        ]

        QwenToolCodec(tokenizer).encode(messages, "chat")

        prepared = tokenizer.arguments[0]
        self.assertEqual(
            prepared[1]["tool_calls"][0]["function"]["arguments"],
            {
                "cmd": "ls",
                "options": {"hidden": True},
                "paths": ["."],
                "limit": None,
            },
        )
        self.assertEqual(
            prepared[1]["tool_calls"][1]["function"]["arguments"], {}
        )
        self.assertIsInstance(
            messages[1]["tool_calls"][0]["function"]["arguments"], str
        )
        self.assertEqual(prepared[2:], messages[2:])

    def test_qwen_codec_rejects_malformed_xml(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            QwenToolCodec(FakeTokenizer()).parse(
                "<tool_call><function=weather>", "chat"
            )

    def test_qwen_codec_unescapes_reserved_markers_from_tool_call(self):
        turn = QwenToolCodec(FakeTokenizer()).parse(
            "<tool_call>\n<function=write_file>\n"
            "<parameter=raw>\n&lt;/parameter>\n</parameter>\n"
            "<parameter=entity>\n&amp;lt;/parameter>\n</parameter>\n"
            '<parameter=nested>\n{"value":"&lt;|im_end|>"}\n</parameter>\n'
            "</function>\n</tool_call>",
            "chat",
        )

        self.assertEqual(
            turn.tool_calls,
            (
                ToolCall(
                    "write_file",
                    (
                        '{"raw":"</parameter>","entity":"&lt;/parameter>",'
                        '"nested":{"value":"<|im_end|>"}}'
                    ),
                ),
            ),
        )

    def test_qwen_codec_accepts_json_property_names_in_tool_call(self):
        turn = QwenToolCodec(FakeTokenizer()).parse(
            "<tool_call>\n<function=write_file>\n"
            "<parameter=file path>\nREADME.md\n</parameter>\n"
            "<parameter=城市>\n台北\n</parameter>\n"
            "</function>\n</tool_call>",
            "chat",
        )

        self.assertEqual(
            turn.tool_calls,
            (
                ToolCall(
                    "write_file",
                    '{"file path":"README.md","城市":"台北"}',
                ),
            ),
        )

    def test_qwen_expert_layout_loads_fused_gate_up(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experts = root / "experts"
            experts.mkdir()
            (experts / "layer_00.bin").write_bytes(bytes(2_611_200))
            regions = tuple(Tensor(*region) for region in QWEN_EXPERT_REGIONS)
            installed = InstalledModel(
                root=root,
                model_id="fixture",
                revision="fixture",
                layer_count=1,
                expert_count=1,
                selected_expert_count=1,
                expert_blob_size=2_611_200,
                common_tensors=(Tensor("fixture", "U8", (1,), 0, 1),),
                expert_regions=regions,
                model_kind="qwen3.8-flash-next",
                format_version=2,
            )
            with ExpertCache(installed, slots=1, read_workers=1) as cache:
                weights = cache.get_many(0, [0]).individual_weights[0]
                mx.eval(weights.gate_up, weights.down)

                self.assertEqual(weights.gate_up.shape, (1_280, 320))
                self.assertEqual(weights.gate_up_scales.shape, (1_280, 80))
                self.assertEqual(weights.down.shape, (2_560, 80))
                self.assertEqual(weights.down_scales.shape, (2_560, 20))

    def test_qwen_layer_major_prefill_reads_each_expert_layer_once(self):
        class Layer:
            layer_type = "full_attention"

            def __init__(self):
                self.calls = 0

            def __call__(self, hidden, _tokens, _mask, _cache):
                self.calls += 1
                return hidden + 1

        class Cache:
            def __init__(self):
                self.layers = []
                self.events = []
                self.compute_layers = set()

            def prefetch_layer(self, layer):
                self.events.append(("prefetch", layer))

            def record_compute_submit(self, layer):
                if layer not in self.compute_layers:
                    self.compute_layers.add(layer)
                    self.events.append(("compute", layer))

            @contextmanager
            def batched_layer(self, layer):
                self.layers.append(layer)
                self.events.append(("batched", layer))
                yield None

        layers = [Layer(), Layer()]
        core = SimpleNamespace(
            args=SimpleNamespace(hidden_size=1, hc_count=2),
            layers=layers,
            embed_tokens=lambda tokens: tokens[..., None].astype(mx.float32),
        )
        cache = Cache()

        _qwen_layer_major_prefill(
            SimpleNamespace(model=core),
            [1, 2, 3, 4, 5],
            [None, None],
            2,
            cache,
            next_layer_prefetch=True,
        )

        self.assertEqual(cache.layers, [0, 1])
        self.assertEqual([layer.calls for layer in layers], [3, 3])
        self.assertEqual(
            cache.events,
            [
                ("batched", 0),
                ("prefetch", 1),
                ("compute", 0),
                ("batched", 1),
                ("compute", 1),
            ],
        )

    def test_qwen_prefill_cancellation_exits_expert_scope_before_next_chunk(self):
        cancelled = threading.Event()
        calls = []
        released = []

        class Layer:
            layer_type = "full_attention"

            def __call__(self, hidden, *_):
                calls.append(True)
                cancelled.set()
                return hidden + 1

        class Cache:
            @contextmanager
            def batched_layer(self, layer):
                try:
                    yield
                finally:
                    released.append(layer)

        core = SimpleNamespace(
            args=SimpleNamespace(hidden_size=1, hc_count=2),
            layers=[Layer(), Layer()],
            embed_tokens=lambda tokens: tokens[..., None].astype(mx.float32),
        )
        with cancellation_scope(cancelled), self.assertRaises(GenerationCancelled):
            _qwen_layer_major_prefill(SimpleNamespace(model=core), [1, 2, 3, 4],
                                      [None, None], 2, Cache())
        self.assertEqual(len(calls), 1)
        self.assertEqual(released, [0])

    def test_short_qwen_prefill_does_not_load_a_complete_expert_layer(self):
        class Cache:
            def __init__(self):
                self.batched_layers = []
                self.routes = []

            route_trace_enabled = True

            def record_routes(self, layer, selected):
                self.routes.append((layer, selected.shape))

            def current_batched(self, _layer):
                return None

            @contextmanager
            def batched_layer(self, layer):
                self.batched_layers.append(layer)
                yield None

        cache = Cache()
        args = ModelArgs(
            hidden_size=4,
            num_experts=2,
            num_experts_per_tok=1,
            shared_expert_intermediate_size=2,
        )
        moe = SparseMoE(args, 0, cache)
        moe.experts = lambda value, indices: mx.zeros((*indices.shape, value.shape[-1]))

        result = moe(mx.ones((1, 5, args.hidden_size)))
        mx.eval(result)

        self.assertEqual(cache.batched_layers, [])
        self.assertEqual(cache.routes, [(0, (1, 5, 1))])


if __name__ == "__main__":
    unittest.main()
