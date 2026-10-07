from __future__ import annotations

import ctypes
import importlib
import json
import os
import sys
import tempfile
import threading
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np
from mlx_lm.models import deepseek_v4
from mlx_lm.models.cache import ArraysCache, CacheList

from deepseek_v4_ssd.expert_cache import (
    BatchedExperts,
    CacheMetrics,
    ExpertCache,
    ExpertUnionLayerMetrics,
    ExpertUnionProfile,
    ExpertWeights,
    ResidentExperts,
    _ReadLimiter,
)
from deepseek_v4_ssd.dspark import (
    DraftResult,
    VerificationMetrics,
    _confidence_prefix_length,
    _should_fallback,
    _verify,
    generate_tokens,
)
from deepseek_v4_ssd.fp8_cache import CorrectPoolingCache, MXFP8PoolingCache
from deepseek_v4_ssd.generation import (
    GenerationOptions,
    ModelRuntime,
    RuntimeMetrics,
    _PromptCacheEntry,
    _RawEvalCacheList,
    _approximation_mode,
    _decode_cache_state,
    _deepseek_v41_prefill,
    _encode_cache_state,
    _persistence_cache_state,
    _restore_persistence_cache,
    _uses_layer_major_prefill,
)
from deepseek_v4_ssd.io_metrics import (
    PageCacheReadClassification,
    PageCacheResidency,
    ProcessDiskIO,
    configure_expert_file_cache_policy,
    page_cache_residency_snapshot,
)
from deepseek_v4_ssd.manifest import InstalledModel, Tensor
from deepseek_v4_ssd.model_support import get_support
from deepseek_v4_ssd.model import (
    RuntimeConfig,
    _ORIGINAL_SPARSE_POOLED_ATTENTION,
    _StreamingSwitchGLU,
    _configure_memory_limits,
    _correct_compressor,
    _select_moe_step_size,
    _select_prefill_step_size,
    _sparse_pooled_attention,
    _stable_topk_indices,
    eval_prompt_cache,
    forward_with_hidden,
    layer_major_prefill,
    sequential_verification_forward_with_hidden,
    verification_forward_with_hidden,
)

_mlx_lm_generate = importlib.import_module("mlx_lm.generate")


class PromptCacheEvaluationTests(unittest.TestCase):
    def test_eval_includes_array_state_without_reading_persistence_state(self):
        class RawArraysCache(ArraysCache):
            @property
            def state(self):
                raise AssertionError("evaluation must use raw cache arrays")

        for wrapped in (False, True):
            with self.subTest(wrapped=wrapped):
                cache = RawArraysCache(size=4)
                recurrent = mx.ones((1, 2, 4)) + 1
                convolution = mx.ones((1, 3, 4)) + 2
                history = mx.array([[1, 2]], dtype=mx.int32)
                cache[0], cache[1], cache[2] = recurrent, convolution, history
                dependency = mx.ones((1, 4)) + 3
                # Repeated cache references must not inflate the array counts.
                caches = [CacheList(cache, cache)] if wrapped else [cache, cache]
                with patch("deepseek_v4_ssd.model.mx.eval", wraps=mx.eval) as evaluate:
                    count, size = eval_prompt_cache(caches, dependency)

                evaluated = {id(value) for value in evaluate.call_args.args}
                self.assertEqual(evaluated, {id(dependency), id(recurrent),
                                             id(convolution), id(history)})
                self.assertEqual(count, 3)
                self.assertEqual(size, recurrent.nbytes + convolution.nbytes + history.nbytes)

    def test_eval_empty_array_cache_still_evaluates_dependencies(self):
        cache = ArraysCache(size=4)
        dependency = mx.ones((1, 4)) + 1
        with patch("deepseek_v4_ssd.model.mx.eval", wraps=mx.eval) as evaluate:
            self.assertEqual(eval_prompt_cache([cache], dependency), (0, 0))
        self.assertEqual(len(evaluate.call_args.args), 1)
        self.assertIs(evaluate.call_args.args[0], dependency)


class ModelRuntimeTests(unittest.TestCase):
    def test_approximation_mode_changes_only_learned_routers_and_restores_them(self):
        layers = []
        for index in range(43):
            gate = SimpleNamespace(top_k=6, hash=index < 3)
            layers.append(SimpleNamespace(ffn=SimpleNamespace(gate=gate)))
        runtime = SimpleNamespace(
            _is_qwen=False,
            model=SimpleNamespace(
                model=SimpleNamespace(layers=layers),
                dspark=None,
            ),
        )

        with self.assertRaisesRegex(RuntimeError, "stop"):
            with _approximation_mode(runtime, "learned-route-drop-lowest-1"):
                self.assertEqual(
                    [layer.ffn.gate.top_k for layer in layers[:3]],
                    [6, 6, 6],
                )
                self.assertEqual(
                    [layer.ffn.gate.top_k for layer in layers[3:]],
                    [5] * 40,
                )
                raise RuntimeError("stop")

        self.assertEqual([layer.ffn.gate.top_k for layer in layers], [6] * 43)

    def test_approximation_scopes_all_learned_routers_and_restores_after_failure(self):
        for kind, count, field in (("qwen3.8-flash-next", 10, "top_k"), ("deepseek-v4.1", 6, "topk")):
            from deepseek_v4_ssd.model_support import get_support
            router = SimpleNamespace(**{field: count})
            layer = (SimpleNamespace(mlp=router) if field == "top_k"
                     else SimpleNamespace(ffn=SimpleNamespace(gate=router)))
            model = SimpleNamespace(model=SimpleNamespace(layers=[layer]))
            support = get_support(kind)
            with self.assertRaisesRegex(RuntimeError, "stop"):
                with support.approximation(model, "learned-route-drop-lowest-1"):
                    self.assertEqual(getattr(router, field), count - 1)
                    raise RuntimeError("stop")
            self.assertEqual(getattr(router, field), count)
            for sidecar in ("mtp", "dspark"):
                setattr(model, sidecar, object())
                with self.assertRaisesRegex(ValueError, "speculative"):
                    with support.approximation(model, "learned-route-drop-lowest-1"):
                        pass
                delattr(model, sidecar)


    def test_v41_warm_prompt_uses_plain_chunked_prefill(self):
        runtime = ModelRuntime.__new__(ModelRuntime)
        runtime._is_qwen = False
        runtime.support = get_support("deepseek-v4.1")
        runtime.model = SimpleNamespace(mtp=None)
        runtime._encode_prompt = lambda _: [1, 2, 3]
        runtime._generation_lock = threading.Lock()
        runtime._generation_stream = mx.new_stream(mx.gpu)
        runtime.config = SimpleNamespace(prefill_step_size=2)
        runtime.expert_cache = object()

        with (
            patch(
                "deepseek_v4_ssd.model_support.state.make_cache",
                return_value=["cache"],
            ),
            patch(
                "deepseek_v4_ssd.model_support.deepseek_v41._deepseek_v41_prefill"
            ) as v41_prefill,
            patch("deepseek_v4_ssd.model.layer_major_prefill") as legacy_prefill,
            patch.object(runtime, "_store_prompt_cache") as store,
        ):
            processed = runtime.warm_prompt("hello")

        self.assertEqual(processed, 2)
        v41_prefill.assert_called_once_with(
            runtime.model,
            [1, 2],
            ["cache"],
            2,
        )
        legacy_prefill.assert_not_called()
        self.assertEqual(store.call_args.args[0].tokens, [1, 2])
        self.assertTrue(store.call_args.kwargs["persist"])

    def test_prompt_cache_is_isolated_by_approximation_mode(self):
        runtime = ModelRuntime.__new__(ModelRuntime)
        runtime.config = RuntimeConfig()
        runtime.model = object()
        runtime._prompt_caches = [
            _PromptCacheEntry(["exact"], [1, 2], "exact"),
            _PromptCacheEntry(
                ["approximate"],
                [1, 2],
                "learned-route-drop-lowest-1",
            ),
        ]
        runtime._persistent_prompt_caches = []

        exact = runtime._acquire_prompt_cache([1, 2, 3], "exact")
        approximate = runtime._acquire_prompt_cache(
            [1, 2, 3],
            "learned-route-drop-lowest-1",
        )

        self.assertEqual(exact.cache, ["exact"])
        self.assertEqual(approximate.cache, ["approximate"])
        self.assertEqual(exact.approximation_mode, "exact")
        self.assertEqual(
            approximate.approximation_mode,
            "learned-route-drop-lowest-1",
        )

    def test_approximate_prompt_cache_is_never_persisted(self):
        runtime = ModelRuntime.__new__(ModelRuntime)
        runtime.config = SimpleNamespace(
            prompt_cache_entries=2,
            prompt_cache_memory_gib=1,
        )
        runtime._prompt_caches = []
        entry = _PromptCacheEntry(
            [],
            [1, 2],
            "learned-route-drop-lowest-1",
        )

        with patch.object(runtime, "_persist_prompt_cache") as persist:
            runtime._store_prompt_cache(entry, persist=True)

        persist.assert_not_called()

    def test_layer_major_prefill_uses_configured_deepseek_threshold(self):
        default = SimpleNamespace(layer_major_prefill=True)
        configured = SimpleNamespace(
            layer_major_prefill=True,
            layer_major_prefill_threshold=2_048,
        )
        self.assertFalse(
            _uses_layer_major_prefill(default, is_qwen=False, token_count=1_023)
        )
        self.assertTrue(
            _uses_layer_major_prefill(default, is_qwen=False, token_count=1_024)
        )
        self.assertFalse(
            _uses_layer_major_prefill(configured, is_qwen=False, token_count=2_047)
        )
        self.assertTrue(
            _uses_layer_major_prefill(configured, is_qwen=False, token_count=2_048)
        )
        self.assertTrue(
            _uses_layer_major_prefill(configured, is_qwen=True, token_count=128)
        )

    def test_generation_passes_sampler_and_presence_processor_options(self):
        installed = SimpleNamespace(
            root=Path("/tmp/tokenizer"),
            is_qwen=True,
            maximum_context=262_144,
        )
        config = SimpleNamespace(
            dspark_enabled=False,
            prefill_step_size=1,
            layer_major_prefill=False,
        )
        response = SimpleNamespace(
            text="OK",
            token=1,
            prompt_tokens=1,
            generation_tokens=1,
            finish_reason="stop",
        )
        sampler = object()
        presence_processor = object()
        received_options = {}

        def fake_stream_generate(*_args, **options):
            received_options.update(options)
            yield response

        with (
            patch(
                "deepseek_v4_ssd.generation.load_model",
                return_value=(object(), SimpleNamespace(close=lambda: None)),
            ),
            patch(
                "deepseek_v4_ssd.generation.AutoTokenizer.from_pretrained",
                return_value=SimpleNamespace(
                    bos_token=None,
                    encode=lambda *_args, **_kwargs: [1],
                ),
            ),
            patch("deepseek_v4_ssd.model_support.state.make_prompt_cache", return_value=[]),
            patch(
                "deepseek_v4_ssd.generation.make_sampler",
                return_value=sampler,
            ) as make_sampler,
            patch(
                "deepseek_v4_ssd.generation.make_logits_processors",
                return_value=[presence_processor],
            ) as make_logits_processors,
            patch(
                "deepseek_v4_ssd.generation.stream_generate",
                side_effect=fake_stream_generate,
            ),
        ):
            runtime = ModelRuntime(installed, config)
            list(
                runtime.stream(
                    "test",
                    GenerationOptions(
                        max_tokens=1,
                        temperature=0.7,
                        top_p=0.8,
                        top_k=20,
                        min_p=0.0,
                        presence_penalty=1.5,
                        repetition_penalty=1.0,
                    ),
                )
            )
            make_sampler.assert_called_once_with(
                temp=0.7,
                top_p=0.8,
                top_k=20,
                min_p=0.0,
            )
            make_logits_processors.assert_called_once_with(presence_penalty=1.5)
            self.assertIs(received_options["sampler"], sampler)
            self.assertEqual(
                received_options["logits_processors"],
                [presence_processor],
            )

            make_sampler.reset_mock()
            make_logits_processors.reset_mock()
            received_options.clear()
            list(
                runtime.stream(
                    "test",
                    GenerationOptions(
                        max_tokens=1,
                        temperature=0,
                        top_p=0.95,
                        top_k=20,
                    ),
                )
            )
            make_sampler.assert_called_once_with(
                temp=0,
                top_p=0.95,
                top_k=20,
                min_p=0.0,
            )
            make_logits_processors.assert_not_called()
            self.assertEqual(received_options["logits_processors"], [])

    def test_model_load_and_request_share_cross_thread_stream(self):
        load_stream = None
        pending = None
        original_stream = _mlx_lm_generate.generation_stream

        def fake_load_model(_installed, _config):
            nonlocal load_stream
            load_stream = mx.default_stream(mx.gpu)
            return object(), SimpleNamespace(close=lambda: None)

        installed = SimpleNamespace(root=Path("/tmp/tokenizer"))
        config = SimpleNamespace(prefill_step_size=1, layer_major_prefill=False)
        with (
            patch("deepseek_v4_ssd.generation.load_model", side_effect=fake_load_model),
            patch(
                "deepseek_v4_ssd.generation.AutoTokenizer.from_pretrained",
                return_value=SimpleNamespace(
                    bos_token=None,
                    encode=lambda *_args, **_kwargs: [1],
                ),
            ),
        ):
            runtime = ModelRuntime(installed, config)

        self.assertEqual(runtime._generation_stream, load_stream)
        response = SimpleNamespace(
            text="OK",
            token=1,
            prompt_tokens=1,
            generation_tokens=1,
            finish_reason="stop",
        )
        pieces = []
        errors = []

        def fake_stream_generate(*_args, **_kwargs):
            nonlocal pending
            with mx.stream(_mlx_lm_generate.generation_stream):
                if pending is not None:
                    mx.eval(pending)
                pending = mx.ones((1,)) + 1
                yield response

        def generate():
            try:
                pieces.extend(runtime.stream("test", GenerationOptions(max_tokens=1)))
            except Exception as error:
                errors.append(error)

        with (
            patch("deepseek_v4_ssd.model_support.state.make_prompt_cache", return_value=[]),
            patch(
                "deepseek_v4_ssd.generation.stream_generate",
                side_effect=fake_stream_generate,
            ),
        ):
            for _ in range(2):
                thread = threading.Thread(target=generate)
                thread.start()
                thread.join()

        self.assertEqual(errors, [])
        self.assertEqual([piece.text for piece in pieces], ["OK", "OK"])
        self.assertIs(_mlx_lm_generate.generation_stream, original_stream)

    def test_generation_tracks_phases_and_evaluates_cache_state(self):
        installed = SimpleNamespace(root=Path("/tmp/tokenizer"))
        config = SimpleNamespace(prefill_step_size=1, layer_major_prefill=False)
        cache = SimpleNamespace(state=(mx.array([1]),))
        responses = [
            SimpleNamespace(
                text="A",
                token=1,
                prompt_tokens=4,
                generation_tokens=1,
                finish_reason=None,
            ),
            SimpleNamespace(
                text="B",
                token=2,
                prompt_tokens=4,
                generation_tokens=2,
                finish_reason="stop",
            ),
        ]
        received_options = {}
        route_phases = []
        observed_phases = []

        class ExpertCache:
            active_phase = None

            @contextmanager
            def trace_routes(self, phase):
                route_phases.append(phase)
                self.active_phase = phase
                try:
                    yield
                finally:
                    self.active_phase = None

            def close(self):
                pass

        expert_cache = ExpertCache()

        def fake_stream_generate(*_args, **options):
            received_options.update(options)
            for response in responses:
                observed_phases.append(expert_cache.active_phase)
                yield response

        with (
            patch(
                "deepseek_v4_ssd.generation.load_model",
                return_value=(object(), expert_cache),
            ),
            patch(
                "deepseek_v4_ssd.generation.AutoTokenizer.from_pretrained",
                return_value=SimpleNamespace(
                    bos_token=None,
                    encode=lambda *_args, **_kwargs: [1, 2, 3, 4],
                ),
            ),
            patch("deepseek_v4_ssd.model_support.state.make_prompt_cache", return_value=[cache]),
            patch(
                "deepseek_v4_ssd.generation.stream_generate",
                side_effect=fake_stream_generate,
            ),
            patch("deepseek_v4_ssd.generation.mx.eval") as evaluate,
        ):
            runtime = ModelRuntime(installed, config)
            pieces = list(runtime.stream("test", GenerationOptions(max_tokens=2)))

        metrics = runtime.metrics.snapshot()
        self.assertEqual([piece.text for piece in pieces], ["A", "B"])
        self.assertEqual(received_options["prompt_cache"], [cache])
        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual(route_phases, ["prefill", "decode", "decode"])
        self.assertEqual(observed_phases, ["prefill", "decode"])
        self.assertEqual(metrics["runtime_prompt_tokens"], 4)
        self.assertEqual(metrics["runtime_generation_tokens"], 2)
        self.assertEqual(metrics["accumulated_generation_tokens"], 2)
        self.assertEqual(metrics["completed_request_count"], 1)
        self.assertEqual(metrics["cache_state_eval_count"], 2)
        self.assertGreaterEqual(metrics["time_to_first_token_seconds"], 0)
        self.assertGreaterEqual(metrics["prefill_tokens_per_second"], 0)
        self.assertGreaterEqual(metrics["decode_seconds"], 0)

    def test_generation_reuses_a_complete_prompt_prefix(self):
        installed = SimpleNamespace(root=Path("/tmp/tokenizer"))
        config = SimpleNamespace(prefill_step_size=1, layer_major_prefill=False)
        cache = SimpleNamespace(state=(mx.array([1]),))
        prompts = {
            "first": [1, 2],
            "continued": [1, 2, 42, 3],
            "branch": [9, 10],
        }
        received_prompts = []

        def fake_stream_generate(_model, _tokenizer, prompt, **_options):
            received_prompts.append(prompt)
            yield SimpleNamespace(
                text="A",
                token=42,
                prompt_tokens=len(prompt),
                generation_tokens=1,
                finish_reason="length",
            )

        tokenizer = SimpleNamespace(
            bos_token=None,
            encode=lambda prompt, **_options: prompts[prompt],
        )
        with (
            patch(
                "deepseek_v4_ssd.generation.load_model",
                return_value=(object(), SimpleNamespace(close=lambda: None)),
            ),
            patch(
                "deepseek_v4_ssd.generation.AutoTokenizer.from_pretrained",
                return_value=tokenizer,
            ),
            patch(
                "deepseek_v4_ssd.model_support.state.make_prompt_cache",
                return_value=[cache],
            ) as make_cache,
            patch(
                "deepseek_v4_ssd.generation.stream_generate",
                side_effect=fake_stream_generate,
            ),
            patch("deepseek_v4_ssd.generation.mx.eval"),
        ):
            runtime = ModelRuntime(installed, config)
            list(runtime.stream("first", GenerationOptions(max_tokens=1)))
            pieces = list(
                runtime.stream("continued", GenerationOptions(max_tokens=1))
            )
            reused_metrics = runtime.metrics.snapshot()
            list(runtime.stream("branch", GenerationOptions(max_tokens=1)))

        self.assertEqual(received_prompts, [[1, 2], [3], [9, 10]])
        self.assertEqual(make_cache.call_count, 2)
        self.assertEqual(pieces[0].prompt_tokens, 4)
        self.assertEqual(reused_metrics["prompt_cache_reused_tokens"], 3)
        self.assertEqual(runtime.metrics.snapshot()["prompt_cache_reused_tokens"], 0)

    def test_generation_keeps_two_independent_prompt_cache_timelines(self):
        installed = SimpleNamespace(root=Path("/tmp/tokenizer"))
        config = SimpleNamespace(
            prefill_step_size=1,
            prompt_cache_entries=2,
            prompt_cache_memory_gib=1,
            layer_major_prefill=False,
        )
        prompts = {
            "first": [1, 2],
            "branch": [9, 10],
            "continued": [1, 2, 42, 3],
        }
        received_prompts = []

        def fake_stream_generate(_model, _tokenizer, prompt, **_options):
            received_prompts.append(prompt)
            yield SimpleNamespace(
                text="A",
                token=42,
                prompt_tokens=len(prompt),
                generation_tokens=1,
                finish_reason="length",
            )

        tokenizer = SimpleNamespace(
            bos_token=None,
            encode=lambda prompt, **_options: prompts[prompt],
        )
        caches = [[SimpleNamespace(state=(), nbytes=1)] for _ in range(2)]
        with (
            patch(
                "deepseek_v4_ssd.generation.load_model",
                return_value=(object(), SimpleNamespace(close=lambda: None)),
            ),
            patch(
                "deepseek_v4_ssd.generation.AutoTokenizer.from_pretrained",
                return_value=tokenizer,
            ),
            patch(
                "deepseek_v4_ssd.model_support.state.make_prompt_cache",
                side_effect=caches,
            ) as make_cache,
            patch(
                "deepseek_v4_ssd.generation.stream_generate",
                side_effect=fake_stream_generate,
            ),
            patch("deepseek_v4_ssd.generation.mx.eval"),
        ):
            runtime = ModelRuntime(installed, config)
            list(runtime.stream("first", GenerationOptions(max_tokens=1)))
            list(runtime.stream("branch", GenerationOptions(max_tokens=1)))
            list(runtime.stream("continued", GenerationOptions(max_tokens=1)))

        self.assertEqual(received_prompts, [[1, 2], [9, 10], [3]])
        self.assertEqual(make_cache.call_count, 2)

    def test_warm_prompt_leaves_one_token_for_the_first_request(self):
        installed = SimpleNamespace(root=Path("/tmp/tokenizer"))
        config = SimpleNamespace(
            prefill_step_size=1,
            prompt_cache_entries=1,
            prompt_cache_memory_gib=1,
            layer_major_prefill=False,
        )
        prompts = {"warm": [1, 2, 3], "request": [1, 2, 3, 4]}
        received_prompts = []
        cache = [SimpleNamespace(state=(), nbytes=1)]

        def fake_stream_generate(_model, _tokenizer, prompt, **_options):
            received_prompts.append(prompt)
            yield SimpleNamespace(
                text="A",
                token=42,
                generation_tokens=1,
                finish_reason="length",
            )

        with (
            patch(
                "deepseek_v4_ssd.generation.load_model",
                return_value=(object(), SimpleNamespace(close=lambda: None)),
            ),
            patch(
                "deepseek_v4_ssd.generation.AutoTokenizer.from_pretrained",
                return_value=SimpleNamespace(
                    bos_token=None,
                    encode=lambda prompt, **_options: prompts[prompt],
                ),
            ),
            patch("deepseek_v4_ssd.model_support.state.make_prompt_cache", return_value=cache),
            patch("deepseek_v4_ssd.model.layer_major_prefill") as prefill,
            patch(
                "deepseek_v4_ssd.generation.stream_generate",
                side_effect=fake_stream_generate,
            ),
            patch("deepseek_v4_ssd.generation.mx.eval"),
        ):
            runtime = ModelRuntime(installed, config)
            warmed = runtime.warm_prompt("warm")
            list(runtime.stream("request", GenerationOptions(max_tokens=1)))

        self.assertEqual(warmed, 2)
        self.assertEqual(prefill.call_args.args[1], [1, 2])
        self.assertEqual(received_prompts, [[3, 4]])
        self.assertEqual(runtime.metrics.snapshot()["prompt_cache_reused_tokens"], 2)

    def test_persistent_prompt_cache_survives_restart(self):
        class Cache:
            def __init__(self):
                self._state = (mx.array([7]), mx.empty((0,), dtype=mx.float32))
                self.nbytes = 4

            @property
            def state(self):
                return self._state

            @state.setter
            def state(self, value):
                self._state = value

        prompts = {
            "first": [1, 2],
            "continued": [1, 2, 42, 3],
        }
        received = []

        def generate(_model, _tokenizer, prompt, **_options):
            received.append(prompt)
            yield SimpleNamespace(
                text="A",
                token=42,
                generation_tokens=1,
                finish_reason="length",
            )

        with tempfile.TemporaryDirectory() as directory:
            installed = SimpleNamespace(
                root=Path("/tmp/tokenizer"),
                revision="fixture-revision",
            )
            config = SimpleNamespace(
                prefill_step_size=1,
                layer_major_prefill=False,
                prompt_cache_entries=2,
                prompt_cache_memory_gib=1,
                persistent_prompt_cache=True,
                persistent_prompt_cache_entries=8,
                prompt_cache_directory=directory,
            )
            tokenizer = SimpleNamespace(
                bos_token=None,
                encode=lambda prompt, **_options: prompts[prompt],
            )
            with (
                patch(
                    "deepseek_v4_ssd.generation.load_model",
                    return_value=(object(), SimpleNamespace(close=lambda: None)),
                ),
                patch(
                    "deepseek_v4_ssd.generation.AutoTokenizer.from_pretrained",
                    return_value=tokenizer,
                ),
                patch(
                    "deepseek_v4_ssd.model_support.state.make_prompt_cache",
                    side_effect=lambda _model: [Cache()],
                ),
                patch(
                    "deepseek_v4_ssd.generation.stream_generate",
                    side_effect=generate,
                ),
            ):
                first = ModelRuntime(installed, config)
                list(first.stream("first", GenerationOptions(max_tokens=1)))
                first.close()

                second = ModelRuntime(installed, config)
                list(second.stream("continued", GenerationOptions(max_tokens=1)))
                reused = second.metrics.snapshot()["prompt_cache_reused_tokens"]
                second.close()

        self.assertEqual(received, [[1, 2], [3]])
        self.assertEqual(reused, 3)

    def test_persistent_prefill_checkpoint_clones_mutable_cache_state(self):
        class MutableCache:
            def __init__(self):
                self.values = [mx.array([0], dtype=mx.int32)]
                self.nbytes = 4

            @property
            def state(self):
                return self.values

            @state.setter
            def state(self, value):
                self.values = value

        prompt_tokens = list(range(48))
        received = []
        restored = []

        def generate(_model, _tokenizer, prompt, **options):
            prompt = list(prompt)
            received.append(prompt)
            cache = options["prompt_cache"][0]
            if len(received) == 1:
                callback = options["prompt_progress_callback"]
                callback(0, len(prompt))
                cache.state[0] = mx.array([47], dtype=mx.int32)
                callback(len(prompt) - 1, len(prompt))
                cache.state[0] = mx.array([999], dtype=mx.int32)
            else:
                restored.append(int(cache.state[0].item()))
            yield SimpleNamespace(
                text="A",
                token=42,
                generation_tokens=1,
                finish_reason="length",
            )

        with tempfile.TemporaryDirectory() as directory:
            installed = SimpleNamespace(
                root=Path("/tmp/tokenizer"),
                revision="fixture-revision",
            )
            config = SimpleNamespace(
                prefill_step_size=128,
                layer_major_prefill=False,
                prompt_cache_entries=2,
                prompt_cache_memory_gib=1,
                persistent_prompt_cache=True,
                persistent_prompt_cache_entries=8,
                prompt_cache_directory=directory,
            )
            tokenizer = SimpleNamespace(
                bos_token=None,
                encode=lambda _prompt, **_options: prompt_tokens,
            )
            with (
                patch(
                    "deepseek_v4_ssd.generation.load_model",
                    return_value=(object(), SimpleNamespace(close=lambda: None)),
                ),
                patch(
                    "deepseek_v4_ssd.generation.AutoTokenizer.from_pretrained",
                    return_value=tokenizer,
                ),
                patch(
                    "deepseek_v4_ssd.model_support.state.make_prompt_cache",
                    side_effect=lambda _model: [MutableCache()],
                ),
                patch(
                    "deepseek_v4_ssd.generation.stream_generate",
                    side_effect=generate,
                ),
            ):
                first = ModelRuntime(installed, config)
                list(first.stream("same", GenerationOptions(max_tokens=1)))
                first.close()

                second = ModelRuntime(installed, config)
                list(second.stream("same", GenerationOptions(max_tokens=1)))
                reused = second.metrics.snapshot()["prompt_cache_reused_tokens"]
                second.close()

        self.assertEqual(received, [prompt_tokens, [prompt_tokens[-1]]])
        self.assertEqual(restored, [47])
        self.assertEqual(reused, 47)

    def test_decode_rate_includes_cache_evaluation(self):
        metrics = RuntimeMetrics()
        metrics.start(4, 0, 1, False, CacheMetrics())
        metrics.record_token(1, 1.0, 0.25)
        metrics.record_token(2, 2.0, 0.5)
        metrics.finish(CacheMetrics())

        snapshot = metrics.snapshot()

        self.assertEqual(snapshot["decode_model_step_seconds"], 2.0)
        self.assertEqual(snapshot["decode_cache_eval_seconds"], 0.5)
        self.assertEqual(snapshot["decode_end_to_end_seconds"], 2.5)
        self.assertEqual(snapshot["decode_model_step_tokens_per_second"], 0.5)
        self.assertEqual(snapshot["decode_tokens_per_second"], 0.4)
        self.assertEqual(snapshot["decode_latency_p50_seconds"], 2.5)
        self.assertEqual(snapshot["decode_latency_p95_seconds"], 2.5)

    def test_metrics_report_actual_approximation_mode(self):
        metrics = RuntimeMetrics()
        metrics.start(
            4,
            0,
            1,
            False,
            CacheMetrics(),
            approximation_mode="learned-route-drop-lowest-1",
        )
        metrics.finish(CacheMetrics())

        self.assertEqual(
            metrics.snapshot()["approximation_mode"],
            "learned-route-drop-lowest-1",
        )

    def test_metrics_report_process_disk_io_delta(self):
        metrics = RuntimeMetrics()
        with patch(
            "deepseek_v4_ssd.generation.process_disk_io_snapshot",
            side_effect=[ProcessDiskIO(100, 20), ProcessDiskIO(175, 50)],
        ):
            metrics.start(4, 0, 1, False, CacheMetrics())
            metrics.finish(CacheMetrics())

        snapshot = metrics.snapshot()

        self.assertEqual(snapshot["request_process_disk_bytes_read"], 75)
        self.assertEqual(snapshot["request_process_disk_bytes_written"], 30)

    def test_output_tokens_accumulate_across_requests(self):
        metrics = RuntimeMetrics()
        metrics.start(4, 0, 1, False, CacheMetrics())
        metrics.record_token(2, 1.0, 0.0)
        self.assertEqual(metrics.snapshot()["accumulated_generation_tokens"], 2)
        metrics.finish(CacheMetrics())

        metrics.start(4, 0, 1, False, CacheMetrics())
        metrics.record_token(3, 1.0, 0.0)
        self.assertEqual(metrics.snapshot()["accumulated_generation_tokens"], 5)
        metrics.finish(CacheMetrics())

        self.assertEqual(metrics.snapshot()["accumulated_generation_tokens"], 5)


class MemoryLimitTests(unittest.TestCase):
    def test_automatic_memory_limit_uses_metals_recommended_maximum(self):
        maximum = 40_200_896_512
        with (
            patch(
                "deepseek_v4_ssd.model.mx.device_info",
                return_value={"max_recommended_working_set_size": maximum},
            ),
            patch("deepseek_v4_ssd.model.mx.set_memory_limit") as set_memory_limit,
            patch("deepseek_v4_ssd.model.mx.set_wired_limit") as set_wired_limit,
        ):
            selected = _configure_memory_limits(RuntimeConfig())

        self.assertEqual(selected, maximum)
        set_memory_limit.assert_called_once_with(maximum)
        set_wired_limit.assert_called_once_with(maximum)

    def test_explicit_memory_limit_caps_wired_memory_at_metals_maximum(self):
        maximum = 40_200_896_512
        requested = 48 * 1024**3
        with (
            patch(
                "deepseek_v4_ssd.model.mx.device_info",
                return_value={"max_recommended_working_set_size": maximum},
            ),
            patch("deepseek_v4_ssd.model.mx.set_memory_limit") as set_memory_limit,
            patch("deepseek_v4_ssd.model.mx.set_wired_limit") as set_wired_limit,
        ):
            selected = _configure_memory_limits(RuntimeConfig(memory_limit_gib=48))

        self.assertEqual(selected, requested)
        set_memory_limit.assert_called_once_with(requested)
        set_wired_limit.assert_called_once_with(maximum)

    def test_automatic_memory_limit_honors_a_model_cap(self):
        maximum = 56 * 1024**3
        cap = 48 * 1024**3
        with (
            patch(
                "deepseek_v4_ssd.model.mx.device_info",
                return_value={"max_recommended_working_set_size": maximum},
            ),
            patch("deepseek_v4_ssd.model.mx.set_memory_limit") as set_memory_limit,
            patch("deepseek_v4_ssd.model.mx.set_wired_limit") as set_wired_limit,
        ):
            selected = _configure_memory_limits(
                RuntimeConfig(), automatic_cap_gib=48
            )

        self.assertEqual(selected, cap)
        set_memory_limit.assert_called_once_with(cap)
        set_wired_limit.assert_called_once_with(cap)


class PrefillTests(unittest.TestCase):
    def test_runtime_uses_1152_expert_slots_by_default(self):
        self.assertEqual(RuntimeConfig().slots, 1_152)
        self.assertEqual(RuntimeConfig().read_workers, 4)
        self.assertEqual(RuntimeConfig().memory_limit_gib, 0)
        self.assertEqual(RuntimeConfig().dspark_slots, 768)
        self.assertFalse(RuntimeConfig().mtp_enabled)
        self.assertEqual(RuntimeConfig().mtp_slots, 32)
        self.assertFalse(RuntimeConfig().dspark_prompt_cache)
        self.assertTrue(RuntimeConfig().dspark_fallback_enabled)
        self.assertFalse(RuntimeConfig().dspark_sequential_verification)
        self.assertTrue(RuntimeConfig().ready_expert_decode)
        self.assertFalse(RuntimeConfig().staged_expert_streaming)
        self.assertFalse(RuntimeConfig().qwen_next_layer_prefetch)
        self.assertTrue(RuntimeConfig().ane_prefill)
        self.assertEqual(RuntimeConfig().ane_prefill_ratio, 0.0)


    def test_automatic_step_size_uses_larger_chunks_for_long_prompts(self):
        self.assertEqual(_select_prefill_step_size(0, 1_000), 128)
        self.assertEqual(_select_prefill_step_size(0, 4_000), 256)
        self.assertEqual(_select_prefill_step_size(0, 14_000), 1_024)
        self.assertEqual(_select_prefill_step_size(192, 14_000), 192)
        self.assertEqual(_select_moe_step_size(0, 14_000), 4_096)
        self.assertEqual(_select_moe_step_size(2_048, 14_000), 2_048)

    def test_metrics_report_layer_major_prefill_token_count(self):
        metrics = RuntimeMetrics()
        metrics.start(4_098, 0, 1_024, True, CacheMetrics())

        self.assertEqual(metrics.snapshot()["layer_major_prefill_tokens"], 4_097)

    def test_layer_major_prefill_finishes_one_layer_before_the_next(self):
        calls = []

        class Cache:
            def __init__(self):
                self.offset = 0

            @property
            def state(self):
                raise AssertionError("layer-major prefill must evaluate raw cache arrays")

        class Layer:
            def __init__(self, index):
                self.index = index

            def __call__(self, hidden, _mask, cache, input_ids):
                calls.append((self.index, input_ids.reshape(-1).tolist()))
                cache.offset += input_ids.shape[1]
                return hidden + 1

        class PinnedCache:
            def pin_layer(self, _layer):
                from contextlib import nullcontext

                return nullcontext()

        core = SimpleNamespace(
            embed_tokens=lambda inputs: inputs[..., None].astype(mx.float32),
            pipeline_layers=[Layer(0), Layer(1)],
            args=SimpleNamespace(hc_mult=1, sliding_window=4),
        )
        model = SimpleNamespace(model=core)
        caches = [Cache(), Cache()]
        with patch(
            "deepseek_v4_ssd.model.deepseek_v4.create_attention_mask",
            return_value=None,
        ):
            layer_major_prefill(model, [1, 2, 3, 4], caches, 2, PinnedCache())

        self.assertEqual(
            calls,
            [(0, [1, 2]), (0, [3, 4]), (1, [1, 2]), (1, [3, 4])],
        )

    def test_layer_major_prefill_matches_standard_next_token_logits(self):
        from contextlib import nullcontext

        arguments = deepseek_v4.ModelArgs(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            n_routed_experts=4,
            num_experts_per_tok=2,
            q_lora_rank=16,
            qk_rope_head_dim=16,
            head_dim=16,
            o_groups=1,
            o_lora_rank=16,
            hc_mult=1,
            compress_ratios=[0, 0],
        )
        model = deepseek_v4.Model(arguments)
        standard_cache = model.make_cache()
        layer_major_cache = model.make_cache()
        prompt = mx.array([[1, 2, 3, 4]])
        model(prompt, cache=standard_cache)

        expert_cache = SimpleNamespace(pin_layer=lambda _layer: nullcontext())
        layer_major_prefill(
            model,
            prompt.reshape(-1).tolist(),
            layer_major_cache,
            2,
            expert_cache,
        )
        expected = model(mx.array([[5]]), cache=standard_cache)
        actual = model(mx.array([[5]]), cache=layer_major_cache)
        mx.eval(expected, actual)

        self.assertLess(mx.max(mx.abs(actual - expected)).item(), 1e-5)

class DSparkTests(unittest.TestCase):


    def test_hardware_scheduler_falls_back_when_dspark_costs_more(self):
        draft = DraftResult([2], [mx.zeros((5,))], [0.8], 0.2)

        self.assertTrue(_should_fallback(0.1, draft, 0, 0.2))
        self.assertFalse(_should_fallback(0.3, draft, 1, 0.1))

    def test_confidence_scheduler_can_skip_the_whole_draft(self):
        self.assertEqual(_confidence_prefix_length([0.4, 0.9], 0.6), 0)
        self.assertEqual(_confidence_prefix_length([0.8, 0.5], 0.6), 1)

    def test_metrics_report_union_and_expert_bytes_per_committed_token(self):
        class DraftCache:
            slots = 8
            resident_count = 2

            def __init__(self):
                self.bytes_read = 0
                self.page_cache_classified_bytes = 0
                self.page_cache_resident_bytes_before_read = 0
                self.page_cache_nonresident_bytes_before_read = 0

            def metrics_snapshot(self):
                return CacheMetrics(
                    bytes_read=self.bytes_read,
                    page_cache_classified_bytes=(
                        self.page_cache_classified_bytes
                    ),
                    page_cache_resident_bytes_before_read=(
                        self.page_cache_resident_bytes_before_read
                    ),
                    page_cache_nonresident_bytes_before_read=(
                        self.page_cache_nonresident_bytes_before_read
                    ),
                )

        cache = DraftCache()
        metrics = RuntimeMetrics()
        metrics.start(
            4,
            0,
            1,
            False,
            CacheMetrics(),
            dspark_enabled=True,
            dspark_cache=cache,
        )
        cache.bytes_read = 300
        cache.page_cache_classified_bytes = 300
        cache.page_cache_resident_bytes_before_read = 60
        cache.page_cache_nonresident_bytes_before_read = 240
        metrics.record_dspark_round(
            DraftResult([2, 3], [], [0.9, 0.8], 0.1),
            accepted=2,
            verification_seconds=0.2,
            verification=VerificationMetrics(
                routed_expert_assignments=100,
                expert_union_experts=70,
                expert_union_misses=20,
                expert_bytes_read=600,
                verification_expert_bytes_read=500,
                replay_expert_bytes_read=100,
                expert_read_seconds=0.15,
                layer_expert_union_layer_ids=(0, 1),
                layer_routed_expert_assignments=(18, 18),
                layer_expert_union_counts=(10, 12),
                layer_expert_union_misses=(3, 4),


                output_budget_trimmed_tokens=3,
                fallback_target_step_seconds=0.05,
                fallback_speculative_seconds=0.30,
                fallback_break_even_seconds=0.15,
                fallback_cost_ratio=2.0,
                fallback_would_trigger=True,
                fallback_triggered=True,
                verification_mode="sequential",
                sequential_verification_positions=3,
                sequential_position_seconds=(0.01, 0.02, 0.03),
            ),
        )
        metrics.finish(
            CacheMetrics(
                bytes_read=600,
                page_cache_classified_bytes=600,
                page_cache_resident_bytes_before_read=100,
                page_cache_nonresident_bytes_before_read=500,
            )
        )

        snapshot = metrics.snapshot()

        self.assertEqual(
            snapshot["dspark_verification_routed_expert_assignments"], 100
        )
        self.assertEqual(snapshot["dspark_verification_expert_union_calls"], 2)
        self.assertEqual(snapshot["dspark_verification_expert_union_experts"], 70)
        self.assertEqual(
            snapshot["dspark_verification_expert_union_reused_assignments"], 30
        )
        self.assertAlmostEqual(
            snapshot["dspark_verification_expert_union_reuse_rate"], 0.3
        )
        self.assertEqual(snapshot["dspark_target_expert_bytes_read"], 600)
        self.assertEqual(snapshot["dspark_verification_expert_bytes_read"], 500)
        self.assertEqual(snapshot["dspark_replay_expert_bytes_read"], 100)
        self.assertEqual(snapshot["dspark_fallback_triggered_rounds"], 1)
        self.assertEqual(snapshot["dspark_fallback_would_trigger_rounds"], 1)
        self.assertEqual(snapshot["dspark_fallback_cost_ratios"], (2.0,))
        self.assertAlmostEqual(
            snapshot["dspark_last_fallback_target_step_seconds"],
            0.05,
        )
        self.assertAlmostEqual(
            snapshot["dspark_last_fallback_speculative_seconds"],
            0.30,
        )
        self.assertAlmostEqual(
            snapshot["dspark_last_fallback_break_even_seconds"],
            0.15,
        )
        self.assertAlmostEqual(snapshot["dspark_last_fallback_cost_ratio"], 2.0)
        self.assertEqual(snapshot["dspark_output_budget_trimmed_tokens"], 3)
        self.assertEqual(snapshot["dspark_block_verification_rounds"], 0)
        self.assertEqual(snapshot["dspark_sequential_verification_rounds"], 1)
        self.assertEqual(snapshot["dspark_last_verification_mode"], "sequential")
        self.assertEqual(
            snapshot["dspark_last_sequential_position_seconds"],
            (0.01, 0.02, 0.03),
        )
        self.assertEqual(snapshot["dspark_draft_expert_bytes_read"], 300)
        self.assertEqual(
            snapshot["dspark_draft_page_cache_classified_bytes"],
            300,
        )
        self.assertEqual(
            snapshot["dspark_draft_page_cache_nonresident_bytes_before_read"],
            240,
        )
        self.assertEqual(
            snapshot[
                "dspark_draft_page_cache_nonresident_bytes_per_committed_token"
            ],
            80,
        )
        self.assertEqual(snapshot["dspark_speculative_expert_bytes_read"], 900)
        self.assertEqual(
            snapshot["request_expert_page_cache_classified_bytes"],
            600,
        )
        self.assertEqual(
            snapshot["request_expert_page_cache_nonresident_bytes_before_read"],
            500,
        )
        self.assertEqual(
            snapshot["dspark_last_verification_expert_union_layer_ids"], (0, 1)
        )
        self.assertEqual(
            snapshot["dspark_last_verification_expert_union_by_layer"], (10, 12)
        )
        self.assertEqual(
            snapshot["dspark_draft_expert_bytes_per_committed_token"], 100
        )
        self.assertEqual(
            snapshot["dspark_target_expert_bytes_per_committed_token"], 200
        )
        self.assertEqual(
            snapshot["dspark_speculative_expert_bytes_per_committed_token"], 300
        )


    def test_next_round_receives_every_committed_hidden_state(self):
        calls = []

        class DSpark:
            target_layers = (0,)

            def reset_cache(self):
                pass

            def prefill_context(self, _hidden, _offset):
                pass

            def draft(
                self,
                _model,
                _anchor,
                hidden,
                start_pos,
                _temperature,
                _top_p,
                _threshold,
            ):
                calls.append((hidden.shape[1], start_pos))
                return DraftResult(
                    [2, 3],
                    [mx.zeros((5,)), mx.zeros((5,))],
                    [1.0, 1.0],
                    0.0,
                )

        def forward(_model, inputs, _cache, _layers):
            logits = mx.zeros((1, inputs.shape[1], 5))
            logits[..., 1] = 1
            hidden = mx.ones((1, inputs.shape[1], 1))
            return logits, hidden

        def verify(_model, tokens, _cache, _layers):
            logits = mx.zeros((1, len(tokens), 5))
            for index, token in enumerate((2, 3, 4)):
                logits[0, index, token] = 1
            hidden = mx.ones((1, len(tokens), 1))
            return logits, hidden, [None], VerificationMetrics()

        with (
            patch("deepseek_v4_ssd.model.forward_with_hidden", side_effect=forward),
            patch("deepseek_v4_ssd.dspark._target_sequence", side_effect=verify),
            patch("deepseek_v4_ssd.dspark._should_fallback", return_value=False),
        ):
            list(
                generate_tokens(
                    [0],
                    object(),
                    DSpark(),
                    [None],
                    max_tokens=6,
                    prefill_step_size=1,
                    temperature=0,
                    top_p=1,
                )
            )

        self.assertEqual(calls, [(1, 1), (3, 2)])

    def test_rejected_draft_replays_only_the_committed_prefix(self):
        replayed = []
        recorded_rounds = []

        class DSpark:
            target_layers = (0,)

            def reset_cache(self):
                pass

            def prefill_context(self, _hidden, _offset):
                pass

            def draft(self, *_args):
                return DraftResult(
                    [2, 4],
                    [mx.zeros((5,)), mx.zeros((5,))],
                    [1.0, 1.0],
                    0.0,
                )

        def forward(_model, inputs, _cache, _layers):
            values = inputs.reshape(-1).tolist()
            if len(values) > 1:
                replayed.append(values)
            logits = mx.zeros((1, inputs.shape[1], 5))
            logits[..., 1] = 1
            return logits, mx.ones((1, inputs.shape[1], 1))

        def verify(_model, tokens, _cache, _layers):
            logits = mx.zeros((1, len(tokens), 5))
            logits[0, 0, 2] = 1
            logits[0, 1, 3] = 1
            return logits, mx.ones((1, len(tokens), 1)), [None], VerificationMetrics()

        class TargetExpertCache:
            def __init__(self):
                self.snapshots = iter(
                    (
                        CacheMetrics(),
                        CacheMetrics(bytes_read=40, read_seconds=0.1),
                        CacheMetrics(bytes_read=60, read_seconds=0.15),
                    )
                )

            def metrics_snapshot(self):
                return next(self.snapshots)

            @contextmanager
            def capture_expert_unions(self):
                yield ExpertUnionProfile(
                    [
                        ExpertUnionLayerMetrics(
                            layer=0,
                            routed_expert_assignments=18,
                            unique_experts=10,
                            cache_misses=4,
                        )
                    ]
                )

        with (
            patch("deepseek_v4_ssd.model.forward_with_hidden", side_effect=forward),
            patch("deepseek_v4_ssd.dspark._target_sequence", side_effect=verify),
            patch("deepseek_v4_ssd.dspark._should_fallback", return_value=True),
            patch("deepseek_v4_ssd.model.eval_prompt_cache", return_value=(0, 0)) as flush_cache,
        ):
            list(
                generate_tokens(
                    [0],
                    object(),
                    DSpark(),
                    [None],
                    max_tokens=5,
                    prefill_step_size=1,
                    temperature=0,
                    top_p=1,
                    record_round=lambda *values: recorded_rounds.append(values),
                    target_expert_cache=TargetExpertCache(),
                )
            )

        self.assertEqual(replayed, [[1, 2]])
        flush_cache.assert_called_once_with([None])
        verification = recorded_rounds[0][3]
        self.assertEqual(verification.routed_expert_assignments, 18)
        self.assertEqual(verification.expert_union_experts, 10)
        self.assertEqual(verification.expert_union_misses, 4)
        self.assertEqual(verification.layer_expert_union_layer_ids, (0,))
        self.assertEqual(verification.verification_expert_bytes_read, 40)
        self.assertEqual(verification.replay_expert_bytes_read, 20)
        self.assertEqual(verification.expert_bytes_read, 60)
        self.assertTrue(verification.fallback_would_trigger)
        self.assertTrue(verification.fallback_triggered)

    def test_fallback_research_control_preserves_would_trigger_signal(self):
        recorded_rounds = []
        fallbacks = []

        class DSpark:
            target_layers = (0,)

            def reset_cache(self):
                pass

            def prefill_context(self, _hidden, _offset):
                pass

            def draft(self, *_args):
                return DraftResult(
                    [2],
                    [mx.zeros((5,))],
                    [0.4],
                    0.1,
                )

        def forward(_model, inputs, _cache, _layers):
            logits = mx.zeros((1, inputs.shape[1], 5))
            logits[..., 1] = 1
            return logits, mx.ones((1, inputs.shape[1], 1))

        def verify(_model, tokens, _cache, _layers):
            logits = mx.zeros((1, len(tokens), 5))
            logits[0, 0, 2] = 1
            logits[0, 1, 3] = 1
            return logits, mx.ones((1, len(tokens), 1)), [None], VerificationMetrics()

        with (
            patch("deepseek_v4_ssd.model.forward_with_hidden", side_effect=forward),
            patch("deepseek_v4_ssd.dspark._target_sequence", side_effect=verify),
            patch("deepseek_v4_ssd.dspark._should_fallback", return_value=True),
        ):
            output = list(
                generate_tokens(
                    [0],
                    object(),
                    DSpark(),
                    [None],
                    max_tokens=4,
                    prefill_step_size=1,
                    temperature=0,
                    top_p=1,
                    record_round=lambda *values: recorded_rounds.append(values),
                    record_fallback=lambda: fallbacks.append(True),
                    fallback_enabled=False,
                )
            )

        self.assertEqual([token for token, _, _ in output], [1, 1, 2, 3])
        self.assertEqual(fallbacks, [])
        verification = recorded_rounds[0][3]
        self.assertTrue(verification.fallback_would_trigger)
        self.assertFalse(verification.fallback_triggered)


    def test_sequential_verification_replays_the_committed_prefix_sequentially(self):
        replayed = []

        class DSpark:
            target_layers = (0,)

            def reset_cache(self):
                pass

            def prefill_context(self, _hidden, _offset):
                pass

            def draft(self, *_args):
                return DraftResult(
                    [2, 4],
                    [mx.zeros((5,)), mx.zeros((5,))],
                    [1.0, 1.0],
                    0.0,
                )

        def forward(_model, inputs, _cache, _layers):
            logits = mx.zeros((1, inputs.shape[1], 5))
            logits[..., 1] = 1
            return logits, mx.ones((1, inputs.shape[1], 1))

        def verify(_model, tokens, _cache, _layers):
            logits = mx.zeros((1, len(tokens), 5))
            logits[0, 0, 2] = 1
            logits[0, 1, 3] = 1
            return (
                logits,
                mx.ones((1, len(tokens), 1)),
                [None],
                VerificationMetrics(verification_mode="sequential"),
            )

        def replay(_model, inputs, _cache, _layers):
            replayed.append(inputs.reshape(-1).tolist())
            return (
                mx.zeros((1, inputs.shape[1], 5)),
                mx.ones((1, inputs.shape[1], 1)),
                (0.01,) * inputs.shape[1],
                0,
                0,
            )

        with (
            patch("deepseek_v4_ssd.model.forward_with_hidden", side_effect=forward),
            patch(
                "deepseek_v4_ssd.model._sequential_forward_with_hidden",
                side_effect=replay,
            ),
            patch(
                "deepseek_v4_ssd.dspark._sequential_target_sequence",
                side_effect=verify,
            ),
            patch("deepseek_v4_ssd.dspark._should_fallback", return_value=True),
        ):
            list(
                generate_tokens(
                    [0],
                    object(),
                    DSpark(),
                    [None],
                    max_tokens=5,
                    prefill_step_size=1,
                    temperature=0,
                    top_p=1,
                    sequential_verification=True,
                )
            )

        self.assertEqual(replayed, [[1, 2]])


    def test_greedy_verification_stops_at_the_first_mismatch(self):
        draft = DraftResult(
            tokens=[2, 4],
            logprobs=[mx.zeros((5,)), mx.zeros((5,))],
            confidence=[0.9, 0.8],
            seconds=0.01,
        )
        target = mx.array(
            [
                [-4.0, -4.0, 0.0, -4.0, -4.0],
                [-4.0, -4.0, -4.0, 0.0, -4.0],
                [-4.0, 0.0, -4.0, -4.0, -4.0],
            ]
        )

        accepted, token, _ = _verify(draft, target, temperature=0)

        self.assertEqual(accepted, 1)
        self.assertEqual(token, 3)

    def test_sampling_verification_accepts_identical_distributions(self):
        distribution = mx.array([-float("inf"), 0.0, -float("inf")])
        draft = DraftResult([1], [distribution], [1.0], 0.01)
        target = mx.stack([distribution, distribution])

        accepted, token, _ = _verify(draft, target, temperature=1)

        self.assertEqual(accepted, 1)
        self.assertEqual(token, 1)

class CacheMetricsTests(unittest.TestCase):

    def test_expert_file_cache_policy_configures_darwin_bypass_flags(self):
        with (
            patch("deepseek_v4_ssd.io_metrics.sys.platform", "darwin"),
            patch("deepseek_v4_ssd.io_metrics.fcntl.fcntl") as control,
        ):
            configure_expert_file_cache_policy(7, "bypass")

        self.assertEqual(control.call_count, 2)
        self.assertEqual(control.call_args_list[0].args[0], 7)
        self.assertEqual(control.call_args_list[0].args[2], 1)
        self.assertEqual(control.call_args_list[1].args[0], 7)
        self.assertEqual(control.call_args_list[1].args[2], 0)

    def test_expert_file_cached_policy_does_not_change_descriptor_flags(self):
        with patch("deepseek_v4_ssd.io_metrics.fcntl.fcntl") as control:
            configure_expert_file_cache_policy(7, "cached")

        control.assert_not_called()

    def test_expert_file_cache_policy_rejects_unknown_values(self):
        with self.assertRaisesRegex(ValueError, "unknown expert-file cache policy"):
            configure_expert_file_cache_policy(7, "cold")

    def test_delta_reports_only_the_current_request(self):
        before = CacheMetrics(
            hits=10,
            misses=4,
            evictions=2,
            bytes_read=100,
            expert_union_calls=2,
            routed_expert_assignments=12,
            expert_union_experts=10,
            expert_union_misses=4,


            page_cache_probe_calls=2,
            page_cache_classified_bytes=80,
            page_cache_resident_bytes_before_read=30,
            page_cache_nonresident_bytes_before_read=50,
        )
        after = CacheMetrics(
            hits=13,
            misses=9,
            evictions=6,
            bytes_read=500,
            expert_union_calls=5,
            routed_expert_assignments=42,
            expert_union_experts=31,
            expert_union_misses=9,


            page_cache_probe_calls=7,
            page_cache_classified_bytes=480,
            page_cache_resident_bytes_before_read=130,
            page_cache_nonresident_bytes_before_read=350,
        )

        delta = after.delta(before)

        self.assertEqual(delta.hits, 3)
        self.assertEqual(delta.misses, 5)
        self.assertEqual(delta.evictions, 4)
        self.assertEqual(delta.bytes_read, 400)
        self.assertEqual(delta.expert_union_calls, 3)
        self.assertEqual(delta.routed_expert_assignments, 30)
        self.assertEqual(delta.expert_union_experts, 21)
        self.assertEqual(delta.expert_union_misses, 5)
        self.assertEqual(delta.page_cache_probe_calls, 5)
        self.assertEqual(delta.page_cache_classified_bytes, 400)
        self.assertEqual(delta.page_cache_resident_bytes_before_read, 100)
        self.assertEqual(delta.page_cache_nonresident_bytes_before_read, 300)

    def test_page_cache_residency_classifies_partial_boundary_pages(self):
        snapshot = PageCacheResidency(
            requested_offset=1_000,
            requested_length=5_000,
            mapped_offset=0,
            page_size=4_096,
            resident_pages=(True, False),
        )

        classification = snapshot.classify(5_000)

        self.assertEqual(classification.classified_bytes, 5_000)
        self.assertEqual(classification.resident_bytes, 3_096)
        self.assertEqual(classification.nonresident_bytes, 1_904)
        self.assertEqual(classification.probe_calls, 1)

    def test_page_cache_classification_preserves_unavailable_bytes(self):
        classified = PageCacheReadClassification(
            classified_bytes=10,
            resident_bytes=4,
            nonresident_bytes=6,
            probe_calls=1,
        )

        combined = classified + PageCacheReadClassification.unavailable(7)

        self.assertEqual(combined.classified_bytes, 10)
        self.assertEqual(combined.resident_bytes, 4)
        self.assertEqual(combined.nonresident_bytes, 6)
        self.assertEqual(combined.unclassified_bytes, 7)
        self.assertEqual(combined.probe_calls, 2)
        self.assertEqual(combined.probe_failures, 1)

    @unittest.skipUnless(sys.platform == "darwin", "Darwin mincore probe")
    def test_darwin_page_cache_probe_classifies_the_complete_range(self):
        with tempfile.TemporaryFile() as file:
            file.write(bytes(32_768))
            file.flush()

            snapshot = page_cache_residency_snapshot(
                file.fileno(),
                123,
                20_000,
            )

        self.assertIsNotNone(snapshot)
        classification = snapshot.classify(20_000)
        self.assertEqual(classification.classified_bytes, 20_000)
        self.assertEqual(
            classification.resident_bytes + classification.nonresident_bytes,
            20_000,
        )

    def test_process_disk_io_delta_does_not_report_counter_regression(self):
        delta = ProcessDiskIO(90, 140).delta(ProcessDiskIO(100, 120))

        self.assertEqual(delta, ProcessDiskIO(0, 20))


class ExpertCacheTests(unittest.TestCase):
    def test_cache_bypass_rejects_unaligned_reads(self):
        cache = object.__new__(ExpertCache)
        cache.direct_io_alignment = 4_096
        backing = bytearray(8_192 + 4_095)
        backing_view = memoryview(backing)
        address = ctypes.addressof(ctypes.c_char.from_buffer(backing_view))
        aligned_start = (-address) % 4_096
        view = backing_view[aligned_start : aligned_start + 8_192]

        cache._validate_direct_read(0, [view])
        with self.assertRaisesRegex(RuntimeError, "file offset"):
            cache._validate_direct_read(1, [view])
        with self.assertRaisesRegex(RuntimeError, "destination address"):
            cache._validate_direct_read(0, [view[1:]])
        with self.assertRaisesRegex(RuntimeError, "iovec length"):
            cache._validate_direct_read(0, [view[:-1]])

    def test_cache_rejects_unknown_file_cache_policy(self):
        model = SimpleNamespace(selected_expert_count=1)
        with self.assertRaisesRegex(ValueError, "unknown expert-file cache policy"):
            ExpertCache(model, slots=1, file_cache_policy="cold")

    def test_read_limiter_caps_aggregate_preadv_rate(self):
        limiter = _ReadLimiter(1_000_000_000)
        with (
            patch("deepseek_v4_ssd.expert_cache.os.preadv", return_value=500_000_000),
            patch(
                "deepseek_v4_ssd.expert_cache.time.perf_counter",
                side_effect=[10.0, 10.25],
            ),
            patch("deepseek_v4_ssd.expert_cache.time.sleep") as sleep,
        ):
            count = limiter.preadv(1, [memoryview(bytearray(1))], 0)

        self.assertEqual(count, 500_000_000)
        sleep.assert_called_once()
        self.assertAlmostEqual(sleep.call_args.args[0], 0.25)

    def test_cache_reads_an_independent_expert_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experts").mkdir()
            dspark_experts = root / "dspark/experts"
            dspark_experts.mkdir(parents=True)
            regions = (
                Tensor("w1.weight", "I8", (1, 4), 0, 4),
                Tensor("w1.scale", "F8_E8M0", (1, 1), 4, 1),
                Tensor("w2.weight", "I8", (1, 4), 5, 4),
                Tensor("w2.scale", "F8_E8M0", (1, 1), 9, 1),
                Tensor("w3.weight", "I8", (1, 4), 10, 4),
                Tensor("w3.scale", "F8_E8M0", (1, 1), 14, 1),
            )
            (root / "experts/layer_00.bin").write_bytes(bytes(15))
            dspark_experts.joinpath("layer_00.bin").write_bytes(bytes(range(15)))
            model = InstalledModel(
                root=root,
                model_id="fixture",
                revision="fixture",
                layer_count=1,
                expert_count=1,
                selected_expert_count=1,
                expert_blob_size=15,
                common_tensors=(),
                expert_regions=regions,
            )

            snapshot = PageCacheResidency(
                requested_offset=0,
                requested_length=15,
                mapped_offset=0,
                page_size=4_096,
                resident_pages=(False,),
            )
            with patch(
                "deepseek_v4_ssd.expert_cache.page_cache_residency_snapshot",
                return_value=snapshot,
            ):
                with ExpertCache(
                    model,
                    slots=1,
                    read_workers=1,
                    layer_count=1,
                    expert_directory=dspark_experts,
                    page_cache_probe=True,
                ) as cache:
                    ready = dict(cache.iter_ready(0, [0]))

            self.assertEqual(cache.metrics.bytes_read, 15)
            self.assertEqual(
                ready[0].w1.view(mx.uint8).tolist(),
                [[0, 1, 2, 3]],
            )
            self.assertEqual(
                ready[0].w13.view(mx.uint8).tolist(),
                [[10, 11, 12, 13], [0, 1, 2, 3]],
            )
            self.assertEqual(cache.metrics.upload_seconds, 0.0)
            self.assertEqual(cache.metrics.page_cache_probe_calls, 1)
            self.assertEqual(cache.metrics.page_cache_probe_failures, 0)
            self.assertEqual(cache.metrics.page_cache_classified_bytes, 15)
            self.assertEqual(
                cache.metrics.page_cache_nonresident_bytes_before_read,
                15,
            )

    def test_staged_cache_admits_only_after_w2_and_reuses_the_split_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experts").mkdir()
            regions = (
                Tensor("w1.weight", "I8", (1, 4), 0, 4),
                Tensor("w1.scale", "F8_E8M0", (1, 1), 4, 1),
                Tensor("w2.weight", "I8", (1, 4), 5, 4),
                Tensor("w2.scale", "F8_E8M0", (1, 1), 9, 1),
                Tensor("w3.weight", "I8", (1, 4), 10, 4),
                Tensor("w3.scale", "F8_E8M0", (1, 1), 14, 1),
            )
            (root / "experts/layer_00.bin").write_bytes(bytes(range(15)))
            model = InstalledModel(
                root=root,
                model_id="fixture",
                revision="fixture",
                layer_count=1,
                expert_count=1,
                selected_expert_count=1,
                expert_blob_size=15,
                common_tensors=(),
                expert_regions=regions,
            )

            with ExpertCache(
                model,
                slots=1,
                read_workers=2,
                staged_expert_streaming=True,
            ) as cache:
                first = []
                for ready in cache.iter_staged_ready(0, [0]):
                    self.assertFalse(ready.finished)
                    self.assertEqual(
                        ready.weights.w13.view(mx.uint8).tolist(),
                        [[10, 11, 12, 13], [0, 1, 2, 3]],
                    )
                    ready.finish_w2()
                    self.assertTrue(ready.finished)
                    first.append(ready)

                second = list(cache.iter_staged_ready(0, [0]))
                self.assertTrue(second[0].finished)
                second[0].finish_w2()
                self.assertEqual(
                    second[0].weights.w2.view(mx.uint8).tolist(),
                    [[5, 6, 7, 8]],
                )
                self.assertEqual(cache.metrics.hits, 1)
                self.assertEqual(cache.metrics.misses, 1)
                self.assertEqual(cache.metrics.bytes_read, 15)
                self.assertEqual(cache.metrics.staged_expert_reads, 1)
                self.assertEqual(cache.metrics.staged_w13_bytes_read, 10)
                self.assertEqual(cache.metrics.staged_w2_bytes_read, 5)

    def test_staged_cache_releases_a_slot_when_w2_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experts").mkdir()
            regions = (
                Tensor("w1.weight", "I8", (1, 4), 0, 4),
                Tensor("w1.scale", "F8_E8M0", (1, 1), 4, 1),
                Tensor("w2.weight", "I8", (1, 4), 5, 4),
                Tensor("w2.scale", "F8_E8M0", (1, 1), 9, 1),
                Tensor("w3.weight", "I8", (1, 4), 10, 4),
                Tensor("w3.scale", "F8_E8M0", (1, 1), 14, 1),
            )
            (root / "experts/layer_00.bin").write_bytes(bytes(range(15)))
            model = InstalledModel(
                root=root,
                model_id="fixture",
                revision="fixture",
                layer_count=1,
                expert_count=1,
                selected_expert_count=1,
                expert_blob_size=15,
                common_tensors=(),
                expert_regions=regions,
            )

            with ExpertCache(
                model,
                slots=1,
                read_workers=1,
                staged_expert_streaming=True,
            ) as cache, patch.object(
                cache,
                "_read_staged_w2_into_pool",
                side_effect=OSError("w2 failed"),
            ):
                iterator = cache.iter_staged_ready(0, [0])
                ready = next(iterator)
                with self.assertRaisesRegex(OSError, "w2 failed"):
                    ready.finish_w2()
                iterator.close()
                self.assertEqual(cache.resident_count, 0)
                self.assertEqual(cache.metrics.bytes_read, 0)

    def test_lfu_cache_reuses_and_evicts_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experts").mkdir()
            regions = (
                Tensor("w1.weight", "I8", (1, 4), 0, 4),
                Tensor("w1.scale", "F8_E8M0", (1, 1), 4, 1),
                Tensor("w2.weight", "I8", (1, 4), 5, 4),
                Tensor("w2.scale", "F8_E8M0", (1, 1), 9, 1),
                Tensor("w3.weight", "I8", (1, 4), 10, 4),
                Tensor("w3.scale", "F8_E8M0", (1, 1), 14, 1),
            )
            blob_size = 15
            (root / "experts/layer_00.bin").write_bytes(bytes(range(blob_size * 2)))
            model = InstalledModel(
                root=root,
                model_id="fixture",
                revision="fixture",
                layer_count=1,
                expert_count=2,
                selected_expert_count=1,
                expert_blob_size=blob_size,
                common_tensors=(),
                expert_regions=regions,
            )

            with ExpertCache(model, slots=1, read_workers=1) as cache:
                first = cache.get_many(0, [0])
                cache.get_many(0, [0])
                second = cache.get_many(0, [1])
                self.assertEqual(cache.metrics.hits, 1)
                self.assertEqual(cache.metrics.misses, 2)
                self.assertEqual(cache.metrics.evictions, 1)
                self.assertEqual(cache.metrics.bytes_read, blob_size * 2)
                self.assertEqual(first.slots[0], second.slots[1])
                individual = cache.get_many(0, [1])
                self.assertEqual(len(individual.individual_weights), 1)
                self.assertEqual(individual.individual_weights[0].w1.shape, (1, 1))

            with ExpertCache(model, slots=2, read_workers=1) as cache:
                with cache.capture_expert_unions() as profile:
                    cache.get_many(0, [0, 1, 0, 1, 1])

                self.assertEqual(cache.metrics.expert_union_calls, 1)
                self.assertEqual(cache.metrics.routed_expert_assignments, 5)
                self.assertEqual(cache.metrics.expert_union_experts, 2)
                self.assertEqual(cache.metrics.expert_union_misses, 2)
                self.assertEqual(profile.routed_expert_assignments, 5)
                self.assertEqual(profile.unique_experts, 2)
                self.assertEqual(profile.cache_misses, 2)
                self.assertEqual(profile.layers[0].layer, 0)


    def test_layer_limits_keep_resident_experts_across_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experts").mkdir()
            regions = (
                Tensor("w1.weight", "I8", (1, 4), 0, 4),
                Tensor("w1.scale", "F8_E8M0", (1, 1), 4, 1),
                Tensor("w2.weight", "I8", (1, 4), 5, 4),
                Tensor("w2.scale", "F8_E8M0", (1, 1), 9, 1),
                Tensor("w3.weight", "I8", (1, 4), 10, 4),
                Tensor("w3.scale", "F8_E8M0", (1, 1), 14, 1),
            )
            for layer in range(2):
                (root / f"experts/layer_{layer:02d}.bin").write_bytes(bytes(range(45)))
            model = InstalledModel(
                root=root,
                model_id="fixture",
                revision="fixture",
                layer_count=2,
                expert_count=3,
                selected_expert_count=1,
                expert_blob_size=15,
                common_tensors=(),
                expert_regions=regions,
            )

            with ExpertCache(model, slots=4, read_workers=1) as cache:
                cache.get_many(0, [0])
                cache.get_many(0, [1])
                cache.get_many(0, [2])
                cache.get_many(1, [0])
                cache.get_many(1, [1])
                misses = cache.metrics.misses
                cache.get_many(1, [0])
                cache.get_many(0, [1])
                self.assertEqual(cache.metrics.misses, misses)
                self.assertEqual(cache.resident_count, 4)

    def test_pinned_prefill_layer_keeps_its_experts_resident(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experts").mkdir()
            regions = (
                Tensor("w1.weight", "I8", (1, 4), 0, 4),
                Tensor("w1.scale", "F8_E8M0", (1, 1), 4, 1),
                Tensor("w2.weight", "I8", (1, 4), 5, 4),
                Tensor("w2.scale", "F8_E8M0", (1, 1), 9, 1),
                Tensor("w3.weight", "I8", (1, 4), 10, 4),
                Tensor("w3.scale", "F8_E8M0", (1, 1), 14, 1),
            )
            for layer in range(2):
                (root / f"experts/layer_{layer:02d}.bin").write_bytes(bytes(range(60)))
            model = InstalledModel(
                root=root,
                model_id="fixture",
                revision="fixture",
                layer_count=2,
                expert_count=4,
                selected_expert_count=1,
                expert_blob_size=15,
                common_tensors=(),
                expert_regions=regions,
            )

            with ExpertCache(model, slots=4, read_workers=1) as cache:
                for _ in range(8):
                    cache.get_many(0, [0, 1])
                with cache.pin_layer(1):
                    cache.get_many(1, [0])
                    cache.get_many(1, [1])
                    cache.get_many(1, [2])
                    misses = cache.metrics.misses
                    cache.get_many(1, [0])

                self.assertEqual(cache.metrics.misses, misses)

    def test_batched_layer_uses_one_strided_expert_view(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experts").mkdir()
            regions = (
                Tensor("w1.weight", "I8", (1, 4), 0, 4),
                Tensor("w1.scale", "F8_E8M0", (1, 4), 4, 4),
                Tensor("w2.weight", "I8", (1, 4), 8, 4),
                Tensor("w2.scale", "F8_E8M0", (1, 4), 12, 4),
                Tensor("w3.weight", "I8", (1, 4), 16, 4),
                Tensor("w3.scale", "F8_E8M0", (1, 4), 20, 4),
            )
            blob_size = 24
            source = bytes(range(blob_size * 2))
            (root / "experts/layer_00.bin").write_bytes(source)
            model = InstalledModel(
                root=root,
                model_id="fixture",
                revision="fixture",
                layer_count=1,
                expert_count=2,
                selected_expert_count=1,
                expert_blob_size=blob_size,
                common_tensors=(),
                expert_regions=regions,
            )

            for staged in (False, True):
                trace_path = root / f"prefetch-{staged}.json"
                with self.subTest(staged=staged), ExpertCache(
                    model,
                    slots=1,
                    read_workers=1,
                    staged_expert_streaming=staged,
                    route_trace_path=trace_path,
                ) as cache:
                    cache.prefetch_layer(0)
                    with cache.batched_layer(0) as batched:
                        mx.eval(batched.w1, batched.w3_scales)
                        cache.record_routes(0, np.array([0], dtype=np.int32))
                        self.assertIs(cache.current_batched(0), batched)
                        cache.record_compute_submit(0)
                        self.assertEqual(batched.w1.shape, (2, 1, 1))
                        self.assertEqual(
                            batched.w1_scales[:, 0, 0].tolist(), [4, 28]
                        )
                    self.assertIsNone(cache.current_batched(0))
                    self.assertEqual(cache.metrics.bytes_read, len(source))

                trace = json.loads(trace_path.read_text(encoding="utf-8"))
                event = trace["prefetch_events"][0]
                self.assertEqual(event["layer"], 0)
                self.assertEqual(event["requested_experts"], 2)
                self.assertEqual(event["used_experts"], 1)
                self.assertEqual(event["useful_bytes"], blob_size)
                self.assertEqual(event["wasted_bytes"], blob_size)
                self.assertLessEqual(
                    event["read_submit_seconds"],
                    event["read_start_seconds"],
                )
                self.assertLessEqual(
                    event["read_start_seconds"],
                    event["read_complete_seconds"],
                )
                self.assertGreaterEqual(
                    event["compute_submit_seconds"],
                    event["expert_deadline_seconds"],
                )
                self.assertEqual(
                    event["ready_by_deadline"],
                    event["read_complete_seconds"]
                    <= event["expert_deadline_seconds"],
                )

    def test_selective_batched_layer_reads_union_rows_at_original_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experts").mkdir()
            regions = (
                Tensor("w1.weight", "I8", (1, 4), 0, 4),
                Tensor("w1.scale", "F8_E8M0", (1, 4), 4, 4),
                Tensor("w2.weight", "I8", (1, 4), 8, 4),
                Tensor("w2.scale", "F8_E8M0", (1, 4), 12, 4),
                Tensor("w3.weight", "I8", (1, 4), 16, 4),
                Tensor("w3.scale", "F8_E8M0", (1, 4), 20, 4),
            )
            blob_size = 24
            source = bytes(range(blob_size * 4))
            (root / "experts/layer_00.bin").write_bytes(source)
            model = InstalledModel(
                root=root,
                model_id="fixture",
                revision="fixture",
                layer_count=1,
                expert_count=4,
                selected_expert_count=1,
                expert_blob_size=blob_size,
                common_tensors=(),
                expert_regions=regions,
            )

            with ExpertCache(model, slots=1, read_workers=1) as cache:
                offsets = []
                original_pread_views = cache._pread_views

                def record_read(layer, views, offset, **kwargs):
                    offsets.append(offset)
                    return original_pread_views(layer, views, offset)

                cache._pread_views = record_read
                cache.prefetch_layer(0, experts=(3, 1, 3))
                with self.assertRaisesRegex(RuntimeError, "different expert set"):
                    cache.prefetch_layer(0, experts=(0, 2))
                with cache.batched_layer(
                    0,
                    experts=(1, 3),
                ) as batched:
                    selected_scales = mx.take(
                        batched.w1_scales[:, 0, 0],
                        mx.array([1, 3]),
                        axis=0,
                    )
                    mx.eval(selected_scales)

                self.assertEqual(selected_scales.tolist(), [28, 76])
                self.assertEqual(offsets, [blob_size, blob_size * 3])
                self.assertEqual(cache.metrics.bytes_read, blob_size * 2)

class MXFP8PoolingCacheTests(unittest.TestCase):
    def test_generation_cache_state_exposes_raw_quantized_arrays(self):
        class QuantizedOnlyCache(MXFP8PoolingCache):
            @property
            def state(self):
                raise AssertionError("generation must not rebuild a BF16 cache")

        cache = QuantizedOnlyCache(ratio=4)
        cache.update(mx.random.uniform(shape=(1, 64, 64)).astype(mx.bfloat16))

        arrays = _RawEvalCacheList(cache).state
        mx.eval(*arrays)

        self.assertGreater(len(arrays), 0)

    def test_persistence_round_trip_keeps_quantized_chunks(self):
        class QuantizedOnlyCache(MXFP8PoolingCache):
            @property
            def state(self):
                raise AssertionError("persistence must not rebuild a BF16 cache")

        source = QuantizedOnlyCache(ratio=4)
        source.update(mx.random.uniform(shape=(1, 130, 64)).astype(mx.bfloat16))
        eval_prompt_cache([CacheList(source)])

        arrays = {}
        schema = _encode_cache_state(
            _persistence_cache_state([CacheList(source)]),
            arrays,
        )
        schema = json.loads(json.dumps(schema))
        decoded = _decode_cache_state(schema, arrays)
        restored = MXFP8PoolingCache(ratio=4)
        _restore_persistence_cache([CacheList(restored)], decoded)
        eval_prompt_cache([CacheList(restored)])

        query = mx.random.uniform(shape=(1, 3, 2, 64)).astype(mx.float32)
        expected = source.quantized_matmul(query)
        actual = restored.quantized_matmul(query)
        mx.eval(expected, actual)

        self.assertEqual(restored.offset, source.offset)
        self.assertEqual(restored.nbytes, source.nbytes)
        self.assertLess(mx.max(mx.abs(actual - expected)).item(), 1e-6)

    def test_persistence_snapshot_keeps_an_independent_active_remainder(self):
        source = MXFP8PoolingCache(ratio=4)
        kv = mx.arange(24).reshape(1, 3, 8).astype(mx.bfloat16)
        gate = mx.arange(12).reshape(1, 3, 4).astype(mx.bfloat16)
        source.accumulate_windows(kv, gate, 0)
        snapshot = source.persistence_state()
        mx.eval(snapshot["buf_kv"], snapshot["buf_gate"])

        source.accumulate_windows(
            mx.full((1, 2, 8), 99, dtype=mx.bfloat16),
            mx.full((1, 2, 4), 99, dtype=mx.bfloat16),
            3,
        )
        restored = MXFP8PoolingCache(ratio=4)
        restored.restore_persistence_state(snapshot)
        mx.eval(restored.buf_kv, restored.buf_gate)

        self.assertEqual(restored.remainder, 3)
        self.assertEqual(restored.buf_kv[:, :3].tolist(), kv.tolist())
        self.assertEqual(restored.buf_gate[:, :3].tolist(), gate.tolist())

    def test_persistence_snapshot_freezes_completed_chunk_collections(self):
        source = MXFP8PoolingCache(ratio=4)
        source.update(mx.ones((1, 64, 64), dtype=mx.bfloat16))
        snapshot = source.persistence_state()
        expected = source.fetch(1, 64, mx.bfloat16)
        mx.eval(expected, *[array for chunk in snapshot["chunks"] for array in chunk])

        source.update(mx.full((1, 64, 64), 2, dtype=mx.bfloat16))
        restored = MXFP8PoolingCache(ratio=4)
        restored.restore_persistence_state(snapshot)
        result = restored.fetch(1, 64, mx.bfloat16)
        mx.eval(result)

        self.assertEqual(len(snapshot["chunks"]), 1)
        self.assertEqual(restored.offset, 64)
        self.assertEqual(result.shape, (1, 64, 64))
        self.assertLess(mx.max(mx.abs(result - expected)).item(), 1e-6)

    def test_completed_chunks_use_less_memory_than_bfloat16(self):
        cache = MXFP8PoolingCache(ratio=4)
        source = mx.random.uniform(shape=(1, 64, 64)).astype(mx.bfloat16)
        restored = cache.update_and_fetch(source)
        mx.eval(restored)

        self.assertEqual(restored.shape, source.shape)
        self.assertEqual(cache.offset, 64)
        self.assertLess(cache.nbytes, source.nbytes)
        self.assertLess(mx.mean(mx.abs(restored - source)).item(), 0.02)
        state = cache.state
        self.assertEqual(cache.offset, 64)
        self.assertEqual(state[2].shape, source.shape)

    def test_quantized_chunks_multiply_without_fetching_complete_cache(self):
        cache = MXFP8PoolingCache(ratio=4)
        source = mx.random.uniform(shape=(1, 130, 64)).astype(mx.bfloat16)
        cache.update(source)
        query = mx.random.uniform(shape=(1, 3, 2, 64)).astype(mx.float32)

        actual = cache.quantized_matmul(query)
        restored = cache._fetch(1, 64, mx.bfloat16)
        expected = query @ restored[:, None].swapaxes(-1, -2).astype(mx.float32)
        mx.eval(actual, expected)

        self.assertEqual(actual.shape, (1, 3, 2, 130))
        self.assertLess(mx.max(mx.abs(actual - expected)).item(), 0.05)

    def test_fp4_index_view_matches_the_fp8_index_shape(self):
        cache = MXFP8PoolingCache(ratio=4)
        source = mx.random.uniform(shape=(1, 130, 64)).astype(mx.bfloat16)
        cache.update(source)
        query = mx.random.uniform(shape=(1, 3, 2, 64)).astype(mx.float32)

        actual = cache.index_matmul(query)
        expected = cache.quantized_matmul(query)
        mx.eval(actual, expected)

        self.assertEqual(actual.shape, expected.shape)
        relative_error = mx.mean(mx.abs(actual - expected)) / mx.mean(mx.abs(expected))
        self.assertLess(relative_error.item(), 0.1)

    def test_packed_chunks_are_reused_until_the_cache_changes(self):
        cache = MXFP8PoolingCache(ratio=4)
        cache.update(mx.random.uniform(shape=(1, 64, 64)).astype(mx.bfloat16))

        packed = cache._packed()
        self.assertIs(cache._packed(), packed)

        cache.update(mx.random.uniform(shape=(1, 64, 64)).astype(mx.bfloat16))
        self.assertIsNot(cache._packed(), packed)

    def test_gather_reads_selected_rows_across_quantized_chunks(self):
        cache = MXFP8PoolingCache(ratio=4)
        source = mx.random.uniform(shape=(1, 130, 64)).astype(mx.bfloat16)
        cache.update(source)
        indices = mx.array([[[0, 63, 64], [65, 128, 129]]])

        actual = cache.gather(indices)
        restored = cache._fetch(1, 64, mx.bfloat16)
        expected = mx.take_along_axis(
            mx.broadcast_to(restored[:, None], (1, 2, 130, 64)),
            mx.broadcast_to(indices[..., None], (1, 2, 3, 64)),
            axis=2,
        )
        mx.eval(actual, expected)

        self.assertEqual(actual.shape, (1, 2, 3, 64))
        self.assertLess(mx.max(mx.abs(actual - expected)).item(), 1e-5)

    def test_gather_reads_pending_rows_before_the_first_chunk(self):
        cache = MXFP8PoolingCache(ratio=4)
        source = mx.random.uniform(shape=(1, 32, 64)).astype(mx.bfloat16)
        cache.update(source)
        indices = mx.array([[[0, 15, 31]]])

        actual = cache.gather(indices)
        expected = mx.take_along_axis(
            source[:, None],
            mx.broadcast_to(indices[..., None], (1, 1, 3, 64)),
            axis=2,
        )
        mx.eval(actual, expected)

        self.assertLess(mx.max(mx.abs(actual - expected)).item(), 1e-5)

    def test_sparse_attention_matches_complete_cache(self):
        cache = MXFP8PoolingCache(ratio=4)
        cache.update(mx.random.uniform(shape=(1, 600, 64)).astype(mx.bfloat16))
        query = mx.random.uniform(shape=(1, 2, 2, 64)).astype(mx.bfloat16)
        local = mx.random.uniform(shape=(1, 1, 3, 64)).astype(mx.bfloat16)
        topk = mx.array([[[0, 63, 511], [64, 512, 599]]])
        sinks = mx.zeros((2,), dtype=mx.float32)

        actual = _sparse_pooled_attention(
            query,
            local,
            cache,
            topk,
            None,
            None,
            0.125,
            sinks,
        )
        complete = cache.fetch(1, 64, mx.bfloat16)
        expected = _ORIGINAL_SPARSE_POOLED_ATTENTION(
            query,
            local,
            complete,
            topk,
            None,
            None,
            0.125,
            sinks,
        )
        mx.eval(actual, expected)

        self.assertLess(mx.max(mx.abs(actual - expected)).item(), 1e-5)

    def test_ratio_four_chunking_matches_one_prefill(self):
        arguments = deepseek_v4.ModelArgs(hidden_size=32)
        compressor = deepseek_v4.Compressor(arguments, compress_ratio=4, head_dim=64)
        source = mx.random.uniform(shape=(1, 8, 32)).astype(mx.bfloat16)

        chunked_cache = CorrectPoolingCache(4)
        _correct_compressor(compressor, source[:, :4], chunked_cache, 0)
        chunked = _correct_compressor(compressor, source[:, 4:], chunked_cache, 4)
        complete = _correct_compressor(compressor, source, CorrectPoolingCache(4), 0)
        mx.eval(chunked, complete)

        self.assertEqual(chunked.shape, complete.shape)
        self.assertLess(mx.max(mx.abs(chunked - complete)).item(), 1e-5)


class MXFP4Tests(unittest.TestCase):


    def test_ready_experts_keep_router_selection_order(self):
        def quantized(value: float):
            return mx.quantize(
                mx.full((32, 32), value),
                group_size=32,
                bits=4,
                mode="mxfp4",
            )

        experts = []
        for value in (0.125, 0.25):
            w1, w1_scales = quantized(value)
            w2, w2_scales = quantized(value + 0.125)
            w3, w3_scales = quantized(value + 0.25)
            experts.append(
                ExpertWeights(w1, w1_scales, w2, w2_scales, w3, w3_scales)
            )

        cache = SimpleNamespace(
            ready_expert_decode=True,
            iter_ready=lambda _layer, _selected: iter(
                [(0, experts[0]), (1, experts[1])]
            ),
        )
        switch = _StreamingSwitchGLU(0, cache, lambda up, _gate: up)
        reference_cache = SimpleNamespace(
            get_many=lambda _layer, _selected: ResidentExperts(
                tuple(experts), {0: 0, 1: 1}
            )
        )
        for source, selected in (
            (
                mx.ones((1, 1, 32), dtype=mx.bfloat16),
                mx.array([[[1, 0]]]),
            ),
            (
                mx.ones((1, 32), dtype=mx.bfloat16),
                mx.array([[1, 0]]),
            ),
        ):
            with self.subTest(source_shape=source.shape):
                actual = switch(source, selected)
                expected = _StreamingSwitchGLU(
                    0,
                    reference_cache,
                    lambda up, _gate: up,
                )(source, selected)
                mx.eval(actual, expected)

                self.assertLess(mx.max(mx.abs(actual - expected)).item(), 1e-5)

    def test_staged_ready_experts_finish_w2_before_down_projection(self):
        def quantized(value: float):
            return mx.quantize(
                mx.full((32, 32), value),
                group_size=32,
                bits=4,
                mode="mxfp4",
            )

        experts = []
        for value in (0.125, 0.25):
            w1, w1_scales = quantized(value)
            w2, w2_scales = quantized(value + 0.125)
            w3, w3_scales = quantized(value + 0.25)
            experts.append(
                ExpertWeights(w1, w1_scales, w2, w2_scales, w3, w3_scales)
            )

        finished = []

        def ready(expert):
            return SimpleNamespace(
                expert=expert,
                weights=experts[expert],
                finish_w2=lambda: finished.append(expert),
            )

        cache = SimpleNamespace(
            ready_expert_decode=True,
            staged_expert_streaming=True,
            iter_staged_ready=lambda _layer, _selected: iter(
                [ready(0), ready(1)]
            ),
            record_staged_first_stage_submit=lambda _seconds: None,
        )
        switch = _StreamingSwitchGLU(0, cache, lambda up, _gate: up)
        source = mx.ones((1, 1, 32), dtype=mx.bfloat16)
        selected = mx.array([[[1, 0]]])

        actual = switch(source, selected)
        reference_cache = SimpleNamespace(
            get_many=lambda _layer, _selected: ResidentExperts(
                tuple(experts), {0: 0, 1: 1}
            )
        )
        expected = _StreamingSwitchGLU(
            0,
            reference_cache,
            lambda up, _gate: up,
        )(source, selected)
        mx.eval(actual, expected)

        self.assertEqual(finished, [0, 1])
        self.assertLess(mx.max(mx.abs(actual - expected)).item(), 1e-5)

    def test_individual_experts_run_in_selected_order(self):
        def quantized(value: float):
            return mx.quantize(
                mx.full((32, 32), value),
                group_size=32,
                bits=4,
                mode="mxfp4",
            )

        experts = []
        for value in (0.125, 0.25):
            w1, w1_scales = quantized(value)
            w2, w2_scales = quantized(value + 0.125)
            w3, w3_scales = quantized(value + 0.25)
            experts.append(
                ExpertWeights(w1, w1_scales, w2, w2_scales, w3, w3_scales)
            )

        def get_many(_layer, _experts):
            return ResidentExperts(tuple(experts), {0: 0, 1: 1})

        cache = SimpleNamespace(
            model=SimpleNamespace(expert_count=2),
            get_many=get_many,
        )
        switch = _StreamingSwitchGLU(0, cache, lambda up, _gate: up)
        def reference(weights):
            def restore(weight, scales):
                return mx.dequantize(
                    weight,
                    scales,
                    group_size=32,
                    bits=4,
                    mode="mxfp4",
                )

            def run(source):
                hidden = source @ restore(weights.w3, weights.w3_scales).T
                return hidden @ restore(weights.w2, weights.w2_scales).T

            return run

        for length in (1, 2):
            with self.subTest(length=length):
                source = mx.ones((1, length, 32), dtype=mx.bfloat16)
                selected = mx.array([[[1, 0] for _ in range(length)]])
                actual = switch(source, selected)
                expected = mx.concatenate(
                    [
                        mx.stack(
                            [
                                reference(experts[expert])(source[:, token : token + 1])
                                for expert in (1, 0)
                            ],
                            axis=-2,
                        )
                        for token in range(length)
                    ],
                    axis=1,
                )
                mx.eval(actual, expected)
                self.assertEqual(actual.shape, (1, length, 2, 32))
                self.assertLess(mx.max(mx.abs(actual - expected)).item(), 1e-5)

    def test_topk_resolves_score_ties_by_position(self):
        scores = mx.array([[1.0, 1.0, 1.0, 0.5]])
        indices = _stable_topk_indices(scores, 2)
        self.assertEqual(indices.tolist(), [[0, 1]])

    def test_batched_experts_match_individual_experts(self):
        def quantized(value: float):
            return mx.quantize(
                mx.full((32, 32), value),
                group_size=32,
                bits=4,
                mode="mxfp4",
            )

        experts = []
        for value in (0.125, 0.25):
            w1, w1_scales = quantized(value)
            w2, w2_scales = quantized(value + 0.125)
            w3, w3_scales = quantized(value + 0.25)
            experts.append(
                ExpertWeights(w1, w1_scales, w2, w2_scales, w3, w3_scales)
            )
        batched = BatchedExperts(
            *(
                mx.stack([getattr(expert, name) for expert in experts])
                for name in (
                    "w1",
                    "w1_scales",
                    "w2",
                    "w2_scales",
                    "w3",
                    "w3_scales",
                )
            ),
            w13=mx.stack(
                [mx.concatenate([expert.w3, expert.w1], axis=0) for expert in experts]
            ),
            w13_scales=mx.stack(
                [
                    mx.concatenate([expert.w3_scales, expert.w1_scales], axis=0)
                    for expert in experts
                ]
            ),
        )
        calls = []
        cache = SimpleNamespace(
            current_batched=lambda _layer: batched,
            record_gather_qmm=lambda count=3: calls.append(count),
        )
        switch = _StreamingSwitchGLU(0, cache, lambda up, _gate: up)
        source = mx.ones((1, 2, 32), dtype=mx.bfloat16)
        selected = mx.array([[[1, 0], [0, 1]]])

        actual = switch(source, selected)
        reference_cache = SimpleNamespace(
            get_many=lambda _layer, _experts: ResidentExperts(
                tuple(experts), {0: 0, 1: 1}
            )
        )
        expected = _StreamingSwitchGLU(
            0, reference_cache, lambda up, _gate: up
        )(source, selected)
        mx.eval(actual, expected)

        self.assertEqual(calls, [2])
        self.assertLess(mx.max(mx.abs(actual - expected)).item(), 1e-5)


if __name__ == "__main__":
    unittest.main()
