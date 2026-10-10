from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from .deepseek_v41 import engram as v41_engram
from .deepseek_v41 import moe as v41_moe
from .deepseek_v41.config import ModelArgs
from .deepseek_v41.packed_cache import PackedRows
from .deepseek_v41.dequant import dequant_fp8_rows
from .deepseek_v41.model import Model
from .expert_cache import ExpertCache
from .io_metrics import request_rows
from .model import _EmptySwitchGLU, _StreamingSwitchGLU


class _EmptyEngramEmbedding(nn.Module):
    def __init__(self, *_: object, **__: object):
        super().__init__()

    def __call__(self, _: mx.array) -> mx.array:
        raise RuntimeError("V4.1 engram SSD store was not installed")


class SSDEngramEmbedding(nn.Module):
    def __init__(
        self,
        weight_path: Path,
        scale_path: Path,
        rows: int,
        dimension: int,
        block_size: int,
    ):
        super().__init__()
        self.rows = rows
        self.dimension = dimension
        self.block_size = block_size
        self.weight = np.memmap(
            weight_path,
            mode="r",
            dtype=np.uint8,
            shape=(rows, dimension),
        )
        self.scale = np.memmap(
            scale_path,
            mode="r",
            dtype=np.uint8,
            shape=(rows, dimension // block_size),
        )

    def __call__(self, indices: mx.array) -> mx.array:
        row_ids = np.asarray(indices, dtype=np.int64)
        if row_ids.size and (row_ids.min() < 0 or row_ids.max() >= self.rows):
            raise ValueError("V4.1 engram row is outside the installed table")
        request_rows(self.weight._mmap, row_ids, self.dimension)
        request_rows(self.scale._mmap, row_ids, self.dimension // self.block_size)
        weight = mx.array(np.array(self.weight[row_ids], copy=True))
        scale = mx.array(np.array(self.scale[row_ids], copy=True))
        return dequant_fp8_rows(weight, scale, self.block_size)


class DeepSeekV41PromptCache:
    def __init__(self, cache):
        self.cache = cache

    @property
    def offset(self) -> int:
        return self.cache.offset

    @offset.setter
    def offset(self, value: int) -> None:
        self.cache.offset = value

    @property
    def state(self):
        arrays = []

        def collect(value):
            if isinstance(value, mx.array):
                arrays.append(value)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    collect(item)
            elif hasattr(value, "__dict__"):
                for item in vars(value).values():
                    collect(item)

        collect(self.cache.layers)
        return arrays

    @state.setter
    def state(self, _: object) -> None:
        raise ValueError("Restore DeepSeek V4.1 cache through its model support")

    def persistence_state(self):
        cache = self.cache
        layers = []
        for layer in cache.layers:
            layers.append({
                "win_kv": layer.win_kv.state() if isinstance(layer.win_kv, PackedRows) else layer.win_kv,
                "comp_kv": layer.comp_kv.state() if isinstance(layer.comp_kv, PackedRows) else layer.comp_kv,
                "index_k": layer.index_k.state() if isinstance(layer.index_k, PackedRows) else layer.index_k,
                "kv_state": layer.comp_state.kv_state if layer.comp_state else None,
                "score_state": layer.comp_state.score_state if layer.comp_state else None,
            })
        return {
            "kind": "deepseek_v41", "version": 2 if cache.packed_kv or cache.packed_index else 1,
            "offset": cache.offset, "capacity": cache.max_seq_len,
            "layers": layers,
            # Engram history is mutable NumPy data; capture it before decoding resumes.
            "engram_ids": mx.array(cache.engram_ids.copy()) if cache.engram_ids is not None else None,
        }

    def restore_persistence_state(self, state):
        from .deepseek_v41.cache import ModelCache

        if not isinstance(state, dict) or state.get("kind") != "deepseek_v41" or state.get("version") != (2 if self.cache.packed_kv or self.cache.packed_index else 1):
            raise ValueError("Invalid DeepSeek V4.1 prompt cache schema")
        offset, capacity = state.get("offset"), state.get("capacity")
        if (type(offset) is not int or type(capacity) is not int
                or not 0 <= offset <= capacity <= self.cache.args.max_seq_len or capacity < 1):
            raise ValueError("Invalid DeepSeek V4.1 prompt cache position")
        saved_layers = state.get("layers")
        if not isinstance(saved_layers, list) or len(saved_layers) != len(self.cache.layers):
            raise ValueError("DeepSeek V4.1 prompt cache layer count does not match")
        first = self.cache.layers[0]
        restored = ModelCache(self.cache.args, first.win_kv.shape[0], capacity, first.dtype, self.cache.packed_kv, self.cache.packed_index)

        def restore_array(saved, expected):
            if isinstance(expected, PackedRows):
                return expected.restore(saved)
            if expected is None:
                if saved is not None:
                    raise ValueError("Unexpected DeepSeek V4.1 prompt cache array")
                return None
            if not isinstance(saved, mx.array) or saved.shape != expected.shape or saved.dtype != expected.dtype:
                raise ValueError("DeepSeek V4.1 prompt cache array shape or dtype does not match")
            return mx.array(saved)

        for layer, saved in zip(restored.layers, saved_layers):
            if not isinstance(saved, dict) or set(saved) != {"win_kv", "comp_kv", "index_k", "kv_state", "score_state"}:
                raise ValueError("Invalid DeepSeek V4.1 layer cache")
            for name in ("win_kv", "comp_kv", "index_k"):
                setattr(layer, name, restore_array(saved[name], getattr(layer, name)))
            for name in ("kv_state", "score_state"):
                expected = getattr(layer.comp_state, name) if layer.comp_state else None
                value = restore_array(saved[name], expected)
                if layer.comp_state:
                    setattr(layer.comp_state, name, value)
        history = state.get("engram_ids")
        if restored.engram_ids is None:
            if history is not None:
                raise ValueError("Unexpected DeepSeek V4.1 Engram history")
        else:
            if not isinstance(history, mx.array) or history.shape != restored.engram_ids.shape or history.dtype != mx.int64:
                raise ValueError("Invalid DeepSeek V4.1 Engram history")
            restored.engram_ids = np.array(history, dtype=np.int64, copy=True)
        restored.offset = offset
        # Publish only after every field has been checked.
        self.cache = restored

    @property
    def nbytes(self) -> int:
        history = self.cache.engram_ids
        return sum(int(array.nbytes) for array in self.state) + (history.nbytes if history is not None else 0)

    @property
    def is_trimmable(self) -> bool:
        return False

    def trim(self, _: int) -> int:
        return 0


class DeepSeekV41ForCausalLM(nn.Module):
    def __init__(self, model: Model):
        super().__init__()
        self.model = model
        self.layers = model.layers
        self.args = model.args
        self.dspark = None
        self.mtp = None
        self.mtp_expert_cache = None
        self.packed_kv = False
        self.packed_index = False

    def make_cache(self):
        return [
            DeepSeekV41PromptCache(
                self.model.make_cache(dtype=mx.bfloat16, packed_kv=self.packed_kv, packed_index=self.packed_index)
            )
        ]

    def forward_with_hidden(self, inputs, cache, target_layers):
        return self.model(inputs, cache[0].cache, target_layers=target_layers)

    def __call__(self, inputs: mx.array, cache=None):
        if cache is None:
            cache = self.make_cache()
        if len(cache) != 1 or not isinstance(cache[0], DeepSeekV41PromptCache):
            raise ValueError("DeepSeek V4.1 requires its singleton model cache")
        return self.model(inputs, cache[0].cache)


@lru_cache(maxsize=2)
def _load_token_map(
    root: str,
    vocab_size: int,
    compressed_vocab_size: int,
) -> tuple[int, ...]:
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(Path(root) / "tokenizer")
    token_map, _ = v41_engram.build_compressed_token_map(tokenizer)
    if len(token_map) != vocab_size:
        raise ValueError(
            f"V4.1 tokenizer map has {len(token_map)} entries; expected {vocab_size}"
        )
    if not token_map or min(token_map) < 0:
        raise ValueError("V4.1 tokenizer map contains an invalid compressed token id")
    compressed = max(token_map) + 1
    if compressed != compressed_vocab_size:
        raise ValueError(
            f"V4.1 compressed tokenizer vocabulary is {compressed}; "
            f"expected {compressed_vocab_size}"
        )
    return tuple(token_map)


def _prepare_common_weights(weights: dict[str, mx.array]):
    quantized_modules = set()
    prepared = dict(weights)
    for name, scale in list(weights.items()):
        if not name.endswith(".scale"):
            continue
        module = name[: -len(".scale")]
        weight_name = f"{module}.weight"
        if weight_name not in weights:
            continue
        weight = weights[weight_name]
        if weight.ndim != 2 or scale.ndim != 2:
            raise ValueError(f"invalid native FP8 tensor pair for {module}")
        if weight.shape[1] % 32:
            raise ValueError(f"native FP8 input dimension is not divisible by 32 for {module}")
        expected_scale_shape = (
            (weight.shape[0] + 31) // 32,
            weight.shape[1] // 32,
        )
        if scale.shape != expected_scale_shape:
            raise ValueError(
                f"native FP8 scale shape is {scale.shape} for {module}; "
                f"expected {expected_scale_shape}"
            )
        expanded = mx.repeat(scale, 32, axis=0)[: weight.shape[0]]
        prepared[weight_name] = weight.view(mx.uint32)
        prepared.pop(name)
        prepared[f"{module}.scales"] = expanded
        quantized_modules.add(module)
    return prepared, quantized_modules


def _validate_common_weight_shapes(model: nn.Module, weights: dict[str, mx.array]) -> None:
    parameters = dict(tree_flatten(model.parameters()))
    expected = set(parameters)
    actual = set(weights)
    missing = {
        name
        for name in expected.difference(actual)
        if not name.endswith(".experts.weight")
        and not name.endswith(".experts.scales")
    }
    unexpected = actual.difference(expected)
    mismatched = sorted(
        name
        for name in expected.intersection(actual)
        if tuple(parameters[name].shape) != tuple(weights[name].shape)
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            "DeepSeek V4.1 common checkpoint does not match the MLX architecture: "
            f"missing={sorted(missing)[:4]}, unexpected={sorted(unexpected)[:4]}, "
            f"shape_mismatches={mismatched[:4]}"
        )


def load(
    installed_model,
    config,
    raw_config: dict,
    common_weights: dict[str, mx.array],
    read_limiter,
):
    if config.mtp_enabled:
        raise ValueError("DeepSeek V4.1 does not yet support MTP")
    if config.staged_expert_streaming:
        raise ValueError("DeepSeek V4.1 does not support staged expert streaming")
    if installed_model.engram is None:
        raise ValueError("installed DeepSeek V4.1 model has no engram descriptor")

    args = ModelArgs.from_dict(raw_config)
    original_switch_glu = v41_moe.SwitchGLU
    original_engram_embedding = v41_engram.EngramEmbedding
    try:
        v41_moe.SwitchGLU = _EmptySwitchGLU
        v41_engram.EngramEmbedding = _EmptyEngramEmbedding
        core = Model(
            args,
            token_map=_load_token_map(
                str(installed_model.root),
                args.vocab_size,
                args.engram_compressed_vocab_size,
            ),
        )
    finally:
        v41_moe.SwitchGLU = original_switch_glu
        v41_engram.EngramEmbedding = original_engram_embedding
    cache = ExpertCache(
        installed_model,
        config.slots,
        config.read_workers,
        config.prefetch_read_workers,
        route_trace_path=config.expert_route_trace,
        ready_expert_decode=config.ready_expert_decode,
        read_limiter=read_limiter,
        page_cache_probe=config.expert_page_cache_probe,
        file_cache_policy=config.expert_file_cache_policy,
        separate_prefill_io=getattr(config, "separate_prefill_io", True),
        eviction_policy=config.expert_eviction_policy,
        staged_expert_streaming=False,
    )
    wrapped = None
    try:
        tables = {table.layer: table for table in installed_model.engram.tables}
        for layer_index, layer in enumerate(core.layers):
            if layer.attn.is_index_source:
                layer.attn.indexer.candidate_only = config.v41_candidate_index
            activation = layer.ffn.experts.activation
            layer.ffn.experts = _StreamingSwitchGLU(layer_index, cache, activation)
            if layer.engram is not None:
                table = tables.get(layer_index)
                if table is None:
                    raise ValueError(f"missing V4.1 engram table for layer {layer_index}")
                layer.engram.embed = SSDEngramEmbedding(
                    installed_model.root / table.weight_file,
                    installed_model.root / table.scale_file,
                    table.rows,
                    table.dimension,
                    table.block_size,
                )

        weights, quantized_modules = _prepare_common_weights(common_weights)
        nn.quantize(
            core,
            group_size=32,
            bits=8,
            mode="mxfp8",
            class_predicate=lambda path, _: path in quantized_modules,
        )
        _validate_common_weight_shapes(core, weights)
        core.load_weights(list(weights.items()), strict=False)
        core.eval()
        mx.eval(core.parameters())
        from .ane_prefill import install_deepseek_ane_prefill
        wrapped = DeepSeekV41ForCausalLM(core)
        wrapped.packed_kv = config.v41_packed_kv
        wrapped.packed_index = config.v41_packed_index
        wrapped.ane_prefill = install_deepseek_ane_prefill(wrapped, config.deepseek_ane_prefill, config.ane_prefill_ratio)
        if config.dspark_enabled:
            from .deepseek_v41.dspark import load_dspark
            wrapped.dspark = load_dspark(installed_model, args, config, read_limiter)
        return wrapped, cache
    except Exception:
        if wrapped is not None:
            wrapped.ane_prefill.close()
        cache.close()
        raise
