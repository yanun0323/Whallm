from __future__ import annotations

import math

from ..expert_layout import FusedMXFP4Layout
from ..mimo.install import EXPERT_BYTES, EXPERT_REGIONS, KIND, MODEL_ID, REVISION
from .base import ModelSupport

PRESERVED_FILES = (
    ".gitattributes", "MiMo_V2_6_technical_report.pdf", "README.md", "assets/architecture.png",
    "audio_tokenizer/chat_template.jinja", "audio_tokenizer/config.json",
    "audio_tokenizer/generation_config.json", "audio_tokenizer/model.safetensors",
    "audio_tokenizer/tokenizer_config.json", "chat_template.jinja", "config.json",
    "configuration_mimo_v2.py", "dflash/config.json", "dflash/dflash.py",
    "dflash/dflash_draft_model.safetensors", "dflash/mask_embedding.pt",
    "dflash/model.safetensors.index.json", "generation_config.json", "merges.txt",
    "model.safetensors.index.json", "model_mtp.safetensors", "modeling_mimo_v2.py",
    "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
)


def required_paths():
    return {
        "common.bin", "config.json", "generation_config.json", "checkpoint-map.json", "preservation.json",
        "checkpoint/common.bin", "vision/common.bin", "audio/common.bin", "mtp/common.bin",
        *(f"tokenizer/{n}" for n in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt")),
        *(f"experts/layer_{i:02d}.bin" for i in range(47)),
        *(f"checkpoint/headers/model_pp0_ep{i}_shard0.safetensors.header" for i in range(64)),
        *(f"checkpoint/{p}" for p in PRESERVED_FILES),
    }


def contract(raw):
    expected = {"modelKind": KIND, "modelID": MODEL_ID, "revision": REVISION,
                "formatVersion": 4, "layerCount": 47, "backboneLayerCount": 48,
                "expertCount": 256, "selectedExpertCount": 8, "expertBlobSize": EXPERT_BYTES,
                "maximumContext": 1048576,
                "expertQuantization": {"mode": "mxfp4", "bits": 4, "groupSize": 32, "conversionVersion": 1},
                "conversion": {"version": 1, "sourceTP": 4, "qkv": "canonical-bf16", "losslessSource": True}}
    if any(raw.get(k) != v for k, v in expected.items()) or any(raw.get(k) is not None for k in ("mtp", "dspark", "ngram", "engram")):
        raise ValueError("installed model does not match the pinned complete MiMo contract")
    components = raw.get("components", {})
    if set(components) != {"backbone", "mtp", "vision", "audio_patch", "audio_tokenizer", "dflash"}:
        raise ValueError("complete MiMo component inventory is missing")
    expected_components = {"backbone": "common.bin", "mtp": "mtp/common.bin", "vision": "vision/common.bin",
                           "audio_patch": "audio/common.bin", "audio_tokenizer": "checkpoint/audio_tokenizer/model.safetensors",
                           "dflash": "checkpoint/dflash/dflash_draft_model.safetensors"}
    for name, path in expected_components.items():
        if components[name].get("file") != path:
            raise ValueError(f"invalid MiMo component file: {name}")
    if components["dflash"].get("mask") != "checkpoint/dflash/mask_embedding.pt":
        raise ValueError("complete MiMo artifact must preserve the DFlash mask")
    if components["backbone"].get("tensors") != raw.get("commonTensors"):
        raise ValueError("MiMo common tensor tables disagree")
    files = {f["path"]: f["size"] for f in raw.get("files", [])}
    from ..manifest import Tensor, _validate_tensors
    for name, count in (("backbone", 331), ("mtp", 36), ("vision", 364), ("audio_patch", 95)):
        table = components[name].get("tensors", [])
        if len(table) != count:
            raise ValueError(f"incomplete MiMo component tensor table: {name}")
        end = 0
        for t in sorted(table, key=lambda t: t["offset"]):
            if (t["dtype"] not in ("BF16", "F32") or not t["shape"]
                    or any(type(n) is not int or n <= 0 for n in t["shape"])
                    or type(t["offset"]) is not int or t["offset"] < end or t["offset"] % 256
                    or t["length"] != math.prod(t["shape"]) * (2 if t["dtype"] == "BF16" else 4)):
                raise ValueError(f"invalid MiMo component tensor layout: {name}")
            end = t["offset"] + t["length"]
        if end != files.get(components[name]["file"], 0):
            raise ValueError(f"MiMo component tensor coverage mismatch: {name}")
        _validate_tensors(tuple(Tensor(t["name"], t["dtype"], tuple(t["shape"]), t["offset"], t["length"])
                                for t in table), end, f"MiMo {name}")
    return {"required": required_paths(), "allowed": required_paths(), "model_kind": KIND,
            "layer_count": 47, "expert_count": 256, "selected_expert_count": 8,
            "expert_blob_size": EXPERT_BYTES, "maximum_context": 1048576,
            "expert_regions": EXPERT_REGIONS}


class MiMoSupport(ModelSupport):
    expert_layout = FusedMXFP4Layout()

    def manifest_contract(self, raw):
        return contract(raw)

    def load(self, installed, config, raw_config, weights, read_limiter):
        import mlx.core as mx
        from ..expert_cache import ExpertCache
        from ..mimo.model import MiMoArgs, Model
        cache = ExpertCache(installed, config.slots, config.read_workers, config.prefetch_read_workers,
                            ready_expert_decode=config.ready_expert_decode, read_limiter=read_limiter,
                            eviction_policy=config.expert_eviction_policy,
                            route_trace_path=config.expert_route_trace,
                            file_cache_policy=config.expert_file_cache_policy,
                            separate_prefill_io=config.separate_prefill_io)
        try:
            model = Model(MiMoArgs.from_config(raw_config), cache)
            model.load_weights(list(weights.items()), strict=True)
            mx.eval(model.parameters())
            return model, cache
        except BaseException:
            cache.close()
            raise

    def validate_media(self, request):
        from ..mimo.inputs import validate_media
        return validate_media(request)

    def prepare_input(self, runtime, request):
        from ..mimo.inputs import prepare
        return prepare(runtime, request)

    def prefill(self, model, tokens, cache, step_size, expert_cache, config):
        """Reuse each layer's experts without changing attention/MoE chunk shapes.

        Bound retained hidden activations to 4096 tokens (or one explicit larger
        chunk). Keep block boundaries aligned with the original chunk schedule.
        No whole-layer weights, gather-QMM, or speculative reads are needed.
        """
        from contextlib import nullcontext
        import mlx.core as mx
        from mlx_lm.models.base import create_attention_mask
        from ..cancellation import check_cancelled
        from ..model import _clear_memory_cache, eval_prompt_cache

        if step_size < 1:
            raise ValueError("MiMo prefill step size must be positive")
        if len(cache) != len(model.layers):
            raise ValueError("MiMo cache layer count mismatch")
        block_size = max(1, 4096 // step_size) * step_size
        for block in range(0, len(tokens), block_size):
            check_cancelled()
            hidden = model.model.embed_tokens(mx.array([tokens[block:block + block_size]]))
            mx.eval(hidden)
            for index, (layer, state) in enumerate(zip(model.layers, cache)):
                outputs = []
                # Backbone layer zero is dense; expert layer IDs are shifted.
                with expert_cache.pin_layer(index - 1) if index else nullcontext():
                    for start in range(0, hidden.shape[1], step_size):
                        check_cancelled()
                        chunk = hidden[:, start:start + step_size]
                        mask = create_attention_mask(chunk, state,
                            window_size=model.model.window if layer.sliding else None)
                        cache_only = index == len(model.layers) - 1
                        output = layer(chunk, mask, state, cache_only=cache_only)
                        # Drain every consumer before a later chunk/layer can
                        # evict its slots, including a partially filled cache.
                        if cache_only:
                            eval_prompt_cache([state])
                        else:
                            eval_prompt_cache([state], output)
                            outputs.append(output)
                if outputs:
                    hidden = mx.concatenate(outputs, axis=1) if len(outputs) > 1 else outputs[0]
                    mx.eval(hidden)
                _clear_memory_cache()

    def open_codec(self, root, tokenizer):
        from ..mimo.codec import MiMoCodec
        return MiMoCodec(tokenizer)

    def make_tool_stream_parser(self, thinking_mode):
        from ..mimo.codec import MiMoToolStreamParser
        return MiMoToolStreamParser(thinking_mode)
