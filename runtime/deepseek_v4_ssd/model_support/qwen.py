from __future__ import annotations

from typing import Any
import mlx.core as mx
from mlx_lm.models.base import create_ssm_mask
from ..model import eval_prompt_cache
from ..cancellation import check_cancelled
from ..expert_layout import GateUpMXFP4Layout
from .base import ModelSupport


class QwenSupport(ModelSupport):
    expert_layout = GateUpMXFP4Layout()

    def manifest_contract(self, raw):
        return _qwen_contract(raw)

    def load(self, installed, config, raw_config, weights, read_limiter):
        from ..qwen4_exp import load
        from ..model import _load_qwen_mtp
        model, cache = load(installed, config, raw_config, weights, read_limiter)
        model.mtp = None
        model.mtp_expert_cache = None
        try:
            if config.mtp_enabled:
                model.mtp, model.mtp_expert_cache = _load_qwen_mtp(
                    installed, model.args, config, read_limiter,
                )
            return model, cache
        except Exception:
            self.close(model, cache)
            raise

    def prefill(self, model, tokens, cache, step_size, expert_cache, config):
        return _qwen_layer_major_prefill(
            model, tokens, cache, step_size, expert_cache,
            getattr(config, "qwen_next_layer_prefetch", False),
            (getattr(config, "batched_expert_prefill", True)
             and not getattr(config, "qwen_expert_wave_slots", 0)),
        )

    def open_codec(self, root, tokenizer):
        from ..tool_codec import QwenToolCodec
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", trust_remote_code=True)
        return QwenToolCodec(tokenizer)

    def make_tool_stream_parser(self, thinking_mode):
        from ..tool_codec import QwenToolStreamParser
        return QwenToolStreamParser(thinking_mode)

    def reasoning_settings(self, thinking_mode, effort):
        mode = "thinking" if thinking_mode == "thinking" or effort not in {None, "none"} else "chat"
        native = {"minimal": "low", "low": "low", "medium": "medium",
                  "high": "xhigh", "xhigh": "xhigh", "max": "xhigh"}.get(effort, "low")
        return mode, native

    def sampling_defaults(self, defaults, thinking_mode):
        thinking = thinking_mode == "thinking"
        values = {
            "temperature": 1.0 if thinking else 0.7,
            "top_p": 0.95 if thinking else 0.8,
            "top_k": 20, "min_p": 0.0,
            "presence_penalty": 0.0 if thinking else 1.5,
            "repetition_penalty": 1.0,
        }
        if not getattr(defaults, "qwen_adaptive_sampling", True):
            values.update(temperature=defaults.temperature, top_p=defaults.top_p, top_k=defaults.top_k)
        return values


    def prefer_tool_first(self, messages, tool_choice, response_tools):
        return (tool_choice.mode == "auto"
                and not any(message.get("role") == "tool" for message in messages)
                and any(tool.get("name") == "exec_command" for tool in response_tools.values()))

    def cli_sampling_defaults(self):
        return self.sampling_defaults(None, "thinking")

def _qwen_contract(raw: dict) -> dict:
    from ..manifest import (QWEN_MODEL_ID, QWEN_REVISION, QWEN_NGRAM_HEAD_OFFSETS, QWEN_NGRAM_HEAD_VOCAB_SIZES, QWEN_EXPERT_REGIONS, ExpertQuantization, NGram)
    quantization = raw.get("expertQuantization") or {}
    ngram = raw.get("ngram") or {}
    expected = (
        raw.get("modelKind") == "qwen3.8-flash-next"
        and raw.get("modelID") == QWEN_MODEL_ID
        and raw.get("revision") == QWEN_REVISION
        and raw.get("layerCount") == 48
        and raw.get("expertCount") == 512
        and raw.get("selectedExpertCount") == 10
        and raw.get("expertBlobSize") == 2_611_200
        and raw.get("maximumContext") == 262_144
        and quantization
        == {"bits": 4, "conversionVersion": 2, "groupSize": 32, "mode": "mxfp4"}
        and ngram.get("file") == "ngram.bin"
        and ngram.get("dtype") == "F8_E4M3"
        and ngram.get("rowBytes") == 160
        and ngram.get("shardCount") == 128
        and ngram.get("shardRowCount") == 2_500_012
        and tuple(ngram.get("headOffsets", [])) == QWEN_NGRAM_HEAD_OFFSETS
        and tuple(ngram.get("headVocabSizes", [])) == QWEN_NGRAM_HEAD_VOCAB_SIZES
        and raw.get("dspark") is None
    )
    if not expected:
        raise ValueError("installed model does not match the pinned Qwen model contract")
    required = {
        "common.bin",
        "ngram.bin",
        "config.json",
        "generation_config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/chat_template.jinja",
        "tokenizer/vocab.json",
        "tokenizer/merges.txt",
        *(f"experts/layer_{layer:02d}.bin" for layer in range(48)),
    }
    file_sizes = {item.get("path"): item.get("size") for item in raw.get("files", [])}
    if file_sizes.get("ngram.bin") != 128 * 2_500_012 * 160:
        raise ValueError("installed Qwen N-gram file has an invalid size")
    mtp = raw.get("mtp")
    if mtp is not None:
        common_tensors = mtp.get("commonTensors")
        if not (
            mtp.get("layerCount") == 1
            and mtp.get("useDedicatedEmbeddings") is False
            and isinstance(common_tensors, list)
            and len(common_tensors) == 29
        ):
            raise ValueError("installed Qwen model has an invalid MTP contract")
        mtp_files = {"mtp/common.bin", "mtp/experts/layer_00.bin"}
        required.update(mtp_files)
        if (
            file_sizes.get("mtp/common.bin") != 181_136_896
            or file_sizes.get("mtp/experts/layer_00.bin") != 512 * 2_611_200
        ):
            raise ValueError("installed Qwen MTP file has an invalid size")
    return {
        "required": required,
        "allowed": required,
        "model_kind": "qwen3.8-flash-next",
        "layer_count": 48,
        "expert_count": 512,
        "selected_expert_count": 10,
        "expert_blob_size": 2_611_200,
        "maximum_context": 262_144,
        "expert_regions": QWEN_EXPERT_REGIONS,
        "expert_quantization": ExpertQuantization("mxfp4", 4, 32, 2),
        "ngram": NGram(
            "ngram.bin",
            "F8_E4M3",
            160,
            128,
            2_500_012,
            tuple(ngram["headOffsets"]),
            tuple(ngram["headVocabSizes"]),
        ),
    }


def _qwen_layer_major_prefill(
    model: Any,
    token_ids: list[int],
    prompt_cache: Any,
    step_size: int,
    expert_cache: Any,
    next_layer_prefetch: bool = False,
    batched_expert_prefill: bool = True,
) -> mx.array | None:
    """Populate Qwen caches layer-major, using whole layers or bounded pair waves."""
    if not token_ids:
        return None
    core = model.model
    if len(prompt_cache) != len(core.layers):
        raise ValueError("prompt cache does not match the Qwen model layers")
    release_slots = getattr(expert_cache, "release_prefill_slots", None)
    if batched_expert_prefill and callable(release_slots):
        release_slots()
    from contextlib import nullcontext
    reuse = getattr(expert_cache, "reuse_layer_buffers", nullcontext)
    with reuse() if batched_expert_prefill else nullcontext():
        record_compute_submit = getattr(expert_cache, "record_compute_submit", None)
        inputs = mx.array(token_ids)[None]
        hidden = mx.tile(core.embed_tokens(inputs), (1, 1, core.args.hc_count))
        for layer_index, (layer, layer_cache) in enumerate(zip(core.layers, prompt_cache)):
            check_cancelled()
            outputs = []
            with expert_cache.batched_layer(layer_index) if batched_expert_prefill else nullcontext():
                for start in range(0, len(token_ids), step_size):
                    check_cancelled()
                    end = min(start + step_size, len(token_ids))
                    chunk = hidden[:, start:end]
                    chunk_ids = inputs[:, start:end]
                    mask = (
                        create_ssm_mask(
                            chunk[..., : core.args.hidden_size],
                            layer_cache,
                        )
                        if layer.layer_type == "linear_attention"
                        else None
                    )
                    output = layer(chunk, chunk_ids, mask, layer_cache)
                    if (
                        batched_expert_prefill and next_layer_prefetch
                        and start == 0
                        and layer_index + 1 < len(core.layers)
                    ):
                        expert_cache.prefetch_layer(layer_index + 1)
                    if callable(record_compute_submit):
                        record_compute_submit(layer_index)
                    eval_prompt_cache([layer_cache], output)
                    outputs.append(output)
            hidden = outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=1)
            mx.eval(hidden)
            if layer_index + 1 == len(core.layers):
                return hidden
