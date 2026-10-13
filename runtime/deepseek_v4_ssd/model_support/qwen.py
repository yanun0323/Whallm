from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import sys
import time
from typing import Any
import mlx.core as mx
from mlx_lm.models.base import create_ssm_mask
from ..model import _select_moe_step_size, eval_prompt_cache
from ..cancellation import GenerationCancelled, check_cancelled
from ..expert_layout import GateUpMXFP4Layout
from .base import ModelSupport


class QwenSupport(ModelSupport):
    expert_layout = GateUpMXFP4Layout()

    def manifest_contract(self, raw):
        return _qwen_contract(raw, self.descriptor)

    def load(self, installed, config, raw_config, weights, read_limiter):
        from ..qwen4_exp import load
        from ..model import _load_qwen_mtp
        model, cache = load(installed, config, raw_config, weights, read_limiter)
        # The model holds the loaded arrays now; without this reference each
        # BF16 projection is freed as soon as its 8-bit form replaces it.
        del weights
        model.mtp = None
        model.mtp_expert_cache = None
        try:
            if config.mtp_enabled:
                model.mtp, model.mtp_expert_cache = _load_qwen_mtp(
                    installed, model.args, config, read_limiter,
                )
            if getattr(config, "qwen_8bit_common_tensors", False):
                from .. import qwen_common_tensors
                qwen_common_tensors.quantize_common_tensors(model)
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
            routed_max_tokens=getattr(config, "qwen_prefill_routed_max_tokens", 512),
            moe_step_size=getattr(config, "moe_prefill_step_size", 0),
        )

    def validate_media_runtime(self, runtime):
        from ..qwen_vision.artifact import validate_runtime
        validate_runtime(runtime)

    def validate_media(self, request):
        from ..qwen_vision.processor import validate_media
        return validate_media(request)

    def prepare_input(self, runtime, request):
        from ..qwen_vision.inputs import prepare
        return prepare(runtime, request)

    def model_for_prompt(self, model, prepared):
        from ..qwen_vision.inputs import generation_model
        return generation_model(model, prepared)

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

def _qwen_contract(raw: dict, descriptor) -> dict:
    from ..manifest import (QWEN_NGRAM_HEAD_OFFSETS, QWEN_NGRAM_HEAD_VOCAB_SIZES, QWEN_EXPERT_REGIONS, ExpertQuantization, NGram)
    # Swift's published artifact uses the original Qwen schema label, but only
    # its own pinned checkpoint identity is accepted by this support instance.
    accepted_kinds = {descriptor.kind}
    if descriptor.kind == "swift1.5-qwen3.8-flash-next":
        accepted_kinds.add("qwen3.8-flash-next")
        if raw.get("mtp") is None:
            raise ValueError("the Swift Qwen artifact must include its own MTP weights")
    quantization = raw.get("expertQuantization") or {}
    ngram = raw.get("ngram") or {}
    expected = (
        raw.get("modelKind") in accepted_kinds
        and raw.get("modelID") == descriptor.checkpoint_model_id
        and raw.get("revision") == descriptor.checkpoint_revision
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
        "model_kind": descriptor.kind,
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


_PREFILL_SYNC_MARKER = Path(__file__).with_name("qwen-prefill-sync.enabled")


def _prefill_sync_enabled() -> bool:
    flag = os.environ.get("WHALLM_QWEN_PREFILL_SYNC")
    return _PREFILL_SYNC_MARKER.is_file() if flag is None else flag == "1"


_MASKED_PREFILL_OFF_MARKER = Path(__file__).with_name("qwen-qsa-masked-off.enabled")


def _masked_prefill_off_enabled() -> bool:
    flag = os.environ.get("WHALLM_QWEN_QSA_MASKED_PREFILL_OFF")
    return _MASKED_PREFILL_OFF_MARKER.is_file() if flag is None else flag == "1"


def masked_prefill_override(config):
    """Diagnostic only: run QSA Prefill on the gathered path despite App tuning."""
    # Without masked prefill the selected-block kernel serves every chunk past the
    # indexer budget, so it goes too.
    if not _masked_prefill_off_enabled() or not (getattr(config, "qwen_qsa_masked_prefill", False)
                                                 or getattr(config, "qwen_qsa_selected_kernel", False)):
        return config
    print("[DEBUG-qwen-qsa-masked-off] masked QSA prefill and the selected-block kernel "
          "disabled for this runtime", file=sys.stderr, flush=True)
    return replace(config, qwen_qsa_masked_prefill=False, qwen_qsa_selected_kernel=False)


class _PrefillSync:
    """Finish the current stage before constructing the next; diagnostic only."""

    def __init__(self, cache):
        self.cache = cache
        self.stage = "layer_input"
        self.last_completed = "none"
        self.started = time.perf_counter()

    def __call__(self, next_stage, *arrays):
        check_cancelled()
        eval_prompt_cache([self.cache], *arrays)
        mx.synchronize()
        check_cancelled()
        self.last_completed, self.stage = self.stage, next_stage
        self.started = time.perf_counter()

    def failure_note(self):
        return (
            f"[DEBUG-qwen-prefill-sync] stage={self.stage} "
            f"last_completed={self.last_completed} "
            f"stage_seconds={time.perf_counter() - self.started:.6f}; "
            "extra waits change scheduling; this is not a GPU-kernel attribution"
        )


def _qwen_layer_major_prefill(
    model: Any,
    token_ids: list[int],
    prompt_cache: Any,
    step_size: int,
    expert_cache: Any,
    next_layer_prefetch: bool = False,
    batched_expert_prefill: bool = True,
    routed_max_tokens: int = 0,
    moe_step_size: int = 0,
) -> mx.array | None:
    """Populate Qwen caches layer-major, using whole layers or bounded pair waves.

    Up to ``routed_max_tokens`` prompt tokens, each layer runs every chunk's
    attention half first and reads only the experts the prompt routes to.
    Chunks, their order and the arithmetic stay the same as whole-layer reads.

    Longer prompts with sorted whole-layer experts run the attention halves of
    consecutive chunks, up to ``moe_step_size`` tokens (0: 4096), then one
    routed-expert call for their MoE halves. Each row is computed as in a call
    per chunk.
    """
    if not token_ids:
        return None
    sync_enabled = _prefill_sync_enabled()
    if sync_enabled:
        print("[DEBUG-qwen-prefill-sync] enabled; extra GPU waits; "
              f"tokens={len(token_ids)} step_size={step_size}", file=sys.stderr, flush=True)
    core = model.model
    if len(prompt_cache) != len(core.layers):
        raise ValueError("prompt cache does not match the Qwen model layers")
    # The sync diagnostic instruments the whole-layer loop below; keep it there.
    routed = bool(batched_expert_prefill and not sync_enabled
                  and 0 < len(token_ids) <= routed_max_tokens)
    if routed:
        return _qwen_routed_layer_major_prefill(model, token_ids, prompt_cache, step_size, expert_cache)
    release_slots = getattr(expert_cache, "release_prefill_slots", None)
    if batched_expert_prefill and callable(release_slots):
        release_slots()
    from contextlib import nullcontext
    reuse = getattr(expert_cache, "reuse_layer_buffers", nullcontext)
    with reuse() if batched_expert_prefill else nullcontext():
        record_compute_submit = getattr(expert_cache, "record_compute_submit", None)
        inputs = mx.array(token_ids)[None]
        hidden = mx.tile(core.embed_tokens(inputs), (1, 1, core.args.hc_count))
        # Every layer's experts share one configuration. The sync diagnostic
        # times each stage of one chunk, so it keeps chunks apart.
        experts = getattr(getattr(core.layers[0], "mlp", None), "experts", None)
        joins = getattr(experts, "shares_expert_call", None)
        spans = _expert_spans(
            len(token_ids), step_size, _select_moe_step_size(moe_step_size, len(token_ids)),
            joins if batched_expert_prefill and not sync_enabled and callable(joins) else None)

        def chunk_mask(layer, chunk, layer_cache):
            if layer.layer_type != "linear_attention":
                return None
            return create_ssm_mask(chunk[..., : core.args.hidden_size], layer_cache)

        def prefetch_next(layer_index):
            if batched_expert_prefill and next_layer_prefetch and layer_index + 1 < len(core.layers):
                expert_cache.prefetch_layer(layer_index + 1)

        for layer_index, (layer, layer_cache) in enumerate(zip(core.layers, prompt_cache)):
            check_cancelled()
            outputs = []
            with expert_cache.batched_layer(layer_index) if batched_expert_prefill else nullcontext():
                for span in spans:
                    if len(span) > 1:
                        outputs.extend(_qwen_merged_span(
                            layer, layer_index, layer_cache, hidden, inputs, span, step_size,
                            chunk_mask, prefetch_next, record_compute_submit))
                        continue
                    start, end = span[0]
                    check_cancelled()
                    chunk = hidden[:, start:end]
                    chunk_ids = inputs[:, start:end]
                    mask = chunk_mask(layer, chunk, layer_cache)
                    diagnostic = _PrefillSync(layer_cache) if sync_enabled else None
                    try:
                        if diagnostic is None:
                            output = layer(chunk, chunk_ids, mask, layer_cache)
                        else:
                            diagnostic("attention_input", chunk, chunk_ids,
                                       *(() if mask is None else (mask,)))
                            output = layer(chunk, chunk_ids, mask, layer_cache,
                                           prefill_boundary=diagnostic)
                        if start == 0:
                            prefetch_next(layer_index)
                        if callable(record_compute_submit):
                            record_compute_submit(layer_index)
                        if diagnostic is None:
                            eval_prompt_cache([layer_cache], output)
                        else:
                            diagnostic("complete", output)
                    except GenerationCancelled:
                        raise
                    except Exception as error:
                        if diagnostic is not None:
                            error.add_note(diagnostic.failure_note())
                        error.add_note(
                            f"Qwen prefill: layer={layer_index} (0-based) "
                            f"token_range=[{start},{end}) step_size={step_size}"
                        )
                        raise
                    outputs.append(output)
            hidden = outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=1)
            mx.eval(hidden)
            if layer_index + 1 == len(core.layers):
                if sync_enabled:
                    print("[DEBUG-qwen-prefill-sync] completed; "
                          f"layers={len(core.layers)} tokens={len(token_ids)}",
                          file=sys.stderr, flush=True)
                return hidden


def _expert_spans(
    token_count: int,
    step_size: int,
    span_tokens: int,
    joins=None,
) -> list[list[tuple[int, int]]]:
    """Group consecutive Prefill chunks whose MoE halves share one expert call.

    A group holds whole chunks and at most ``span_tokens`` tokens. A chunk for
    which ``joins(tokens)`` is false, or every chunk without ``joins``, is alone.
    """
    spans = []
    for start in range(0, token_count, step_size):
        chunk = (start, min(start + step_size, token_count))
        if (spans and joins is not None and joins(chunk[1] - chunk[0])
                and joins(spans[-1][0][1] - spans[-1][0][0])
                and chunk[1] - spans[-1][0][0] <= span_tokens):
            spans[-1].append(chunk)
        else:
            spans.append([chunk])
    return spans


def _qwen_merged_span(
    layer: Any,
    layer_index: int,
    layer_cache: Any,
    hidden: mx.array,
    inputs: mx.array,
    span: list[tuple[int, int]],
    step_size: int,
    chunk_mask,
    prefetch_next,
    record_compute_submit,
) -> list[mx.array]:
    """One layer's output for the chunks of ``span``.

    Attention never depends on the same layer's MoE output, so each chunk's
    attention half runs first, in order, as in the per-chunk loop. Their MoE
    halves then share one routed-expert call.
    """
    attended = []
    for start, end in span:
        check_cancelled()
        chunk = hidden[:, start:end]
        try:
            value = layer.attend(chunk, inputs[:, start:end], chunk_mask(layer, chunk, layer_cache),
                                 layer_cache)
            if start == 0:
                prefetch_next(layer_index)
            eval_prompt_cache([layer_cache], value)
        except GenerationCancelled:
            raise
        except Exception as error:
            error.add_note(f"Qwen prefill: layer={layer_index} (0-based) "
                           f"token_range=[{start},{end}) step_size={step_size}")
            raise
        attended.append(value)
    check_cancelled()
    try:
        outputs = layer.feed_forward_merged(attended)
        if callable(record_compute_submit):
            record_compute_submit(layer_index)
        mx.eval(*outputs)
    except GenerationCancelled:
        raise
    except Exception as error:
        error.add_note(f"Qwen prefill experts: layer={layer_index} (0-based) "
                       f"token_range=[{span[0][0]},{span[-1][1]}) step_size={step_size}")
        raise
    return outputs


def _qwen_routed_layer_major_prefill(
    model: Any,
    token_ids: list[int],
    prompt_cache: Any,
    step_size: int,
    expert_cache: Any,
) -> mx.array:
    """Whole-layer Prefill that reads only the experts this prompt routes to.

    Attention never depends on the same layer's MoE output, so every chunk's
    attention half runs first. The router then runs on those outputs, the union
    of selected experts is read into the layer buffer, and the MoE halves run on
    the same chunks. Unread experts are never indexed by gather_qmm. Next-layer
    prefetch is not possible here: that layer's routes are not known yet.
    """
    import numpy as np
    from contextlib import nullcontext
    core = model.model
    release_slots = getattr(expert_cache, "release_prefill_slots", None)
    if callable(release_slots):
        release_slots()
    reuse = getattr(expert_cache, "reuse_layer_buffers", nullcontext)
    with reuse():
        record_compute_submit = getattr(expert_cache, "record_compute_submit", None)
        inputs = mx.array(token_ids)[None]
        hidden = mx.tile(core.embed_tokens(inputs), (1, 1, core.args.hc_count))
        spans = [(start, min(start + step_size, len(token_ids)))
                 for start in range(0, len(token_ids), step_size)]
        for layer_index, (layer, layer_cache) in enumerate(zip(core.layers, prompt_cache)):
            check_cancelled()
            attended = []
            for start, end in spans:
                check_cancelled()
                chunk = hidden[:, start:end]
                mask = (
                    create_ssm_mask(chunk[..., : core.args.hidden_size], layer_cache)
                    if layer.layer_type == "linear_attention"
                    else None
                )
                try:
                    output = layer.attend(chunk, inputs[:, start:end], mask, layer_cache)
                    eval_prompt_cache([layer_cache], output)
                except GenerationCancelled:
                    raise
                except Exception as error:
                    error.add_note(f"Qwen routed prefill attention: layer={layer_index} (0-based) "
                                   f"token_range=[{start},{end}) step_size={step_size}")
                    raise
                attended.append(output)
            # The MoE halves below reuse these mixer and router results.
            prepared = [layer.prepare_feed_forward(value) for value in attended]
            mx.eval(*(item[3][1] for item in prepared))
            experts = np.unique(np.concatenate([np.asarray(item[3][1]).reshape(-1) for item in prepared]))
            outputs = []
            with expert_cache.batched_layer(layer_index, experts=tuple(experts.tolist())):
                for (start, end), value, ready in zip(spans, attended, prepared):
                    check_cancelled()
                    try:
                        output = layer.feed_forward(value, prepared=ready)
                        if callable(record_compute_submit):
                            record_compute_submit(layer_index)
                        mx.eval(output)
                    except GenerationCancelled:
                        raise
                    except Exception as error:
                        error.add_note(f"Qwen routed prefill experts: layer={layer_index} (0-based) "
                                       f"token_range=[{start},{end}) experts={len(experts)} "
                                       f"step_size={step_size}")
                        raise
                    outputs.append(output)
            del attended, prepared
            hidden = outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=1)
            mx.eval(hidden)
        return hidden
