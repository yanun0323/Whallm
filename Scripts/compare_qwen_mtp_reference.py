#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import numpy as np

from deepseek_v4_ssd.expert_cache import ExpertCache
from deepseek_v4_ssd.manifest import InstalledModel, Tensor
from deepseek_v4_ssd.model import RuntimeConfig, _load_tensor_file, _tensor_from_buffer, load_model
from deepseek_v4_ssd.qwen4_exp import MTPModel, mtp_prefill_pairs


def _packed_expert_view(path: Path, model: InstalledModel, region: Tensor) -> mx.array:
    dtype = {
        "U8": np.uint8,
        "U32": np.uint32,
    }[region.dtype]
    mapped = np.memmap(path, mode="r", dtype=np.uint8)
    item_size = np.dtype(dtype).itemsize
    strides = [model.expert_blob_size]
    for axis in range(len(region.shape)):
        strides.append(item_size * int(np.prod(region.shape[axis + 1 :])))
    view = np.ndarray(
        (model.expert_count, *region.shape),
        dtype=dtype,
        buffer=mapped,
        offset=region.offset,
        strides=tuple(strides),
    )
    return mx.array(view)


def prepare(model_path: Path, output: Path) -> None:
    installed = InstalledModel.open(model_path)
    if installed.mtp is None:
        raise ValueError("installed model has no MTP sidecar")

    config = RuntimeConfig(slots=128, read_workers=4, prefetch_read_workers=2)
    model, main_cache = load_model(installed, config)
    mtp_installed = replace(
        installed,
        root=installed.root / "mtp",
        layer_count=1,
        common_tensors=installed.mtp.common_tensors,
        ngram=None,
        mtp=None,
    )
    mtp_cache = ExpertCache(
        mtp_installed,
        slots=32,
        read_workers=4,
        prefetch_read_workers=2,
    )
    try:
        prompt_tokens = mx.array([[9419, 1814]], dtype=mx.int32)
        _, target_hidden = model.forward_with_hidden(prompt_tokens, model.make_cache())
        paired_hidden, paired_tokens = mtp_prefill_pairs(target_hidden, prompt_tokens)

        mtp = MTPModel(model.args, mtp_cache)
        mtp_weights = _load_tensor_file(
            installed.root / "mtp/common.bin",
            installed.mtp.common_tensors,
        )
        mtp.load_weights(list(mtp.sanitize(mtp_weights).items()), strict=True)
        mtp.eval()
        mx.eval(mtp.parameters(), paired_hidden)

        embedding = mx.take(model.model.embed_tokens.weight, paired_tokens, axis=0)
        local_logits, local_wide = mtp(
            paired_hidden,
            paired_tokens,
            model.model.embed_tokens.weight,
            model.lm_head,
            mtp.make_cache(),
        )
        local_sample = mtp.hyper_connection_mixer(local_wide)
        projected_embedding = mtp.fc_embedding(mtp.pre_fc_norm_embedding(embedding))
        projected_hidden = mtp.fc_hidden(
            mtp.pre_fc_norm_hidden(paired_hidden).reshape(1, 1, 4, 2560)
        )
        layer_input = (projected_hidden + projected_embedding[..., None, :]).reshape(
            1, 1, 10240
        )
        layer = mtp.layers[0]
        attention_input, attention_residual, attention_injection = (
            layer.attn_hyper_connection(layer_input)
        )
        attention_output = layer.self_attn(attention_input, mtp.make_cache())
        after_attention = attention_residual + (
            attention_output[..., None, :] * attention_injection[..., None]
        ).reshape(1, 1, 10240)
        moe_input, moe_residual, moe_injection = layer.mlp_hyper_connection(
            after_attention
        )
        moe_output = layer.mlp(moe_input)
        staged_wide = moe_residual + (
            moe_output[..., None, :] * moe_injection[..., None]
        ).reshape(1, 1, 10240)
        mx.eval(
            embedding,
            local_logits,
            local_wide,
            local_sample,
            projected_embedding,
            projected_hidden,
            layer_input,
            attention_input,
            attention_output,
            after_attention,
            moe_input,
            moe_output,
            staged_wide,
        )

        output.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(
            str(output / "local-output.safetensors"),
            {
                "local.logits": local_logits,
                "local.sample_hidden": local_sample,
                "local.wide_hidden": local_wide,
                "stage.01_projected_embedding": projected_embedding,
                "stage.02_projected_hidden": projected_hidden,
                "stage.03_layer_input": layer_input,
                "stage.04_attention_input": attention_input,
                "stage.05_attention_output": attention_output,
                "stage.06_after_attention": after_attention,
                "stage.07_moe_input": moe_input,
                "stage.08_moe_output": moe_output,
                "stage.09_wide_hidden": staged_wide,
            },
        )

        bundle = dict(mtp_weights)
        bundle["reference.embedding.weight"] = embedding.reshape(1, -1)
        bundle["reference.target_hidden"] = paired_hidden
        expert_path = installed.root / "mtp/experts/layer_00.bin"
        names = {
            "gate_up.weight": "mtp.layers.0.mlp.experts.gate_up_proj.weight",
            "gate_up.scale": "mtp.layers.0.mlp.experts.gate_up_proj.weight_scale",
            "down.weight": "mtp.layers.0.mlp.experts.down_proj.weight",
            "down.scale": "mtp.layers.0.mlp.experts.down_proj.weight_scale",
        }
        for region in installed.expert_regions:
            bundle[names[region.name]] = _packed_expert_view(
                expert_path,
                installed,
                region,
            )
        mx.save_safetensors(str(output / "reference-input.safetensors"), bundle)

        metadata = {
            "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "model_id": installed.model_id,
            "revision": installed.revision,
            "prompt": "Hello world",
            "prompt_token_ids": [9419, 1814],
            "paired_token_id": 1814,
            "target_hidden_shape": list(paired_hidden.shape),
        }
        (output / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
    finally:
        mtp_cache.close()
        main_cache.close()


def _metrics(expected: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    expected = expected.astype(np.float32).reshape(-1)
    actual = actual.astype(np.float32).reshape(-1)
    difference = np.abs(expected - actual)
    denominator = float(np.linalg.norm(expected) * np.linalg.norm(actual))
    return {
        "maximum_absolute_error": float(difference.max()),
        "mean_absolute_error": float(difference.mean()),
        "cosine_similarity": float(np.dot(expected, actual) / denominator),
    }


def compare(model_path: Path, local_path: Path, reference_path: Path, output: Path) -> None:
    installed = InstalledModel.open(model_path)
    local = mx.load(str(local_path))
    reference = mx.load(str(reference_path))

    mapped = np.memmap(installed.root / "common.bin", mode="r", dtype=np.uint8)
    lm_descriptor = next(
        item for item in installed.common_tensors if item.name == "lm_head.weight"
    )
    lm_head = _tensor_from_buffer(mapped, lm_descriptor)
    reference_logits = reference["reference.sample_hidden"] @ lm_head.T
    mx.eval(reference_logits)

    local_logits = np.asarray(local["local.logits"].astype(mx.float32))
    reference_logits_np = np.asarray(reference_logits.astype(mx.float32))
    local_top = np.argsort(local_logits.reshape(-1))[-10:][::-1]
    reference_top = np.argsort(reference_logits_np.reshape(-1))[-10:][::-1]
    stage_metrics = {
        key: _metrics(
            np.asarray(local[key].astype(mx.float32)),
            np.asarray(reference[key].astype(mx.float32)),
        )
        for key in sorted(local)
        if key.startswith("stage.") and key in reference
    }
    distribution_metrics = _metrics(local_logits, reference_logits_np)
    passed = bool(
        local_top.tolist() == reference_top.tolist()
        and distribution_metrics["cosine_similarity"] >= 0.9999
        and distribution_metrics["maximum_absolute_error"] <= 0.25
    )
    result = {
        "wide_hidden": _metrics(
            np.asarray(local["local.wide_hidden"].astype(mx.float32)),
            np.asarray(reference["reference.wide_hidden"].astype(mx.float32)),
        ),
        "sample_hidden": _metrics(
            np.asarray(local["local.sample_hidden"].astype(mx.float32)),
            np.asarray(reference["reference.sample_hidden"].astype(mx.float32)),
        ),
        "distribution": distribution_metrics,
        "stages": stage_metrics,
        "local": {
            "argmax_token_id": int(local_top[0]),
            "top10_token_ids": local_top.tolist(),
            "sha256": hashlib.sha256(local_logits.tobytes()).hexdigest(),
        },
        "reference": {
            "argmax_token_id": int(reference_top[0]),
            "top10_token_ids": reference_top.tolist(),
            "sha256": hashlib.sha256(reference_logits_np.tobytes()).hexdigest(),
        },
        "top10_overlap": len(set(local_top.tolist()) & set(reference_top.tolist())),
        "acceptance": {
            "ordered_top10_equal": local_top.tolist() == reference_top.tolist(),
            "minimum_distribution_cosine_similarity": 0.9999,
            "maximum_distribution_absolute_error": 0.25,
            "passed": passed,
        },
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--model", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--model", type=Path, required=True)
    compare_parser.add_argument("--local", type=Path, required=True)
    compare_parser.add_argument("--reference", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        prepare(arguments.model, arguments.output)
    else:
        compare(arguments.model, arguments.local, arguments.reference, arguments.output)


if __name__ == "__main__":
    main()
