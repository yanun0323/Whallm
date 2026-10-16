"""Reference data from the MLX DeepSeek V4.1 runtime for the native engine.

Writes, into OUT:
  token_map.i32        compressed Engram token id for every vocabulary id (int32)
  prompt.txt           the prompt token ids, comma-separated
  greedy.txt           the greedy continuation (MLX engine, whole prompt in one call)
  layer_XX.f32         with --dump-layers N: the hidden stream [tokens, 4, 5120] after
                       layers 0 .. N-1 of the prompt call, float32
  logits.f32           the logits of the last prompt token, float32

Usage: python -I Scripts/deepseek_v41_reference.py MODEL_DIR OUT [--prompt TEXT | --tokens 1,2,3]
           [--greedy N] [--dump-layers N] [--slots N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("out")
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file")
    parser.add_argument("--tokens")
    parser.add_argument("--greedy", type=int, default=16)
    parser.add_argument("--dump-layers", type=int, default=0)
    parser.add_argument("--slots", type=int, default=512)
    parser.add_argument("--token-map-only", action="store_true")
    parser.add_argument("--f32", action="store_true", help="run the activations in float32 instead of bfloat16")
    args = parser.parse_args()

    import numpy as np
    from deepseek_v4_ssd.deepseek_v41.config import ModelArgs
    from deepseek_v4_ssd.deepseek_v41_ssd import _load_token_map
    import json

    root = Path(args.model).expanduser().resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = json.loads((root / "config.json").read_text())
    margs = ModelArgs.from_dict(raw)
    token_map = np.asarray(
        _load_token_map(str(root), margs.vocab_size, margs.engram_compressed_vocab_size), dtype=np.int32)
    token_map.tofile(out / "token_map.i32")
    if args.token_map_only:
        return 0

    from transformers import PreTrainedTokenizerFast
    if args.tokens:
        prompt = [int(t) for t in args.tokens.split(",") if t]
    else:
        text = args.prompt if args.prompt is not None else Path(args.prompt_file).read_text()
        tokenizer = PreTrainedTokenizerFast.from_pretrained(root / "tokenizer")
        prompt = [0] + tokenizer.encode(text, add_special_tokens=False)
    (out / "prompt.txt").write_text(",".join(map(str, prompt)))

    import mlx.core as mx
    from deepseek_v4_ssd.manifest import InstalledModel
    from deepseek_v4_ssd.model import RuntimeConfig, load_model
    from deepseek_v4_ssd.deepseek_v41.hyper_connections import hc_pre, make_identity_pre_mix
    from deepseek_v4_ssd.deepseek_v41.model import SharedState

    installed = InstalledModel.open(root)
    # V4.1 runs on the native engine; the reference is its MLX runtime.
    from dataclasses import replace
    from unittest import mock
    from deepseek_v4_ssd.model_support.catalog import BY_KIND
    from deepseek_v4_ssd.model_support.deepseek_v41 import DeepSeekV41Support
    mlx = DeepSeekV41Support(replace(BY_KIND["deepseek-v4.1"], features=frozenset({"promptCache", "dspark"})))
    with mock.patch("deepseek_v4_ssd.model_support.support_for_installed", return_value=mlx):
        model, expert_cache = load_model(installed, RuntimeConfig(slots=args.slots, v41_layer_major_prefill=False))
    core = model.model
    if args.f32:
        import mlx.nn as nn

        class _F32Embedding(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def __call__(self, ids):
                return self.inner(ids).astype(mx.float32)

        core.embed = _F32Embedding(core.embed)
    cache = model.make_cache()
    state = cache[0].cache

    ids = mx.array(prompt)[None]
    if args.dump_layers:
        # Model.__call__ with the stream written out after each layer.
        state.ensure_capacity(len(prompt))
        hashes = mx.array(core.engram_hasher(np.array(ids, dtype=np.int64), 0, state.engram_ids))
        h = core.embed(ids)
        h = mx.broadcast_to(h[:, :, None, :], (1, len(prompt), core.hc_mult, h.shape[-1]))
        pre = make_identity_pre_mix(1, len(prompt), core.hc_mult)
        shared = SharedState()
        for layer in core.layers:
            if layer.engram is not None:
                h = layer.engram(h, hashes[:, :, layer.engram.layer_hash_index])
            h, pre = layer(h, pre, 0, state, shared)
            mx.eval(h, pre)
            if layer.layer_id < args.dump_layers:
                np.asarray(h[0].astype(mx.float32)).tofile(out / f"layer_{layer.layer_id:02d}.f32")
        h = core.norm(hc_pre(h, pre))[:, -1:]
        logits = core.head(h.astype(mx.float32))
        state.offset = len(prompt)
    else:
        logits = model(ids, cache=cache)[:, -1:]
    mx.eval(logits)
    np.asarray(logits[0, -1]).astype(np.float32).tofile(out / "logits.f32")
    tokens = []
    token = int(mx.argmax(logits[0, -1]).item())
    for _ in range(args.greedy):
        tokens.append(token)
        logits = model(mx.array([[token]]), cache=cache)
        token = int(mx.argmax(logits[0, -1]).item())
    (out / "greedy.txt").write_text(" ".join(map(str, tokens)))
    print("prompt", ",".join(map(str, prompt)))
    print("greedy", " ".join(map(str, tokens)))
    expert_cache.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
