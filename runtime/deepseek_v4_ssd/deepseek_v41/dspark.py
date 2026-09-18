"""V4.1's three trained DSpark stages; candidates use the shared exact verifier."""
from dataclasses import replace
import time

import mlx.core as mx
import mlx.nn as nn

from .attention import Attention
from .fakequant import fake_quant_fp8_ue8m0
from .hyper_connections import hc_pre, make_identity_pre_mix
from .layers import RMSNorm, rope_tail
from .model import Block
from ..cancellation import check_cancelled
from ..dspark import DraftResult, _ConfidenceHead, _confidence_prefix_length, sampling_logprobs


class DraftAttention(Attention):
    def __init__(self, layer, args):
        super().__init__(layer, args)
        self.context = None
        self.offset = 0

    def seed(self, main_x, start):
        if start != self.offset:
            raise ValueError("DSpark context position does not match committed tokens")
        end = start + main_x.shape[1]
        cos, sin = self._freqs(end)
        kv = rope_tail(self.kv_norm(self.wkv(main_x)), self.rope_head_dim,
                       cos[start:end], sin[start:end])
        kv = fake_quant_fp8_ue8m0(kv, 32)
        previous = self.context
        self.context = (mx.concatenate([previous, kv], axis=1) if previous is not None else kv)[:, -self.window_size:]
        self.offset = end

    def __call__(self, x, start, main_x, shared):
        self.seed(main_x, start)
        b, n = x.shape[:2]
        end = self.offset + n
        cos, sin = self._freqs(end)
        c, s = cos[self.offset:end], sin[self.offset:end]
        q = self.wq_b(self.q_norm(self.wq_a(x))).reshape(b, n, self.n_heads, self.head_dim)
        q = rope_tail(q, self.rope_head_dim, c, s)
        kv = fake_quant_fp8_ue8m0(rope_tail(self.kv_norm(self.wkv(x)), self.rope_head_dim, c, s), 32)
        kv = mx.concatenate([self.context.astype(kv.dtype), kv], axis=1)
        indices = mx.broadcast_to(mx.arange(kv.shape[1], dtype=mx.int32), (b, n, kv.shape[1]))
        from .sparse_attention import sparse_attn
        out = sparse_attn(q, kv, self.attn_sink, indices, self.softmax_scale)
        return self.project_output(rope_tail(out, self.rope_head_dim, c, s, inverse=True), x.dtype)


class MarkovHead(nn.Module):
    def __init__(self, vocab, rank):
        super().__init__()
        self.embed = nn.Embedding(vocab, rank)
        self.head = nn.Linear(rank, vocab, bias=False)


class DSpark(nn.Module):
    def __init__(self, args, expert_cache=None, *, block_size=5, noise_token_id=128799,
                 target_layers=(37, 38, 39), markov_rank=256, expert_count=128, topk=3):
        super().__init__()
        self.block_size, self.noise_token_id = block_size, noise_token_id
        self.target_layers = target_layers
        self.expert_cache = expert_cache
        draft_args = replace(args, n_routed_experts=expert_count, n_activated_experts=topk,
                             engram_layer_ids=(), kv_source_layers=(), index_source_layers=(),
                             compress_ratios=(), candidate_source_layer=-1)
        from . import moe
        from ..model import _EmptySwitchGLU, _StreamingSwitchGLU
        original = moe.SwitchGLU
        try:
            if expert_cache is not None:
                moe.SwitchGLU = _EmptySwitchGLU
            self.mtp = [Block(i, draft_args) for i in range(3)]
        finally:
            moe.SwitchGLU = original
        for i, layer in enumerate(self.mtp):
            layer.attn = DraftAttention(i, draft_args)
            if expert_cache is not None:
                layer.ffn.experts = _StreamingSwitchGLU(i, expert_cache, layer.ffn.experts.activation)
        self.mtp[0].main_proj = nn.Linear(args.dim * len(target_layers), args.dim, bias=False)
        self.mtp[0].main_norm = RMSNorm(args.dim, args.norm_eps)
        self.mtp[-1].norm = RMSNorm(args.dim, args.norm_eps)
        self.mtp[-1].markov_head = MarkovHead(args.vocab_size, markov_rank)
        self.mtp[-1].confidence_head = _ConfidenceHead(args.dim + markov_rank)

    def reset_cache(self):
        for stage in self.mtp:
            stage.attn.context, stage.attn.offset = None, 0

    def cache_state(self):
        return tuple((mx.array(stage.attn.context) if stage.attn.context is not None else None,
                      stage.attn.offset) for stage in self.mtp)

    def restore_cache_state(self, state):
        if state is None:
            self.reset_cache()
            return
        if len(state) != len(self.mtp):
            raise ValueError("Invalid V4.1 DSpark state")
        for stage, saved in zip(self.mtp, state):
            # A missing stage state starts that stage empty, as V4 DSpark does.
            context, offset = saved if saved is not None else (None, 0)
            stage.attn.context = mx.array(context) if context is not None else None
            stage.attn.offset = offset

    def _main_x(self, hidden):
        first = self.mtp[0]
        return first.main_norm(first.main_proj(hidden))

    def prefill_context(self, main_hidden, offset):
        # Bound projection temporaries and materialize each context before the
        # next chunk. The stored context itself remains limited to window_size.
        for begin in range(0, main_hidden.shape[1], 4096):
            check_cancelled()
            main_x = self._main_x(main_hidden[:, begin:begin + 4096])
            for stage in self.mtp:
                stage.attn.seed(main_x, offset + begin)
            mx.eval([stage.attn.context for stage in self.mtp])

    def draft(self, main_model, anchor, main_hidden, start_pos, temperature, top_p, confidence_threshold):
        started = time.perf_counter()
        main_x = self._main_x(main_hidden)
        ids = mx.full((1, self.block_size), self.noise_token_id, mx.int32)
        ids[:, 0] = anchor
        h = main_model.model.embed(ids)
        h = mx.broadcast_to(h[:, :, None, :], (*h.shape[:2], self.mtp[0].hc_mult, h.shape[-1]))
        pre = make_identity_pre_mix(1, self.block_size, self.mtp[0].hc_mult)
        for stage in self.mtp:
            check_cancelled()
            h, pre = stage(h, pre, start_pos, main_x, None)
        head_hidden = hc_pre(h, pre)
        last = self.mtp[-1]
        logits = main_model.model.head(last.norm(head_hidden).astype(mx.float32))
        previous = mx.array([anchor], mx.int32)
        tokens, distributions, embeddings = [], [], []
        for i in range(self.block_size):
            embedding = last.markov_head.embed(previous)
            bias = last.markov_head.head(embedding.astype(mx.float32))
            logprobs = sampling_logprobs(logits[:, i] + bias, temperature, top_p)
            previous = (mx.argmax(logprobs, axis=-1) if temperature == 0 else mx.random.categorical(logprobs)).astype(mx.int32)
            tokens.append(previous)
            distributions.append(logprobs[0])
            embeddings.append(embedding)
        confidence = mx.sigmoid(last.confidence_head(head_hidden, mx.stack(embeddings, axis=1)))[0]
        mx.eval(confidence, tokens, distributions)
        values = confidence.tolist()
        keep = _confidence_prefix_length(values, confidence_threshold) if confidence_threshold > 0 else self.block_size
        return DraftResult([int(t.item()) for t in tokens[:keep]], distributions[:keep], values[:keep], time.perf_counter()-started)


def load_dspark(installed, args, config, read_limiter):
    from ..model import _load_tensor_file
    from ..expert_cache import ExpertCache
    from ..deepseek_v41_ssd import _prepare_common_weights, _validate_common_weight_shapes
    if not installed.has_dspark:
        raise ValueError("Install the V4.1 DSpark weights before enabling DSpark")
    sidecar = replace(installed, expert_count=128, selected_expert_count=3)
    cache = ExpertCache(sidecar, config.dspark_slots, config.read_workers, config.prefetch_read_workers,
                        layer_count=3, expert_directory=installed.root / 'dspark/experts',
                        read_limiter=read_limiter, file_cache_policy=config.expert_file_cache_policy,
        separate_prefill_io=getattr(config, "separate_prefill_io", True),
                        eviction_policy=config.expert_eviction_policy)
    try:
        model = DSpark(args, cache)
        weights, modules = _prepare_common_weights(_load_tensor_file(installed.root / 'dspark/common.bin', installed.dspark_common_tensors))
        nn.quantize(model, group_size=32, bits=8, mode='mxfp8', class_predicate=lambda path, _: path in modules)
        _validate_common_weight_shapes(model, weights)
        model.load_weights(list(weights.items()), strict=False)
        model.eval()
        mx.eval(model.parameters())
        return model
    except Exception:
        cache.close()
        raise
