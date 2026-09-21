"""Pure Qwen tensor candidate: no mutable cache, weight or RNG capture."""
import mlx.core as mx


def group_rms_norm(value, weight, eps):
    normalized = value.astype(mx.float32) * mx.rsqrt(
        mx.mean(mx.square(value.astype(mx.float32)), axis=-1, keepdims=True) + eps)
    return (normalized * weight).astype(value.dtype)


compiled_group_rms_norm = mx.compile(group_rms_norm)


def scaled_silu(value, divisor):
    value = value / divisor
    return value * mx.sigmoid(value)


def hyper_mix(normalized, mix_logits, stream_count):
    shape = (*normalized.shape[:-1], stream_count, normalized.shape[-1] // stream_count)
    return (mx.sigmoid(mix_logits).reshape(shape) * normalized.reshape(shape)).mean(axis=-2)


def injection_gate(logits, stream_count):
    return 2 * mx.sigmoid(logits / stream_count)


def hyper_inject(residual, value, injection):
    return residual + (value[..., None, :] * injection[..., None]).reshape(residual.shape)


# No module, weight, cache or RNG is captured: parameters are always live inputs.
compiled_scaled_silu = mx.compile(scaled_silu)
compiled_hyper_mix = mx.compile(hyper_mix)
compiled_injection_gate = mx.compile(injection_gate)
compiled_hyper_inject = mx.compile(hyper_inject)


def dense_causal_attention(query, key, value, offset):
    """SDPA on nonduplicated KV for QSA contexts below the sparse budget.

    Offset is absolute and cannot be replaced with a top-left triangular mask
    when cached prompt tokens precede the current query chunk.
    """
    query_length, key_length = query.shape[2], key.shape[2]
    if query_length == 1 and offset == key_length - 1:
        mask = None
    elif offset == 0 and query_length == key_length:
        mask = "causal"
    else:
        mask = mx.arange(key_length)[None, :] <= (mx.arange(query_length) + offset)[:, None]
    return mx.fast.scaled_dot_product_attention(
        query, key, value, scale=query.shape[-1] ** -0.5, mask=mask)
