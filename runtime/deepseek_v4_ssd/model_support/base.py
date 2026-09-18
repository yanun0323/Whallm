from __future__ import annotations

from contextlib import contextmanager

from .catalog import ModelDescriptor


class ModelSupport:
    """Model-specific loading, prefill, state and chat behind one interface.

    One loaded model serves one request at a time. Cache objects belong to the
    model; the caller may only clone or serialize them through these methods.
    """

    # True when DSpark requests reuse prompt snapshots of the target cache and
    # draft context without the experimental dspark_prompt_cache option.
    dspark_reuses_prompt_cache = False

    def __init__(self, descriptor: ModelDescriptor):
        self.descriptor = descriptor

    def validate_config(self, config) -> None:
        features = self.descriptor.features
        if self.descriptor.kind == "deepseek-v4.1" and getattr(config, "dspark_enabled", False):
            for name in ("v41_ced_prefill", "dspark_sequential_verification"):
                if getattr(config, name, False):
                    raise ValueError(f"{name} is not supported by V4.1 DSpark")
        # The two V4.1 prefill defaults are consulted only by its support package.
        # They must not prevent a default RuntimeConfig from loading V4 or Qwen.
        for name in ("qwen_quantized_kv", "qwen_quantized_index", "v41_packed_kv", "v41_packed_index",
                     "v41_candidate_index", "v41_ced_prefill"):
            expected = "qwen3.8-flash-next" if name.startswith("qwen_") else "deepseek-v4.1"
            if getattr(config, name, False) and self.descriptor.kind != expected:
                raise ValueError(f"{name} is supported only by {expected}")
        if getattr(config, "deepseek_ane_prefill", False) and self.descriptor.kind not in ("deepseek-v4", "deepseek-v4.1"):
            raise ValueError("DeepSeek ANE prefill requires a DeepSeek model")
        if getattr(config, "v41_ced_prefill", False) and not (config.layer_major_prefill and config.v41_layer_major_prefill):
            raise ValueError("CED prefill requires layer-major prefill")
        for option, feature, label in (
            ("dspark_enabled", "dspark", "DSpark"),
            ("staged_expert_streaming", "stagedExpertStreaming", "staged expert streaming"),
        ):
            if getattr(config, option, False) and feature not in features:
                raise ValueError(f"{self.option_error_name} does not support {label}")
        if getattr(config, "mtp_enabled", False) and "mtp" not in features:
            raise ValueError("MTP is supported only by Qwen3.8-Flash-Next")

    @property
    def option_error_name(self) -> str:
        return self.descriptor.display_name

    def uses_layer_major_prefill(self, config, token_count: int) -> bool:
        if self.descriptor.kind == "deepseek-v4.1" and not getattr(config, "v41_layer_major_prefill", False):
            return False
        threshold = self.descriptor.prefill_threshold
        if threshold is None:
            threshold = getattr(config, "layer_major_prefill_threshold", 1_024)
        return bool(self.descriptor.supports("layerMajorPrefill")
                    and getattr(config, "layer_major_prefill", True)
                    and token_count >= threshold)

    def load(self, installed, config, raw_config, weights, read_limiter):
        raise NotImplementedError

    def manifest_contract(self, raw):
        raise NotImplementedError

    def prefill(self, model, tokens, cache, step_size, expert_cache, config):
        raise NotImplementedError

    def open_codec(self, root, tokenizer):
        raise NotImplementedError

    def make_tool_stream_parser(self, thinking_mode):
        raise NotImplementedError

    def reasoning_settings(self, thinking_mode, effort):
        mode = thinking_mode or ("chat" if effort in {None, "none"} else "thinking")
        effort = {"minimal": "low", "low": "low", "medium": "low",
                  "high": "high", "xhigh": "max", "max": "max"}.get(effort, "low")
        return mode, effort

    def sampling_defaults(self, defaults, thinking_mode):
        return {
            "temperature": defaults.temperature, "top_p": defaults.top_p,
            "top_k": defaults.top_k, "min_p": 0.0,
            "presence_penalty": 0.0, "repetition_penalty": 1.0,
        }

    def cli_sampling_defaults(self):
        defaults = self.descriptor.defaults
        return {"temperature": defaults["temperature"], "top_p": defaults["topP"],
                "top_k": defaults["topK"]}

    def default_approximation(self, dspark_enabled=False, mode="exact"):
        return (mode
                if self.descriptor.supports("approximation") and not dspark_enabled else "exact")

    def prefer_tool_first(self, messages, tool_choice, response_tools):
        return False

    def new_cache(self, model):
        from .state import make_cache
        return make_cache(model)

    def evaluate_cache(self, cache):
        from ..model import eval_prompt_cache
        return eval_prompt_cache(cache)

    def clone_cache(self, cache):
        import copy
        if not self.descriptor.supports("promptCache"):
            raise ValueError("model does not support prompt cache restoration")
        return copy.deepcopy(cache)

    def snapshot_cache(self, cache):
        from .state import persistence_cache_state
        if not self.descriptor.supports("promptCache"):
            raise ValueError("model does not support prompt cache persistence")
        return persistence_cache_state(cache)

    def restore_cache(self, cache, state):
        from .state import restore_persistence_cache
        if not self.descriptor.supports("promptCache"):
            raise ValueError("model does not support prompt cache restoration")
        restore_persistence_cache(cache, state)

    @contextmanager
    def approximation(self, model, mode):
        if mode == "exact":
            yield
            return
        if mode != "learned-route-drop-lowest-1" or not self.descriptor.supports("approximation"):
            raise ValueError(f"approximation mode is not supported for {self.option_error_name}")
        if getattr(model, "dspark", None) is not None or getattr(model, "mtp", None) is not None:
            raise ValueError("approximation mode cannot be combined with speculative decoding")
        changed = []
        try:
            for layer in model.model.layers:
                router, attribute = ((layer.mlp, "top_k") if hasattr(layer, "mlp")
                                     else (layer.ffn.gate, "topk"))
                count = getattr(router, attribute)
                if count <= 1:
                    raise ValueError("approximation requires at least two routed experts")
                changed.append((router, attribute, count))
                setattr(router, attribute, count - 1)
            yield
        finally:
            for router, attribute, count in changed:
                setattr(router, attribute, count)

    def close(self, model, expert_cache):
        resources = []
        dspark = getattr(model, "dspark", None)
        if dspark is not None:
            dspark.reset_cache()
            resources.append(dspark.expert_cache)
        resources.extend((getattr(model, "mtp_expert_cache", None),
                          getattr(model, "ane_prefill", None), expert_cache))
        error = None
        seen = set()
        for resource in resources:
            if resource is None or id(resource) in seen:
                continue
            seen.add(id(resource))
            try:
                resource.close()
            except Exception as cause:
                error = error or cause
        if error is not None:
            raise error
