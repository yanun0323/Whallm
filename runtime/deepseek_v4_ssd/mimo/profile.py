"""Opt-in MiMo diagnostics. Never attached by the server or model loader.

Reuse the existing profiling collector; only the model attachment differs.
Host spans preserve laziness. Sync spans include CPU/I/O/barriers, NOT kernel
execution time. Backbone layer zero is dense; expert-cache ID = backbone ID - 1.
"""
from ..qwen_profile import QwenProfile


def layer_metrics(report):
    """Flatten only decoder totals; child spans are separate, nested columns."""
    rows = report['rows']
    by_key = {(r['request'], r['phase'], r['layer'], r['component']): r for r in rows}
    result = []
    for row in rows:
        if row['component'] != 'decoder':
            continue
        counters = row['counters']
        hits, misses = counters.get('expert_hits', 0), counters.get('expert_misses', 0)
        flat = {k: row[k] for k in ('request', 'phase', 'layer', 'expert_layer', 'attention_kind', 'calls')}
        flat.update(wall_seconds=row['inclusive_seconds'],
            owner_cpu_seconds=counters.get('owner_cpu_seconds', 0),
            expert_hits=hits, expert_misses=misses,
            miss_fraction=misses/(hits+misses) if hits+misses else None,
            logical_expert_bytes=counters.get('expert_bytes_read', 0))
        for name in ('wait', 'read', 'pack', 'eviction'):
            flat[name+'_seconds'] = counters.get('expert_'+name+'_seconds', 0)
        for component in ('attention', 'qkv_projection', 'sdpa', 'attention_output_projection',
                          'kv_update', 'router', 'routed_experts', 'expert_acquire', 'dense_mlp'):
            child = by_key.get((row['request'], row['phase'], row['layer'], component), {})
            flat[component+'_inclusive_seconds'] = child.get('inclusive_seconds', 0)
        result.append(flat)
    return result


class MiMoProfile(QwenProfile):
    def attach(self, runtime):
        from . import model as mimo_model
        from .. import model as expert_model
        if not isinstance(runtime.model, mimo_model.Model):
            raise ValueError('MiMo profiler requires a MiMo model')
        if self.patches:
            raise ValueError('MiMo profiler is already attached')
        self.cache = runtime.expert_cache
        self.layer_kinds = ['sliding' if layer.sliding else 'global' for layer in runtime.model.layers]
        groups = {}

        def add(obj, name, layer=None, method='__call__'):
            groups.setdefault((type(obj), method), {})[id(obj)] = (name, layer)

        model = runtime.model
        add(model.model.embed_tokens, 'embedding', -1)
        add(model.model.norm, 'final_norm', -1)
        add(model.lm_head, 'lm_head', -1)
        for i, layer in enumerate(model.layers):
            add(layer, 'decoder', i)
            add(layer.input_layernorm, 'input_norm', i)
            add(layer.post_attention_layernorm, 'post_attention_norm', i)
            add(layer.self_attn, 'attention', i)
            add(layer.self_attn.qkv_proj, 'qkv_projection', i)
            add(layer.self_attn.o_proj, 'attention_output_projection', i)
            add(layer.self_attn.rope, 'rope', i)
            add(layer.mlp, 'moe' if i else 'dense_mlp', i)
            if i:
                add(layer.mlp.gate, 'router', i)
                add(layer.mlp.switch_mlp, 'routed_experts', i)
        for (cls, method), labels in groups.items():
            self.hook(cls, method, labels)
        # Cache objects are request-owned and replaced on later requests. Class
        # hooks inherit the active decoder ID; use only in a dedicated process.
        from mlx_lm.models.cache import KVCache, RotatingKVCache
        for cls in (KVCache, RotatingKVCache):
            self.hook(cls, 'update_and_fetch', name='kv_update')
        self.hook(mimo_model, 'scaled_dot_product_attention', name='sdpa')
        self.hook(type(self.cache), 'get_many', {id(self.cache): ('expert_acquire', None)}, evaluate=False)
        # iter_ready is a generator: counters in routed_experts cover actual
        # consumption. Timing its construction would report bogus near-zero I/O.
        self.hook(type(runtime.support), 'prefill', {id(runtime.support): ('prefill_driver', -1)}, evaluate=False)
        self.hook(expert_model, 'eval_prompt_cache', name='cache_eval', evaluate=False)
        self.hook(expert_model, '_clear_memory_cache', name='allocator_clear', evaluate=False)

    def report(self):
        report = super().report()
        report['model_kind'] = 'mimo-v2.6-flash-rl'
        kinds = getattr(self, 'layer_kinds', ())
        for row in report['rows']:
            layer = row['layer']
            row['expert_layer'] = layer - 1 if layer > 0 else None
            row['attention_kind'] = kinds[layer] if 0 <= layer < len(kinds) else None
        report['notes'] += ' MiMo rows use backbone IDs0..47; dense0 has no expert-cache layer. Sum decoder rows only for non-nested layer totals. Router host time is lazy construction; route materialization can be charged to routed_experts. KV/SDPA hooks are process-global but owner-thread-only; run this diagnostic in a dedicated process.'
        return report
