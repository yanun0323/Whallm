"""QSA fast-path lifetime checks with quantized KV and persisted index state."""
import unittest

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import CacheList, KVCache
from deepseek_v4_ssd.model import _fork_prompt_cache, eval_prompt_cache
from deepseek_v4_ssd.model_support.state import persistence_cache_state, restore_persistence_cache
from deepseek_v4_ssd.qwen4_exp import QSAAttention
from deepseek_v4_ssd.qwen_pooled_cache import QSAPooledIndexCache, QSAPooledQuantizedIndexCache
from deepseek_v4_ssd.qwen_quantized_cache import QSAQuantizedCache
from runtime.tests.test_qwen_speed import tiny_args, bits, close


class DenseQSACacheLifecycleTests(unittest.TestCase):
    def test_dense_sparse_fork_restore_and_replay(self):
        for packed in (False, True):
            for dtype, tolerance in ((mx.float32, 3e-5), (mx.bfloat16, 5e-3)):
                with self.subTest(packed=packed, dtype=dtype):
                    mx.random.seed(934)
                    attention = QSAAttention(tiny_args())
                    attention.set_dtype(dtype)
                    hidden = (mx.random.normal((1, 17, 64)) * .1).astype(dtype)
                    def make_cache(pooled):
                        main = QSAQuantizedCache(8, 32) if packed else KVCache()
                        if pooled:
                            index = QSAPooledQuantizedIndexCache(4, 32) if packed else QSAPooledIndexCache()
                        else:
                            index = QSAQuantizedCache(4, 32) if packed else KVCache()
                        return CacheList(main, index)
                    reference, optimized = make_cache(False), make_cache(True)
                    start = 0
                    for count in (3, 5, 1):
                        chunk = hidden[:, start:start + count]
                        attention.sparse_sdpa = False
                        expected = attention(chunk, reference)
                        attention.sparse_sdpa = True
                        actual = attention(chunk, optimized)
                        close(actual, expected, tolerance)
                        eval_prompt_cache([optimized, reference], actual, expected)
                        start += count
                    saved = persistence_cache_state([optimized])
                    forked, dependencies = _fork_prompt_cache([optimized])
                    mx.eval(*dependencies)
                    restored = make_cache(True)
                    restore_persistence_cache([restored], saved)
                    chunk = hidden[:, 9:13]
                    expected = attention(chunk, optimized)
                    eval_prompt_cache([optimized], expected)
                    for other in (forked[0], restored):
                        actual = attention(chunk, other)
                        eval_prompt_cache([other], actual)
                        np.testing.assert_array_equal(bits(actual), bits(expected))
                    attention.sparse_sdpa = False
                    close(expected, attention(chunk, reference), tolerance)
                    # Rewind back below the dense threshold, then cross it again.
                    for cache in (reference, optimized, restored):
                        for branch in cache.caches:
                            branch.trim(7)
                    chunk = hidden[:, 6:11]
                    expected = attention(chunk, reference)
                    attention.sparse_sdpa = True
                    for cache in (optimized, restored):
                        actual = attention(chunk, cache)
                        close(actual, expected, tolerance)
                        self.assertTrue(mx.all(mx.isfinite(actual)).item())
                        self.assertEqual([c.offset for c in cache.caches], [11, 11])


if __name__ == '__main__':
    unittest.main()
