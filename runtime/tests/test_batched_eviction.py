"""Bulk reservation must match scalar eviction without rescanning reserves."""
import heapq
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from deepseek_v4_ssd.expert_cache import ExpertCache
from deepseek_v4_ssd.manifest import InstalledModel, Tensor


def model_fixture(root, layers=8, experts=256):
    # Tiny canonical blobs; no installed checkpoint or local research archive.
    regions = tuple(Tensor(f'w{i}.{kind}', dtype, shape, (i - 1) * 5 + offset, length)
                    for i in (1, 2, 3)
                    for kind, dtype, shape, offset, length in
                    [('weight', 'I8', (1, 4), 0, 4), ('scale', 'F8_E8M0', (1, 1), 4, 1)])
    (root / 'experts').mkdir()
    model = InstalledModel(root, 'fixture', 'fixture', layers, experts, 1, 15, (), regions)
    payload = bytes(range(256)) * ((experts * model.expert_blob_size + 255) // 256)
    for layer in range(layers):
        (root / 'experts' / f'layer_{layer:02d}.bin').write_bytes(payload[:experts * model.expert_blob_size])
    return model


class ScalarCache(ExpertCache):
    def _evict_batch(self, protected):
        while True:
            yield self._evict(protected)


class BatchedEvictionTests(unittest.TestCase):
    def test_bulk_reservation_does_not_rescan_reserved_entries_per_victim(self):
        with tempfile.TemporaryDirectory() as directory:
            with ExpertCache(model_fixture(Path(directory)), slots=128, eviction_policy='lru') as cache:
                for layer in range(7):
                    cache.get_many(layer, list(range(8)))
                cache.get_many(7, list(range(72)))
                with patch('deepseek_v4_ssd.expert_cache.heapq.heappop', wraps=heapq.heappop) as pops:
                    cache.get_many(7, list(range(72, 136)))
                self.assertEqual(set(cache._entries),
                    {(layer, expert) for layer in range(7) for expert in range(8)} |
                    {(7, expert) for expert in range(64, 136)})
                self.assertLessEqual(pops.call_count, 2 * cache.slots,
                    'bulk eviction must not pop/reinsert the same reserved entries for every missing expert')

    def test_bulk_matches_scalar_with_pinning_protection_decay_and_resize(self):
        with tempfile.TemporaryDirectory() as directory:
            model = model_fixture(Path(directory), layers=4, experts=128)
            for policy in ('lru', 'lfu'):
                with ExpertCache(model, slots=96, eviction_policy=policy) as actual, \
                     ScalarCache(model, slots=96, eviction_policy=policy) as expected:
                    rng = random.Random(42)
                    for step in range(100):
                        layer = rng.randrange(4)
                        experts = rng.sample(range(128), rng.choice((1, 10, 40, 64)))
                        pinned = set(sorted(actual._entries)[:2]) if step % 3 else set()
                        pinned_layers = {rng.randrange(4)} if step % 2 else set()
                        for cache in (actual, expected):
                            cache._pinned_expert_keys = pinned.copy()
                            cache._pinned_layers = pinned_layers.copy()
                            cache.get_many(layer, experts + experts[:3])
                        self.assertEqual(actual._entries, expected._entries, (policy, step))
                        self.assertEqual(actual._layer_counts, expected._layer_counts)
                        self.assertEqual(actual.metrics.bytes_read, expected.metrics.bytes_read)
                        self.assertEqual(actual.metrics.evictions, expected.metrics.evictions)
                        if step in (35, 70):
                            for cache in (actual, expected):
                                cache._pinned_expert_keys.clear()
                                cache._pinned_layers.clear()
                                cache._resize_phase_slots(80 if step == 35 else 96)
                    self.assertGreater(actual._last_decay, 0)
                    self.assertLess(len(actual._heap), actual.slots * 10)

    def test_failure_and_release_leave_scalar_heap_usable(self):
        with tempfile.TemporaryDirectory() as directory:
            with ExpertCache(model_fixture(Path(directory), layers=2), slots=64, eviction_policy='lru') as cache:
                cache.get_many(0, list(range(64)))
                with patch.object(cache, '_read_expert_into_slot', side_effect=OSError('read failure')):
                    with self.assertRaises(OSError):
                        cache.get_many(1, list(range(40)))
                cache.get_many(1, list(range(40)))
                cache.get_many(0, [100])
                self.assertEqual(len({e.slot for e in cache._entries.values()}), cache.resident_count)
                cache.release_prefill_slots()
                cache.get_many(0, [0])
                self.assertEqual(set(cache._entries), {(0, 0)})

    def test_ready_and_staged_consumers_match_scalar_reservations(self):
        with tempfile.TemporaryDirectory() as directory:
            model = model_fixture(Path(directory), layers=2)
            for staged in (False, True):
                with ExpertCache(model, slots=64, eviction_policy='lru', staged_expert_streaming=staged) as actual, \
                     ScalarCache(model, slots=64, eviction_policy='lru') as expected:
                    for cache in (actual, expected):
                        cache.get_many(0, list(range(64)))
                    expected.get_many(1, list(range(40)))
                    if staged:
                        for handle in actual.iter_staged_ready(1, list(range(40))):
                            handle.finish_w2()
                    else:
                        list(actual.iter_ready(1, list(range(40))))
                    self.assertEqual(actual._entries, expected._entries)
                    self.assertEqual(actual.metrics.bytes_read, expected.metrics.bytes_read)

    def test_route_policy_keeps_its_existing_selector(self):
        with tempfile.TemporaryDirectory() as directory:
            with ExpertCache(model_fixture(Path(directory), layers=2), slots=64, eviction_policy='route') as cache:
                cache.get_many(0, list(range(64)))
                with patch.object(cache, '_evict_batch', side_effect=AssertionError('route uses its own policy')):
                    cache.get_many(1, list(range(40)))

    def test_no_eligible_victim_preserves_hard_pins(self):
        with tempfile.TemporaryDirectory() as directory:
            with ExpertCache(model_fixture(Path(directory), layers=2), slots=64, eviction_policy='lfu') as cache:
                cache.get_many(0, list(range(64)))
                cache._pinned_expert_keys = set(cache._entries)
                before = dict(cache._entries)
                with self.assertRaisesRegex(RuntimeError, 'no expert cache slot'):
                    cache.get_many(1, list(range(40)))
                self.assertEqual(cache._entries, before)
                cache._pinned_expert_keys.clear()
                cache.get_many(1, list(range(40)))


if __name__ == '__main__':
    unittest.main()
