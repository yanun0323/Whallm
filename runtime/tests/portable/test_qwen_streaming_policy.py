"""Host-only contracts for exact streaming schedules and redacted diagnostics."""
import argparse
from dataclasses import dataclass
from types import SimpleNamespace
import unittest
import numpy as np

if __package__:
    from .test_qwen_flash_portable import load_host_module
else:
    from test_qwen_flash_portable import load_host_module

policy = load_host_module('qwen_streaming_policy')
settings = load_host_module('qwen_flash_config')
evidence = load_host_module('throughput_diagnostics')


class StreamingPolicyTests(unittest.TestCase):
    def test_adjacent_batches_cover_exactly_the_input_without_gaps(self):
        rng = np.random.default_rng(492)
        for maximum in (1, 2, 4, 8, 32):
            for count in (0, 1, 2, 64, 400):
                ids = sorted(rng.choice(512, count, replace=False).tolist())
                batches = list(policy.contiguous_batches(ids, maximum))
                self.assertEqual([e for b in batches for e in b], ids)
                for batch in batches:
                    self.assertLessEqual(len(batch), maximum)
                    self.assertEqual(list(batch), list(range(batch[0], batch[-1] + 1)))
        self.assertEqual(list(policy.contiguous_batches([0,1,2,5,6,9], 2)), [(0,1),(2,),(5,6),(9,)])

    def test_invalid_schedules_and_defaults_are_rejected(self):
        for ids in ([1,0], [0,0], [-1], [True], [1.2]):
            with self.assertRaises(ValueError): list(policy.contiguous_batches(ids, 4))
        for size in (0,33,False,1.2):
            with self.assertRaises(ValueError): list(policy.contiguous_batches([0], size))

    def test_hot_tail_counts_then_recency_with_stable_insertion_order(self):
        rows = np.array([[1,2],[1,3],[4,1],[2,5]])
        self.assertEqual(policy.hot_tail_experts(rows, 8, 3), [5,2,1])
        self.assertEqual(policy.hot_tail_experts(rows[None], 8, 3), [5,2,1])
        self.assertEqual(policy.hot_tail_experts(rows, 8, 0), [])
        self.assertEqual(policy.hot_tail_experts(np.empty((0,2),dtype=int), 8, 5), [])
        tail = np.concatenate([np.ones((256,1),dtype=int),np.full((128,1),5)])
        self.assertEqual(policy.hot_tail_experts(tail, 8, 128), [5])
        for invalid in (np.array([1,2]), np.array([[8]]),np.array([[-1]]), np.ones((3,2))):
            with self.assertRaises(ValueError): policy.hot_tail_experts(invalid, 8, 2)

    def test_read_seed_cli_range_and_dependencies(self):
        parser = argparse.ArgumentParser()
        settings.add_flash_arguments(parser)
        raw = settings.flash_arguments(parser.parse_args([]))
        self.assertEqual((raw['qwen_prefill_read_experts'],raw['qwen_prefill_seed_experts'],raw['qwen_shared_expert_overlap']), (1,0,False))
        raw = settings.flash_arguments(parser.parse_args(['--qwen-prefill-read-experts','4','--qwen-prefill-seed-experts','32','--qwen-shared-expert-overlap']))
        settings.validate_flash_config(SimpleNamespace(**raw))
        for changes in ({'layer_major_prefill':False}, {'batched_expert_prefill':False}, {'qwen_expert_wave_slots':32},
                        {'qwen_prefill_read_experts':True}, {'qwen_prefill_seed_experts':129}, {'qwen_shared_expert_overlap':1}):
            with self.assertRaises(ValueError): settings.validate_flash_config(SimpleNamespace(**(raw|changes)))
        settings.validate_flash_config(SimpleNamespace(layer_major_prefill=False,batched_expert_prefill=False))

    def test_configuration_evidence_redacts_paths_and_is_stable(self):
        cfg = SimpleNamespace(slots=64, qwen_ngram_io='mmap', secret_token='never-export',
                              prompt_cache_directory='/private/user', trace_path='/private/trace', qwen_prefill_seed_experts=32)
        text, digest = evidence.config_evidence(cfg)
        self.assertNotIn('private', text)
        self.assertNotIn('never-export', text)
        self.assertNotIn('secret_token', text)
        reverse = SimpleNamespace(**dict(reversed(list(vars(cfg).items()))))
        self.assertEqual(evidence.config_evidence(reverse), (text,digest))
        cfg.slots=65
        self.assertNotEqual(evidence.config_evidence(cfg)[1], digest)

    def test_counters_use_request_deltas_and_unknown_is_not_zero(self):
        @dataclass
        class Counters:
            bytes_read: int = 100
            wait_seconds: float = .5
        cache = SimpleNamespace(metrics_snapshot=lambda:Counters(),prefill_io_snapshot=lambda:{'read_calls':4,'direct_bytes':0,'copied_bytes':100})
        before=evidence.cache_counters(cache)
        after=before|{'bytes_read':160,'wait_seconds':.75}
        delta=evidence.difference(after,before)
        self.assertEqual(delta['bytes_read'],60)
        self.assertEqual(delta['wait_seconds'],.25)
        self.assertIsNone(evidence.cache_counters(None))
        self.assertIsNone(evidence.difference(after,None))
        report=evidence.make_report(SimpleNamespace(slots=64),before,after,after,{},1,5)
        self.assertEqual(report['through_first_token']['bytes_read'],60)
        self.assertEqual(report['after_first_token']['bytes_read'],0)
        self.assertEqual(len(report['source_files_sha256']),64)
        self.assertIsNone(report['process_disk_bytes_read'])
