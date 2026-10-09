from __future__ import annotations

import unittest
from dataclasses import asdict

from deepseek_v4_ssd.cli import _select_approximation_mode
from deepseek_v4_ssd.model import RuntimeConfig
from deepseek_v4_ssd.model_manager import ModelCatalogError, ModelDefaults, parse_model_catalog
from deepseek_v4_ssd.model_support import get_support
from deepseek_v4_ssd.server import OpenAIHandler


class AdvancedDefaultsTests(unittest.TestCase):
    def test_qwen_adaptive_and_manual_values_reach_request_options(self):
        support = get_support('qwen3.8-flash-next')
        for adaptive in (True, False):
            defaults = ModelDefaults(128, 0.4, 0.6, 7, qwen_adaptive_sampling=adaptive)
            for mode in ('chat', 'thinking'):
                with self.subTest(adaptive=adaptive, mode=mode):
                    options, _ = OpenAIHandler._common(None, {}, defaults,
                        support=support, thinking_mode=mode)
                    expected = ((0.7, 0.8, 20) if mode == 'chat' else (1.0, 0.95, 20)) if adaptive else (0.4, 0.6, 7)
                    self.assertEqual((options.temperature, options.top_p, options.top_k), expected)
                    self.assertEqual(options.presence_penalty, 1.5 if mode == 'chat' else 0.0)
            # Explicit client values retain their existing priority over defaults.
            options, _ = OpenAIHandler._common(None, {'temperature': 0, 'top_p': 0.9, 'top_k': 1}, defaults, support=support)
            self.assertEqual((options.temperature, options.top_p, options.top_k), (0, 0.9, 1))

    def test_deepseek_default_exact_and_optional_approximation_reach_requests(self):
        support = get_support('deepseek-v4')
        for mode in ('exact', 'learned-route-drop-lowest-1'):
            defaults = ModelDefaults(128, 0.2, 0.98, 0, approximation_mode=mode)
            options, _ = OpenAIHandler._common(None, {}, defaults, support=support)
            self.assertEqual(options.approximation_mode, mode)
            exact, _ = OpenAIHandler._common(None, {'approximation': {'mode': 'exact'}}, defaults, support=support)
            self.assertEqual(exact.approximation_mode, 'exact')
            dspark, _ = OpenAIHandler._common(None, {}, defaults, support=support, dspark=True)
            self.assertEqual(dspark.approximation_mode, 'exact')
        self.assertEqual(ModelDefaults(128, 0.2, 0.98, 0).approximation_mode, 'exact')
        self.assertEqual(_select_approximation_mode(None, dspark_enabled=False), 'exact')

    def test_catalog_accepts_old_defaults_and_validates_new_fields(self):
        for kind in ('deepseek-v4', 'qwen3.8-flash-next'):
            model = {
                'id': get_support(kind).descriptor.api_model_id, 'alias': None,
                'path': '/tmp/model', 'model_kind': kind,
                'runtime': asdict(RuntimeConfig()),
                'defaults': {'max_tokens': 128, 'temperature': 0.4, 'top_p': 0.6, 'top_k': 7},
                'warmup_prompt_path': None,
            }
            catalog = {'version': 1, 'models': [model]}
            defaults = parse_model_catalog(catalog)[0].defaults
            self.assertTrue(defaults.qwen_adaptive_sampling)
            self.assertEqual(defaults.approximation_mode, 'exact')
            model['defaults']['qwen_adaptive_sampling'] = False
            self.assertFalse(parse_model_catalog(catalog)[0].defaults.qwen_adaptive_sampling)
            model['defaults']['approximation_mode'] = 'learned-route-drop-lowest-1'
            if kind == 'deepseek-v4':
                self.assertEqual(parse_model_catalog(catalog)[0].defaults.approximation_mode, 'learned-route-drop-lowest-1')
            else:
                # Removed from both Qwen models on 2026-10-08.
                with self.assertRaisesRegex(ModelCatalogError, 'approximation_mode is not supported'):
                    parse_model_catalog(catalog)
            model['defaults']['approximation_mode'] = 'exact'
            for invalid in (None, 1, 'false'):
                model['defaults']['qwen_adaptive_sampling'] = invalid
                with self.assertRaises(ModelCatalogError):
                    parse_model_catalog(catalog)
