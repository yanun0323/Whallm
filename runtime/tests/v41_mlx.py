"""The MLX DeepSeek V4.1 support with the features the catalog listed before V4.1 moved
to the native engine (2026-10-16). Tests of the MLX V4.1 code use it."""
from dataclasses import replace

from deepseek_v4_ssd.model_support.catalog import BY_KIND
from deepseek_v4_ssd.model_support.deepseek_v41 import DeepSeekV41Support

MLX_FEATURES = frozenset({
    "promptCache", "approximation", "layerMajorPrefill", "readyExpertDecode", "batchedExpertPrefill",
    "deepseekANEPrefill", "packedKVCache", "packedIndexCache", "candidateIndex", "cedPrefill",
    "nextLayerPrefetch", "dspark",
})


def mlx_v41_support() -> DeepSeekV41Support:
    return DeepSeekV41Support(replace(BY_KIND["deepseek-v4.1"], features=MLX_FEATURES))
