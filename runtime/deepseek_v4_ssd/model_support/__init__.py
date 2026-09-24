"""Built-in model packages. Installed data cannot name executable modules."""

from functools import lru_cache

from .catalog import BY_KIND, DESCRIPTORS


def support_types():
    from .deepseek_v4 import DeepSeekV4Support
    from .deepseek_v41 import DeepSeekV41Support
    from .qwen import QwenSupport
    from .mimo import MiMoSupport

    return {
        "deepseek-v4": DeepSeekV4Support,
        "deepseek-v4.1": DeepSeekV41Support,
        "qwen3.8-flash-next": QwenSupport,
        "mimo-v2.6-flash-rl": MiMoSupport,
    }


@lru_cache(maxsize=None)
def get_support(kind: str):
    try:
        return support_types()[kind](BY_KIND[kind])
    except KeyError as error:
        raise ValueError(f"unsupported model kind: {kind}") from error


def support_for_manifest(raw: dict):
    version = raw.get("formatVersion")
    if type(version) is not int:
        raise ValueError("installed model has an unsupported manifest format")
    kind = raw.get("modelKind")
    if kind is None:
        # Only old manifests may omit modelKind. New packages share schema versions.
        kind = {1: "deepseek-v4", 2: "qwen3.8-flash-next", 3: "deepseek-v4.1"}.get(version)
    if not isinstance(kind, str):
        raise ValueError("installed model has an unsupported manifest format")
    support = get_support(kind)
    if version != support.descriptor.manifest_version:
        raise ValueError("installed model kind does not match the manifest format")
    return support


def support_for_installed(installed):
    kind = getattr(installed, "model_kind", None)
    if kind is None:
        # Legacy research fixtures predate model_kind; real manifests supply it.
        kind = "deepseek-v4.1" if getattr(installed, "is_deepseek_v41", False) else (
            "qwen3.8-flash-next" if getattr(installed, "is_qwen", False) else "deepseek-v4"
        )
    return get_support(kind)


def support_for_runtime(runtime):
    if hasattr(runtime, "support"):
        return runtime.support
    if hasattr(runtime, "installed"):
        return support_for_installed(runtime.installed)
    # Compatibility for research fixtures that do not construct ModelRuntime.
    kind = "deepseek-v4.1" if getattr(runtime, "_is_deepseek_v41", False) else (
        "qwen3.8-flash-next" if getattr(runtime, "_is_qwen", False) else "deepseek-v4"
    )
    return get_support(kind)
