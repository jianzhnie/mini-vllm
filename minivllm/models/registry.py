"""Model registry and factory function.

Centralizes model type detection and instantiation so that both
models/__init__.py and models/manager.py can share the same logic
without creating circular imports.
"""

from typing import Any

from minivllm.models.qwen3 import Qwen3ForCausalLM

SUPPORTED_MODELS = {
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
}

TYPE_TO_ARCH = {
    "qwen3": "Qwen3ForCausalLM",
}


def create_model(hf_config: Any) -> Any:
    """Instantiate a model from a HuggingFace config object.

    Detects the architecture from ``hf_config.architectures`` or
    ``hf_config.model_type`` and returns the corresponding model.
    """
    architectures = getattr(hf_config, "architectures", None) or []
    model_type = getattr(hf_config, "model_type", "").lower()

    arch_name = None
    for arch in architectures:
        if arch in SUPPORTED_MODELS:
            arch_name = arch
            break
    if arch_name is None and model_type in TYPE_TO_ARCH:
        arch_name = TYPE_TO_ARCH[model_type]
    if arch_name is None:
        raise ValueError(
            f"Unsupported model: model_type={model_type!r}, "
            f"architectures={architectures}"
        )

    return SUPPORTED_MODELS[arch_name](hf_config)
