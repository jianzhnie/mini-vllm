"""Models module for mini-vLLM.

This module contains model-specific implementations and utilities
for loading and managing different language model architectures.

Currently supported models:
- Qwen3: A transformer-based language model with optimized attention mechanisms

The module provides:
- Model definitions with tensor-parallel support
- Integration with flash attention for high-performance inference
- Support for various attention patterns (MHA, MQA, GQA)
- RMSNorm normalization for improved training stability
"""

from typing import Any

from transformers import PretrainedConfig

from .qwen3 import Qwen3ForCausalLM, Qwen3Model

__all__ = [
    'Qwen3ForCausalLM',
    'Qwen3Model',
    'get_model_cls',
    'create_model',
]


def get_model_cls(hf_config: PretrainedConfig) -> Any:
    """Get the appropriate model class based on the HuggingFace config.

    Args:
        hf_config: HuggingFace model configuration object.

    Returns:
        The corresponding model class for the given configuration.

    Raises:
        ValueError: If the model architecture is not supported.
    """
    model_type: str = getattr(hf_config, 'model_type', '')

    # Map model types to their corresponding model classes
    model_cls_map = {
        'qwen3': Qwen3ForCausalLM,
        # Add more model types here as they are supported
        # 'llama': LlamaForCausalLM,
        # 'mistral': MistralForCausalLM,
    }

    # Check if we have a direct match for the model_type
    if model_type in model_cls_map:
        return model_cls_map[model_type]

    # Check if it's a Qwen3 model by architecture name
    arch_name: str = getattr(hf_config, 'architectures', [''])[0]
    if 'Qwen3' in arch_name:
        return Qwen3ForCausalLM

    raise ValueError(
        f'Unsupported model architecture: {model_type} ({arch_name}). '
        f'Currently supported models: {list(model_cls_map.keys())}')


def create_model(hf_config: PretrainedConfig, **kwargs) -> Any:
    """Create a model instance based on the HuggingFace config.

    Args:
        hf_config: HuggingFace model configuration object.
        **kwargs: Additional keyword arguments to pass to the model constructor.

    Returns:
        A model instance initialized with the given configuration.
    """
    model_cls = get_model_cls(hf_config)
    return model_cls(hf_config, **kwargs)
