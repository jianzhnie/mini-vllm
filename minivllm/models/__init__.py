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

from .qwen3 import Qwen3ForCausalLM, Qwen3Model

__all__ = [
    'Qwen3ForCausalLM',
    'Qwen3Model',
]
