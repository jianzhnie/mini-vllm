"""Model layers module for mini-vLLM.

This module provides essential building blocks for constructing language models,
including attention mechanisms, linear layers, normalization, and more.

The module is organized into several categories:

1. **Activation Functions**: Non-linear transformations for neural networks
2. **Attention Components**: Self-attention mechanisms with KV-cache support
3. **Embedding Layers**: Token embeddings and language model heads with tensor parallelism
4. **Normalization**: RMSNorm and other normalization techniques
5. **Linear Layers**: Various tensor-parallel linear transformations
6. **Rotary Embeddings**: Positional encoding using rotary embeddings

All components are optimized for tensor-parallel inference and support
both prefill and decode phases of language model execution.
"""

from .activation import SiluAndMul
from .attention import Attention
from .embed_head import ParallelLMHead, VocabParallelEmbedding
from .layernorm import RMSNorm
from .linear import (
    ColumnParallelLinear,
    LinearBase,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    divide,
)
from .rotary_embedding import RotaryEmbedding, apply_rotary_emb, get_rope

__all__ = [
    # Activation functions
    'SiluAndMul',
    # Attention components
    'Attention',
    # Embedding layers
    'ParallelLMHead',
    'VocabParallelEmbedding',
    # Normalization
    'RMSNorm',
    # Linear layers
    'ColumnParallelLinear',
    'LinearBase',
    'MergedColumnParallelLinear',
    'QKVParallelLinear',
    'ReplicatedLinear',
    'RowParallelLinear',
    'divide',
    # Rotary embeddings
    'RotaryEmbedding',
    'apply_rotary_emb',
    'get_rope',
]
