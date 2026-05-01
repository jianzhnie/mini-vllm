"""Qwen3 model implementation.

Thin specialization of the shared Qwen base, overriding defaults
for Qwen3 (qkv_bias=False, rope_theta=10000).
"""

from minivllm.models.qwen_base import (
    QwenAttention,
    QwenDecoderLayer,
    QwenForCausalLM,
    QwenMLP,
    QwenModel,
)

__all__ = [
    'Qwen3Attention',
    'Qwen3MLP',
    'Qwen3Model',
    'Qwen3ForCausalLM',
]


class Qwen3Attention(QwenAttention):
    default_qkv_bias: bool = False
    default_rope_theta: float = 10000


Qwen3MLP = QwenMLP


class Qwen3DecoderLayer(QwenDecoderLayer):
    attention_cls = Qwen3Attention


class Qwen3Model(QwenModel):
    decoder_layer_cls = Qwen3DecoderLayer


class Qwen3ForCausalLM(QwenForCausalLM):
    model_cls = Qwen3Model
