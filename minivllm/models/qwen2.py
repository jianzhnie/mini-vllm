"""Qwen2/Qwen2.5 model implementation.

Thin specialization of the shared Qwen base, overriding defaults
for Qwen2 (qkv_bias=True, rope_theta=1000000).
"""

from minivllm.models.qwen_base import (
    QwenAttention,
    QwenDecoderLayer,
    QwenForCausalLM,
    QwenMLP,
    QwenModel,
)

__all__ = [
    'Qwen2Attention',
    'Qwen2MLP',
    'Qwen2Model',
    'Qwen2ForCausalLM',
]


class Qwen2Attention(QwenAttention):
    default_qkv_bias: bool = True
    default_rope_theta: float = 1000000


Qwen2MLP = QwenMLP


class Qwen2DecoderLayer(QwenDecoderLayer):
    attention_cls = Qwen2Attention


class Qwen2Model(QwenModel):
    decoder_layer_cls = Qwen2DecoderLayer


class Qwen2ForCausalLM(QwenForCausalLM):
    model_cls = Qwen2Model
