"""Shared base implementation for Qwen-style models.

Qwen2 and Qwen3 share the same architecture, differing only in defaults:
- qkv_bias default: True (Qwen2) vs False (Qwen3)
- rope_theta default in Attention: 1000000 vs 10000
- attention_bias config fallback: True vs False
"""

from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from minivllm.models.layers import (
    Attention,
    MergedColumnParallelLinear,
    ParallelLMHead,
    QKVParallelLinear,
    RMSNorm,
    RowParallelLinear,
    SiluAndMul,
    VocabParallelEmbedding,
    get_rope,
)
from minivllm.utils.logger_utils import get_logger

logger = get_logger(__name__)


def _resolve_rope_theta(config, default: float) -> float:
    """Resolve rope_theta from config, handling nested storage formats."""
    if hasattr(config, 'rope_theta'):
        return config.rope_theta
    rope_params = getattr(config, 'rope_parameters', None) or {}
    if isinstance(rope_params, dict) and 'rope_theta' in rope_params:
        return rope_params['rope_theta']
    rope_scaling = getattr(config, 'rope_scaling', None) or {}
    if isinstance(rope_scaling, dict) and 'rope_theta' in rope_scaling:
        return rope_scaling['rope_theta']
    return default


def _resolve_rope_scaling(config) -> dict[str, Any] | None:
    """Resolve rope_scaling from config, filtering out non-scaling params."""
    raw = getattr(config, 'rope_scaling', None)
    if raw is None:
        return None
    if isinstance(raw, dict):
        # Some configs (e.g. Qwen3) store rope_theta/rope_type alongside scaling params
        scaling_keys = {
            'factor', 'type', 'low_freq_factor', 'high_freq_factor',
            'original_max_position_embeddings', 'short_factor', 'long_factor'
        }
        scaling = {k: v for k, v in raw.items() if k in scaling_keys}
        return scaling or None
    return None


class QwenAttention(nn.Module):
    """Multi-head attention block for Qwen-style models."""

    default_qkv_bias: bool = True
    default_rope_theta: float = 1000000

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool | None = None,
        rope_theta: float | None = None,
        rope_scaling: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if qkv_bias is None:
            qkv_bias = self.default_qkv_bias
        if rope_theta is None:
            rope_theta = self.default_rope_theta

        tp_size = 1
        if dist.is_available() and dist.is_initialized():
            tp_size = dist.get_world_size()

        self.total_num_heads = num_heads
        if self.total_num_heads % tp_size != 0:
            raise ValueError(
                f"total_num_heads ({self.total_num_heads}) must be divisible by tp_size ({tp_size})"
            )
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads % tp_size != 0:
            raise ValueError(
                f"total_num_kv_heads ({self.total_num_kv_heads}) must be divisible by tp_size ({tp_size})"
            )
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = int(head_dim or (hidden_size // self.total_num_heads))
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(self, positions: torch.Tensor,
                hidden_states: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)

        if hidden_states.dim() == 3:
            batch_size, seq_len, _ = hidden_states.shape
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size],
                                dim=-1)

            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            if self.q_norm is not None and self.k_norm is not None:
                q, _ = self.q_norm(q)
                k, _ = self.k_norm(k)

            q, k = self.rotary_emb(positions, q, k)
            o = self.attn(q, k, v)

            o = o.transpose(1, 2).contiguous()
            output = self.o_proj(o.view(batch_size * seq_len, -1))
            return output.view(batch_size, seq_len, -1)
        else:
            total_tokens, _ = hidden_states.shape
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size],
                                dim=-1)

            q = q.view(total_tokens, self.num_heads, self.head_dim)
            k = k.view(total_tokens, self.num_kv_heads, self.head_dim)
            v = v.view(total_tokens, self.num_kv_heads, self.head_dim)

            if self.q_norm is not None and self.k_norm is not None:
                q, _ = self.q_norm(q)
                k, _ = self.k_norm(k)

            q, k = self.rotary_emb(positions, q, k)
            o = self.attn(q, k, v)

            o = o.reshape(total_tokens, self.num_heads * self.head_dim)
            output = self.o_proj(o)
            return output

    def extra_repr(self) -> str:
        return (
            f"num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, qkv_bias={self.qkv_bias}")


class QwenMLP(nn.Module):
    """Feed-forward (MLP) block with gated activation."""

    _ACTIVATION_FN_MAP = {
        'silu': SiluAndMul,
    }

    def __init__(self, hidden_size: int, intermediate_size: int,
                 hidden_act: str) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )

        if hidden_act not in self._ACTIVATION_FN_MAP:
            raise ValueError(
                f"Unsupported activation function: '{hidden_act}'. "
                f"Supported activations: {list(self._ACTIVATION_FN_MAP.keys())}"
            )

        self.act_fn = self._ACTIVATION_FN_MAP[hidden_act]()
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x

    def extra_repr(self) -> str:
        return (f"hidden_size={self.hidden_size}, "
                f"intermediate_size={self.intermediate_size}, "
                f"hidden_act={self.hidden_act}")


class QwenDecoderLayer(nn.Module):
    """A single transformer decoder layer for Qwen-style models."""

    attention_cls = QwenAttention

    def __init__(self, config) -> None:
        super().__init__()
        self.self_attn = self.attention_cls(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias',
                             self.attention_cls.default_qkv_bias),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=_resolve_rope_theta(
                config, self.attention_cls.default_rope_theta),
            rope_scaling=_resolve_rope_scaling(config),
        )
        self.mlp = QwenMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if residual is None:
            residual = hidden_states
            hidden_states, _ = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states)

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)

        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class QwenModel(nn.Module):
    """Qwen model backbone: token embedding followed by stacked decoder layers."""

    decoder_layer_cls = QwenDecoderLayer

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size,
                                                   config.hidden_size)
        self.layers = nn.ModuleList([
            self.decoder_layer_cls(config)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class QwenForCausalLM(nn.Module):
    """Causal language-modeling wrapper with LM head."""

    model_cls = QwenModel
    packed_modules_mapping = {
        'q_proj': ('qkv_proj', 'q'),
        'k_proj': ('qkv_proj', 'k'),
        'v_proj': ('qkv_proj', 'v'),
        'gate_proj': ('gate_up_proj', 0),
        'up_proj': ('gate_up_proj', 1),
    }

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.model = self.model_cls(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if getattr(config, 'tie_word_embeddings', False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def set_kv_cache(
            self, kv_cache: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        for layer_idx, layer in enumerate(self.model.layers):
            if layer_idx < len(kv_cache):
                k_cache, v_cache = kv_cache[layer_idx]
                layer.self_attn.attn.k_cache = k_cache
                layer.self_attn.attn.v_cache = v_cache

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def load_weights(self, weights: dict[str, Any]) -> None:
        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights.items():
            if 'rotary_emb.inv_freq' in name:
                continue

            param = None
            loaded_shard_id = None

            for key, (new_key,
                      shard_id) in self.packed_modules_mapping.items():
                if name.endswith(f"{key}.weight") or name.endswith(
                        f"{key}.bias"):
                    parts = name.split('.')
                    if len(parts) >= 2 and parts[-2] == key:
                        parts[-2] = new_key
                        internal_name = '.'.join(parts)
                        if internal_name in params_dict:
                            param = params_dict[internal_name]
                            loaded_shard_id = shard_id
                        break

            if param is None:
                if name in params_dict:
                    param = params_dict[name]

            if param is not None:
                if hasattr(param, 'weight_loader'):
                    param.weight_loader(param, loaded_weight, loaded_shard_id)
                else:
                    param.data.copy_(loaded_weight)
