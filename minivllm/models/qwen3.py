"""Qwen3 model implementation using mini-vLLM building blocks.

This module adapts the smaller building blocks into a full model
implementation compatible with the rest of the engine.
"""

from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn
from transformers import Qwen3Config

from minivllm.models.layers import (Attention, MergedColumnParallelLinear,
                                    ParallelLMHead, QKVParallelLinear, RMSNorm,
                                    RowParallelLinear, SiluAndMul,
                                    VocabParallelEmbedding, get_rope)


class Qwen3Attention(nn.Module):
    """Multi-head attention block for Qwen3-style models.

    This module encapsulates Q/K/V projection, rotary embedding, and the
    attention computation. It supports tensor-parallel (TP) execution by
    splitting heads across ranks.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: Optional[int] = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: Optional[Tuple] = None,
    ) -> None:
        super().__init__()
        # Determine tensor parallel (TP) size. When distributed is not
        # initialized (e.g., single-process testing), default to 1.
        try:
            # Prefer to check initialization to avoid calling get_world_size
            # before the process group is ready.
            if dist.is_available() and dist.is_initialized():
                tp_size = dist.get_world_size()
            else:
                tp_size = 1
        except Exception:
            tp_size = 1

        self.total_num_heads = num_heads
        if self.total_num_heads % tp_size != 0:
            raise ValueError(
                f'total_num_heads ({self.total_num_heads}) must be divisible by tp_size ({tp_size})'
            )
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads % tp_size != 0:
            raise ValueError(
                f'total_num_kv_heads ({self.total_num_kv_heads}) must be divisible by tp_size ({tp_size})'
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

    def forward(self, positions: torch.Tensor,
                hidden_states: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):
    """Feed-forward (MLP) block with gated activation used in Qwen3.

    The implementation uses a merged column-parallel linear layer for the
    gate and up projections, followed by a down projection. Only SiLU+mul
    activation ('silu') is supported at the moment.
    """

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
        assert hidden_act == 'silu'
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):
    """A single transformer decoder layer for Qwen3.

    This layer composes attention + MLP blocks with RMS normalization
    before and after attention for residual connections.
    """

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, 'rope_theta', 1000000),
            rope_scaling=getattr(config, 'rope_scaling', None),
        )
        self.mlp = Qwen3MLP(
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
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(
                hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states: torch.Tensor = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states: torch.Tensor = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):
    """Qwen3 model backbone: token embedding followed by stacked decoder layers.

    This class is kept intentionally small and relies on distributed
    parallel embedding/LM-head implementations for efficiency.
    """

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size,
                                                   config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)
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


class Qwen3ForCausalLM(nn.Module):
    """Causal language-modeling wrapper adding an LM head on top of the
    Qwen3Model backbone.

    The `packed_modules_mapping` helps tools that unpack/model convert
    parameters to map standard module names to the model's internal names.
    """

    packed_modules_mapping = {
        'q_proj': ('qkv_proj', 'q'),
        'k_proj': ('qkv_proj', 'k'),
        'v_proj': ('qkv_proj', 'v'),
        'gate_proj': ('gate_up_proj', 0),
        'up_proj': ('gate_up_proj', 1),
    }

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if getattr(config, 'tie_word_embeddings', False):
            # Tie embeddings if requested by configuration
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(self, input_ids: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
