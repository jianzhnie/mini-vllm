"""OPT model implementation compatible with mini-vLLM.

This module implements the OPT architecture using the mini-vLLM building blocks.
"""

from typing import Optional, Tuple

import torch
from torch import nn
from transformers import OPTConfig

from minivllm.models.layers import (
    Attention,
    ColumnParallelLinear,
    ParallelLMHead,
    QKVParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from minivllm.utils.logger_utils import get_logger

logger = get_logger(__name__)

__all__ = [
    'OPTAttention',
    'OPTDecoderLayer',
    'OPTModel',
    'OPTForCausalLM',
]


class OPTLearnedPositionalEmbedding(nn.Embedding):
    """Learned positional embeddings for OPT."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT uses offset of 2 for positional embeddings
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return super().forward(positions + self.offset)


class OPTAttention(nn.Module):
    """OPT Attention layer."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5

        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}'
                f' and `num_heads`: {num_heads}).')

        self.qkv_proj = QKVParallelLinear(
            embed_dim,
            self.head_dim,
            num_heads,
            bias=bias,
        )
        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=bias,
        )
        self.attn = Attention(
            num_heads,
            self.head_dim,
            self.scaling,
            num_heads,  # OPT uses MHA (num_kv_heads = num_heads)
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)

        # Split qkv and reshape
        if hidden_states.dim() == 3:
            batch_size, seq_len, _ = hidden_states.shape
            # q, k, v are same size
            q, k, v = qkv.chunk(3, dim=-1)

            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

            # Attention expects (batch, num_heads, seq, dim) ?
            # Wait, Attention backend expects (batch, num_heads, seq, dim)
            # Or flattened (total_tokens, num_heads, dim)
            # The backend handles reshape if 3D input is passed?
            # StandardAttentionBackend handles 3D input (tokens, heads, dim) -> (1, heads, tokens, dim)
            # But here we have (batch, seq, heads, dim).
            # We should pass (batch, heads, seq, dim) to Attention.

            # Let's flatten to (total_tokens, hidden_size) style for consistent processing
            # Or just pass (batch, heads, seq, dim) if supported.
            # Attention.forward calls self.backend.forward(q, k, v)

            # StandardAttentionBackend:
            # if query.dim() == 3: (tokens, heads, dim)
            # else: (batch, heads, seq, dim)

            # So we should reshape to (batch, heads, seq, dim)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # No RoPE in OPT

            o = self.attn(q, k, v)

            # o is (batch, heads, seq, dim)
            o = o.transpose(1, 2).contiguous()
            output = self.out_proj(o.view(batch_size * seq_len, -1))
            return output.view(batch_size, seq_len, -1)

        else:
            # Flattened input (total_tokens, hidden_size)
            total_tokens, _ = hidden_states.shape
            q, k, v = qkv.chunk(3, dim=-1)

            q = q.view(total_tokens, self.num_heads, self.head_dim)
            k = k.view(total_tokens, self.num_heads, self.head_dim)
            v = v.view(total_tokens, self.num_heads, self.head_dim)

            # No RoPE

            o = self.attn(q, k, v)

            # o is (total_tokens, num_heads, head_dim)
            o = o.reshape(total_tokens, self.embed_dim)
            output = self.out_proj(o)
            return output


class OPTDecoderLayer(nn.Module):
    """OPT Decoder Layer."""

    def __init__(self, config: OPTConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = OPTAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            bias=config.enable_bias,
        )
        self.do_layer_norm_before = config.do_layer_norm_before
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.fc1 = ColumnParallelLinear(
            self.embed_dim,
            config.ffn_dim,
            bias=config.enable_bias,
        )
        self.activation_fn = nn.ReLU()
        self.fc2 = RowParallelLinear(
            config.ffn_dim,
            self.embed_dim,
            bias=config.enable_bias,
        )
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # OPT can be Pre-LN or Post-LN depending on config, but standard OPT is Pre-LN
        # config.do_layer_norm_before defaults to True for OPT

        if self.do_layer_norm_before:
            # Pre-LN
            # Attention block
            temp = self.self_attn_layer_norm(hidden_states)
            temp = self.self_attn(positions, temp)
            hidden_states = hidden_states + temp

            # MLP block
            temp = self.final_layer_norm(hidden_states)
            temp = self.fc1(temp)
            temp = self.activation_fn(temp)
            temp = self.fc2(temp)
            hidden_states = hidden_states + temp
        else:
            # Post-LN
            # Attention block
            temp = self.self_attn(positions, hidden_states)
            hidden_states = hidden_states + temp
            hidden_states = self.self_attn_layer_norm(hidden_states)

            # MLP block
            temp = self.fc1(hidden_states)
            temp = self.activation_fn(temp)
            temp = self.fc2(temp)
            hidden_states = hidden_states + temp
            hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states, None


class OPTDecoder(nn.Module):
    """OPT Decoder."""

    def __init__(self, config: OPTConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            self.embed_dim,
        )

        self.embed_positions = OPTLearnedPositionalEmbedding(
            config.max_position_embeddings,
            self.embed_dim,
        )

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(config.word_embed_proj_dim,
                                        config.hidden_size,
                                        bias=False)
            self.project_out = nn.Linear(config.hidden_size,
                                         config.word_embed_proj_dim,
                                         bias=False)
        else:
            self.project_in = None
            self.project_out = None

        if config.do_layer_norm_before:
            self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        else:
            self.final_layer_norm = None

        self.layers = nn.ModuleList(
            [OPTDecoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        pos_embeds = self.embed_positions(positions)

        if self.project_in is not None:
            hidden_states = self.project_in(hidden_states)
            pos_embeds = self.project_in(pos_embeds)

        hidden_states = hidden_states + pos_embeds

        for layer in self.layers:
            hidden_states, _ = layer(positions, hidden_states, None)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        return hidden_states


class OPTModel(nn.Module):
    """OPT Model."""

    def __init__(self, config: OPTConfig):
        super().__init__()
        self.decoder = OPTDecoder(config)

        # Mapping for weight loading
        self.packed_modules_mapping = {
            'q_proj': ('qkv_proj', 'q'),
            'k_proj': ('qkv_proj', 'k'),
            'v_proj': ('qkv_proj', 'v'),
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(input_ids, positions)


class OPTForCausalLM(nn.Module):
    """OPT for Causal LM."""

    def __init__(self, config: OPTConfig):
        super().__init__()
        self.config = config
        self.model = OPTModel(config)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            bias=False,
        )

        # Tie weights if requested
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.decoder.embed_tokens.weight

        # Support for packed modules mapping (delegate to model)
        self.packed_modules_mapping = self.model.packed_modules_mapping

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute logits from hidden states."""
        return self.lm_head(hidden_states)
