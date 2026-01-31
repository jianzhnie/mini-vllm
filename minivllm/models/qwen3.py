"""Qwen3 model implementation using mini-vLLM building blocks.

This module adapts the smaller building blocks into a full model
implementation compatible with the rest of the engine.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn
from transformers import Qwen3Config

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

__all__ = [
    'Qwen3Attention',
    'Qwen3MLP',
    'Qwen3Model',
    'Qwen3ForCausalLM',
]


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
        rope_scaling: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        # Determine tensor parallel (TP) size.
        tp_size = 1
        if dist.is_available() and dist.is_initialized():
            tp_size = dist.get_world_size()

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

        # Qwen models typically use RMSNorm for Q/K if bias is not present
        # or if explicitly configured. Here we follow the logic:
        # if no bias, we assume we need normalization (based on original implementation logic).
        # Note: Some Qwen versions use QK-Norm even with bias.
        # Ideally this should be controlled by config.use_sliding_window or specific config.
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(self, positions: torch.Tensor,
                hidden_states: torch.Tensor) -> torch.Tensor:
        """Execute attention forward pass.

        Args:
            positions: Tensor of shape (batch_size, seq_len) containing position indices.
            hidden_states: Tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            Tensor of shape (batch_size, seq_len, hidden_size) containing attention output.
        """
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        if self.q_norm is not None and self.k_norm is not None:
            q, _ = self.q_norm(q)
            k, _ = self.k_norm(k)

        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output

    def extra_repr(self) -> str:
        return (
            f'num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads}, '
            f'head_dim={self.head_dim}, qkv_bias={self.qkv_bias}')


class Qwen3MLP(nn.Module):
    """Feed-forward (MLP) block with gated activation used in Qwen3.

    The implementation uses a merged column-parallel linear layer for the
    gate and up projections, followed by a down projection.

    Supported activation functions:
    - 'silu': SiLU activation with gating (SwiGLU variant)
    - Additional activations can be added by extending the mapping

    Args:
        hidden_size: Dimension of the hidden layer.
        intermediate_size: Dimension of the intermediate layer.
        hidden_act: Name of the activation function (e.g., 'silu').

    Raises:
        ValueError: If activation function is not supported.
    """

    # Mapping of activation function names to implementations
    _ACTIVATION_FN_MAP = {
        'silu': SiluAndMul,
        # Future activations can be added here
        # 'gelu': GeluAndMul,
        # 'relu': ReluAndMul,
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

        # Select activation function
        if hidden_act not in self._ACTIVATION_FN_MAP:
            raise ValueError(
                f"Unsupported activation function: '{hidden_act}'. "
                f'Supported activations: {list(self._ACTIVATION_FN_MAP.keys())}'
            )

        activation_cls = self._ACTIVATION_FN_MAP[hidden_act]
        self.act_fn = activation_cls()
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute MLP forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x

    def extra_repr(self) -> str:
        return (f'hidden_size={self.hidden_size}, '
                f'intermediate_size={self.intermediate_size}, '
                f'hidden_act={self.hidden_act}')


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
        """Execute layer forward pass.

        Args:
            positions: Position indices.
            hidden_states: Input hidden states.
            residual: Residual connection from previous layer (if any).

        Returns:
            Tuple of (hidden_states, residual).
        """
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
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size,
                                                   config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        """Execute model backbone forward pass.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len).
            positions: Position indices of shape (batch_size, seq_len).

        Returns:
            Last hidden state of shape (batch_size, seq_len, hidden_size).
        """
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
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if getattr(config, 'tie_word_embeddings', False):
            # Tie embeddings if requested by configuration
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(self, input_ids: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        """Execute full model forward pass (backbone only).

        Args:
            input_ids: Input token IDs.
            positions: Position indices.

        Returns:
            Hidden states from the backbone.
        """
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute logits from hidden states.

        Args:
            hidden_states: Last hidden states from backbone.

        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size).
        """
        return self.lm_head(hidden_states)

    def load_weights(self, weights: Dict[str, Any]) -> None:
        """Load weights from a dictionary (e.g., state_dict).

        Args:
            weights: Dictionary containing model weights.
        """
        params_dict = dict(self.named_parameters())

        # Iterate over weights with progress bar
        # Note: We can't easily use tqdm here without adding it to imports,
        # but for now we'll stick to simple logging if needed, or just iterate.
        # Adding tqdm import if not present is good practice.

        for name, loaded_weight in weights.items():
            if 'rotary_emb.inv_freq' in name:
                continue

            param = None
            loaded_shard_id = None

            # Handle packed modules (e.g., q_proj -> qkv_proj)
            # This mapping handles merging Q/K/V and Gate/Up projections
            for key, (new_key,
                      shard_id) in self.packed_modules_mapping.items():
                if name.endswith(f'{key}.weight'):
                    parts = name.split('.')
                    # Ensure we are replacing the module name, not just a substring
                    if len(parts) >= 2 and parts[-2] == key:
                        parts[-2] = new_key
                        internal_name = '.'.join(parts)
                        if internal_name in params_dict:
                            param = params_dict[internal_name]
                            loaded_shard_id = shard_id
                        break

            # Handle standard modules (direct name match)
            if param is None:
                if name in params_dict:
                    param = params_dict[name]

            if param is not None:
                # Use custom weight loader if available (critical for Tensor Parallelism)
                if hasattr(param, 'weight_loader'):
                    param.weight_loader(param, loaded_weight, loaded_shard_id)
                else:
                    # Standard parameter load (e.g., LayerNorm weights)
                    param.data.copy_(loaded_weight)
            else:
                # Log warning for unused weights, but skip for some known non-mapped keys
                pass
