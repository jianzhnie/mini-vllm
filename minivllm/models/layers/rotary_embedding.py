"""Rotary positional embedding helpers."""

import os
from functools import lru_cache
from typing import Any

import torch
from torch import nn

from minivllm.utils.device import is_torch_npu_available
from minivllm.utils.logger_utils import get_logger

logger = get_logger(__name__)

_NPU_ROPE_AVAILABLE = False
if is_torch_npu_available():
    try:
        import torch_npu

        if hasattr(torch_npu, "npu_rotary_mul"):
            _NPU_ROPE_AVAILABLE = True
            logger.info("NPU RoPE kernel available")
    except ImportError:
        pass

_USE_NPU_ROPE = _NPU_ROPE_AVAILABLE and os.getenv(
    "MINIVLLM_USE_NPU_ROPE", "1"
).lower() not in {"0", "false", "no"}


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary positional embeddings to last dimension of `x`.

    Optimized for NPU with minimized shape manipulation overhead.
    """
    if _USE_NPU_ROPE and x.device.type == "npu":
        needs_unsqueeze = x.dim() == 3
        if needs_unsqueeze:
            x = x.unsqueeze(1)

        head_dim = x.shape[-1]
        if cos.shape[-1] != head_dim:
            cos = torch.cat([cos, cos], dim=-1)
            sin = torch.cat([sin, sin], dim=-1)

        # Reshape cos/sin to match x: (batch, 1, seq, head_dim) or broadcastable
        if cos.dim() == 2:
            cos = cos.unsqueeze(0).unsqueeze(1)
            sin = sin.unsqueeze(0).unsqueeze(1)
        elif cos.dim() == 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        # Expand only if shapes don't match (avoid unnecessary memory allocation)
        if cos.shape != x.shape:
            cos = cos.expand_as(x).contiguous()
            sin = sin.expand_as(x).contiguous()
        else:
            if not cos.is_contiguous():
                cos = cos.contiguous()
            if not sin.is_contiguous():
                sin = sin.contiguous()

        if cos.dtype != x.dtype:
            cos = cos.to(x.dtype)
            sin = sin.to(x.dtype)

        out = torch_npu.npu_rotary_mul(x, cos, sin)

        if needs_unsqueeze:
            out = out.squeeze(1)

        return out

    # Fallback: compute in float for numerical stability
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding module for applying RoPE to query/key tensors.

    This class implements the core rotary position embedding mechanism by
    precomputing cosine and sine values for a range of positions and providing
    efficient application to query and key tensors during forward pass.

    The rotary embedding is computed as:
    cos(m * θ) and sin(m * θ) for position m, where θ are the base frequencies.

    Attributes:
        head_size: The size of attention heads (must equal rotary_dim).
        cos_sin_cache: Precomputed cosine and sine values for positions.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        rope_scaling: dict[str, Any] | None = None,
    ) -> None:
        """Initialize rotary embedding with precomputed cos/sin values.

        Args:
            head_size: Size of attention heads.
            rotary_dim: Rotary embedding dimension (must equal head_size).
            max_position_embeddings: Maximum sequence length to support.
            base: Base frequency for the rotary embedding calculation.
            rope_scaling: Dictionary containing scaling configuration.
                Supported types: 'linear', 'dynamic'.

        Raises:
            ValueError: If rotary_dim != head_size.
        """
        super().__init__()
        self.head_size: int = head_size
        self.max_position_embeddings: int = max_position_embeddings
        if rotary_dim != head_size:
            raise ValueError("rotary_dim must equal head_size")

        # Compute inverse frequencies for rotary embedding
        inv_freq = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
        )

        # Handle RoPE scaling
        if rope_scaling:
            scaling_type = rope_scaling.get("type")
            factor = rope_scaling.get("factor", 1.0)

            if scaling_type == "linear":
                # Linear scaling: divide position indices by factor
                # We can achieve this by dividing inv_freq by factor?
                # No, cos(pos/factor * freq) = cos(pos * freq/factor).
                inv_freq /= factor
            elif scaling_type == "dynamic":
                # NTK-Aware scaling
                # Scale base by factor^(dim / (dim-2))
                # This is a simplified approximation
                base = base * (factor ** (rotary_dim / (rotary_dim - 2)))
                inv_freq = 1.0 / (
                    base
                    ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
                )

        # Create position indices
        t = torch.arange(max_position_embeddings, dtype=torch.float)

        # Compute frequencies for all positions
        freqs = torch.einsum("i,j -> ij", t, inv_freq)

        # Compute cosine and sine values
        cos = freqs.cos()
        sin = freqs.sin()

        # Store cos and sin values separately for efficient lookup
        # Shape: (max_pos, dim)
        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)

    def forward(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embedding to query and key tensors.

        Args:
            positions: Position indices for each token in the batch.
            query: Query tensor of shape (batch, num_heads, seq_len, head_size).
            key: Key tensor of shape (batch, num_heads, seq_len, head_size).

        Returns:
            Tuple of (query_with_rope, key_with_rope) with rotary embeddings applied.
        """
        # Get cos/sin for positions
        # self.cos_cache, self.sin_cache: (max_pos, dim)
        # positions: (batch, seq_len)
        # cos, sin: (batch, seq_len, dim)
        cos = self.cos_cache[positions]
        sin = self.sin_cache[positions]

        # Unsqueeze dim 1 for broadcasting across heads
        # cos, sin: (batch, 1, seq_len, dim)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict[str, Any] | None = None,
) -> RotaryEmbedding:
    """Get a cached RotaryEmbedding instance with the specified parameters.

    This function provides a convenient way to obtain a RotaryEmbedding
    instance with caching to avoid redundant computations. The LRU cache
    ensures that only one instance is created for each unique set of parameters.

    Args:
        head_size: Size of attention heads.
        rotary_dim: Rotary embedding dimension (must equal head_size).
        max_position: Maximum sequence length to support.
        base: Base frequency for the rotary embedding calculation.
        rope_scaling: Optional rope scaling parameters.

    Returns:
        RotaryEmbedding instance configured with the specified parameters.
    """
    # Note: dict is not hashable, so we can't directly pass it to lru_cache wrapped function
    # if we want to use it as a key. However, for this helper, we assume the dict content
    # determines uniqueness.
    # To make it hashable for lru_cache, we'd need to convert it to a tuple of items.
    # For now, we'll instantiate it directly if scaling is present, bypassing cache for simplicity
    # or rely on the user to pass hashable args if we were strict.
    # Given the simplicity, we'll just create the instance.

    # Check if we need to bypass cache due to unhashable dict
    if rope_scaling is not None:
        return RotaryEmbedding(head_size, rotary_dim, max_position, base, rope_scaling)

    return _get_rope_cached(head_size, rotary_dim, max_position, base)


@lru_cache(4)
def _get_rope_cached(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
) -> RotaryEmbedding:
    """Internal cached helper."""
    return RotaryEmbedding(head_size, rotary_dim, max_position, base)
