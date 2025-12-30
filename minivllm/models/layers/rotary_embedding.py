"""Rotary positional embedding helpers.

This module provides a lightweight rotary embedding implementation and a
cached factory (`get_rope`). The `RotaryEmbedding` stores precomputed
cos/sin values for positions and applies them to query/key tensors.
"""

from functools import lru_cache
from typing import Optional, Tuple

import torch
from torch import nn


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor,
                     sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embeddings to last dimension of `x`.

    This function implements the core rotary embedding transformation:
    - Split input tensor into real and imaginary parts along last dimension
    - Apply rotation using trigonometric interpolation
    - Concatenate results while preserving original dtype

    The rotation formula for position m is:
    [cos(mθ)  -sin(mθ)] [x₁]
    [sin(mθ)   cos(mθ)] [x₂]

    where x₁, x₂ are the split halves and θ are the base frequencies.

    Args:
        x: Tensor of shape (..., dim), where dim is even and split into two
           halves for rotary rotation. The first half represents the real part
           and the second half represents the imaginary part.
        cos: Cosine terms broadcastable to `x`'s last-half shape.
        sin: Sine terms broadcastable to `x`'s last-half shape.

    Returns:
        Tensor with rotary embeddings applied, preserving original dtype.

    Examples:
        >>> x = torch.randn(2, 8, 64)  # (batch, seq_len, hidden_dim)
        >>> cos = torch.randn(8, 32)   # (seq_len, rotary_dim//2)
        >>> sin = torch.randn(8, 32)   # (seq_len, rotary_dim//2)
        >>> rotated = apply_rotary_emb(x, cos, sin)
        >>> print(rotated.shape)  # torch.Size([2, 8, 64])
    """
    # Convert to float for stable mathematical operations
    # The rotation uses trigonometric functions that work best in float precision
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)

    # Apply rotary transformation using the rotation matrix:
    # [cos  -sin] [x1] = x1*cos - x2*sin
    # [sin   cos] [x2] = x2*cos + x1*sin
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin

    # Concatenate rotated halves and restore original dtype
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
    ) -> None:
        """Initialize rotary embedding with precomputed cos/sin values.

        Args:
            head_size: Size of attention heads.
            rotary_dim: Rotary embedding dimension (must equal head_size).
            max_position_embeddings: Maximum sequence length to support.
            base: Base frequency for the rotary embedding calculation.

        Raises:
            ValueError: If rotary_dim != head_size.
        """
        super().__init__()
        self.head_size: int = head_size
        if rotary_dim != head_size:
            raise ValueError('rotary_dim must equal head_size')

        # Compute inverse frequencies for rotary embedding
        inv_freq = 1.0 / (base**(
            torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))

        # Create position indices
        t = torch.arange(max_position_embeddings, dtype=torch.float)

        # Compute frequencies for all positions
        freqs = torch.einsum('i,j -> ij', t, inv_freq)

        # Compute cosine and sine values
        cos = freqs.cos()
        sin = freqs.sin()

        # Store concatenated cos/sin values for efficient lookup
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer('cos_sin_cache', cache, persistent=False)

    def forward(self, positions: torch.Tensor, query: torch.Tensor,
                key: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embedding to query and key tensors.

        Args:
            positions: Position indices for each token in the batch.
            query: Query tensor of shape (batch, num_heads, seq_len, head_size).
            key: Key tensor of shape (batch, num_heads, seq_len, head_size).

        Returns:
            Tuple of (query_with_rope, key_with_rope) with rotary embeddings applied.
        """
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Optional[dict] = None,
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
        rope_scaling: Optional rope scaling parameters (not supported in this helper).

    Returns:
        RotaryEmbedding instance configured with the specified parameters.

    Raises:
        ValueError: If rope_scaling is provided (not supported).
    """
    if rope_scaling is not None:
        raise ValueError('rope_scaling not supported in this helper')
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
