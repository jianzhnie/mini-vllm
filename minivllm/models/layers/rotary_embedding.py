"""Rotary positional embedding helpers.

This module provides a lightweight rotary embedding implementation and a
cached factory (`get_rope`). The `RotaryEmbedding` stores precomputed
cos/sin values for positions and applies them to query/key tensors.
"""

from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from minivllm.utils.device import is_torch_npu_available
from minivllm.utils.logger_utils import get_logger

logger = get_logger(__name__)

# Try to import NPU specific kernels
_NPU_ROPE_AVAILABLE = False
if is_torch_npu_available():
    try:
        import torch_npu

        if hasattr(torch_npu, 'npu_rotary_mul'):
            _NPU_ROPE_AVAILABLE = True
            logger.info('NPU RoPE kernel available')
    except ImportError:
        pass


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
    # NPU optimization
    if _NPU_ROPE_AVAILABLE and x.device.type == 'npu':
        import torch_npu

        return torch_npu.npu_rotary_mul(x, cos, sin)

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
        rope_scaling: Optional[Dict[str, Any]] = None,
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
        if rotary_dim != head_size:
            raise ValueError('rotary_dim must equal head_size')

        # Compute inverse frequencies for rotary embedding
        inv_freq = 1.0 / (base**(
            torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))

        # Handle RoPE scaling
        if rope_scaling:
            scaling_type = rope_scaling.get('type')
            factor = rope_scaling.get('factor', 1.0)

            if scaling_type == 'linear':
                # Linear scaling: divide position indices by factor
                # We can achieve this by dividing inv_freq by factor?
                # No, cos(pos/factor * freq) = cos(pos * freq/factor).
                inv_freq /= factor
            elif scaling_type == 'dynamic':
                # NTK-Aware scaling
                # Scale base by factor^(dim / (dim-2))
                # This is a simplified approximation
                base = base * (factor**(rotary_dim / (rotary_dim - 2)))
                inv_freq = 1.0 / (base**(torch.arange(
                    0, rotary_dim, 2, dtype=torch.float) / rotary_dim))

        # Create position indices
        t = torch.arange(max_position_embeddings, dtype=torch.float)

        # Compute frequencies for all positions
        freqs = torch.einsum('i,j -> ij', t, inv_freq)

        # Compute cosine and sine values
        cos = freqs.cos()
        sin = freqs.sin()

        # Store cos and sin values separately for efficient lookup
        # Shape: (max_pos, dim)
        self.register_buffer('cos_cache', cos, persistent=False)
        self.register_buffer('sin_cache', sin, persistent=False)

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
    rope_scaling: Optional[Dict[str, Any]] = None,
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
        return RotaryEmbedding(head_size, rotary_dim, max_position, base,
                               rope_scaling)

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
