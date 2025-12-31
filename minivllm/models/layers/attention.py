"""Attention layer utilities and KV-cache helpers.

This module implements attention primitives which rely on external
high-performance libraries (Triton / flash-attn). The implementation
here focuses on correctness, type annotations and clean imports.

The module provides:
- High-performance attention computation using FlashAttention
- KV-cache management for efficient inference
- Support for both prefill and decode phases
- Tensor-parallel attention patterns

Key components:
- store_kvcache_kernel: Triton kernel for efficient KV cache updates
- store_kvcache: Python wrapper for KV cache storage
- Attention: Main attention module with flash attention integration
"""

import logging

import torch
from torch import nn

from minivllm.utils.context import get_context

logger = logging.getLogger(__name__)

# Optional imports for high-performance attention
try:
    import triton
    import triton.language as tritonlang
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False
    logger.warning(
        'Triton not available. Some optimizations will be disabled.')

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    _FLASH_ATTN_AVAILABLE = False
    logger.warning(
        'FlashAttention not available. Falling back to standard attention.')

if _TRITON_AVAILABLE:

    @triton.jit
    def store_kvcache_kernel(
        key_ptr,
        key_stride,
        value_ptr,
        value_stride,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_ptr,
        D: tritonlang.constexpr,
    ):
        """Triton kernel for storing key-value pairs to cache.

        This kernel efficiently writes K/V tensors to their cache locations
        based on slot mapping, enabling fast KV-cache updates during inference.
        """
        idx = tritonlang.program_id(0)
        slot = tritonlang.load(slot_mapping_ptr + idx)
        if slot == -1:
            return
        key_offsets = idx * key_stride + tritonlang.arange(0, D)
        value_offsets = idx * value_stride + tritonlang.arange(0, D)
        key = tritonlang.load(key_ptr + key_offsets)
        value = tritonlang.load(value_ptr + value_offsets)
        cache_offsets = slot * D + tritonlang.arange(0, D)
        tritonlang.store(k_cache_ptr + cache_offsets, key)
        tritonlang.store(v_cache_ptr + cache_offsets, value)
else:
    # Fallback implementation when Triton is not available
    def store_kvcache_kernel(*args, **kwargs):
        """Fallback kernel when Triton is not available.

        This fallback raises an error to prevent silent failures.
        Users should install Triton for optimal performance.
        """
        raise RuntimeError('Triton is required for KV cache operations. '
                           'Please install triton: pip install triton')


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Store key-value pairs to cache using Triton kernel.

    This function efficiently writes K/V tensors to their cache locations
    based on slot mapping, enabling fast KV-cache updates during inference.

    Args:
        key: Key tensor of shape (N, num_heads, head_dim)
        value: Value tensor of shape (N, num_heads, head_dim)
        k_cache: Key cache tensor
        v_cache: Value cache tensor
        slot_mapping: Slot mapping tensor of shape (N,)

    Raises:
        AssertionError: If tensor strides or shapes don't match expected patterns
    """
    batch_size, num_heads, head_dim = key.shape
    hidden_size = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == hidden_size and v_cache.stride(
        1) == hidden_size
    assert slot_mapping.numel() == batch_size
    store_kvcache_kernel[(batch_size, )](key, key.stride(0), value,
                                         value.stride(0), k_cache, v_cache,
                                         slot_mapping, hidden_size)


class Attention(nn.Module):
    """Flash Attention module with KV-cache support.

    This class provides high-performance attention computation using FlashAttention
    with integrated KV-cache management for efficient inference. It supports both
    prefill (prompt processing) and decode (token generation) phases.

    The module expects pre-split query/key/value tensors with specific shapes:
    - Query: (N, num_heads, head_dim)
    - Key/Value: (N, num_kv_heads, head_dim)

    Args:
        num_heads: Number of attention heads for queries
        head_dim: Dimension of each attention head
        scale: Attention scaling factor (typically 1/sqrt(head_dim))
        num_kv_heads: Number of attention heads for keys/values (for GQA/MQA)

    Attributes:
        num_heads: Number of query attention heads
        head_dim: Dimension of each attention head
        scale: Attention scaling factor
        num_kv_heads: Number of key/value attention heads
        k_cache: Key cache tensor (initialized later by ModelRunner)
        v_cache: Value cache tensor (initialized later by ModelRunner)

    Examples:
        >>> attention = Attention(num_heads=32, head_dim=128, scale=0.088, num_kv_heads=32)
        >>> q = torch.randn(2, 32, 128)  # (batch, num_heads, head_dim)
        >>> k = torch.randn(2, 32, 128)  # (batch, num_kv_heads, head_dim)
        >>> v = torch.randn(2, 32, 128)  # (batch, num_kv_heads, head_dim)
        >>> output = attention(q, k, v)
        >>> print(output.shape)  # torch.Size([2, 32, 128])
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
    ) -> None:
        super().__init__()
        self.num_heads: int = num_heads
        self.head_dim: int = head_dim
        self.scale: float = float(scale)
        self.num_kv_heads: int = num_kv_heads

        # KV cache tensors are set later by the ModelRunner; initialize
        # as empty tensors to avoid attribute errors during initialization
        self.k_cache: torch.Tensor = torch.tensor([])
        self.v_cache: torch.Tensor = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                v: torch.Tensor) -> torch.Tensor:
        """Apply attention computation with KV-cache management.

        This method handles both prefill and decode phases, automatically
        selecting the appropriate FlashAttention function based on the
        inference context.

        Args:
            q: Query tensor of shape (N, num_heads, head_dim)
            k: Key tensor of shape (N, num_kv_heads, head_dim)
            v: Value tensor of shape (N, num_kv_heads, head_dim)

        Returns:
            Attention output tensor of shape (N, num_heads, head_dim)

        Note:
            The method interacts with the global inference context to:
            - Store new K/V pairs to cache during prefill
            - Use cached K/V pairs for decode operations
            - Handle prefix caching when available
        """
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # Store K/V to cache if cache is initialized
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if _FLASH_ATTN_AVAILABLE:
            if context.is_prefill:
                # Prefill phase: process entire prompt sequence
                # In prefill, if a prefix cache exists we use cached KV tensors
                if context.block_tables is not None:  # prefix cache
                    k, v = k_cache, v_cache

                attn_out: torch.Tensor = flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    max_seqlen_q=context.max_seqlen_q,
                    cu_seqlens_q=context.cum_seqlens_q,
                    max_seqlen_k=context.max_seqlen_k,
                    cu_seqlens_k=context.cum_seqlens_k,
                    softmax_scale=self.scale,
                    causal=True,
                    block_table=context.block_tables,
                )
            else:
                # Decode phase: generate single token using cached K/V
                attn_out = flash_attn_with_kvcache(
                    q.unsqueeze(1),
                    k_cache,
                    v_cache,
                    cache_seqlens=context.context_lens,
                    block_table=context.block_tables,
                    softmax_scale=self.scale,
                    causal=True,
                )
        else:
            # Fallback to standard attention when FlashAttention is not available
            # This is less efficient but ensures compatibility
            if context.is_prefill:
                # For prefill, we need to reshape tensors for standard attention
                # This is a simplified implementation that doesn't handle all edge cases
                # but provides basic functionality when FlashAttention is unavailable
                batch_size = context.cum_seqlens_q.size(0) - 1
                q_reshaped = q.view(batch_size, -1, self.num_heads,
                                    self.head_dim)
                k_reshaped = k.view(batch_size, -1, self.num_kv_heads,
                                    self.head_dim)
                v_reshaped = v.view(batch_size, -1, self.num_kv_heads,
                                    self.head_dim)

                # Repeat k/v heads to match q heads if needed (for MQA/GQA)
                if self.num_kv_heads != self.num_heads:
                    k_reshaped = k_reshaped.repeat_interleave(
                        self.num_heads // self.num_kv_heads, dim=2)
                    v_reshaped = v_reshaped.repeat_interleave(
                        self.num_heads // self.num_kv_heads, dim=2)

                # Compute attention scores
                attn_weights = torch.matmul(
                    q_reshaped, k_reshaped.transpose(-2, -1)) * self.scale

                # Apply causal mask
                seq_len = q_reshaped.size(1)
                mask = torch.triu(torch.ones(seq_len,
                                             seq_len,
                                             device=attn_weights.device),
                                  diagonal=1).bool()
                attn_weights.masked_fill_(mask, float('-inf'))

                # Softmax and attention output
                attn_probs = torch.softmax(attn_weights, dim=-1)
                attn_out = torch.matmul(attn_probs, v_reshaped)

                # Extract last token output for each sequence
                attn_out = attn_out[:, -1, :, :].contiguous().view(
                    -1, self.num_heads, self.head_dim)
            else:
                # For decode, we use cached k/v
                batch_size = q.size(0)

                # Get cached k/v from previous steps
                cached_k = []
                cached_v = []
                for i in range(batch_size):
                    seq_len = context.context_lens[i]
                    block_table = context.block_tables[i]
                    # Collect cached k/v blocks for this sequence
                    k_blocks = [
                        k_cache[:, block_idx] for block_idx in block_table
                        if block_idx != -1
                    ]
                    v_blocks = [
                        v_cache[:, block_idx] for block_idx in block_table
                        if block_idx != -1
                    ]

                    if k_blocks:
                        # Concatenate blocks and truncate to actual sequence length
                        cached_k_seq = torch.cat(k_blocks, dim=0)[:seq_len]
                        cached_v_seq = torch.cat(v_blocks, dim=0)[:seq_len]
                    else:
                        # No cached k/v yet
                        cached_k_seq = k[i].unsqueeze(0)
                        cached_v_seq = v[i].unsqueeze(0)

                    cached_k.append(cached_k_seq)
                    cached_v.append(cached_v_seq)

                # Pad to maximum sequence length
                max_len = max(k_seq.size(0) for k_seq in cached_k)
                cached_k_padded = torch.zeros(batch_size,
                                              max_len,
                                              self.num_kv_heads,
                                              self.head_dim,
                                              device=q.device)
                cached_v_padded = torch.zeros(batch_size,
                                              max_len,
                                              self.num_kv_heads,
                                              self.head_dim,
                                              device=q.device)

                for i, (k_seq, v_seq) in enumerate(zip(cached_k, cached_v)):
                    cached_k_padded[i, :k_seq.size(0)] = k_seq
                    cached_v_padded[i, :v_seq.size(0)] = v_seq

                # Expand q to match k/v dimensions
                q_expanded = q.unsqueeze(1)  # [batch, 1, num_heads, head_dim]

                # Repeat k/v heads to match q heads if needed (for MQA/GQA)
                if self.num_kv_heads != self.num_heads:
                    cached_k_padded = cached_k_padded.repeat_interleave(
                        self.num_heads // self.num_kv_heads, dim=2)
                    cached_v_padded = cached_v_padded.repeat_interleave(
                        self.num_heads // self.num_kv_heads, dim=2)

                # Compute attention scores
                attn_weights = torch.matmul(
                    q_expanded, cached_k_padded.transpose(-2, -1)) * self.scale

                # Apply causal mask (not needed for decode with single token)

                # Softmax and attention output
                attn_probs = torch.softmax(attn_weights, dim=-1)
                attn_out = torch.matmul(attn_probs, cached_v_padded)

                # Remove sequence dimension
                attn_out = attn_out.squeeze(1)

        return attn_out
