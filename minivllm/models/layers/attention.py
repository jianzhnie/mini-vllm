"""Attention layer utilities and KV-cache helpers.

This module implements attention primitives which rely on external
high-performance libraries (Triton / flash-attn). The implementation
here focuses on correctness, type annotations and clean imports.

The module provides:
- High-performance attention computation using FlashAttention
- KV-cache management for efficient inference
- Support for both prefill and decode phases
- Tensor-parallel attention patterns
- Fallback implementations when FlashAttention is unavailable

Key components:
- store_kvcache_kernel: Triton kernel for efficient KV cache updates
- store_kvcache: Python wrapper for KV cache storage
- Attention: Main attention module with flash attention integration

Performance Notes:
    FlashAttention provides significant speedups (2-4x) over standard
    attention implementations by:
    - Fusing attention operations
    - Optimizing memory access patterns
    - Reducing memory allocations

    When FlashAttention is unavailable, the module falls back to a
    standard PyTorch implementation with reduced performance.

Dependencies:
    - flash-attn: Required for optimal performance
    - triton: Required for KV cache operations
    - torch: Always required
"""

import torch
from torch import nn

from minivllm.utils.context import get_context
from minivllm.utils.logger_utils import get_logger

logger = get_logger(__name__)

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

        # KV cache tensors are set later by the ModelRunner
        # Initialize as empty tensors to avoid attribute errors during initialization
        self.k_cache: torch.Tensor = torch.tensor([])
        self.v_cache: torch.Tensor = torch.tensor([])
        # Track whether KV cache has been properly initialized by ModelRunner
        self._cache_initialized: bool = False

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

        Raises:
            RuntimeError: If KV cache is accessed before being initialized by ModelRunner

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
            self._cache_initialized = True
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        elif not context.is_prefill and not self._cache_initialized:
            # In decode phase, KV cache must be initialized
            raise RuntimeError(
                'KV cache has not been initialized. Ensure ModelRunner.allocate_kv_cache() '
                'is called before inference.')

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
            import warnings
            warnings.warn(
                'FlashAttention not available. Using fallback implementation which is significantly slower. '
                'For optimal performance, install flash-attn: pip install flash-attn',
                RuntimeWarning,
                stacklevel=2)

            if context.is_prefill:
                # For prefill, process complete sequences with causal masking
                # Split concatenated batch by cumulative sequence lengths
                if context.cum_seqlens_q is None or context.cum_seqlens_k is None:
                    raise RuntimeError(
                        'Cumulative sequence lengths not set in context for prefill phase. '
                        'This is a bug in the inference pipeline.')

                batch_size = context.cum_seqlens_q.size(0) - 1
                outputs = []

                for i in range(batch_size):
                    # Extract sequence i from concatenated batch
                    q_start = context.cum_seqlens_q[i].item()
                    q_end = context.cum_seqlens_q[i + 1].item()
                    k_start = context.cum_seqlens_k[i].item()
                    k_end = context.cum_seqlens_k[i + 1].item()

                    q_seq = q[q_start:q_end]  # [seqlen_q, num_heads, head_dim]
                    k_seq = k[k_start:
                              k_end]  # [seqlen_k, num_kv_heads, head_dim]
                    v_seq = v[k_start:
                              k_end]  # [seqlen_k, num_kv_heads, head_dim]

                    # Expand for attention computation: [1, seqlen, num_heads, head_dim]
                    q_seq = q_seq.unsqueeze(0)
                    k_seq = k_seq.unsqueeze(0)
                    v_seq = v_seq.unsqueeze(0)

                    # Handle GQA/MQA: repeat k/v heads to match q heads
                    if self.num_kv_heads != self.num_heads:
                        repeat_factor = self.num_heads // self.num_kv_heads
                        k_seq = k_seq.repeat_interleave(repeat_factor, dim=2)
                        v_seq = v_seq.repeat_interleave(repeat_factor, dim=2)

                    # Compute attention: [1, seqlen_q, num_heads, seqlen_k]
                    attn_weights = torch.matmul(
                        q_seq.transpose(
                            1, 2),  # [1, num_heads, seqlen_q, head_dim]
                        k_seq.transpose(1, 2).transpose(
                            -2, -1)  # [1, num_heads, head_dim, seqlen_k]
                    ) * self.scale  # [1, num_heads, seqlen_q, seqlen_k]

                    # Apply causal mask (lower triangular)
                    seqlen_q = q_seq.size(1)
                    seqlen_k = k_seq.size(1)
                    if seqlen_q > 1:
                        # Create causal mask: allow attending to all keys up to current position
                        causal_mask = torch.triu(torch.ones(seqlen_q,
                                                            seqlen_k,
                                                            device=q.device,
                                                            dtype=torch.bool),
                                                 diagonal=seqlen_k - seqlen_q +
                                                 1)
                        attn_weights = attn_weights.masked_fill(
                            causal_mask, float('-inf'))

                    # Softmax and weighted sum
                    attn_probs = torch.softmax(attn_weights, dim=-1)
                    out_seq = torch.matmul(
                        attn_probs,  # [1, num_heads, seqlen_q, seqlen_k]
                        v_seq.transpose(
                            1, 2)  # [1, num_heads, seqlen_k, head_dim]
                    )  # [1, num_heads, seqlen_q, head_dim]

                    # Reshape back to original format
                    out_seq = out_seq.squeeze(0).transpose(
                        0, 1)  # [seqlen_q, num_heads, head_dim]
                    outputs.append(out_seq)

                # Concatenate outputs from all sequences
                attn_out = torch.cat(
                    outputs, dim=0)  # [total_tokens, num_heads, head_dim]

            else:
                # For decode, use cached k/v for efficient single-token generation
                if not self._cache_initialized:
                    raise RuntimeError(
                        'KV cache must be initialized before decode phase. '
                        'This indicates a problem with the inference pipeline.'
                    )

                batch_size = q.size(0)

                # Validate context information
                if context.context_lens is None or context.block_tables is None:
                    raise RuntimeError(
                        'Context lengths or block tables not set for decode phase. '
                        'This is a bug in the inference pipeline.')

                # Collect cached k/v for all sequences in batch
                max_seqlen = context.context_lens.max().item()
                cached_k = torch.zeros(batch_size,
                                       max_seqlen,
                                       self.num_kv_heads,
                                       self.head_dim,
                                       device=q.device,
                                       dtype=q.dtype)
                cached_v = torch.zeros(batch_size,
                                       max_seqlen,
                                       self.num_kv_heads,
                                       self.head_dim,
                                       device=q.device,
                                       dtype=q.dtype)

                block_size = k_cache.size(1)  # Tokens per block
                for i in range(batch_size):
                    seqlen = context.context_lens[i].item()
                    block_table = context.block_tables[i]

                    # Gather tokens from blocks
                    token_idx = 0
                    for block_id in block_table:
                        if block_id == -1:
                            break
                        tokens_in_block = min(block_size, seqlen - token_idx)
                        cached_k[i, token_idx:token_idx +
                                 tokens_in_block] = k_cache[:tokens_in_block,
                                                            block_id]
                        cached_v[i, token_idx:token_idx +
                                 tokens_in_block] = v_cache[:tokens_in_block,
                                                            block_id]
                        token_idx += tokens_in_block
                        if token_idx >= seqlen:
                            break

                # Handle GQA/MQA: repeat k/v heads to match q heads
                if self.num_kv_heads != self.num_heads:
                    repeat_factor = self.num_heads // self.num_kv_heads
                    cached_k = cached_k.repeat_interleave(repeat_factor, dim=2)
                    cached_v = cached_v.repeat_interleave(repeat_factor, dim=2)

                # Compute attention for single query token per sequence
                # q: [batch, num_heads, head_dim]
                # cached_k: [batch, seqlen, num_heads, head_dim]
                q_expanded = q.unsqueeze(2)  # [batch, num_heads, 1, head_dim]
                attn_weights = torch.matmul(
                    q_expanded,  # [batch, num_heads, 1, head_dim]
                    cached_k.transpose(1, 2).transpose(
                        -2, -1)  # [batch, num_heads, head_dim, seqlen]
                ) * self.scale  # [batch, num_heads, 1, seqlen]

                # Create attention mask based on actual sequence lengths
                seqlen_mask = torch.arange(max_seqlen, device=q.device).expand(
                    batch_size, max_seqlen)
                seqlen_mask = seqlen_mask >= context.context_lens.unsqueeze(1)
                seqlen_mask = seqlen_mask.unsqueeze(1).unsqueeze(
                    2)  # [batch, 1, 1, seqlen]
                attn_weights = attn_weights.masked_fill(
                    seqlen_mask, float('-inf'))

                # Softmax and weighted sum
                attn_probs = torch.softmax(attn_weights, dim=-1)
                attn_out = torch.matmul(
                    attn_probs,  # [batch, num_heads, 1, seqlen]
                    cached_v.transpose(
                        1, 2)  # [batch, num_heads, seqlen, head_dim]
                ).squeeze(2)  # [batch, num_heads, head_dim]

        return attn_out
