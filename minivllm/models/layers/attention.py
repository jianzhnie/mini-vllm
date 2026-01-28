"""Attention layer utilities and KV-cache helpers.

This module implements attention primitives which rely on external
high-performance libraries (Triton / flash-attn). The implementation
here focuses on correctness, type annotations, clean imports, and
multi-device support.

The module provides:
- High-performance attention computation using FlashAttention (CUDA/NPU)
- KV-cache management for efficient inference
- Support for both prefill and decode phases
- Tensor-parallel attention patterns
- Fallback implementations when FlashAttention is unavailable
- Multi-device support (CUDA, NPU, XPU, etc.)

Key components:
- store_kvcache_kernel: Triton kernel for efficient KV cache updates
- store_kvcache: Python wrapper for KV cache storage
- Attention: Main attention module with flash attention integration
- Helper methods for GQA/MQA and attention computation

Performance Notes:
    FlashAttention provides significant speedups (2-4x) over standard
    attention implementations by:
    - Fusing attention operations
    - Optimizing memory access patterns
    - Reducing memory allocations

    When FlashAttention is unavailable, the module falls back to a
    standard PyTorch implementation with reduced performance but full
    device compatibility.

Dependencies:
    - flash-attn: Required for optimal performance on CUDA devices
    - transformers (NPU Flash Attention): Required for NPU devices
    - triton: Required for KV cache operations
    - torch: Always required
"""

import math
from typing import Any, Tuple

import torch
from torch import nn
from transformers.utils import is_torch_npu_available

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

# Try to import FlashAttention for CUDA devices
_FLASH_ATTN_AVAILABLE = False
flash_attn_varlen_func = None
flash_attn_with_kvcache = None

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    # Try NPU Flash Attention if CUDA FlashAttention is not available
    if is_torch_npu_available():
        try:
            from transformers.integrations.npu_flash_attention import \
                npu_flash_attn_varlen_func as flash_attn_varlen_func
            _FLASH_ATTN_AVAILABLE = True
            logger.info('Using NPU Flash Attention')
        except ImportError:
            logger.warning(
                'FlashAttention not available. Falling back to standard attention.'
            )
    else:
        logger.warning(
            'FlashAttention not available. Falling back to standard attention.'
        )

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
    """Store key-value pairs to cache using Triton kernel or PyTorch fallback.

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

    if _TRITON_AVAILABLE:
        # Use Triton kernel for optimal performance
        store_kvcache_kernel[(batch_size, )](key, key.stride(0), value,
                                             value.stride(0), k_cache, v_cache,
                                             slot_mapping, hidden_size)
    else:
        # PyTorch fallback implementation
        for i in range(batch_size):
            slot = slot_mapping[i].item()
            if slot != -1:
                # Reshape key and value to flat tensors for indexing
                key_flat = key[i].view(hidden_size)
                value_flat = value[i].view(hidden_size)

                # Calculate cache offsets
                cache_start = slot * hidden_size
                cache_end = cache_start + hidden_size

                # Store to cache
                k_cache.view(-1)[cache_start:cache_end] = key_flat
                v_cache.view(-1)[cache_start:cache_end] = value_flat


class Attention(nn.Module):
    """Flash Attention module with KV-cache support and multi-device compatibility.

    This class provides high-performance attention computation using FlashAttention
    (CUDA/NPU) with integrated KV-cache management for efficient inference. It supports
    both prefill (prompt processing) and decode (token generation) phases, with automatic
    fallback to standard PyTorch attention when FlashAttention is unavailable.

    The module expects pre-split query/key/value tensors with specific shapes:
    - Query: (N, num_heads, head_dim)
    - Key/Value: (N, num_kv_heads, head_dim)

    Device Support:
        - CUDA: Uses flash-attn library for optimal performance
        - NPU: Uses transformers NPU Flash Attention integration
        - Other devices: Falls back to standard PyTorch attention

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
        _cache_initialized: Flag indicating if KV cache has been initialized

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
        self.scale: float = float(scale) or 1.0 / math.sqrt(head_dim)
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

        if _FLASH_ATTN_AVAILABLE and flash_attn_varlen_func is not None:
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
                if flash_attn_with_kvcache is None:
                    attn_out = self._fallback_attention(q, k, v, context)
                else:
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
            attn_out = self._fallback_attention(q, k, v, context)

        return attn_out

    def _fallback_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context: Any,
    ) -> torch.Tensor:
        """Fallback attention implementation when FlashAttention is not available.

        This method provides a standard PyTorch implementation of attention
        computation. It is significantly slower than FlashAttention but ensures
        compatibility across all devices.

        Args:
            q: Query tensor of shape (N, num_heads, head_dim)
            k: Key tensor of shape (N, num_kv_heads, head_dim)
            v: Value tensor of shape (N, num_kv_heads, head_dim)
            context: Inference context containing sequence information

        Returns:
            Attention output tensor of shape (N, num_heads, head_dim)
        """
        import warnings
        warnings.warn(
            'FlashAttention not available. Using fallback implementation which is significantly slower. '
            'For optimal performance, install flash-attn: pip install flash-attn',
            RuntimeWarning,
            stacklevel=3)

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
                k_seq = k[k_start:k_end]  # [seqlen_k, num_kv_heads, head_dim]
                v_seq = v[k_start:k_end]  # [seqlen_k, num_kv_heads, head_dim]

                # Expand for attention computation: [1, seqlen, num_heads, head_dim]
                q_seq = q_seq.unsqueeze(0)
                k_seq = k_seq.unsqueeze(0)
                v_seq = v_seq.unsqueeze(0)

                # Handle GQA/MQA: repeat k/v heads to match q heads
                k_seq, v_seq = self._repeat_kv_heads(k_seq, v_seq)

                # Compute attention for this sequence
                out_seq = self._compute_attention_weights(
                    q_seq, k_seq, v_seq, q.device)
                outputs.append(out_seq)

            # Concatenate outputs from all sequences
            # [total_tokens, num_heads, head_dim]
            attn_out = torch.cat(outputs, dim=0)
        else:
            # For decode, use cached k/v for efficient single-token generation
            if not self._cache_initialized:
                raise RuntimeError(
                    'KV cache must be initialized before decode phase. '
                    'This indicates a problem with the inference pipeline.')

            batch_size = q.size(0)

            # Validate context information
            if context.context_lens is None or context.block_tables is None:
                raise RuntimeError(
                    'Context lengths or block tables not set for decode phase. '
                    'This is a bug in the inference pipeline.')

            # Collect cached k/v for all sequences in batch
            max_seqlen = context.context_lens.max().item()
            # Use device and dtype from input tensors for consistency
            device = q.device
            dtype = q.dtype
            cached_k = torch.zeros(batch_size,
                                   max_seqlen,
                                   self.num_kv_heads,
                                   self.head_dim,
                                   device=device,
                                   dtype=dtype)
            cached_v = torch.zeros(batch_size,
                                   max_seqlen,
                                   self.num_kv_heads,
                                   self.head_dim,
                                   device=device,
                                   dtype=dtype)

            block_size = self.k_cache.size(1)  # Tokens per block
            # Optimize block gathering with vectorized operations where possible
            for i in range(batch_size):
                seqlen = context.context_lens[i].item()
                block_table = context.block_tables[i]

                # Gather tokens from blocks
                token_idx = 0
                for block_id in block_table:
                    if block_id == -1:
                        break
                    tokens_in_block = min(block_size, seqlen - token_idx)
                    if tokens_in_block > 0:
                        cached_k[i, token_idx:token_idx +
                                 tokens_in_block] = self.k_cache[
                                     block_id, :tokens_in_block]
                        cached_v[i, token_idx:token_idx +
                                 tokens_in_block] = self.v_cache[
                                     block_id, :tokens_in_block]
                    token_idx += tokens_in_block
                    if token_idx >= seqlen:
                        break

            # Handle GQA/MQA: repeat k/v heads to match q heads
            cached_k, cached_v = self._repeat_kv_heads(cached_k.unsqueeze(0),
                                                       cached_v.unsqueeze(0))
            cached_k = cached_k.squeeze(0)
            cached_v = cached_v.squeeze(0)

            # Compute attention for single query token per sequence
            # q: [batch, num_heads, head_dim]
            # cached_k: [batch, seqlen, num_heads, head_dim]
            # q_expanded: [batch, num_heads, 1, head_dim]
            q_expanded = q.unsqueeze(2)
            # [batch, num_heads, 1, head_dim] @ [batch, num_heads, head_dim, seqlen]
            # -> [batch, num_heads, 1, seqlen]
            attn_weights = torch.matmul(
                q_expanded,
                cached_k.transpose(1, 2).transpose(-2, -1)) * self.scale

            # Create attention mask based on actual sequence lengths
            seqlen_mask = self._create_seqlen_mask(max_seqlen, batch_size,
                                                   context.context_lens,
                                                   device)
            attn_weights = attn_weights.masked_fill(seqlen_mask, float('-inf'))

            # Softmax and weighted sum
            attn_probs = torch.softmax(attn_weights, dim=-1)
            # [batch, num_heads, 1, seqlen] @ [batch, num_heads, seqlen, head_dim]
            # -> [batch, num_heads, 1, head_dim]
            attn_out = torch.matmul(attn_probs,
                                    cached_v.transpose(1, 2)).squeeze(2)

        return attn_out

    def _repeat_kv_heads(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Repeat key/value heads to match query heads for GQA/MQA.

        Args:
            k: Key tensor
            v: Value tensor

        Returns:
            Tuple of (repeated_k, repeated_v) if GQA/MQA, otherwise (k, v)
        """
        if self.num_kv_heads != self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            # Dynamically determine the dimension to repeat based on tensor shape
            # Rank 4: [batch, seqlen, num_kv_heads, head_dim] -> repeat at dim=2
            # Rank 5: [1, batch, seqlen, num_kv_heads, head_dim] -> repeat at dim=3
            head_dim = 3 if k.dim() == 5 else 2
            k = k.repeat_interleave(repeat_factor, dim=head_dim)
            v = v.repeat_interleave(repeat_factor, dim=head_dim)
        return k, v

    def split_head(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the input tensor into multiple attention heads.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            torch.Tensor: Tensor of shape (batch_size, num_heads, seq_len, head_dim).
        """
        batch_size, seq_len, num_heads, head_dim = x.size()
        return x.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    def _compute_attention_weights(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute attention weights and output for a single sequence.

        Args:
            q: Query tensor of shape [1, seqlen_q, num_heads, head_dim]
            k: Key tensor of shape [1, seqlen_k, num_heads, head_dim]
            v: Value tensor of shape [1, seqlen_k, num_heads, head_dim]
            device: Device for creating masks

        Returns:
            Output tensor of shape [seqlen_q, num_heads, head_dim]
        """
        # Compute attention: [1, num_heads, seqlen_q, seqlen_k]
        # [1, num_heads, seqlen_q, head_dim] @ [1, num_heads, seqlen_k, head_dim]
        # -> [1, num_heads, seqlen_q, seqlen_k]
        attn_weights = torch.matmul(q.transpose(1, 2),
                                    k.transpose(1, 2).transpose(
                                        -2, -1)) * self.scale
        # Apply causal mask (lower triangular)
        seqlen_q = q.size(1)
        seqlen_k = k.size(1)
        if seqlen_q > 1:
            # Create causal mask: allow attending to all keys up to current position
            causal_mask = torch.triu(torch.ones(seqlen_q,
                                                seqlen_k,
                                                device=device,
                                                dtype=torch.bool),
                                     diagonal=seqlen_k - seqlen_q + 1)
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        # Softmax and weighted sum
        attn_probs = torch.softmax(attn_weights, dim=-1)
        # [1, num_heads, seqlen_q, seqlen_k] @ [1, num_heads, seqlen_k, head_dim]
        # -> [1, num_heads, seqlen_q, head_dim]
        out_seq = torch.matmul(attn_probs, v.transpose(1, 2))
        # Reshape back to original format
        out_seq = out_seq.squeeze(0).transpose(
            0, 1)  # [seqlen_q, num_heads, head_dim]
        return out_seq

    def _create_seqlen_mask(
        self,
        max_seqlen: int,
        batch_size: int,
        context_lens: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Create attention mask based on actual sequence lengths.

        Args:
            max_seqlen: Maximum sequence length
            batch_size: Batch size
            context_lens: Tensor of actual sequence lengths
            device: Device for creating the mask

        Returns:
            Mask tensor of shape [batch, 1, 1, seqlen]
        """
        seqlen_mask = torch.arange(max_seqlen, device=device,
                                   dtype=torch.long).expand(
                                       batch_size, max_seqlen)
        seqlen_mask = seqlen_mask >= context_lens.unsqueeze(1)
        seqlen_mask = seqlen_mask.unsqueeze(1).unsqueeze(2)
        # [batch, 1, 1, seqlen]
        return seqlen_mask
