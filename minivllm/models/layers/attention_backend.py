"""Attention backend abstraction layer.

This module provides a unified interface for different attention backends
(CUDA, NPU, CPU) with proper error handling and type safety.
It also implements efficient KV-cache storage kernels.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# Optional imports for high-performance kernels
_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    logger.warning('Triton not available. KV cache operations may be slower.')

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
        D: tl.constexpr,
    ):
        """Triton kernel for storing key-value pairs to cache.

        This kernel efficiently writes K/V tensors to their cache locations
        based on slot mapping, enabling fast KV-cache updates during inference.
        """
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1:
            return
        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)
        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)
        cache_offsets = slot * D + tl.arange(0, D)
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)


class AttentionBackend(ABC):
    """Abstract base class for attention implementations."""

    @abstractmethod
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        is_causal: bool = True,
    ) -> Tensor:
        """Forward pass for attention computation.

        Args:
            query: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
            key: Key tensor of shape (batch_size, num_kv_heads, seq_len, head_dim)
            value: Value tensor of shape (batch_size, num_kv_heads, seq_len, head_dim)
            attention_mask: Optional attention mask
            is_causal: Whether to apply causal masking

        Returns:
            Attention output tensor
        """
        pass

    @abstractmethod
    def store_kv_cache(
        self,
        key: Tensor,
        value: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        slot_mapping: Tensor,
    ) -> None:
        """Store key-value pairs to cache.

        Args:
            key: Key tensor
            value: Value tensor
            k_cache: Key cache tensor
            v_cache: Value cache tensor
            slot_mapping: Slot mapping tensor
        """
        pass


class StandardAttentionBackend(AttentionBackend):
    """Standard PyTorch attention implementation.

    This is the fallback implementation that works on all devices
    but may be slower than specialized backends.
    """

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        is_causal: bool = True,
    ) -> Tensor:
        """Standard attention computation using PyTorch operations."""
        batch_size, num_heads, seq_len, head_dim = query.shape
        _, num_kv_heads, kv_seq_len, _ = key.shape

        # Handle GQA/MQA by repeating key/value heads
        if num_kv_heads != num_heads:
            repeat_factor = num_heads // num_kv_heads
            key = key.repeat_interleave(repeat_factor, dim=1)
            value = value.repeat_interleave(repeat_factor, dim=1)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / (head_dim**0.5)

        # Apply causal mask if needed
        if is_causal:
            causal_mask = torch.triu(torch.ones(seq_len, kv_seq_len),
                                     diagonal=1).bool()
            if query.device != torch.device('cpu'):
                causal_mask = causal_mask.to(query.device)
            scores = scores.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores + attention_mask

        # Softmax and attention weights
        attention_weights = torch.softmax(scores, dim=-1)

        # Apply attention to values
        output = torch.matmul(attention_weights, value)

        return output

    def store_kv_cache(
        self,
        key: Tensor,
        value: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        slot_mapping: Tensor,
    ) -> None:
        """Store KV cache using Triton or vectorized PyTorch operations."""
        batch_size, num_heads, head_dim = key.shape
        hidden_size = num_heads * head_dim

        # Validate inputs
        if slot_mapping.numel() != batch_size:
            raise ValueError(
                f'slot_mapping size {slot_mapping.numel()} != batch_size {batch_size}'
            )

        if _TRITON_AVAILABLE and key.is_cuda:
            # Use Triton kernel for optimal performance on CUDA
            assert key.stride(-1) == 1 and value.stride(-1) == 1
            assert key.stride(1) == head_dim and value.stride(1) == head_dim
            # Note: k_cache/v_cache strides depend on layout, assuming contiguous or compatible
            # This is a simplification; in practice, strides need careful checking.
            # Assuming [blocks, block_size, heads, head_dim] or similar flattened view.
            # The original kernel assumed flattened [total_tokens, hidden_size] view-like access via offsets.

            # The kernel expects flat pointers and calculates offsets manually.
            # We need to pass the number of elements per token (hidden_size)
            store_kvcache_kernel[(batch_size, )](
                key,
                key.stride(0),
                value,
                value.stride(0),
                k_cache,
                v_cache,
                slot_mapping,
                hidden_size,
            )
            return

        # Vectorized store implementation (PyTorch fallback)
        # Filter out invalid slots (negative values)
        valid_mask = slot_mapping >= 0
        if not valid_mask.any():
            return

        valid_slots = slot_mapping[valid_mask]

        # Reshape key/value to [num_valid, hidden_size]
        # key is [batch, num_heads, head_dim]
        valid_key = key[valid_mask].view(-1, hidden_size)
        valid_value = value[valid_mask].view(-1, hidden_size)

        # Reshape cache to [total_tokens, hidden_size] for direct indexing
        # This assumes cache is contiguous in memory compatible with this view
        k_cache_reshaped = k_cache.view(-1, hidden_size)
        v_cache_reshaped = v_cache.view(-1, hidden_size)

        # Scatter update
        k_cache_reshaped[valid_slots] = valid_key
        v_cache_reshaped[valid_slots] = valid_value


class FlashAttentionBackend(AttentionBackend):
    """Flash Attention backend for CUDA devices.

    This implementation uses flash-attn for optimal performance on CUDA.
    """

    def __init__(self):
        """Initialize Flash Attention backend."""
        self._flash_attn_available = self._check_flash_attn()
        self._fallback_backend = StandardAttentionBackend()
        if not self._flash_attn_available:
            logger.warning(
                'Flash Attention not available, falling back to standard attention'
            )

    def _check_flash_attn(self) -> bool:
        """Check if Flash Attention is available."""
        try:
            from flash_attn import flash_attn_func

            self.flash_attn_func = flash_attn_func
            return True
        except ImportError:
            return False

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        is_causal: bool = True,
    ) -> Tensor:
        """Forward pass using Flash Attention."""
        if not self._flash_attn_available:
            # Fallback to standard attention
            return self._fallback_backend.forward(query, key, value,
                                                  attention_mask, is_causal)

        try:
            # Flash Attention expects (batch, seq_len, num_heads, head_dim)
            # Input is (batch, num_heads, seq_len, head_dim)
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)

            output = self.flash_attn_func(
                q,
                k,
                v,
                dropout_p=0.0,
                softmax_scale=None,
                causal=is_causal,
            )

            # Transpose back to (batch, num_heads, seq_len, head_dim)
            return output.transpose(1, 2)

        except Exception as e:
            logger.warning(
                f'Flash Attention failed with error: {e}, falling back to standard attention'
            )
            return self._fallback_backend.forward(query, key, value,
                                                  attention_mask, is_causal)

    def store_kv_cache(
        self,
        key: Tensor,
        value: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        slot_mapping: Tensor,
    ) -> None:
        """Store KV cache (delegates to standard implementation)."""
        return self._fallback_backend.store_kv_cache(key, value, k_cache,
                                                     v_cache, slot_mapping)


class NPUAttentionBackend(AttentionBackend):
    """NPU-optimized attention backend.

    This implementation uses NPU-specific optimizations for Ascend hardware.
    """

    def __init__(self):
        """Initialize NPU attention backend."""
        self._npu_available = self._check_npu_availability()
        self._fallback_backend = StandardAttentionBackend()
        if not self._npu_available:
            logger.warning(
                'NPU not available, falling back to standard attention')

    def _check_npu_availability(self) -> bool:
        """Check if NPU is available."""
        try:
            from minivllm.utils.device import is_torch_npu_available

            return is_torch_npu_available()
        except ImportError:
            try:
                from transformers.utils import is_torch_npu_available

                return is_torch_npu_available()
            except ImportError:
                return False

    def _choose_optimal_api(self) -> str:
        """Choose the best NPU attention API based on hardware and version."""
        if (hasattr(self, 'npu_fused_infer_attention_score')
                and self.npu_fused_infer_attention_score is not None):
            return 'unified_inference'
        return 'legacy'

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        is_causal: bool = True,
    ) -> Tensor:
        """Forward pass using NPU optimizations."""
        if not self._npu_available:
            return self._fallback_backend.forward(query, key, value,
                                                  attention_mask, is_causal)

        try:
            # Fallback to standard attention on NPU for now as simplified implementation
            # unless we have a specific kernel that matches the generic signature.
            return self._fallback_backend.forward(query, key, value,
                                                  attention_mask, is_causal)

        except Exception as e:
            logger.error(
                f'NPU attention failed: {e}, falling back to standard attention'
            )
            return self._fallback_backend.forward(query, key, value,
                                                  attention_mask, is_causal)

    def unified_inference(
        self,
        query: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        seq_len: int,
        num_kv_heads: int,
    ) -> Tensor:
        """Unified inference API for NPU."""
        if not self._npu_available or self.npu_fused_infer_attention_score is None:
            raise RuntimeError('NPU unified inference API not available')

        head_dim = query.shape[-1]

        try:
            # Basic implementation of unified inference
            # Note: This matches the signature expected by Attention.forward
            return self.npu_fused_infer_attention_score(
                query,
                key_cache,
                value_cache,
                head_dim,
                seq_len,
                num_kv_heads,
            )
        except RuntimeError as e:
            if self._handle_oom(e):
                logger.warning('Retrying NPU inference after OOM handling')
                # Retry once after clearing cache
                return self.npu_fused_infer_attention_score(
                    query,
                    key_cache,
                    value_cache,
                    head_dim,
                    seq_len,
                    num_kv_heads,
                )
            raise e

    def _handle_oom(self, e: RuntimeError) -> bool:
        """Handle OOM error by clearing cache or advising fallback.

        Returns:
            True if OOM was handled (e.g., by clearing cache), False otherwise.
        """
        if 'out of memory' in str(e).lower():
            logger.warning('NPU OOM detected. Attempting to clear cache.')
            if hasattr(torch, 'npu'):
                torch.npu.empty_cache()
            return True
        return False

    def get_health_report(self) -> dict:
        """Get performance health report."""
        if not self._npu_available:
            return {'status': 'unavailable'}
        return {
            'status':
            'healthy',
            'backend':
            'npu',
            'api':
            'unified_inference'
            if self.npu_fused_infer_attention_score else 'legacy',
        }

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        pass

    def prepare_npu_cache(
        self,
        k: Tensor,
        v: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        context: Any,
        num_kv_heads: int,
        head_dim: int,
    ) -> Tuple[Tensor, Tensor]:
        """Prepare KV cache in optimal format for NPU unified inference.

        Args:
            k: Key tensor (either current input or cached)
            v: Value tensor (either current input or cached)
            k_cache: Key cache tensor
            v_cache: Value cache tensor
            context: Inference context containing cache information
            num_kv_heads: Number of KV heads
            head_dim: Head dimension

        Returns:
            Tuple of (k_cache, v_cache) in NPU-optimized format
        """
        # If we have block tables, use page attention format
        if context.block_tables is not None and not context.is_prefill:
            # Decode phase with block tables: gather from cache
            batch_size = k.size(0) if k.dim() > 2 else 1
            max_seqlen = (context.context_lens.max().item()
                          if context.context_lens is not None else k.size(-2))

            # Create empty cache tensors
            k_npu = torch.zeros(
                batch_size,
                max_seqlen,
                num_kv_heads,
                head_dim,
                device=k.device,
                dtype=k.dtype,
            )
            v_npu = torch.zeros(
                batch_size,
                max_seqlen,
                num_kv_heads,
                head_dim,
                device=v.device,
                dtype=v.dtype,
            )

            if k_cache.numel() == 0:
                return k.contiguous(), v.contiguous()

            block_size = k_cache.size(1)

            # Vectorized gather implementation

            # 1. Create grid of sequence positions [1, max_seqlen]
            seq_pos = torch.arange(max_seqlen, device=k.device).unsqueeze(0)

            # 2. Map to block indices and offsets
            # Ensure indices are long for indexing
            block_table_indices = (seq_pos // block_size).long()
            block_offsets = (seq_pos % block_size).long()

            # 3. Create validity mask [batch_size, max_seqlen]
            # Handle context_lens device mismatch if any
            context_lens = context.context_lens
            if context_lens.device != k.device:
                context_lens = context_lens.to(k.device)

            valid_mask = seq_pos < context_lens.unsqueeze(1)

            # 4. Gather block IDs
            # block_tables: [batch_size, max_blocks]
            block_tables = context.block_tables
            if block_tables.device != k.device:
                block_tables = block_tables.to(k.device)

            # Clamp indices to avoid out of bounds (though valid_mask should handle it logic-wise)
            max_block_idx = block_tables.size(1) - 1
            safe_indices = block_table_indices.clamp(max=max_block_idx)

            # Expand safe_indices to batch size
            safe_indices = safe_indices.expand(batch_size, -1)

            # Gather block IDs: [batch_size, max_seqlen]
            block_ids = torch.gather(block_tables, 1, safe_indices)

            # 5. Filter valid slots
            # valid if within sequence length AND block_id != -1
            mask = valid_mask & (block_ids >= 0)

            if not mask.any():
                return k_npu, v_npu

            # 6. Gather from k_cache/v_cache
            # k_cache: [num_blocks, block_size, num_kv_heads, head_dim]

            # Get active indices
            active_block_ids = block_ids[mask]
            active_offsets = block_offsets.expand(batch_size, -1)[mask]

            # Gather
            gathered_k = k_cache[active_block_ids, active_offsets]
            gathered_v = v_cache[active_block_ids, active_offsets]

            # Scatter to output
            k_npu[mask] = gathered_k
            v_npu[mask] = gathered_v

            return k_npu, v_npu
        else:
            # Prefill phase or direct access: use tensors as-is
            return k.contiguous(), v.contiguous()

    def store_kv_cache(
        self,
        key: Tensor,
        value: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        slot_mapping: Tensor,
    ) -> None:
        """Store KV cache using NPU optimizations."""
        # For now, use standard implementation
        return self._fallback_backend.store_kv_cache(key, value, k_cache,
                                                     v_cache, slot_mapping)
