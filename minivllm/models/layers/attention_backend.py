"""Attention backend abstraction layer.

This module provides a unified interface for different attention backends
(CUDA, NPU, CPU) with proper error handling and type safety.
It also implements efficient KV-cache storage kernels.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

# Optional imports for NPU Flash Attention
try:
    from minivllm.models.layers.npu_flash_attention import (
        SPARSE_MODE,
        get_attn_mask_npu,
        npu_flash_attn_func,
    )
except ImportError:
    SPARSE_MODE = 0
    get_attn_mask_npu = None

# Optional imports for NPU unified inference
try:
    from torch_npu import npu_fused_infer_attention_score
except ImportError:
    npu_fused_infer_attention_score = None

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
                   or (num_tokens, num_heads, head_dim) for flattened inputs.
            key: Key tensor of shape (batch_size, num_kv_heads, seq_len, head_dim)
                 or (num_tokens, num_kv_heads, head_dim) for flattened inputs.
            value: Value tensor of shape (batch_size, num_kv_heads, seq_len, head_dim)
                   or (num_tokens, num_kv_heads, head_dim) for flattened inputs.
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
        """Standard attention computation using PyTorch operations.

        Handles both 4D (batch, heads, seq, dim) and 3D (tokens, heads, dim) inputs.
        For 3D inputs, assumes flattened batch (effectively batch_size=1).
        """
        # Handle 3D inputs (num_tokens, num_heads, head_dim) from flattened batches
        is_3d = query.dim() == 3
        if is_3d:
            num_tokens, num_heads, head_dim = query.shape
            batch_size = 1
            seq_len = num_tokens
            # Reshape to (batch, num_heads, seq_len, head_dim)
            query = query.view(batch_size, num_heads, seq_len, head_dim)
            key = key.view(batch_size, key.shape[1], seq_len, head_dim)
            value = value.view(batch_size, value.shape[1], seq_len, head_dim)
        else:
            batch_size, num_heads, seq_len, head_dim = query.shape

        _, num_kv_heads, kv_seq_len, _ = key.shape

        # Handle GQA/MQA by repeating key/value heads
        # Note: SDPA supports broadcasting, so explicit repeat might be unnecessary for some versions.
        # But for safety on NPU, we keep it or rely on broadcasting if verified.
        # Let's try explicit repeat first to ensure correctness, then optimize if needed.
        if num_kv_heads != num_heads:
            repeat_factor = num_heads // num_kv_heads
            key = key.repeat_interleave(repeat_factor, dim=1)
            value = value.repeat_interleave(repeat_factor, dim=1)

        # Use PyTorch 2.0+ scaled_dot_product_attention (SDPA)
        # This is optimized for NPU (and CUDA/CPU) and avoids manual mask creation
        if attention_mask is not None:
            # If custom mask provided, use it (is_causal=False to avoid conflict)
            output = F.scaled_dot_product_attention(query,
                                                    key,
                                                    value,
                                                    attn_mask=attention_mask,
                                                    is_causal=False)
        else:
            # Use is_causal flag
            output = F.scaled_dot_product_attention(query,
                                                    key,
                                                    value,
                                                    is_causal=is_causal)

        # If output is not contiguous, make it contiguous? SDPA returns contiguous?
        # Standard implementation returned matmul result which is contiguous.

        if is_3d:
            output = output.view(num_tokens, num_heads, head_dim)

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
        # NPU optimization: use int32 indices for better compatibility
        if k_cache.device.type == 'npu':
            valid_slots = valid_slots.to(torch.int32)

        # Reshape key/value to [num_valid, hidden_size]
        # key is [batch, num_heads, head_dim]
        valid_key = key[valid_mask].view(-1, hidden_size)
        valid_value = value[valid_mask].view(-1, hidden_size)

        # Ensure valid_key/valid_value match cache dtype
        if valid_key.dtype != k_cache.dtype:
            valid_key = valid_key.to(k_cache.dtype)
        if valid_value.dtype != v_cache.dtype:
            valid_value = valid_value.to(v_cache.dtype)

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

    This implementation uses NPU-specific optimizations for Ascend hardware,
    specifically leveraging Flash Attention (npu_flash_attn_func) when available.
    It handles necessary layout transformations (BNSD <-> BSND) transparently.
    """

    def __init__(self):
        """Initialize NPU attention backend."""
        self._npu_available = self._check_npu_availability()
        self._fallback_backend = StandardAttentionBackend()
        self.npu_fused_infer_attention_score = npu_fused_infer_attention_score

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
            q_dtype = query.dtype
            k_dtype = key.dtype
            v_dtype = value.dtype
            if q_dtype != torch.bfloat16:
                query = query.to(torch.bfloat16)
            if k_dtype != torch.bfloat16:
                key = key.to(torch.bfloat16)
            if v_dtype != torch.bfloat16:
                value = value.to(torch.bfloat16)
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)
            output = npu_flash_attn_func(
                q,
                k,
                v,
                dropout_p=0.0,
                softmax_scale=None,
                causal=is_causal,
                input_layout='BSND',
            )

            if q_dtype != torch.bfloat16:
                output = output.to(q_dtype)
            return output.transpose(1, 2)

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
        scale: float,
        context_lens: Optional[Tensor] = None,
        is_prefill: bool = False,
    ) -> Tensor:
        """
        Unified inference interface for NPU.
        Automatically selects between prefill and decode kernels.
        """
        batch_size = query.shape[0]

        # Prepare sequence lengths
        if context_lens is not None:
            # Ensure context_lens is a list of ints
            if isinstance(context_lens, torch.Tensor):
                kv_seq_lens = context_lens.tolist()
            else:
                kv_seq_lens = list(context_lens)
        else:
            # Fallback to fixed seq_len if not provided
            kv_seq_lens = [seq_len] * batch_size

        if is_prefill:
            # In prefill, query length equals context length
            q_seq_lens = kv_seq_lens

            # Handle packed prefill input: (TotalTokens, NumHeads, HeadDim)
            # We need to unflatten/pad to (Batch, NumHeads, MaxSeq, HeadDim) -> BNSD
            if context_lens is not None:
                max_seq_len = max(kv_seq_lens)
                num_heads = query.shape[1]
                head_dim = query.shape[2]

                # Create padded tensors in BNSD layout
                padded_q = torch.zeros(batch_size,
                                       num_heads,
                                       max_seq_len,
                                       head_dim,
                                       dtype=query.dtype,
                                       device=query.device)

                # Check if key/value are already in BNSD layout (4D) or need padding (3D)
                if key_cache.dim() == 4:
                    # Already BNSD, assume correct shape
                    padded_k = key_cache
                    padded_v = value_cache
                else:
                    # Flattened 3D, need padding
                    padded_k = torch.zeros(batch_size,
                                           num_kv_heads,
                                           max_seq_len,
                                           head_dim,
                                           dtype=key_cache.dtype,
                                           device=key_cache.device)
                    padded_v = torch.zeros(batch_size,
                                           num_kv_heads,
                                           max_seq_len,
                                           head_dim,
                                           dtype=value_cache.dtype,
                                           device=value_cache.device)

                start = 0
                for i, s_len in enumerate(kv_seq_lens):
                    end = start + s_len
                    # query[start:end] is (s_len, num_heads, head_dim)
                    # permute to (num_heads, s_len, head_dim)
                    padded_q[i, :, :s_len, :] = query[start:end].permute(
                        1, 0, 2)

                    if key_cache.dim() == 3:
                        padded_k[
                            i, :, :s_len, :] = key_cache[start:end].permute(
                                1, 0, 2)
                        padded_v[
                            i, :, :s_len, :] = value_cache[start:end].permute(
                                1, 0, 2)

                    start = end

                query = padded_q
                key_cache = padded_k
                value_cache = padded_v
            elif query.dim() == 3:
                # Fallback for single sequence if context_lens missing
                # Input is (s_len, num_heads, head_dim)
                # Output BNSD: (1, num_heads, s_len, head_dim)
                query = query.permute(1, 0, 2).unsqueeze(0)
                key_cache = key_cache.permute(1, 0, 2).unsqueeze(0)
                value_cache = value_cache.permute(1, 0, 2).unsqueeze(0)

        else:
            # In decode, query length is 1
            q_seq_lens = [1] * batch_size

            # Ensure query is BNSD for NPU kernel
            # Input query is (batch, num_heads, head_dim)
            if query.dim() == 3:
                query = query.unsqueeze(2)  # (batch, num_heads, 1, head_dim)

        try:
            # Basic implementation of unified inference
            # Note: This matches the signature expected by Attention.forward
            # Cast to bfloat16 for NPU kernel support
            q_dtype = query.dtype
            if q_dtype != torch.bfloat16:
                query = query.to(torch.bfloat16)
                key_cache = key_cache.to(torch.bfloat16)
                value_cache = value_cache.to(torch.bfloat16)

            # Construct attention mask for prefill (causal)
            q_len = query.shape[2]
            atten_mask = None
            if q_len > 1:
                # Causal mask for prefill: 1 for mask, 0 for keep (BOOL)
                # Use upper triangle as mask (standard for NPU FlashAttention)
                atten_mask = torch.triu(torch.ones(q_len,
                                                   q_len,
                                                   device=query.device,
                                                   dtype=torch.bool),
                                        diagonal=1)
                atten_mask = atten_mask.view(1, 1, q_len, q_len)

            try:
                out = self.npu_fused_infer_attention_score(
                    query,
                    key_cache,
                    value_cache,
                    atten_mask,
                    seq_len,
                    num_kv_heads,
                    input_layout='BNSD',
                    actual_seq_lengths=q_seq_lens,
                    actual_seq_lengths_kv=kv_seq_lens,
                    num_heads=query.shape[1],
                    sparse_mode=0 if is_prefill else SPARSE_MODE,
                    pre_tokens=65535,
                    next_tokens=0,
                    scale=scale,
                )
            except TypeError:
                out = self.npu_fused_infer_attention_score(
                    query,
                    key_cache,
                    value_cache,
                    num_heads=query.shape[1],
                    num_key_value_heads=num_kv_heads,
                    input_layout='BNSD',
                    actual_seq_lengths=q_seq_lens,
                    actual_seq_lengths_kv=kv_seq_lens,
                    scale=scale,
                    sparse_mode=0 if is_prefill else SPARSE_MODE,
                    atten_mask=atten_mask,
                    pre_tokens=65535,
                    next_tokens=0,
                )

            # Handle tuple return (attn_out, softmax_lse)
            if isinstance(out, tuple):
                out = out[0]

            if q_dtype != torch.bfloat16:
                out = out.to(q_dtype)

            if is_prefill and context_lens is not None:
                # Flatten back to (TotalTokens, NumHeads, HeadDim)
                # out is BNSD: (batch, num_heads, max_seq_len, head_dim)
                out_flat = torch.empty(sum(kv_seq_lens),
                                       out.shape[1],
                                       out.shape[3],
                                       dtype=out.dtype,
                                       device=out.device)
                start = 0
                for i, s_len in enumerate(kv_seq_lens):
                    end = start + s_len
                    # out[i, :, :s_len, :] is (num_heads, s_len, head_dim)
                    # permute to (s_len, num_heads, head_dim)
                    out_flat[start:end] = out[i, :, :s_len, :].permute(1, 0, 2)
                    start = end
                return out_flat

            # Transpose back to BNSD for attention.py compatibility (Decode phase)
            # Output is already BNSD, but attention.py expects (batch, num_heads, head_dim) for decode
            # out: (batch, num_heads, 1, head_dim)
            if out.dim() == 4 and out.shape[2] == 1:
                out = out.squeeze(2)

            return out
        except RuntimeError as e:
            if self._handle_oom(e):
                logger.warning('Retrying NPU inference after OOM handling')
                try:
                    return self.npu_fused_infer_attention_score(
                        query,
                        key_cache,
                        value_cache,
                        None,
                        seq_len,
                        num_kv_heads,
                        input_layout='BNSD',
                        actual_seq_lengths=q_seq_lens,
                        actual_seq_lengths_kv=kv_seq_lens,
                        num_heads=query.shape[1],
                        sparse_mode=SPARSE_MODE,
                        pre_tokens=65535,
                        next_tokens=0,
                        scale=scale,
                    )
                except TypeError:
                    return self.npu_fused_infer_attention_score(
                        query,
                        key_cache,
                        value_cache,
                        num_heads=query.shape[1],
                        num_key_value_heads=num_kv_heads,
                        input_layout='BNSD',
                        actual_seq_lengths=q_seq_lens,
                        actual_seq_lengths_kv=kv_seq_lens,
                        scale=scale,
                        sparse_mode=SPARSE_MODE,
                        pre_tokens=65535,
                        next_tokens=0,
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

            # Transpose to BNSD format as required by unified_inference
            return k_npu.transpose(1, 2).contiguous(), v_npu.transpose(
                1, 2).contiguous()

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
        """Store KV cache using NPU optimizations.

        Uses vectorized operations optimized for NPU memory layout.
        """
        batch_size, num_heads, head_dim = key.shape
        hidden_size = num_heads * head_dim

        # Filter out invalid slots (negative values)
        valid_mask = slot_mapping >= 0
        if not valid_mask.any():
            return

        valid_slots = slot_mapping[valid_mask]

        # Reshape key/value to [num_valid, hidden_size]
        valid_key = key[valid_mask].view(-1, hidden_size)
        valid_value = value[valid_mask].view(-1, hidden_size)

        # Reshape cache to [total_tokens, hidden_size] for direct indexing
        k_cache_reshaped = k_cache.view(-1, hidden_size)
        v_cache_reshaped = v_cache.view(-1, hidden_size)

        # Ensure valid_key/valid_value match cache dtype
        if valid_key.dtype != k_cache.dtype:
            valid_key = valid_key.to(k_cache.dtype)
        if valid_value.dtype != v_cache.dtype:
            valid_value = valid_value.to(v_cache.dtype)

        # NPU optimization: ensure contiguous memory for faster transfer
        if not valid_key.is_contiguous():
            valid_key = valid_key.contiguous()
        if not valid_value.is_contiguous():
            valid_value = valid_value.contiguous()

        # Ensure indices are int32 for NPU performance/compatibility
        if k_cache_reshaped.device.type == 'npu' and valid_slots.dtype != torch.int32:
            valid_slots = valid_slots.to(torch.int32)

        # Scatter update - Prefer index_copy_ on NPU for stability with bbfloat16
        try:
            if k_cache_reshaped.device.type == 'npu':
                # Use index_copy_ which is verified to work on NPU with bbfloat16
                k_cache_reshaped.index_copy_(0, valid_slots, valid_key)
                v_cache_reshaped.index_copy_(0, valid_slots, valid_value)
            else:
                k_cache_reshaped.index_put_((valid_slots, ), valid_key)
                v_cache_reshaped.index_put_((valid_slots, ), valid_value)
        except RuntimeError as e:
            # Fallback
            logger.warning(f'KV Cache update failed: {e}')
            # Try CPU fallback if desperate? No, that would be too slow.
            # Just raise for now.
            raise e
