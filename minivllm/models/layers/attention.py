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
- NPU-optimized implementations with BNSD layout and unified inference API

Key components:
- store_kvcache_kernel: Triton kernel for efficient KV cache updates
- store_kvcache: Python wrapper for KV cache storage
- NPUAttentionEngine: NPU-optimized attention engine with monitoring
- Attention: Main attention module with flash attention integration
- Helper methods for GQA/MQA and attention computation

Performance Notes:
    FlashAttention provides significant speedups (2-4x) over standard
    attention implementations by:
    - Fusing attention operations
    - Optimizing memory access patterns
    - Reducing memory allocations

    NPU Flash Attention optimizations:
    - BNSD data layout for optimal hardware performance
    - Unified inference API with automatic prefill/decode selection
    - Sparse mode 3 for GPT-style causal attention
    - Memory-efficient PageAttention support
    - Quantized inference for memory-constrained scenarios

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
import os
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from minivllm.models.layers.attention_backend import (
    AttentionBackend,
    FlashAttentionBackend,
    NPUAttentionBackend,
    StandardAttentionBackend,
)
from minivllm.utils.context import get_context
from minivllm.utils.device import is_torch_npu_available
from minivllm.utils.logger_utils import get_logger

logger = get_logger(__name__)

# Optional imports for high-performance attention
_FLASH_ATTN_AVAILABLE = False
_NPU_FLASH_ATTN_AVAILABLE = False

# Global variables for optional functions
flash_attn_varlen_func = None
flash_attn_with_kvcache = None
npu_incre_flash_attention = None

# Try to import FlashAttention for CUDA devices
try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    logger.warning(
        "GPU FlashAttention not available. Falling back to standard attention."
    )

# Try to import native torch_npu functions
if is_torch_npu_available():
    try:
        import torch_npu

        npu_incre_flash_attention = torch_npu.npu_incre_flash_attention

        _NPU_FLASH_ATTN_AVAILABLE = True
        logger.info("NPU Flash Attention available")
    except ImportError:
        logger.warning("Native NPU functions not available")


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
        self.scale: float = float(scale) if scale != 0.0 else 1.0 / math.sqrt(head_dim)
        self.num_kv_heads: int = num_kv_heads

        # Initialize appropriate backend
        self.backend: AttentionBackend
        # NPU FA is opt-in via MINIVLLM_USE_NPU_FA=1 (known compatibility issues
        # with packed prefill 3D inputs on CANN 8.2.RC1). SDPA is well-optimized
        # on NPU and is the safe default.
        use_npu_fa = os.getenv("MINIVLLM_USE_NPU_FA", "0").lower() in {
            "1",
            "true",
            "yes",
        }
        if _NPU_FLASH_ATTN_AVAILABLE and use_npu_fa:
            self.backend = NPUAttentionBackend()
            logger.debug("NPU Flash Attention backend initialized")
        elif _FLASH_ATTN_AVAILABLE:
            self.backend = FlashAttentionBackend()
        else:
            self.backend = StandardAttentionBackend()

        # KV cache tensors are set later by the ModelRunner
        # Initialize as empty tensors to avoid attribute errors during initialization
        self.k_cache: torch.Tensor = torch.tensor([])
        self.v_cache: torch.Tensor = torch.tensor([])
        # Track whether KV cache has been properly initialized by ModelRunner
        self._cache_initialized: bool = False
        # Buffer caching for decode attention to avoid per-step allocation
        self._buf_k: torch.Tensor | None = None
        self._buf_v: torch.Tensor | None = None

    def extra_repr(self) -> str:
        return (
            f"num_heads={self.num_heads}, head_dim={self.head_dim}, "
            f"num_kv_heads={self.num_kv_heads}, scale={self.scale}, "
            f"backend={self.backend.__class__.__name__}"
        )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Apply attention computation with KV-cache management.

        This method handles both prefill and decode phases, automatically
        selecting the optimal FlashAttention function based on:
        1. NPU unified inference API (highest priority)
        2. CUDA FlashAttention
        3. Fallback PyTorch implementation

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
            - Leverage NPU BNSD layout and unified inference for optimal performance
        """
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # Ensure input tensors match cache dtype (critical for NPU operations)
        if k_cache.numel() > 0:
            target_dtype = k_cache.dtype
            if q.dtype != target_dtype:
                q = q.to(target_dtype)
            if k.dtype != target_dtype:
                k = k.to(target_dtype)
            if v.dtype != target_dtype:
                v = v.to(target_dtype)

        # Store K/V to cache if cache is initialized
        if k_cache.numel() and v_cache.numel():
            self._cache_initialized = True
            # Only store if slot_mapping is available and valid
            if context.slot_mapping is not None and context.slot_mapping.numel() > 0:
                self.backend.store_kv_cache(
                    k, v, k_cache, v_cache, context.slot_mapping
                )
        elif not context.is_prefill and not self._cache_initialized:
            # In decode phase, KV cache must be initialized
            raise RuntimeError(
                "KV cache has not been initialized. Ensure ModelRunner.allocate_kv_cache() "
                "is called before inference."
            )

        # NPU FA APIs (npu_fusion_attention, npu_fused_infer_attention_score)
        # do not support GQA (num_heads != num_kv_heads). Skip all NPU FA paths
        # for GQA models to avoid garbled output.
        _npu_fa_safe = self.num_heads == self.num_kv_heads

        # NPU Flash Attention paths (Priority 1 + 2): GQA-safe check only once
        if isinstance(self.backend, NPUAttentionBackend) and _npu_fa_safe:
            # Priority 1: NPU unified inference API (most optimized)
            # Skip NPU path during warmup (empty/all-invalid block tables)
            _bt = context.block_tables
            is_warmup = (
                not context.is_prefill
                and _bt is not None
                and _bt.numel() > 0
                and (_bt.shape[1] == 0 or not (_bt >= 0).any())
            )
            if not is_warmup:
                try:
                    seq_length = (
                        context.context_lens.max().item()
                        if context.context_lens is not None
                        else q.shape[0]
                    )

                    npu_k_cache, npu_v_cache = self.backend.prepare_npu_cache(
                        k,
                        v,
                        k_cache,
                        v_cache,
                        context,
                        self.num_kv_heads,
                        self.head_dim,
                    )

                    attn_out = self.backend.unified_inference(
                        q,
                        npu_k_cache,
                        npu_v_cache,
                        seq_length,
                        self.num_kv_heads,
                        scale=self.scale,
                        context_lens=context.context_lens,
                        is_prefill=context.is_prefill,
                    )

                    if attn_out.dim() == 4 and attn_out.shape[2] == 1:
                        attn_out = attn_out.squeeze(2)
                    elif attn_out.dim() == 4:
                        attn_out = (
                            attn_out.transpose(1, 2)
                            .contiguous()
                            .view(-1, self.num_heads, self.head_dim)
                        )

                    return attn_out

                except Exception as e:
                    logger.debug(
                        "NPU unified inference failed: %s, trying NPU fallback paths", e
                    )

            # Priority 2: NPU-specific prefill/decode paths
            if _NPU_FLASH_ATTN_AVAILABLE:
                if context.is_prefill:
                    if q.dim() == 3 and context.cum_seqlens_q is not None:
                        batch_size = context.cum_seqlens_q.size(0) - 1
                        max_seq = context.max_seqlen_q
                        padded_q = torch.zeros(
                            batch_size,
                            self.num_heads,
                            max_seq,
                            self.head_dim,
                            dtype=q.dtype,
                            device=q.device,
                        )
                        padded_k = torch.zeros(
                            batch_size,
                            self.num_kv_heads,
                            max_seq,
                            self.head_dim,
                            dtype=k.dtype,
                            device=k.device,
                        )
                        padded_v = torch.zeros(
                            batch_size,
                            self.num_kv_heads,
                            max_seq,
                            self.head_dim,
                            dtype=v.dtype,
                            device=v.device,
                        )
                        for i in range(batch_size):
                            start = context.cum_seqlens_q[i].item()
                            end = context.cum_seqlens_q[i + 1].item()
                            s_len = end - start
                            padded_q[i, :, :s_len] = q[start:end].transpose(0, 1)
                            padded_k[i, :, :s_len] = k[start:end].transpose(0, 1)
                            padded_v[i, :, :s_len] = v[start:end].transpose(0, 1)

                        try:
                            attn_out = self.backend.forward(
                                padded_q, padded_k, padded_v, is_causal=True
                            )
                            attn_out = attn_out[:, :, :max_seq].contiguous()
                            attn_out = (
                                attn_out.transpose(1, 2)
                                .contiguous()
                                .view(-1, self.num_heads, self.head_dim)
                            )
                            total_tokens = context.cum_seqlens_q[-1].item()
                            attn_out = attn_out[:total_tokens]
                            return attn_out
                        except Exception as e:
                            logger.warning("NPU prefill fallback failed: %s", e)
                else:
                    # Decode: use legacy npu_incre_flash_attention
                    # Only attempt NPU FA if block_tables are valid (no empty blocks)
                    block_tables_valid = (
                        context.block_tables is not None
                        and context.block_tables.numel() > 0
                        and (context.block_tables >= 0).all()
                    )
                    if (
                        npu_incre_flash_attention is not None
                        and k_cache.numel() > 0
                        and block_tables_valid
                    ):
                        try:
                            batch_size = q.size(0)
                            target_dtype = torch.float16
                            q_npu = q.to(target_dtype).unsqueeze(2)
                            k_cache_native = (
                                k_cache
                                if k_cache.dtype == target_dtype
                                else k_cache.to(target_dtype)
                            )
                            v_cache_native = (
                                v_cache
                                if v_cache.dtype == target_dtype
                                else v_cache.to(target_dtype)
                            )
                            attn_out = npu_incre_flash_attention(
                                q_npu,
                                k_cache_native,
                                v_cache_native,
                                num_heads=self.num_heads,
                                num_key_value_heads=self.num_kv_heads,
                                input_layout="BNSD",
                                scale_value=self.scale,
                                actual_seq_lengths=context.context_lens,
                                block_table=context.block_tables,
                            )
                            attn_out = attn_out.to(q.dtype).squeeze(2)
                            return attn_out
                        except Exception as e:
                            logger.warning("NPU decode fallback failed: %s", e)

        # Priority 3: CUDA FlashAttention
        if _FLASH_ATTN_AVAILABLE:
            if context.is_prefill:
                # Prefill phase: process entire prompt sequence
                # In prefill, if a prefix cache exists we use cached KV tensors
                if context.block_tables is not None:  # prefix cache
                    k, v = k_cache, v_cache

                if flash_attn_varlen_func is not None:
                    # Flatten BNSD to (total_tokens, num_heads, head_dim) if needed
                    if q.dim() == 4:
                        q = (
                            q.transpose(1, 2)
                            .contiguous()
                            .view(-1, self.num_heads, self.head_dim)
                        )
                        if k.dim() == 4:
                            k = (
                                k.transpose(1, 2)
                                .contiguous()
                                .view(-1, self.num_kv_heads, self.head_dim)
                            )
                            v = (
                                v.transpose(1, 2)
                                .contiguous()
                                .view(-1, self.num_kv_heads, self.head_dim)
                            )

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
                    attn_out = self._fallback_attention(q, k, v, context)
            else:
                # Decode phase: generate single token using cached K/V
                if flash_attn_with_kvcache is not None:
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
                    attn_out = self._fallback_attention(q, k, v, context)
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
        """Standard PyTorch attention fallback for all devices."""
        # No warning here — fallback is the expected path on NPU and other non-CUDA devices

        if context.is_prefill:
            # For prefill, process complete sequences with causal masking
            # Split concatenated batch by cumulative sequence lengths
            if context.cum_seqlens_q is None or context.cum_seqlens_k is None:
                raise RuntimeError(
                    "Cumulative sequence lengths not set in context for prefill phase. "
                    "This is a bug in the inference pipeline."
                )

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
                out_seq = self._compute_attention_weights(q_seq, k_seq, v_seq, q.device)
                outputs.append(out_seq)

            # Concatenate outputs from all sequences
            # [total_tokens, num_heads, head_dim]
            attn_out = torch.cat(outputs, dim=0)
        else:
            # For decode, use cached k/v for efficient single-token generation
            if not self._cache_initialized:
                raise RuntimeError(
                    "KV cache must be initialized before decode phase. "
                    "This indicates a problem with the inference pipeline."
                )

            batch_size = q.size(0)

            # Validate context information
            if context.context_lens is None or context.block_tables is None:
                raise RuntimeError(
                    "Context lengths or block tables not set for decode phase. "
                    "This is a bug in the inference pipeline."
                )

            # Collect cached k/v for all sequences in batch
            max_seqlen = context.context_lens.max().item()
            # Use device and dtype from input tensors for consistency
            device = q.device
            dtype = q.dtype

            # Reuse buffers to avoid per-step allocation
            buf_shape = (batch_size, max_seqlen, self.num_kv_heads, self.head_dim)
            if (
                self._buf_k is None
                or self._buf_k.shape != buf_shape
                or self._buf_k.dtype != dtype
            ):
                self._buf_k = torch.zeros(buf_shape, device=device, dtype=dtype)
                self._buf_v = torch.zeros(buf_shape, device=device, dtype=dtype)
            cached_k = self._buf_k.zero_()
            cached_v = self._buf_v.zero_()

            # Vectorized KV cache gather (replaces triple-nested Python for-loop)
            block_size = self.k_cache.size(1)

            # Position grid: [1, max_seqlen] → which token position we're at
            seq_pos = torch.arange(
                max_seqlen, dtype=torch.int64, device=device
            ).unsqueeze(0)

            # Map token positions to block index and intra-block offset
            block_indices = seq_pos // block_size
            block_offsets = seq_pos % block_size

            # Ensure context tensors are on the right device
            context_lens = context.context_lens
            if context_lens.device != device:
                context_lens = context_lens.to(device)
            block_tables = context.block_tables
            if block_tables.device != device:
                block_tables = block_tables.to(device)

            # Guard: empty block tables (warmup path, no blocks allocated yet)
            if block_tables.size(1) > 0:
                # Validity mask: token position < actual sequence length
                valid_mask = seq_pos < context_lens.unsqueeze(1)

                # Gather block IDs from block tables using safe clamping
                max_block_idx = block_tables.size(1) - 1
                safe_indices = block_indices.clamp(0, max_block_idx).expand(
                    batch_size, -1
                )
                block_ids = torch.gather(block_tables, 1, safe_indices)

                # Valid slots: within seqlen AND block_id >= 0
                mask = valid_mask & (block_ids >= 0)
                if mask.any():
                    active_block_ids = block_ids[mask]
                    active_offsets = block_offsets.expand(batch_size, -1)[mask]
                    cached_k[mask] = self.k_cache[active_block_ids, active_offsets]
                    cached_v[mask] = self.v_cache[active_block_ids, active_offsets]

            # Handle GQA/MQA: SDPA handles head broadcasting natively on CUDA,
            # but CPU backend requires explicit head expansion
            if self.num_kv_heads != self.num_heads:
                k_sdpa = cached_k.repeat_interleave(
                    self.num_heads // self.num_kv_heads, dim=2
                )
                v_sdpa = cached_v.repeat_interleave(
                    self.num_heads // self.num_kv_heads, dim=2
                )
            else:
                k_sdpa = cached_k
                v_sdpa = cached_v

            # Permute from [batch, seqlen, heads, dim] to [batch, heads, seqlen, dim]
            k_sdpa = k_sdpa.permute(0, 2, 1, 3)
            v_sdpa = v_sdpa.permute(0, 2, 1, 3)

            q_sdpa = q.unsqueeze(2)  # [batch, heads, 1, dim]

            # SDPA mask: True = attend, False = masked (padding)
            sdpa_mask = torch.arange(max_seqlen, device=device).expand(
                batch_size, max_seqlen
            )
            sdpa_mask = sdpa_mask < context.context_lens.unsqueeze(1)
            sdpa_mask = sdpa_mask.unsqueeze(1).unsqueeze(2)

            attn_out = F.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                attn_mask=sdpa_mask,
                scale=self.scale,
            )
            attn_out = attn_out.squeeze(2)  # [batch, heads, dim]

        return attn_out

    def _repeat_kv_heads(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        attn_weights = (
            torch.matmul(q.transpose(1, 2), k.transpose(1, 2).transpose(-2, -1))
            * self.scale
        )
        # Apply causal mask (lower triangular)
        seqlen_q = q.size(1)
        seqlen_k = k.size(1)
        if seqlen_q > 1:
            # Create causal mask: allow attending to all keys up to current position
            causal_mask = torch.triu(
                torch.ones(seqlen_q, seqlen_k, device=device, dtype=torch.bool),
                diagonal=seqlen_k - seqlen_q + 1,
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

        # Softmax and weighted sum
        attn_probs = torch.softmax(attn_weights, dim=-1)
        # [1, num_heads, seqlen_q, seqlen_k] @ [1, num_heads, seqlen_k, head_dim]
        # -> [1, num_heads, seqlen_q, head_dim]
        out_seq = torch.matmul(attn_probs, v.transpose(1, 2))
        # Reshape back to original format
        out_seq = out_seq.squeeze(0).transpose(0, 1)  # [seqlen_q, num_heads, head_dim]
        return out_seq
