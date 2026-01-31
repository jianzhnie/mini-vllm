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
from typing import Any, Dict, Tuple

import torch
from torch import nn

try:
    from transformers.utils import is_torch_npu_available
except ImportError:

    def is_torch_npu_available() -> bool:
        return False


from minivllm.models.layers.attention_backend import (
    AttentionBackend,
    FlashAttentionBackend,
    NPUAttentionBackend,
    StandardAttentionBackend,
)
from minivllm.utils.context import get_context
from minivllm.utils.logger_utils import get_logger

logger = get_logger(__name__)

# Optional imports for high-performance attention
_FLASH_ATTN_AVAILABLE = False
_NPU_FLASH_ATTN_AVAILABLE = False

# Global variables for optional functions
flash_attn_varlen_func = None
flash_attn_with_kvcache = None
npu_fusion_attention = None
npu_incre_flash_attention = None
npu_fused_infer_attention_score = None
npu_prompt_flash_attention = None

# Try to import FlashAttention for CUDA devices
try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    logger.warning(
        'GPU FlashAttention not available. Falling back to standard attention.'
    )

# Try to import native torch_npu functions
if is_torch_npu_available():
    try:
        import torch_npu

        npu_fusion_attention = torch_npu.npu_fusion_attention
        npu_incre_flash_attention = torch_npu.npu_incre_flash_attention

        # Check for newer unified inference API
        if hasattr(torch_npu, 'npu_fused_infer_attention_score'):
            npu_fused_infer_attention_score = torch_npu.npu_fused_infer_attention_score
            logger.info('NPU unified inference API available')

        if hasattr(torch_npu, 'npu_prompt_flash_attention'):
            npu_prompt_flash_attention = torch_npu.npu_prompt_flash_attention

        _NPU_FLASH_ATTN_AVAILABLE = True
        logger.info('NPU Flash Attention available')
    except ImportError:
        logger.warning('Native NPU functions not available')


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
        enable_npu_monitoring: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads: int = num_heads
        self.head_dim: int = head_dim
        self.scale: float = float(scale) or 1.0 / math.sqrt(head_dim)
        self.num_kv_heads: int = num_kv_heads

        # Initialize appropriate backend
        self.backend: AttentionBackend
        if _NPU_FLASH_ATTN_AVAILABLE:
            self.backend = NPUAttentionBackend(
                enable_monitoring=enable_npu_monitoring)
            logger.info('NPU Flash Attention backend initialized')
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

    def extra_repr(self) -> str:
        return (f'num_heads={self.num_heads}, head_dim={self.head_dim}, '
                f'num_kv_heads={self.num_kv_heads}, scale={self.scale}, '
                f'backend={self.backend.__class__.__name__}')

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                v: torch.Tensor) -> torch.Tensor:
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

        # Store K/V to cache if cache is initialized
        if k_cache.numel() and v_cache.numel():
            self._cache_initialized = True
            self.backend.store_kv_cache(k, v, k_cache, v_cache,
                                        context.slot_mapping)
        elif not context.is_prefill and not self._cache_initialized:
            # In decode phase, KV cache must be initialized
            raise RuntimeError(
                'KV cache has not been initialized. Ensure ModelRunner.allocate_kv_cache() '
                'is called before inference.')

        # Priority 1: NPU unified inference API (most optimized)
        if isinstance(self.backend, NPUAttentionBackend):
            try:
                # Use NPU unified inference engine with BNSD layout
                seq_length = (context.context_lens.max().item() if
                              context.context_lens is not None else q.shape[0])

                # Prepare tensors for NPU format
                if context.is_prefill:
                    # Use current K/V for prefill
                    npu_k_cache, npu_v_cache = self.backend.prepare_npu_cache(
                        k,
                        v,
                        k_cache,
                        v_cache,
                        context,
                        self.num_kv_heads,
                        self.head_dim,
                    )
                else:
                    # Use cached K/V for decode
                    npu_k_cache, npu_v_cache = self.backend.prepare_npu_cache(
                        k_cache,
                        v_cache,
                        k_cache,
                        v_cache,
                        context,
                        self.num_kv_heads,
                        self.head_dim,
                    )

                attn_out = self.backend.unified_inference(
                    q, npu_k_cache, npu_v_cache, seq_length, self.num_kv_heads)

                # Reshape output back to expected format [N, num_heads, head_dim]
                if attn_out.dim() == 4 and attn_out.shape[2] == 1:
                    attn_out = attn_out.squeeze(
                        2)  # Remove seq_len dimension if present
                elif attn_out.dim() == 4:
                    attn_out = (attn_out.transpose(1, 2).contiguous().view(
                        -1, self.num_heads, self.head_dim))

                return attn_out.contiguous()

            except Exception as e:
                logger.warning(
                    f'NPU Flash Attention failed: {e}, falling back to CUDA/PyTorch implementation'
                )

        # Priority 2: CUDA FlashAttention
        if _FLASH_ATTN_AVAILABLE:
            if context.is_prefill:
                # Prefill phase: process entire prompt sequence
                # In prefill, if a prefix cache exists we use cached KV tensors
                if context.block_tables is not None:  # prefix cache
                    k, v = k_cache, v_cache

                if flash_attn_varlen_func is not None:
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
                elif (_NPU_FLASH_ATTN_AVAILABLE
                      and npu_incre_flash_attention is not None):
                    # Legacy NPU incremental FlashAttention
                    # Reshape tensors for NPU format
                    batch_size = q.size(0)
                    q_npu = q.view(batch_size, self.num_heads,
                                   self.head_dim)  # [B, H, D]

                    # Prepare KV cache in NPU format
                    k_cache_npu = k_cache  # Should be in correct format already
                    v_cache_npu = v_cache

                    attn_out = npu_incre_flash_attention(
                        q_npu,
                        k_cache_npu,
                        v_cache_npu,
                        num_heads=self.num_heads,
                        num_key_value_heads=self.num_kv_heads,
                        input_layout='BNSD',  # Use optimal layout
                        scale_value=self.scale,
                        actual_seq_lengths=context.context_lens,
                        block_table=context.block_tables,
                    )

                    # Reshape back to original format [batch, num_heads, head_dim]
                    attn_out = attn_out.view(batch_size, self.num_heads,
                                             self.head_dim)
                else:
                    attn_out = self._fallback_attention(q, k, v, context)
        else:
            # Fallback to standard attention when FlashAttention is not available
            # This is less efficient but ensures compatibility
            attn_out = self._fallback_attention(q, k, v, context)

        return attn_out

    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report from NPU attention backend.

        Returns:
            Performance metrics dictionary or empty dict if NPU backend not available
        """
        if isinstance(self.backend, NPUAttentionBackend) and hasattr(
                self.backend, 'get_health_report'):
            return self.backend.get_health_report()
        return {
            'status': 'unavailable',
            'message': 'NPU attention backend not initialized',
        }

    def reset_performance_metrics(self) -> None:
        """Reset NPU attention performance metrics for fresh monitoring."""
        if isinstance(self.backend, NPUAttentionBackend) and hasattr(
                self.backend, 'reset_metrics'):
            self.backend.reset_metrics()
        else:
            logger.warning(
                'NPU attention backend not available for metrics reset')

    def get_optimization_suggestions(self) -> Dict[str, Any]:
        """Get optimization suggestions based on current configuration and environment.

        Returns:
            Dictionary with optimization suggestions and best practices
        """
        suggestions = {
            'environment': {},
            'configuration': {},
            'performance': {},
            'best_practices': []
        }

        # Environment suggestions
        if torch.npu.is_available():
            suggestions['environment']['npu_available'] = True
            suggestions['best_practices'].append('✅ NPU设备已就绪')

            if npu_fused_infer_attention_score is not None:
                suggestions['environment']['optimal_api'] = 'unified_inference'
                suggestions['best_practices'].append('✅ 推荐使用统一推理接口')
            else:
                suggestions['environment']['optimal_api'] = 'legacy'
                suggestions['best_practices'].append(
                    '⚠️ 建议升级到支持统一接口的PyTorch版本')
        else:
            suggestions['environment']['npu_available'] = False
            suggestions['best_practices'].append('❌ NPU设备不可用，将使用CPU/CUDA实现')

        # Configuration suggestions
        if self.head_dim % 16 != 0:
            suggestions['configuration']['head_dim_alignment'] = 'misaligned'
            suggestions['best_practices'].append(
                f'⚠️ head_dim={self.head_dim}未16对齐，建议填充到{(self.head_dim//16+1)*16}'
            )
        else:
            suggestions['configuration']['head_dim_alignment'] = 'aligned'
            suggestions['best_practices'].append('✅ head_dim已16对齐')

        if self.num_kv_heads != self.num_heads:
            ratio = self.num_heads // self.num_kv_heads
            suggestions['configuration']['gqa_ratio'] = ratio
            suggestions['best_practices'].append(f'✅ GQA优化已启用，比例={ratio}')
        else:
            suggestions['configuration']['gqa_ratio'] = 1
            suggestions['best_practices'].append('ℹ️ 标准多头注意力')

        # Performance suggestions
        if isinstance(self.backend, NPUAttentionBackend):
            report = self.backend.get_health_report()
            if report.get('status') == 'warning':
                suggestions['performance']['health'] = 'warning'
                suggestions['best_practices'].extend(
                    report.get('recommendations', []))
            elif report.get('status') == 'unhealthy':
                suggestions['performance']['health'] = 'unhealthy'
                suggestions['best_practices'].extend(
                    report.get('recommendations', []))
            else:
                suggestions['performance']['health'] = 'healthy'
                suggestions['best_practices'].append('✅ NPU注意力运行健康')

        # General best practices
        suggestions['best_practices'].extend([
            '💡 使用BNSD数据布局以获得最佳NPU性能', '💡 启用稀疏模式3用于GPT类模型', '💡 长序列考虑使用分块处理',
            '💡 内存受限时考虑量化推理'
        ])

        return suggestions

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
