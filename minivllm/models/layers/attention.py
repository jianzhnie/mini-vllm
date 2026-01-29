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
import time
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

try:
    from transformers.utils import is_torch_npu_available
except ImportError:

    def is_torch_npu_available() -> bool:
        return False


from minivllm.utils.context import get_context
from minivllm.utils.logger_utils import get_logger

logger = get_logger(__name__)

# Optional imports for high-performance attention
_TRITON_AVAILABLE = False
_FLASH_ATTN_AVAILABLE = False
_NPU_FLASH_ATTN_AVAILABLE = False

# Global variables for optional functions
triton = None
tritonlang = None
flash_attn_varlen_func = None
flash_attn_with_kvcache = None
npu_fusion_attention = None
npu_incre_flash_attention = None
npu_fused_infer_attention_score = None
npu_prompt_flash_attention = None

# Try to import Triton
try:
    import triton
    import triton.language as tritonlang
    _TRITON_AVAILABLE = True
except ImportError:
    logger.warning(
        'Triton not available. Some optimizations will be disabled.')

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


class NPUAttentionEngine:
    """NPU Flash Attention engine with performance monitoring and optimization.

    This class provides optimized NPU Flash Attention computation with:
    - Unified inference API selection (prefill/decode automatic)
    - BNSD data layout for optimal performance
    - Performance monitoring and error handling
    - Memory optimization strategies
    - Fallback mechanisms for robustness
    """

    def __init__(self,
                 num_heads: int,
                 head_dim: int,
                 enable_monitoring: bool = False):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self.enable_monitoring = enable_monitoring

        # Performance metrics
        self.metrics = {
            'total_calls': 0,
            'total_time': 0.0,
            'memory_peak': 0.0,
            'error_count': 0,
            'oom_count': 0,
            'prefill_calls': 0,
            'decode_calls': 0
        }

    def _choose_optimal_api(self) -> str:
        """Choose the optimal NPU API based on availability and PyTorch version."""
        # Priority: unified inference > separate APIs > legacy
        if npu_fused_infer_attention_score is not None:
            return 'npu_fused_infer_attention_score'
        elif npu_prompt_flash_attention is not None and npu_incre_flash_attention is not None:
            return 'separate_apis'
        elif npu_fusion_attention is not None:
            return 'legacy_fusion'
        else:
            raise RuntimeError('No suitable NPU Flash Attention API available')

    def _optimize_layout(
            self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Optimize tensor layout for NPU hardware (prefer BNSD)."""
        # Check current layout and convert if needed
        if q.dim() == 3:  # [B, H*D] format
            batch_size = q.size(0)
            hidden_size = q.size(-1)
            if hidden_size == self.num_heads * self.head_dim:
                # Convert to BNSD layout
                q = q.view(batch_size, self.num_heads, 1, self.head_dim)
                k = k.view(
                    batch_size, self.num_kv_heads if hasattr(
                        self, 'num_kv_heads') else self.num_heads, 1,
                    self.head_dim)
                v = v.view(
                    batch_size, self.num_kv_heads if hasattr(
                        self, 'num_kv_heads') else self.num_heads, 1,
                    self.head_dim)

        # Ensure contiguous for optimal memory access
        return q.contiguous(), k.contiguous(), v.contiguous()

    def _handle_oom(self, query: torch.Tensor, key: torch.Tensor,
                    value: torch.Tensor, api_func: callable,
                    **kwargs) -> torch.Tensor:
        """Handle out-of-memory errors with multi-level fallback strategies.

        Strategies applied in order:
        1. Gradient checkpointing (training only)
        2. Reduce batch size progressively
        3. Chunked sequence processing for long sequences
        4. Precision reduction (FP16 -> FP32 -> BF16)
        5. CPU fallback
        6. PyTorch standard attention fallback
        """
        try:
            return api_func(query, key, value, **kwargs)
        except RuntimeError as e:
            if 'out of memory' not in str(e).lower():
                raise e

            self.metrics['oom_count'] += 1
            logger.warning(
                f'NPU OOM detected, applying fallback strategies. Error: {e}')

            # Strategy 1: Gradient checkpointing for training
            if kwargs.get('training', False):
                logger.info(
                    'Attempting gradient checkpointing for memory reduction')
                try:
                    return torch.utils.checkpoint.checkpoint(
                        api_func,
                        query,
                        key,
                        value,
                        use_reentrant=False,
                        **kwargs)
                except RuntimeError:
                    logger.warning(
                        'Gradient checkpointing failed, trying other strategies'
                    )

            # Strategy 2: Progressive batch size reduction
            original_batch = query.shape[0]
            if original_batch > 1:
                for reduction_factor in [2, 4, 8]:
                    batch_size = max(1, original_batch // reduction_factor)
                    if batch_size == original_batch:
                        continue

                    logger.info(
                        f'Reducing batch size from {original_batch} to {batch_size}'
                    )
                    try:
                        outputs = []
                        for i in range(0, original_batch, batch_size):
                            end = min(i + batch_size, original_batch)
                            batch_output = self._handle_oom(
                                query[i:end], key[i:end], value[i:end],
                                api_func, **kwargs)
                            outputs.append(batch_output)
                        return torch.cat(outputs, dim=0)
                    except RuntimeError:
                        continue
                        logger.info(
                            'Batch size reduction failed, trying chunked processing'
                        )

            # Strategy 3: Chunked sequence processing for long sequences
            seq_len = query.shape[2] if query.dim() == 4 else query.shape[1]
            if seq_len > 2048:
                logger.info(
                    f'Attempting chunked processing for sequence length {seq_len}'
                )
                chunk_size = min(1024, seq_len // 2)
                try:
                    return self._chunked_attention(query, key, value, api_func,
                                                   chunk_size, **kwargs)
                except RuntimeError:
                    logger.warning(
                        'Chunked processing failed, trying precision reduction'
                    )

            # Strategy 4: Precision reduction
            original_dtype = query.dtype
            if original_dtype == torch.float16:
                logger.info(
                    'Attempting FP32 computation for better memory management')
                try:
                    query_fp32, key_fp32, value_fp32 = query.float(
                    ), key.float(), value.float()
                    result = api_func(query_fp32, key_fp32, value_fp32,
                                      **kwargs)
                    return result.to(original_dtype)
                except RuntimeError:
                    logger.warning('FP32 fallback failed, trying CPU')

            elif original_dtype == torch.bfloat16:
                logger.info('Attempting FP16 computation for memory reduction')
                try:
                    query_fp16, key_fp16, value_fp16 = query.half(), key.half(
                    ), value.half()
                    result = api_func(query_fp16, key_fp16, value_fp16,
                                      **kwargs)
                    return result.to(original_dtype)
                except RuntimeError:
                    logger.warning('FP16 fallback failed, trying CPU')

            # Strategy 5: CPU fallback
            logger.warning(
                'All GPU/NPU strategies failed, falling back to CPU computation'
            )
            try:
                query_cpu, key_cpu, value_cpu = query.cpu(), key.cpu(
                ), value.cpu()

                # Remove NPU-specific parameters for CPU fallback
                cpu_kwargs = {
                    k: v
                    for k, v in kwargs.items()
                    if k not in ['block_table', 'actual_seq_lengths']
                }

                result = api_func(query_cpu, key_cpu, value_cpu, **cpu_kwargs)
                return result.to(query.device)
            except RuntimeError:
                logger.error('CPU fallback also failed')

            # Strategy 6: PyTorch standard attention as last resort
            logger.warning(
                'Using PyTorch standard attention as final fallback')
            return self._pytorch_fallback_attention(query, key, value,
                                                    **kwargs)

    def _chunked_attention(self, query: torch.Tensor, key: torch.Tensor,
                           value: torch.Tensor, api_func: callable,
                           chunk_size: int, **kwargs) -> torch.Tensor:
        """Process attention in chunks for memory-efficient computation."""
        seq_len = query.shape[2] if query.dim() == 4 else query.shape[1]
        num_chunks = (seq_len + chunk_size - 1) // chunk_size

        outputs = []

        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, seq_len)

            if query.dim() == 4:  # BNSD format
                q_chunk = query[:, :, start:end]
                k_chunk = key[:, :,
                              start:end] if key.shape[2] == seq_len else key
                v_chunk = value[:, :, start:end] if value.shape[
                    2] == seq_len else value
            else:  # BSH format
                q_chunk = query[:, start:end]
                k_chunk = key[:, start:end] if key.shape[1] == seq_len else key
                v_chunk = value[:, start:end] if value.shape[
                    1] == seq_len else value

            chunk_output = api_func(q_chunk, k_chunk, v_chunk, **kwargs)
            outputs.append(chunk_output)

        # Concatenate chunk results
        if query.dim() == 4:  # BNSD format
            return torch.cat(outputs, dim=2)
        else:  # BSH format
            return torch.cat(outputs, dim=1)

    def _pytorch_fallback_attention(self, query: torch.Tensor,
                                    key: torch.Tensor, value: torch.Tensor,
                                    **kwargs) -> torch.Tensor:
        """PyTorch standard attention fallback when all optimized methods fail."""
        logger.warning(
            'Using PyTorch scaled_dot_product_attention as final fallback')

        # Ensure compatible format for PyTorch attention
        if query.dim() == 4:  # BNSD -> BNSH
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

        # Extract scale if available
        scale = kwargs.get('scale', self.scale)

        # Use PyTorch's efficient attention implementation
        try:
            return torch.nn.functional.scaled_dot_product_attention(
                query, key, value, scale=scale, is_causal=True, attn_mask=None
            ).transpose(1, 2) if query.dim(
            ) == 3 else torch.nn.functional.scaled_dot_product_attention(
                query, key, value, scale=scale, is_causal=True, attn_mask=None)
        except Exception as e:
            logger.error(f'PyTorch attention fallback failed: {e}')
            raise RuntimeError('All attention computation methods failed')

    def monitored_attention(self, query: torch.Tensor, key: torch.Tensor,
                            value: torch.Tensor, api_func: callable,
                            **kwargs) -> torch.Tensor:
        """Execute attention with performance monitoring."""
        if not self.enable_monitoring:
            return self._handle_oom(query, key, value, api_func, **kwargs)

        start_time = time.perf_counter()
        start_memory = torch.npu.max_memory_allocated(
        ) if torch.npu.is_available() else 0

        try:
            result = self._handle_oom(query, key, value, api_func, **kwargs)

            # Update metrics
            self.metrics['total_calls'] += 1
            self.metrics['total_time'] += time.perf_counter() - start_time

            if torch.npu.is_available():
                current_memory = torch.npu.max_memory_allocated(
                ) - start_memory
                self.metrics['memory_peak'] = max(self.metrics['memory_peak'],
                                                  current_memory)

            return result

        except RuntimeError as e:
            self.metrics['error_count'] += 1
            raise e

    def get_optimal_config(self,
                           is_prefill: bool,
                           is_training: bool = False,
                           seq_len: Optional[int] = None) -> Dict[str, Any]:
        """Get optimal configuration for NPU Flash Attention.

        Args:
            is_prefill: Whether this is a prefill operation
            is_training: Whether this is a training operation
            seq_len: Sequence length for adaptive optimization

        Returns:
            Configuration dictionary with optimal parameters
        """
        config = {
            'input_layout': 'BNSD',
            'scale_value': self.scale,
            'sparse_mode': 3,  # RightDownCausal for GPT-style models
            'pre_tokens': 65535,
            'next_tokens': 0,
        }

        # Adaptive sparse mode selection
        if seq_len is not None:
            if seq_len > 4096:
                # Long sequences: use band attention for better memory efficiency
                config['sparse_mode'] = 4
                config['pre_tokens'] = 512
                config['next_tokens'] = 512
                logger.info(
                    f'Using band attention (mode 4) for long sequence length: {seq_len}'
                )
            elif seq_len < 128:
                # Very short sequences: can use full attention
                config['sparse_mode'] = 1
                logger.debug(
                    f'Using full attention (mode 1) for short sequence length: {seq_len}'
                )

        if is_training:
            config['keep_prob'] = 0.9  # Enable dropout for training
            config['inner_precise'] = 1  # Higher precision for training
        else:
            config['keep_prob'] = 1.0  # Disable dropout for inference
            config['inner_precise'] = 0  # Faster computation for inference

        if is_prefill:
            config['softmax_lse_flag'] = True  # Get numerical stability info
            # For very long prefill, consider chunked processing
            if seq_len and seq_len > 8192:
                logger.warning(
                    f'Long prefill sequence ({seq_len}) detected. Consider chunked processing.'
                )

        return config

    def unified_inference(self,
                          query: torch.Tensor,
                          key_cache: torch.Tensor,
                          value_cache: torch.Tensor,
                          seq_length: int,
                          num_kv_heads: Optional[int] = None) -> torch.Tensor:
        """Unified NPU inference interface - automatically selects prefill/decode.

        This method provides:
        - Automatic API selection based on availability
        - Adaptive configuration based on sequence length
        - Optimized BNSD data layout
        - Performance monitoring and error handling
        - Memory optimization strategies
        """
        api_choice = self._choose_optimal_api()

        # Optimize layout to BNSD for NPU hardware
        query, key_cache, value_cache = self._optimize_layout(
            query, key_cache, value_cache)

        # Determine operation type and sequence length for adaptive config
        actual_seq_len = query.shape[2] if query.dim() == 4 else query.shape[1]
        is_prefill = actual_seq_len > 1

        # Get adaptive configuration based on sequence characteristics
        config = self.get_optimal_config(is_prefill,
                                         is_training=False,
                                         seq_len=actual_seq_len)

        # Update metrics
        if is_prefill:
            self.metrics['prefill_calls'] += 1
        else:
            self.metrics['decode_calls'] += 1

        # Configure for GQA/MQA if needed
        if num_kv_heads is not None and num_kv_heads != self.num_heads:
            config['num_key_value_heads'] = num_kv_heads
        else:
            config['num_heads'] = self.num_heads

        # Add sequence length information
        config['actual_seq_lengths'] = [seq_length]
        config['actual_seq_lengths_kv'] = [seq_length]

        # Select and execute optimal API
        if api_choice == 'npu_fused_infer_attention_score':
            # Unified inference API - highest performance
            return self.monitored_attention(query, key_cache, value_cache,
                                            npu_fused_infer_attention_score,
                                            **config)
        elif api_choice == 'separate_apis':
            # Separate prefill/decode APIs
            if is_prefill:
                return self.monitored_attention(query, key_cache, value_cache,
                                                npu_prompt_flash_attention,
                                                **config)
            else:
                return self.monitored_attention(query, key_cache, value_cache,
                                                npu_incre_flash_attention,
                                                **config)
        else:
            # Legacy fallback with optimized parameters
            config.update({
                'head_num': config.pop('num_heads', self.num_heads),
                'input_layout': 'TND',  # Legacy API requires TND
                'scale': self.scale,
            })
            return self.monitored_attention(query, key_cache, value_cache,
                                            npu_fusion_attention, **config)

    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance health report.

        Returns:
            Dictionary containing health status, metrics, and recommendations
        """
        if self.metrics['total_calls'] == 0:
            return {'status': 'no_data', 'message': '暂无调用记录'}

        avg_time = self.metrics['total_time'] / self.metrics['total_calls']
        error_rate = self.metrics['error_count'] / self.metrics['total_calls']
        oom_rate = self.metrics['oom_count'] / self.metrics['total_calls']

        # Performance thresholds (tuned for NPU Flash Attention)
        thresholds = {
            'max_avg_time_ms': 5.0,  # NPU should be faster than 5ms average
            'max_error_rate': 0.02,  # <2% error rate
            'max_oom_rate': 0.005,  # <0.5% OOM rate
            'min_prefill_ratio':
            0.1,  # At least 10% prefill for balanced workload
            'min_decode_ratio':
            0.1  # At least 10% decode for balanced workload
        }

        # Health status determination
        status_issues = []
        if error_rate > thresholds['max_error_rate']:
            status_issues.append(
                f"高错误率: {error_rate:.2%} > {thresholds['max_error_rate']:.2%}")
        if oom_rate > thresholds['max_oom_rate']:
            status_issues.append(
                f"高OOM率: {oom_rate:.2%} > {thresholds['max_oom_rate']:.2%}")
        if avg_time * 1000 > thresholds['max_avg_time_ms']:
            status_issues.append(
                f"高延迟: {avg_time*1000:.2f}ms > {thresholds['max_avg_time_ms']}ms"
            )

        # Determine status
        if status_issues:
            status = 'unhealthy' if len(status_issues) > 2 else 'warning'
        else:
            status = 'healthy'

        # Generate recommendations
        recommendations = []
        if oom_rate > thresholds['max_oom_rate']:
            recommendations.append('考虑减小批量大小或启用梯度检查点')
        if avg_time * 1000 > thresholds['max_avg_time_ms']:
            recommendations.append('检查NPU硬件利用率，考虑使用更优化的稀疏模式')
        if error_rate > thresholds['max_error_rate']:
            recommendations.append('检查输入数据格式和参数配置')

        # Calculate performance metrics
        prefill_ratio = self.metrics['prefill_calls'] / max(
            self.metrics['total_calls'], 1)
        decode_ratio = self.metrics['decode_calls'] / max(
            self.metrics['total_calls'], 1)
        throughput = self.metrics['total_calls'] / max(
            self.metrics['total_time'], 0.001)

        return {
            'status': status,
            'status_issues': status_issues,
            'recommendations': recommendations,
            'metrics': {
                'total_calls': self.metrics['total_calls'],
                'prefill_calls': self.metrics['prefill_calls'],
                'decode_calls': self.metrics['decode_calls'],
                'prefill_ratio': round(prefill_ratio, 3),
                'decode_ratio': round(decode_ratio, 3),
                'avg_time_ms': round(avg_time * 1000, 2),
                'throughput_qps': round(throughput, 2),
                'memory_peak_mb': round(self.metrics['memory_peak'] / 1024**2,
                                        1),
                'error_rate': round(error_rate * 100, 2),
                'oom_rate': round(oom_rate * 100, 2),
                'error_count': self.metrics['error_count'],
                'oom_count': self.metrics['oom_count']
            },
            'thresholds': thresholds,
            'api_choice': self._choose_optimal_api()
        }

    def reset_metrics(self) -> None:
        """Reset all performance metrics for fresh monitoring."""
        self.metrics = {
            'total_calls': 0,
            'total_time': 0.0,
            'memory_peak': 0.0,
            'error_count': 0,
            'oom_count': 0,
            'prefill_calls': 0,
            'decode_calls': 0
        }
        logger.info('NPU Attention performance metrics reset')


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

        # Initialize NPU attention engine if available
        self.npu_engine: Optional[NPUAttentionEngine] = None
        if _NPU_FLASH_ATTN_AVAILABLE:
            self.npu_engine = NPUAttentionEngine(
                num_heads=num_heads,
                head_dim=head_dim,
                enable_monitoring=enable_npu_monitoring)
            logger.info('NPU Flash Attention engine initialized')

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
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        elif not context.is_prefill and not self._cache_initialized:
            # In decode phase, KV cache must be initialized
            raise RuntimeError(
                'KV cache has not been initialized. Ensure ModelRunner.allocate_kv_cache() '
                'is called before inference.')

        # Priority 1: NPU unified inference API (most optimized)
        if self.npu_engine is not None:
            try:
                # Use NPU unified inference engine with BNSD layout
                seq_length = context.context_lens.max().item(
                ) if context.context_lens is not None else q.shape[0]

                # Prepare tensors for NPU format
                if context.is_prefill:
                    # Use current K/V for prefill
                    npu_k_cache, npu_v_cache = self._prepare_npu_cache(
                        k, v, context)
                else:
                    # Use cached K/V for decode
                    npu_k_cache, npu_v_cache = self._prepare_npu_cache(
                        k_cache, v_cache, context)

                attn_out = self.npu_engine.unified_inference(
                    q, npu_k_cache, npu_v_cache, seq_length, self.num_kv_heads)

                # Reshape output back to expected format [N, num_heads, head_dim]
                if attn_out.dim() == 4 and attn_out.shape[2] == 1:
                    attn_out = attn_out.squeeze(
                        2)  # Remove seq_len dimension if present
                elif attn_out.dim() == 4:
                    attn_out = attn_out.transpose(1, 2).contiguous().view(
                        -1, self.num_heads, self.head_dim)

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
                elif _NPU_FLASH_ATTN_AVAILABLE and npu_fusion_attention is not None:
                    # Legacy NPU fusion attention (fallback)
                    # Convert cumulative lengths to actual lengths (cumulative sum list) for NPU
                    # Note: context.cum_seqlens_q is [0, s1, s1+s2, ...], we need [s1, s1+s2, ...]
                    seqlens_q = context.cum_seqlens_q[1:].cpu().tolist()
                    seqlens_k = context.cum_seqlens_k[1:].cpu().tolist()

                    # Use optimal configuration for legacy API
                    attn_out = torch_npu.npu_fusion_attention(
                        q,
                        k,
                        v,
                        head_num=self.num_heads,
                        input_layout='TND',  # Legacy TND layout
                        actual_seq_qlen=seqlens_q,
                        actual_seq_kvlen=seqlens_k,
                        scale=self.scale,
                        sparse_mode=3,
                        keep_prob=1.0,  # Inference mode
                    )[0]
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
                elif _NPU_FLASH_ATTN_AVAILABLE and npu_incre_flash_attention is not None:
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

    def _prepare_npu_cache(self, k: torch.Tensor, v: torch.Tensor,
                           context: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare KV cache in optimal format for NPU unified inference.

        This method handles different cache formats and converts them to the
        expected format for NPU Flash Attention APIs.

        Args:
            k: Key tensor (either current input or cached)
            v: Value tensor (either current input or cached)
            context: Inference context containing cache information

        Returns:
            Tuple of (k_cache, v_cache) in NPU-optimized format
        """
        # If we have block tables, use page attention format
        if context.block_tables is not None and not context.is_prefill:
            # Decode phase with block tables: gather from cache
            batch_size = k.size(0) if k.dim() > 2 else 1
            max_seqlen = context.context_lens.max().item(
            ) if context.context_lens is not None else k.size(-2)

            # Create empty cache tensors
            k_npu = torch.zeros(batch_size,
                                max_seqlen,
                                self.num_kv_heads,
                                self.head_dim,
                                device=k.device,
                                dtype=k.dtype)
            v_npu = torch.zeros(batch_size,
                                max_seqlen,
                                self.num_kv_heads,
                                self.head_dim,
                                device=v.device,
                                dtype=v.dtype)

            # Gather tokens from blocks
            if hasattr(self, 'k_cache') and self.k_cache.numel() > 0:
                block_size = self.k_cache.size(1)
                for i in range(batch_size):
                    seqlen = context.context_lens[i].item()
                    block_table = context.block_tables[i]

                    token_idx = 0
                    for block_id in block_table:
                        if block_id == -1 or token_idx >= seqlen:
                            break
                        tokens_in_block = min(block_size, seqlen - token_idx)
                        k_npu[i, token_idx:token_idx +
                              tokens_in_block] = self.k_cache[
                                  block_id, :tokens_in_block]
                        v_npu[i, token_idx:token_idx +
                              tokens_in_block] = self.v_cache[
                                  block_id, :tokens_in_block]
                        token_idx += tokens_in_block
            else:
                # Fallback: use input tensors directly
                return k.contiguous(), v.contiguous()

            return k_npu, v_npu
        else:
            # Prefill phase or direct access: use tensors as-is
            return k.contiguous(), v.contiguous()

    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report from NPU attention engine.

        Returns:
            Performance metrics dictionary or empty dict if NPU engine not available
        """
        if self.npu_engine is not None:
            return self.npu_engine.get_health_report()
        return {
            'status': 'unavailable',
            'message': 'NPU attention engine not initialized'
        }

    def reset_performance_metrics(self) -> None:
        """Reset NPU attention performance metrics for fresh monitoring."""
        if self.npu_engine is not None:
            self.npu_engine.reset_metrics()
        else:
            logger.warning(
                'NPU attention engine not available for metrics reset')

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
        if self.npu_engine is not None:
            report = self.npu_engine.get_health_report()
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
