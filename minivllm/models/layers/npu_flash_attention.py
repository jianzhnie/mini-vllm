import math
import os
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from minivllm.utils.device import is_torch_npu_available

if is_torch_npu_available():
    try:
        from torch_npu import npu_fusion_attention
    except ImportError:
        npu_fusion_attention = None

# FlashAttention2 is supported on Ascend NPU with down-right aligned causal mask by default.
# Set environment variable `NPU_FA2_SPARSE_MODE` to 2 when using top-left aligned causal mask.
TOP_LEFT_ALIGNED_CAUSAL_MASK_MODE = 2
DOWN_RIGHT_ALIGNED_CAUSAL_MASK_MODE = 3

SPARSE_MODE = int(
    os.getenv('NPU_FA2_SPARSE_MODE',
              default=str(DOWN_RIGHT_ALIGNED_CAUSAL_MASK_MODE)))
if SPARSE_MODE not in [
        TOP_LEFT_ALIGNED_CAUSAL_MASK_MODE, DOWN_RIGHT_ALIGNED_CAUSAL_MASK_MODE
]:
    raise ValueError(
        'Environment variable `NPU_FA2_SPARSE_MODE` can only be set as 2 (top-left aligned causal mask) '
        'or 3 (down-right aligned causal mask).')

ATTN_MASK_NPU_CACHE: Dict[torch.device, Tensor] = {}


def get_attn_mask_npu(device: torch.device, size: int = 2048) -> Tensor:
    """Get or create attention mask for the specified device and size.

    This function maintains a cache of attention masks to avoid unnecessary
    memory allocations. It automatically expands the mask if the requested
    size is larger than the cached one.

    Args:
        device: Device to create mask on
        size: Size of the mask (default: 2048, but will expand if needed)

    Returns:
        Triangular causal mask tensor
    """
    if device not in ATTN_MASK_NPU_CACHE or ATTN_MASK_NPU_CACHE[device].size(
            0) < size:
        # Round up size to next power of 2 or multiple of 2048 for efficiency
        # For now, just ensure it's large enough
        new_size = max(size, 2048)
        if device in ATTN_MASK_NPU_CACHE:
            current_size = ATTN_MASK_NPU_CACHE[device].size(0)
            if current_size >= new_size:
                return ATTN_MASK_NPU_CACHE[device][:size, :size]

        # Create new mask
        mask = torch.triu(torch.ones((new_size, new_size), device=device),
                          diagonal=1).bool()
        ATTN_MASK_NPU_CACHE[device] = mask

    return ATTN_MASK_NPU_CACHE[device][:size, :size]


def is_npu_fa2_top_left_aligned_causal_mask() -> bool:
    """Check if NPU FlashAttention2 uses top-left aligned causal mask.

    Returns:
        True if configured for top-left alignment, False otherwise.
    """
    return (SPARSE_MODE == TOP_LEFT_ALIGNED_CAUSAL_MASK_MODE
            if is_torch_npu_available() else False)


def npu_flash_attn_func(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    **kwargs: Any,
) -> Tensor:
    """Execute NPU FlashAttention function.

    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        dropout_p: Dropout probability
        softmax_scale: Scaling factor for softmax
        causal: Whether to apply causal masking
        **kwargs: Additional arguments

    Returns:
        Output tensor from attention computation
    """
    if npu_fusion_attention is None:
        raise RuntimeError('torch_npu is not available or failed to import')

    keep_prob = 1.0 - dropout_p

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    if not causal:
        head_num = q.shape[2]
        output = npu_fusion_attention(q,
                                      k,
                                      v,
                                      head_num,
                                      'BSND',
                                      keep_prob=keep_prob,
                                      scale=softmax_scale)[0]
    else:
        seq_len = q.shape[1]
        attn_mask_npu = get_attn_mask_npu(q.device, size=seq_len)
        head_num = q.shape[2]
        output = npu_fusion_attention(
            q,
            k,
            v,
            head_num,
            'BSND',
            keep_prob=keep_prob,
            scale=softmax_scale,
            atten_mask=attn_mask_npu,
            sparse_mode=SPARSE_MODE,
        )[0]

    return output


def npu_flash_attn_varlen_func(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    **kwargs: Any,
) -> Tensor:
    """Execute NPU FlashAttention function for variable length sequences.

    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        cu_seqlens_q: Cumulative sequence lengths for query
        cu_seqlens_k: Cumulative sequence lengths for key
        max_seqlen_q: Maximum sequence length for query
        max_seqlen_k: Maximum sequence length for key
        dropout_p: Dropout probability
        softmax_scale: Scaling factor for softmax
        causal: Whether to apply causal masking
        **kwargs: Additional arguments

    Returns:
        Output tensor from attention computation
    """
    if npu_fusion_attention is None:
        raise RuntimeError('torch_npu is not available or failed to import')

    keep_prob = 1.0 - dropout_p

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    if not causal:
        head_num = q.shape[1]
        output = npu_fusion_attention(
            q,
            k,
            v,
            head_num,
            pse=None,
            atten_mask=None,
            scale=softmax_scale,
            keep_prob=keep_prob,
            input_layout='TND',
            actual_seq_qlen=tuple(cu_seqlens_q[1:].cpu().numpy().tolist()),
            actual_seq_kvlen=tuple(cu_seqlens_k[1:].cpu().numpy().tolist()),
        )[0]
    else:
        # Determine max sequence length from cu_seqlens if not provided
        if max_seqlen_q is None and cu_seqlens_q is not None:
            max_seqlen_q = int(torch.max(torch.diff(cu_seqlens_q)).item())

        attn_mask_npu = get_attn_mask_npu(
            q.device, size=max_seqlen_q if max_seqlen_q else 2048)
        head_num = q.shape[1]
        output = npu_fusion_attention(
            q,
            k,
            v,
            head_num,
            pse=None,
            padding_mask=None,
            atten_mask=attn_mask_npu,
            scale=softmax_scale,
            keep_prob=keep_prob,
            input_layout='TND',
            actual_seq_qlen=tuple(cu_seqlens_q[1:].cpu().numpy().tolist()),
            actual_seq_kvlen=tuple(cu_seqlens_k[1:].cpu().numpy().tolist()),
            sparse_mode=SPARSE_MODE,
        )[0]

    return output
