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
    input_layout: str = 'BSND',
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
        input_layout: Data layout ("BSND" or "BNSD")
        **kwargs: Additional arguments

    Returns:
        Output tensor from attention computation
    """
    if npu_fusion_attention is None:
        raise RuntimeError('torch_npu is not available or failed to import')

    keep_prob = 1.0 - dropout_p

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    if input_layout == 'BSND':
        head_num = q.shape[2]
    else:
        # BNSD
        head_num = q.shape[1]

    if not causal:
        output = npu_fusion_attention(q,
                                      k,
                                      v,
                                      head_num,
                                      input_layout,
                                      keep_prob=keep_prob,
                                      scale=softmax_scale)[0]
    else:
        seq_len = q.shape[1] if input_layout == 'BSND' else q.shape[2]
        attn_mask_npu = get_attn_mask_npu(q.device, size=seq_len)
        output = npu_fusion_attention(
            q,
            k,
            v,
            head_num,
            input_layout,
            keep_prob=keep_prob,
            scale=softmax_scale,
            atten_mask=attn_mask_npu,
            sparse_mode=SPARSE_MODE,
        )[0]

    return output

