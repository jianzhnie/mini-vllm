"""Page Attention — memory-efficient attention over paged KV cache.

Implements block-gather + SDPA fused attention for the decode phase,
enabling non-contiguous KV cache storage without materializing full
contiguous K/V tensors per step.

For CUDA devices with flash-attn available, the Attention module
prefers flash_attn_with_kvcache (which supports block_tables natively).
This module provides the fallback path that works on all devices.

Classes:
    PageAttention: Allocates fresh gather buffers per call (stateless).

Functions:
    page_attention_decode: Convenience wrapper around a default PageAttention instance.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


class PageAttention:
    """Page attention that caches position grids for NPU efficiency.

    Optimized with cached seq_pos tensors to avoid per-call arange allocation
    on NPU, which has high kernel launch overhead.
    """

    def __init__(self) -> None:
        self._seq_pos_cache: dict[tuple[int, torch.device], Tensor] = {}

    def __call__(
        self,
        q: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        block_tables: Tensor,
        context_lens: Tensor,
        scale: float,
    ) -> Tensor:
        """Compute attention over paged KV cache for the decode phase."""
        if k_cache.numel() == 0 or v_cache.numel() == 0:
            raise ValueError("KV cache is empty — cannot compute page attention")

        batch_size = q.size(0)
        num_heads = q.size(1)
        head_dim = q.size(2)
        num_kv_heads = k_cache.size(2)
        block_size = k_cache.size(1)

        device = q.device
        dtype = q.dtype

        if context_lens.device != device:
            context_lens = context_lens.to(device)
        if block_tables.device != device:
            block_tables = block_tables.to(device)

        if block_tables.size(1) == 0:
            return torch.zeros(
                batch_size, num_heads, head_dim, device=device, dtype=dtype
            )

        max_context = int(context_lens.max().item())

        # Cache position grid per (max_context, device)
        cache_key = (max_context, device)
        if cache_key in self._seq_pos_cache:
            seq_pos = self._seq_pos_cache[cache_key]
        else:
            seq_pos = torch.arange(
                max_context, dtype=torch.int64, device=device
            ).unsqueeze(0)
            self._seq_pos_cache[cache_key] = seq_pos

        block_indices = seq_pos // block_size
        block_offsets = seq_pos % block_size
        valid_mask = seq_pos < context_lens.unsqueeze(1)

        max_block_idx = block_tables.size(1) - 1
        safe_indices = block_indices.clamp(0, max_block_idx).expand(batch_size, -1)
        block_ids = torch.gather(block_tables, 1, safe_indices)

        mask = valid_mask & (block_ids >= 0)
        if not mask.any():
            raise RuntimeError(
                "No valid KV cache entries found. Check block_tables and context_lens."
            )

        cached_k = torch.zeros(
            batch_size,
            max_context,
            num_kv_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )
        cached_v = torch.zeros(
            batch_size,
            max_context,
            num_kv_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )

        active_block_ids = block_ids[mask]
        active_offsets = block_offsets.expand(batch_size, -1)[mask]
        cached_k[mask] = k_cache[active_block_ids, active_offsets]
        cached_v[mask] = v_cache[active_block_ids, active_offsets]

        # SDPA
        if num_kv_heads != num_heads:
            k_sdpa = cached_k.repeat_interleave(num_heads // num_kv_heads, dim=2)
            v_sdpa = cached_v.repeat_interleave(num_heads // num_kv_heads, dim=2)
        else:
            k_sdpa = cached_k
            v_sdpa = cached_v

        k_sdpa = k_sdpa.permute(0, 2, 1, 3)
        v_sdpa = v_sdpa.permute(0, 2, 1, 3)
        q_sdpa = q.unsqueeze(2)

        sdpa_mask = seq_pos < context_lens.unsqueeze(1)
        sdpa_mask = sdpa_mask.unsqueeze(1).unsqueeze(2)

        attn_out = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa, attn_mask=sdpa_mask, scale=scale
        )
        return attn_out.squeeze(2)


# Convenience wrapper: module-level function backed by a default instance.
_default_page_attention = PageAttention()


def page_attention_decode(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    block_tables: Tensor,
    context_lens: Tensor,
    scale: float,
) -> Tensor:
    """Compute attention over paged KV cache for the decode phase.

    Convenience wrapper around a default :class:`PageAttention` instance.
    See that class for full documentation.
    """
    return _default_page_attention(
        q, k_cache, v_cache, block_tables, context_lens, scale
    )
