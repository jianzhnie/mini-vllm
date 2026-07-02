"""Buffered gather-then-SDPA attention — the original decode attention path.

Preserved for reference and as an alternative to page_attention_decode.
This implementation reuses pre-allocated gather buffers to avoid per-step
tensor allocations, at the cost of materializing full contiguous K/V tensors.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


class BufferedPageAttention:
    """Page attention with reusable gather buffers.

    Optimized with cached position grids and pre-allocated K/V buffers
    to minimize per-step allocation overhead on NPU.
    Usage:
        attn = BufferedPageAttention()
        output = attn(q, k_cache, v_cache, block_tables, context_lens, scale)
    """

    def __init__(self) -> None:
        self._buf_k: Tensor | None = None
        self._buf_v: Tensor | None = None
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
        """Compute attention over paged KV cache with buffered gather.

        Args:
            q: Query tensor of shape (batch_size, num_heads, head_dim).
            k_cache: Key cache of shape (num_blocks, block_size, num_kv_heads, head_dim).
            v_cache: Value cache of shape (num_blocks, block_size, num_kv_heads, head_dim).
            block_tables: Block table of shape (batch_size, max_blocks_per_seq).
            context_lens: Context lengths of shape (batch_size,).
            scale: Attention scaling factor.

        Returns:
            Attention output of shape (batch_size, num_heads, head_dim).
        """
        batch_size = q.size(0)
        num_heads = q.size(1)
        num_kv_heads = k_cache.size(2)
        head_dim = k_cache.size(3)
        device = q.device
        dtype = q.dtype

        max_seqlen = int(context_lens.max().item())

        # Move context tensors to the right device if needed
        if context_lens.device != device:
            context_lens = context_lens.to(device)
        if block_tables.device != device:
            block_tables = block_tables.to(device)

        # Reuse buffers to avoid per-step allocation
        buf_shape = (batch_size, max_seqlen, num_kv_heads, head_dim)
        if (
            self._buf_k is None
            or self._buf_k.shape != buf_shape
            or self._buf_k.dtype != dtype
        ):
            self._buf_k = torch.zeros(buf_shape, device=device, dtype=dtype)
            self._buf_v = torch.zeros(buf_shape, device=device, dtype=dtype)
        assert self._buf_k is not None and self._buf_v is not None
        cached_k = self._buf_k.zero_()
        cached_v = self._buf_v.zero_()

        # Vectorized KV cache gather
        block_size = k_cache.size(1)

        # Cache position grid per (max_seqlen, device)
        cache_key = (max_seqlen, device)
        if cache_key in self._seq_pos_cache:
            seq_pos = self._seq_pos_cache[cache_key]
        else:
            seq_pos = torch.arange(
                max_seqlen, dtype=torch.int64, device=device
            ).unsqueeze(0)
            self._seq_pos_cache[cache_key] = seq_pos

        # Map token positions to block index and intra-block offset
        block_indices = seq_pos // block_size
        block_offsets = seq_pos % block_size

        # Guard: empty block tables (warmup path, no blocks allocated yet)
        if block_tables.size(1) > 0:
            # Validity mask: token position < actual sequence length
            valid_mask = seq_pos < context_lens.unsqueeze(1)

            # Gather block IDs from block tables using safe clamping
            max_block_idx = block_tables.size(1) - 1
            safe_indices = block_indices.clamp(0, max_block_idx).expand(batch_size, -1)
            block_ids = torch.gather(block_tables, 1, safe_indices)

            # Valid slots: within seqlen AND block_id >= 0
            mask = valid_mask & (block_ids >= 0)
            if mask.any():
                active_block_ids = block_ids[mask]
                active_offsets = block_offsets.expand(batch_size, -1)[mask]
                cached_k[mask] = k_cache[active_block_ids, active_offsets]
                cached_v[mask] = v_cache[active_block_ids, active_offsets]

        # Handle GQA/MQA: SDPA handles head broadcasting natively on CUDA,
        # but CPU backend requires explicit head expansion
        if num_kv_heads != num_heads:
            k_sdpa = cached_k.repeat_interleave(num_heads // num_kv_heads, dim=2)
            v_sdpa = cached_v.repeat_interleave(num_heads // num_kv_heads, dim=2)
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
        sdpa_mask = sdpa_mask < context_lens.unsqueeze(1)
        sdpa_mask = sdpa_mask.unsqueeze(1).unsqueeze(2)

        attn_out = F.scaled_dot_product_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            attn_mask=sdpa_mask,
            scale=scale,
        )
        return attn_out.squeeze(2)  # [batch, heads, dim]
