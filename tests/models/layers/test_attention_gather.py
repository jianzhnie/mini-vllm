"""Tests for minivllm.models.layers.attention_gather.BufferedPageAttention.

Compares the buffered gather-then-SDPA path against a direct reference that
gathers K/V from the paged cache and runs scaled_dot_product_attention.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from minivllm.models.layers.attention_gather import BufferedPageAttention


def _reference(q, k_cache, v_cache, block_tables, context_lens, scale):
    num_heads = q.size(1)
    num_kv_heads = k_cache.size(2)
    head_dim = k_cache.size(3)
    block_size = k_cache.size(1)
    batch = q.size(0)
    max_seqlen = int(context_lens.max())

    K = torch.zeros(batch, max_seqlen, num_kv_heads, head_dim)
    V = torch.zeros(batch, max_seqlen, num_kv_heads, head_dim)
    for b in range(batch):
        for pos in range(int(context_lens[b])):
            blk = int(block_tables[b, pos // block_size])
            off = pos % block_size
            K[b, pos] = k_cache[blk, off]
            V[b, pos] = v_cache[blk, off]

    if num_kv_heads != num_heads:
        rep = num_heads // num_kv_heads
        K = K.repeat_interleave(rep, dim=2)
        V = V.repeat_interleave(rep, dim=2)
    K = K.permute(0, 2, 1, 3)
    V = V.permute(0, 2, 1, 3)
    mask = (
        torch.arange(max_seqlen).expand(batch, max_seqlen) < context_lens.unsqueeze(1)
    )
    mask = mask.unsqueeze(1).unsqueeze(2)
    out = F.scaled_dot_product_attention(
        q.unsqueeze(2), K, V, attn_mask=mask, scale=scale
    )
    return out.squeeze(2)


def _paged(num_blocks, block_size, num_kv_heads, head_dim):
    g = torch.Generator().manual_seed(0)
    k = torch.randn(
        num_blocks, block_size, num_kv_heads, head_dim, generator=g
    )
    v = torch.randn(
        num_blocks, block_size, num_kv_heads, head_dim, generator=g
    )
    return k, v


def test_mha_matches_reference():
    batch, num_heads, head_dim, block_size = 2, 2, 4, 4
    context_lens = torch.tensor([3, 2])
    block_tables = torch.tensor([[5, 0], [6, 0]], dtype=torch.int32)
    k, v = _paged(8, block_size, num_kv_heads=num_heads, head_dim=head_dim)
    q = torch.randn(batch, num_heads, head_dim)
    scale = 1.0 / head_dim**0.5

    out = BufferedPageAttention()(q, k, v, block_tables, context_lens, scale)
    assert torch.allclose(out, _reference(q, k, v, block_tables, context_lens, scale), atol=1e-5)


def test_gqa_matches_reference():
    batch, num_heads, num_kv_heads, head_dim, block_size = 2, 4, 2, 4, 4
    context_lens = torch.tensor([3, 2])
    block_tables = torch.tensor([[5, 0], [6, 0]], dtype=torch.int32)
    k, v = _paged(8, block_size, num_kv_heads=num_kv_heads, head_dim=head_dim)
    q = torch.randn(batch, num_heads, head_dim)
    scale = 1.0 / head_dim**0.5

    out = BufferedPageAttention()(q, k, v, block_tables, context_lens, scale)
    ref = _reference(q, k, v, block_tables, context_lens, scale)
    assert out.shape == (batch, num_heads, head_dim)
    assert torch.allclose(out, ref, atol=1e-5)


def test_empty_block_tables_warmup_runs():
    batch, num_heads, head_dim, block_size = 2, 2, 4, 4
    context_lens = torch.tensor([3, 2])
    block_tables = torch.zeros((batch, 0), dtype=torch.int32)
    k, v = _paged(4, block_size, num_kv_heads=num_heads, head_dim=head_dim)
    q = torch.randn(batch, num_heads, head_dim)

    out = BufferedPageAttention()(q, k, v, block_tables, context_lens, 1.0)
    assert out.shape == (batch, num_heads, head_dim)
    assert torch.isfinite(out).all()
