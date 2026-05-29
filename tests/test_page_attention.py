"""Tests for page attention module.

Verifies that page_attention_decode produces correct results compared
to a reference contiguous-attention implementation.
"""

import pytest
import torch
import torch.nn.functional as F

from minivllm.models.layers.page_attention import page_attention_decode


def _reference_decode_attention(
    q: torch.Tensor,
    k_contiguous: torch.Tensor,
    v_contiguous: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Reference: contiguous attention with padding mask."""
    _, num_heads, _ = q.shape
    _, max_seqlen, num_kv_heads, _ = k_contiguous.shape

    # Expand GQA if needed
    if num_kv_heads != num_heads:
        k = k_contiguous.repeat_interleave(num_heads // num_kv_heads, dim=2)
        v = v_contiguous.repeat_interleave(num_heads // num_kv_heads, dim=2)
    else:
        k = k_contiguous
        v = v_contiguous

    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    q_in = q.unsqueeze(2)

    mask = torch.arange(max_seqlen, device=q.device).unsqueeze(0).unsqueeze(
        1
    ).unsqueeze(2) < context_lens.unsqueeze(1).unsqueeze(2).unsqueeze(3)

    out = F.scaled_dot_product_attention(q_in, k, v, attn_mask=mask, scale=scale)
    return out.squeeze(2)


def _build_kv_cache_from_contiguous(
    k_contiguous: torch.Tensor,
    v_contiguous: torch.Tensor,
    block_tables: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter contiguous K/V into paged cache using block_tables + sequential offsets.

    Given a contiguous (batch, seqlen, heads, dim) K/V and block_tables,
    scatter tokens into a (num_blocks, block_size, heads, dim) cache so
    that page_attention_decode can gather them back.
    """
    batch_size, max_seqlen, num_kv_heads, head_dim = k_contiguous.shape
    num_blocks = int(block_tables.max().item()) + 1

    k_cache = torch.zeros(
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        device=k_contiguous.device,
        dtype=k_contiguous.dtype,
    )
    v_cache = torch.zeros(
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        device=v_contiguous.device,
        dtype=v_contiguous.dtype,
    )

    for b in range(batch_size):
        for pos in range(max_seqlen):
            logical_block = pos // block_size
            if logical_block >= block_tables.size(1):
                break
            phys_block = int(block_tables[b, logical_block].item())
            if phys_block < 0:
                break
            offset = pos % block_size
            k_cache[phys_block, offset] = k_contiguous[b, pos]
            v_cache[phys_block, offset] = v_contiguous[b, pos]

    return k_cache, v_cache


class TestPageAttentionDecode:
    """Tests for page_attention_decode."""

    def test_single_sequence_single_block(self) -> None:
        """Single sequence, all tokens in one block."""
        device = "cpu"
        batch_size, num_heads, head_dim = 1, 8, 64
        num_kv_heads = 8
        block_size = 16
        seqlen = 12

        q = torch.randn(batch_size, num_heads, head_dim, device=device)
        k_cont = torch.randn(batch_size, seqlen, num_kv_heads, head_dim, device=device)
        v_cont = torch.randn(batch_size, seqlen, num_kv_heads, head_dim, device=device)
        block_tables = torch.tensor([[0]], device=device)
        context_lens = torch.tensor([seqlen], device=device)
        scale = 1.0 / (head_dim**0.5)

        k_cache, v_cache = _build_kv_cache_from_contiguous(
            k_cont, v_cont, block_tables, block_size
        )

        out = page_attention_decode(
            q, k_cache, v_cache, block_tables, context_lens, scale
        )
        ref = _reference_decode_attention(q, k_cont, v_cont, context_lens, scale)

        assert out.shape == (batch_size, num_heads, head_dim)
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)

    def test_single_sequence_multi_block(self) -> None:
        """Single sequence spanning multiple blocks."""
        device = "cpu"
        batch_size, num_heads, head_dim = 1, 8, 64
        num_kv_heads = 8
        block_size = 4
        seqlen = 10  # 2.5 blocks → 3 blocks

        q = torch.randn(batch_size, num_heads, head_dim, device=device)
        k_cont = torch.randn(batch_size, seqlen, num_kv_heads, head_dim, device=device)
        v_cont = torch.randn(batch_size, seqlen, num_kv_heads, head_dim, device=device)
        block_tables = torch.tensor([[5, 2, 7]], device=device)  # non-contiguous
        context_lens = torch.tensor([seqlen], device=device)
        scale = 1.0 / (head_dim**0.5)

        k_cache, v_cache = _build_kv_cache_from_contiguous(
            k_cont, v_cont, block_tables, block_size
        )

        out = page_attention_decode(
            q, k_cache, v_cache, block_tables, context_lens, scale
        )
        ref = _reference_decode_attention(q, k_cont, v_cont, context_lens, scale)

        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)

    def test_multi_sequence(self) -> None:
        """Multiple sequences with different block allocations."""
        device = "cpu"
        batch_size, num_heads, head_dim = 3, 8, 64
        num_kv_heads = 8
        block_size = 8
        seqlens = [15, 8, 20]

        q = torch.randn(batch_size, num_heads, head_dim, device=device)
        max_seqlen = max(seqlens)

        k_cont = torch.zeros(
            batch_size, max_seqlen, num_kv_heads, head_dim, device=device
        )
        v_cont = torch.zeros(
            batch_size, max_seqlen, num_kv_heads, head_dim, device=device
        )
        for i, sl in enumerate(seqlens):
            k_cont[i, :sl] = torch.randn(sl, num_kv_heads, head_dim)
            v_cont[i, :sl] = torch.randn(sl, num_kv_heads, head_dim)

        # Non-contiguous block table (physical blocks in reverse order)
        block_tables = torch.tensor(
            [
                [3, 1, 0],
                [2, -1, -1],
                [5, 4, 6],
            ],
            device=device,
        )
        context_lens = torch.tensor(seqlens, device=device)
        scale = 1.0 / (head_dim**0.5)

        k_cache, v_cache = _build_kv_cache_from_contiguous(
            k_cont, v_cont, block_tables, block_size
        )

        out = page_attention_decode(
            q, k_cache, v_cache, block_tables, context_lens, scale
        )
        ref = _reference_decode_attention(q, k_cont, v_cont, context_lens, scale)

        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)

    def test_gqa(self) -> None:
        """Grouped-query attention: num_kv_heads < num_heads."""
        device = "cpu"
        batch_size, num_heads, head_dim = 2, 8, 64
        num_kv_heads = 2  # GQA: 4 query heads per KV head
        block_size = 8
        seqlens = [16, 10]

        q = torch.randn(batch_size, num_heads, head_dim, device=device)
        max_seqlen = max(seqlens)

        k_cont = torch.zeros(
            batch_size, max_seqlen, num_kv_heads, head_dim, device=device
        )
        v_cont = torch.zeros(
            batch_size, max_seqlen, num_kv_heads, head_dim, device=device
        )
        for i, sl in enumerate(seqlens):
            k_cont[i, :sl] = torch.randn(sl, num_kv_heads, head_dim)
            v_cont[i, :sl] = torch.randn(sl, num_kv_heads, head_dim)

        block_tables = torch.tensor(
            [
                [0, 1],
                [2, 3],
            ],
            device=device,
        )
        context_lens = torch.tensor(seqlens, device=device)
        scale = 1.0 / (head_dim**0.5)

        k_cache, v_cache = _build_kv_cache_from_contiguous(
            k_cont, v_cont, block_tables, block_size
        )

        out = page_attention_decode(
            q, k_cache, v_cache, block_tables, context_lens, scale
        )
        ref = _reference_decode_attention(q, k_cont, v_cont, context_lens, scale)

        assert out.shape == (batch_size, num_heads, head_dim)
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)

    def test_varying_context_lengths(self) -> None:
        """Sequences with different context lengths, shared block table."""
        device = "cpu"
        batch_size, num_heads, head_dim = 3, 4, 32
        num_kv_heads = 4
        block_size = 4
        seqlens = [4, 7, 3]  # 1 block, 2 blocks, 1 block

        q = torch.randn(batch_size, num_heads, head_dim, device=device)
        max_seqlen = max(seqlens)

        k_cont = torch.zeros(
            batch_size, max_seqlen, num_kv_heads, head_dim, device=device
        )
        v_cont = torch.zeros(
            batch_size, max_seqlen, num_kv_heads, head_dim, device=device
        )
        for i, sl in enumerate(seqlens):
            k_cont[i, :sl] = torch.randn(sl, num_kv_heads, head_dim)
            v_cont[i, :sl] = torch.randn(sl, num_kv_heads, head_dim)

        block_tables = torch.tensor(
            [
                [0, 1],
                [2, 3],
                [4, -1],
            ],
            device=device,
        )
        context_lens = torch.tensor(seqlens, device=device)
        scale = 1.0 / (head_dim**0.5)

        k_cache, v_cache = _build_kv_cache_from_contiguous(
            k_cont, v_cont, block_tables, block_size
        )

        out = page_attention_decode(
            q, k_cache, v_cache, block_tables, context_lens, scale
        )
        ref = _reference_decode_attention(q, k_cont, v_cont, context_lens, scale)

        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)

    def test_output_shape(self) -> None:
        """Verify output tensor shape."""
        device = "cpu"
        batch_size, num_heads, head_dim = 4, 8, 64
        num_kv_heads = 8
        block_size = 16

        q = torch.randn(batch_size, num_heads, head_dim, device=device)
        k_cache = torch.randn(10, block_size, num_kv_heads, head_dim, device=device)
        v_cache = torch.randn(10, block_size, num_kv_heads, head_dim, device=device)
        block_tables = torch.zeros(batch_size, 2, dtype=torch.int32, device=device)
        block_tables[:, 0] = torch.arange(batch_size)
        block_tables[:, 1] = -1
        context_lens = torch.full((batch_size,), block_size, device=device)
        scale = 1.0 / (head_dim**0.5)

        out = page_attention_decode(
            q, k_cache, v_cache, block_tables, context_lens, scale
        )

        assert out.shape == (batch_size, num_heads, head_dim)
        assert out.dtype == q.dtype

    def test_empty_cache_raises(self) -> None:
        """Empty KV cache should raise ValueError."""
        device = "cpu"
        q = torch.randn(2, 8, 64, device=device)
        k_cache = torch.tensor([], device=device)
        v_cache = torch.tensor([], device=device)
        block_tables = torch.zeros(2, 2, dtype=torch.int32, device=device)
        context_lens = torch.tensor([8, 8], device=device)
        scale = 0.125

        with pytest.raises(ValueError, match="empty"):
            page_attention_decode(
                q, k_cache, v_cache, block_tables, context_lens, scale
            )

    def test_no_valid_blocks_raises(self) -> None:
        """All-negative block tables should raise RuntimeError."""
        device = "cpu"
        q = torch.randn(2, 8, 64, device=device)
        k_cache = torch.randn(4, 16, 8, 64, device=device)
        v_cache = torch.randn(4, 16, 8, 64, device=device)
        block_tables = torch.full((2, 2), -1, dtype=torch.int32, device=device)
        context_lens = torch.tensor([8, 8], device=device)
        scale = 0.125

        with pytest.raises(RuntimeError, match="No valid KV cache"):
            page_attention_decode(
                q, k_cache, v_cache, block_tables, context_lens, scale
            )

    def test_dtype_preservation(self) -> None:
        """Output dtype should match query dtype."""
        device = "cpu"
        for dt in [torch.float32, torch.float64]:
            q = torch.randn(2, 4, 32, device=device, dtype=dt)
            k_cache = torch.randn(4, 8, 4, 32, device=device, dtype=dt)
            v_cache = torch.randn(4, 8, 4, 32, device=device, dtype=dt)
            block_tables = torch.tensor([[0, -1], [1, -1]], device=device)
            context_lens = torch.tensor([8, 8], device=device)
            scale = 1.0 / (32**0.5)

            out = page_attention_decode(
                q, k_cache, v_cache, block_tables, context_lens, scale
            )
            assert out.dtype == dt

    def test_device_mismatch_tensors(self) -> None:
        """Context tensors on different device should be handled gracefully."""
        # This test only runs on CPU since we can't guarantee multi-device
        device = "cpu"
        q = torch.randn(1, 4, 32, device=device)
        # Create block_tables and context_lens on CPU explicitly
        block_tables = torch.tensor([[0]], dtype=torch.int32)
        context_lens = torch.tensor([8], dtype=torch.int32)

        k_cache = torch.randn(4, 8, 4, 32, device=device)
        v_cache = torch.randn(4, 8, 4, 32, device=device)
        scale = 1.0 / (32**0.5)

        out = page_attention_decode(
            q, k_cache, v_cache, block_tables, context_lens, scale
        )
        assert out.shape == (1, 4, 32)
