"""Regression test: dense (SDPA) fallback prefill with a prefix-cache hit.

cum_seqlens_k counts the FULL context while q/k/v carry only the NEW tokens, so
the old code sliced k out of range and produced wrong attention whenever
``num_cached > 0`` on a non-flash device (CPU/NPU-without-FA). The fallback must
instead gather the full K/V from the paged cache and attend causally over the
cached prefix.
"""

from __future__ import annotations

import torch

from minivllm.models.layers.attention import Attention
from minivllm.utils.context import reset_context, set_context

BLOCK = 4
HEADS = 2
KV_HEADS = 2
DIM = 8
SCALE = 1.0 / DIM**0.5


def _fill_cache(cache: torch.Tensor, all_tokens: torch.Tensor) -> None:
    """Scatter per-token values into a paged [num_blocks, BLOCK, kv, dim] cache."""
    for t in range(all_tokens.shape[0]):
        cache[t // BLOCK, t % BLOCK] = all_tokens[t]


def _reference_new_token_output(q_all, k_all, v_all, start: int) -> torch.Tensor:
    """Causal attention for the query rows at global positions >= start.

    Shapes: q_all/k_all/v_all are [len, heads, dim]. For query row p it attends
    to key/value rows 0..p: logits[h, t] = sum_d q[p,h,d] * k[t,h,d].
    """
    full_len = q_all.shape[0]
    out = []
    for p in range(start, full_len):
        logits = torch.einsum("hd,thd->ht", q_all[p], k_all[: p + 1]) * SCALE
        probs = torch.softmax(logits, dim=-1)
        out.append(torch.einsum("ht,thd->hd", probs, v_all[: p + 1]))
    return torch.stack(out)


def test_prefix_cached_prefill_fallback_matches_reference():
    torch.manual_seed(0)
    full_len, num_cached = 6, 2
    q_all = torch.randn(full_len, HEADS, DIM)
    k_all = torch.randn(full_len, KV_HEADS, DIM)
    v_all = torch.randn(full_len, KV_HEADS, DIM)

    k_cache = torch.zeros(2, BLOCK, KV_HEADS, DIM)
    v_cache = torch.zeros(2, BLOCK, KV_HEADS, DIM)
    _fill_cache(k_cache, k_all)
    _fill_cache(v_cache, v_all)

    attn = Attention(num_heads=HEADS, head_dim=DIM, scale=SCALE, num_kv_heads=KV_HEADS)
    attn.k_cache = k_cache
    attn.v_cache = v_cache

    # New tokens are the tail [num_cached, full_len); pass them as q/k/v.
    q = q_all[num_cached:]
    k = k_all[num_cached:]
    v = v_all[num_cached:]

    set_context(
        is_prefill=True,
        max_seqlen_q=full_len - num_cached,
        max_seqlen_k=full_len,
        cum_seqlens_q=torch.tensor([0, full_len - num_cached]),
        cum_seqlens_k=torch.tensor([0, full_len]),
        slot_mapping=None,  # cache already populated; skip the store
        block_tables=torch.tensor([[0, 1]]),
    )
    try:
        got = attn(q, k, v)
    finally:
        reset_context()

    expected = _reference_new_token_output(q_all, k_all, v_all, num_cached)
    assert got.shape == expected.shape
    assert torch.allclose(got, expected, atol=1e-5), (
        f"max abs diff {(got - expected).abs().max().item()}"
    )


def test_fresh_prefill_fallback_still_correct():
    # num_cached == 0: gather path must equal a plain full-context reference.
    torch.manual_seed(1)
    full_len = 5
    q_all = torch.randn(full_len, HEADS, DIM)
    k_all = torch.randn(full_len, KV_HEADS, DIM)
    v_all = torch.randn(full_len, KV_HEADS, DIM)

    k_cache = torch.zeros(2, BLOCK, KV_HEADS, DIM)
    v_cache = torch.zeros(2, BLOCK, KV_HEADS, DIM)
    _fill_cache(k_cache, k_all)
    _fill_cache(v_cache, v_all)

    attn = Attention(num_heads=HEADS, head_dim=DIM, scale=SCALE, num_kv_heads=KV_HEADS)
    attn.k_cache = k_cache
    attn.v_cache = v_cache

    set_context(
        is_prefill=True,
        max_seqlen_q=full_len,
        max_seqlen_k=full_len,
        cum_seqlens_q=torch.tensor([0, full_len]),
        cum_seqlens_k=torch.tensor([0, full_len]),
        slot_mapping=None,
        block_tables=torch.tensor([[0, 1]]),
    )
    try:
        got = attn(q_all, k_all, v_all)
    finally:
        reset_context()

    expected = _reference_new_token_output(q_all, k_all, v_all, 0)
    assert torch.allclose(got, expected, atol=1e-5)
