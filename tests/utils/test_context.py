"""Tests for minivllm.utils.context (thread-local inference context)."""

from __future__ import annotations

import pytest
import torch

from minivllm.utils.context import Context, get_context, reset_context, set_context


@pytest.fixture(autouse=True)
def _clean_context():
    reset_context()
    yield
    reset_context()


def test_default_context_has_defaults():
    ctx = get_context()
    assert isinstance(ctx, Context)
    assert ctx.is_prefill is False
    assert ctx.max_seqlen_q == 0
    assert ctx.max_seqlen_k == 0
    assert ctx.slot_mapping is None
    assert ctx.block_tables is None


def test_get_context_never_returns_none():
    assert get_context() is not None


def test_set_get_roundtrip():
    slot = torch.tensor([1, 2, 3])
    ctx_len = torch.tensor([4, 5])
    blocks = torch.zeros((2, 3), dtype=torch.int32)
    set_context(
        is_prefill=True,
        max_seqlen_q=8,
        max_seqlen_k=10,
        slot_mapping=slot,
        context_lens=ctx_len,
        block_tables=blocks,
    )
    ctx = get_context()
    assert ctx.is_prefill is True
    assert ctx.max_seqlen_q == 8
    assert ctx.max_seqlen_k == 10
    assert torch.equal(ctx.slot_mapping, slot)
    assert torch.equal(ctx.context_lens, ctx_len)
    assert torch.equal(ctx.block_tables, blocks)


def test_reset_restores_defaults():
    set_context(is_prefill=True, max_seqlen_q=16)
    reset_context()
    ctx = get_context()
    assert ctx.is_prefill is False
    assert ctx.max_seqlen_q == 0


def test_context_isolation_across_set_calls():
    set_context(is_prefill=True, max_seqlen_q=4)
    set_context(is_prefill=False, max_seqlen_q=9)
    ctx = get_context()
    assert ctx.is_prefill is False
    assert ctx.max_seqlen_q == 9
