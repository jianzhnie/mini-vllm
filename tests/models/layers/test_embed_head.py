"""Tests for minivllm.models.layers.embed_head (single-rank / tp_size=1 path)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from minivllm.models.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from minivllm.utils.context import reset_context, set_context


# --- VocabParallelEmbedding ------------------------------------------------


def test_embedding_weight_shape():
    e = VocabParallelEmbedding(100, 16)
    assert e.weight.shape == (100, 16)


def test_embedding_forward_matches_torch():
    e = VocabParallelEmbedding(100, 16)
    ids = torch.tensor([0, 5, 99])
    assert torch.allclose(e(ids), F.embedding(ids, e.weight))


def test_embedding_single_rank_allows_any_vocab_size():
    e = VocabParallelEmbedding(7, 4)  # 7 % tp_size(1) == 0
    assert e.weight.shape == (7, 4)


def test_embedding_weight_loader_full_shard():
    e = VocabParallelEmbedding(100, 8)
    full = torch.arange(100 * 8, dtype=torch.float32).reshape(100, 8)
    e.weight_loader(e.weight, full)
    assert torch.allclose(e.weight, full)


# --- ParallelLMHead ---------------------------------------------------------


def test_lm_head_decode_logits_equal_linear():
    head = ParallelLMHead(50, 16)
    x = torch.randn(3, 16)
    assert torch.allclose(head(x), F.linear(x, head.weight))


def test_lm_head_prefill_picks_last_token_per_seq():
    head = ParallelLMHead(30, 8)
    try:
        set_context(is_prefill=True, cum_seqlens_q=torch.tensor([0, 2, 5]))
        x = torch.randn(5, 8)  # 5 tokens across 2 sequences of length [2, 3]
        logits = head(x)
        assert logits.shape[0] == 2
    finally:
        reset_context()


def test_lm_head_bias_not_supported():
    with pytest.raises(ValueError):
        ParallelLMHead(10, 4, bias=True)
