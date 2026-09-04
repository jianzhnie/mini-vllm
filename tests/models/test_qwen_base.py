"""Tests for minivllm.models.qwen_base (shared Qwen backbone).

Qwen2/Qwen3 both subclass this module, so it is the core model code. These tests
run entirely on CPU with a tiny config and cover: the rope resolution helpers,
the MLP and attention blocks (including the prefill / decode reshape branches),
the stacked model, and ``QwenForCausalLM.load_weights`` (packed q/k/v -> qkv_proj,
gate/up -> gate_up_proj, rotary freqs skipped).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from minivllm.models.qwen_base import (
    QwenAttention,
    QwenDecoderLayer,
    QwenForCausalLM,
    QwenMLP,
    QwenModel,
    _resolve_rope_scaling,
    _resolve_rope_theta,
)
from minivllm.utils.context import reset_context, set_context

HIDDEN = 32
HEADS = 2
KV_HEADS = 2
HEAD_DIM = 16
VOCAB = 64
SEQ = 8


def _config(**overrides) -> SimpleNamespace:
    base = {
        "hidden_size": HIDDEN,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "attention_bias": False,
        "head_dim": None,
        "intermediate_size": 64,
        "hidden_act": "silu",
        "vocab_size": VOCAB,
        "num_hidden_layers": 1,
        "rope_theta": None,
        "rope_scaling": None,
        "tie_word_embeddings": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _prefill_context(seq_len: int = SEQ) -> None:
    set_context(
        is_prefill=True,
        max_seqlen_q=seq_len,
        max_seqlen_k=seq_len,
        cum_seqlens_q=torch.tensor([0, seq_len]),
        cum_seqlens_k=torch.tensor([0, seq_len]),
        slot_mapping=torch.arange(seq_len),
    )


# --- rope resolution helpers ----------------------------------------------


class TestResolveRopeTheta:
    def test_direct_attribute(self):
        assert _resolve_rope_theta(SimpleNamespace(rope_theta=1234), 1.0) == 1234

    def test_from_rope_parameters(self):
        cfg = SimpleNamespace(rope_parameters={"rope_theta": 777})
        assert _resolve_rope_theta(cfg, 1.0) == 777

    def test_from_rope_scaling(self):
        cfg = SimpleNamespace(rope_scaling={"rope_theta": 555})
        assert _resolve_rope_theta(cfg, 1.0) == 555

    def test_default_fallback(self):
        assert _resolve_rope_theta(SimpleNamespace(), 42.0) == 42.0


class TestResolveRopeScaling:
    def test_none(self):
        assert _resolve_rope_scaling(SimpleNamespace(rope_scaling=None)) is None

    def test_filters_non_scaling_keys(self):
        raw = {
            "factor": 1.0,
            "type": "yarn",
            "rope_theta": 1000,  # not a scaling key -> dropped
            "low_freq_factor": 1,
        }
        assert _resolve_rope_scaling(SimpleNamespace(rope_scaling=raw)) == {
            "factor": 1.0,
            "type": "yarn",
            "low_freq_factor": 1,
        }

    def test_non_dict_returns_none(self):
        assert _resolve_rope_scaling(SimpleNamespace(rope_scaling=42)) is None


# --- QwenMLP --------------------------------------------------------------


class TestQwenMLP:
    def test_forward_shape(self):
        mlp = QwenMLP(hidden_size=HIDDEN, intermediate_size=64, hidden_act="silu")
        out = mlp(torch.randn(5, HIDDEN))
        assert out.shape == (5, HIDDEN)

    def test_invalid_activation_raises(self):
        with pytest.raises(ValueError, match="Unsupported activation"):
            QwenMLP(hidden_size=HIDDEN, intermediate_size=64, hidden_act="gelu")


# --- QwenAttention ---------------------------------------------------------


class TestQwenAttention:
    def test_derived_attributes(self):
        att = QwenAttention(
            hidden_size=HIDDEN,
            num_heads=HEADS,
            num_kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            qkv_bias=False,
        )
        assert att.head_dim == HEAD_DIM
        assert att.q_size == HEADS * HEAD_DIM
        assert att.kv_size == KV_HEADS * HEAD_DIM
        # qkv_bias=False enables per-head q/k RMSNorm
        assert att.q_norm is not None and att.k_norm is not None

    def test_qkv_bias_disabled_norms(self):
        att = QwenAttention(
            hidden_size=HIDDEN,
            num_heads=HEADS,
            num_kv_heads=KV_HEADS,
            qkv_bias=True,
        )
        assert att.q_norm is None and att.k_norm is None

    def test_prefill_2d_branch(self):
        # QwenModel always feeds a flat (tokens, hidden) tensor, hitting the
        # 2-D reshape path.
        att = QwenAttention(
            hidden_size=HIDDEN,
            num_heads=HEADS,
            num_kv_heads=KV_HEADS,
            qkv_bias=False,
        )
        _prefill_context()
        try:
            out = att(torch.arange(SEQ), torch.randn(SEQ, HIDDEN))
        finally:
            reset_context()
        assert out.shape == (SEQ, HIDDEN)

    def test_gqa_forward(self):
        # num_heads != num_kv_heads exercises the group-query path.
        att = QwenAttention(
            hidden_size=HIDDEN,
            num_heads=2,
            num_kv_heads=1,
            head_dim=HEAD_DIM,
            qkv_bias=False,
        )
        _prefill_context()
        try:
            out = att(torch.arange(SEQ), torch.randn(SEQ, HIDDEN))
        finally:
            reset_context()
        assert out.shape == (SEQ, HIDDEN)


# --- QwenDecoderLayer ------------------------------------------------------


class TestQwenDecoderLayer:
    def test_forward_with_none_residual(self):
        layer = QwenDecoderLayer(_config())
        _prefill_context()
        try:
            hidden = torch.randn(SEQ, HIDDEN)
            out, residual = layer(torch.arange(SEQ), hidden, None)
        finally:
            reset_context()
        assert out.shape == (SEQ, HIDDEN)
        assert residual.shape == (SEQ, HIDDEN)

    def test_forward_with_provided_residual(self):
        layer = QwenDecoderLayer(_config())
        _prefill_context()
        try:
            hidden = torch.randn(SEQ, HIDDEN)
            residual = torch.randn(SEQ, HIDDEN)
            out, new_residual = layer(torch.arange(SEQ), hidden, residual)
        finally:
            reset_context()
        assert out.shape == (SEQ, HIDDEN)
        assert new_residual.shape == (SEQ, HIDDEN)


# --- QwenModel -------------------------------------------------------------


class TestQwenModel:
    def test_forward_shape(self):
        model = QwenModel(_config())
        _prefill_context()
        try:
            ids = torch.randint(0, VOCAB, (SEQ,))
            out = model(ids, torch.arange(SEQ))
        finally:
            reset_context()
        assert out.shape == (SEQ, HIDDEN)


# --- QwenForCausalLM.load_weights ------------------------------------------


class TestLoadWeights:
    def test_packed_and_direct_assignment(self):
        model = QwenForCausalLM(_config())
        torch.manual_seed(0)
        q, k, v = (torch.randn(HIDDEN, HIDDEN) for _ in range(3))
        gate, up = torch.randn(64, HIDDEN), torch.randn(64, HIDDEN)
        o = torch.randn(HIDDEN, HIDDEN)
        emb = torch.randn(VOCAB, HIDDEN)
        lmh = torch.randn(VOCAB, HIDDEN)

        model.load_weights(
            {
                "model.embed_tokens.weight": emb,
                "model.layers.0.self_attn.q_proj.weight": q,
                "model.layers.0.self_attn.k_proj.weight": k,
                "model.layers.0.self_attn.v_proj.weight": v,
                "model.layers.0.self_attn.o_proj.weight": o,
                "model.layers.0.mlp.gate_proj.weight": gate,
                "model.layers.0.mlp.up_proj.weight": up,
                "model.layers.0.input_layernorm.weight": torch.ones(HIDDEN),
                "model.layers.0.post_attention_layernorm.weight": torch.ones(HIDDEN),
                "model.norm.weight": torch.ones(HIDDEN),
                "lm_head.weight": lmh,
                # must be skipped by load_weights
                "model.rotary_emb.inv_freq": torch.ones(8),
            }
        )

        attn = model.model.layers[0].self_attn
        qkv = attn.qkv_proj.weight
        # layout: [Q(2 heads) | K(2 heads) | V(2 heads)] * head_dim, each block 32 rows
        assert torch.allclose(qkv[0:32], q)
        assert torch.allclose(qkv[32:64], k)
        assert torch.allclose(qkv[64:96], v)
        assert torch.allclose(attn.o_proj.weight, o)

        gu = model.model.layers[0].mlp.gate_up_proj.weight
        assert gu.shape == (128, HIDDEN)
        assert torch.allclose(gu[0:64], gate)
        assert torch.allclose(gu[64:128], up)

        assert torch.allclose(model.model.embed_tokens.weight, emb)
        assert torch.allclose(model.lm_head.weight, lmh)
        assert torch.allclose(model.model.layers[0].input_layernorm.weight, torch.ones(HIDDEN))

    def test_tie_word_embeddings_shares_weight(self):
        model = QwenForCausalLM(_config(tie_word_embeddings=True))
        assert model.lm_head.weight is model.model.embed_tokens.weight

    def test_compute_logits_shape(self):
        model = QwenForCausalLM(_config())
        logits = model.compute_logits(torch.randn(3, HIDDEN))
        assert logits.shape == (3, VOCAB)

    def test_set_kv_cache_assigns_per_layer(self):
        model = QwenForCausalLM(_config())
        k_cache = torch.zeros(2, 4, KV_HEADS, HEAD_DIM)
        v_cache = torch.zeros(2, 4, KV_HEADS, HEAD_DIM)
        model.set_kv_cache([(k_cache, v_cache)])
        attn = model.model.layers[0].self_attn
        assert attn.attn.k_cache is k_cache
        assert attn.attn.v_cache is v_cache


if __name__ == "__main__":
    pytest.main([__file__])
