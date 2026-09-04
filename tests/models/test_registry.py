"""Tests for minivllm.models.registry.create_model."""

from __future__ import annotations

import pytest
from transformers import Qwen3Config

from minivllm.models.qwen3 import Qwen3ForCausalLM
from minivllm.models.registry import SUPPORTED_MODELS, TYPE_TO_ARCH, create_model


def _cfg(**overrides):
    kwargs = {
        "vocab_size": 64,
        "hidden_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 64,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
    }
    kwargs.update(overrides)
    return Qwen3Config(**kwargs)


def test_create_by_model_type():
    model = create_model(_cfg())
    assert isinstance(model, Qwen3ForCausalLM)


def test_create_by_architectures():
    cfg = _cfg()
    cfg.model_type = "unknown_type"  # force the architectures branch
    cfg.architectures = ["Qwen3ForCausalLM"]
    assert isinstance(create_model(cfg), Qwen3ForCausalLM)


def test_create_unknown_raises():
    class _Bad:
        model_type = "llama"
        architectures = ["LlamaForCausalLM"]

    with pytest.raises(ValueError):
        create_model(_Bad())


def test_type_to_arch_matches_supported():
    assert set(TYPE_TO_ARCH.values()) <= set(SUPPORTED_MODELS.keys())
