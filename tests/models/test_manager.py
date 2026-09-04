"""Tests for minivllm.models.manager.ModelManager.

ModelManager orchestrates device setup, tokenizer/model loading and validation.
Real weight/tokenizer loading is patched out; these tests exercise the manager's
own logic (state, validation, info, cleanup, context protocol and the dtype
resolution performed inside ``_load_model``).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from minivllm.models.manager import ModelManager


def _config(**overrides) -> SimpleNamespace:
    """Build a lightweight Config stand-in with only the attrs ModelManager uses."""
    base = {
        "model": "/tmp/model",
        "dtype": "auto",
        "tensor_parallel_size": 1,
        "hf_config": None,
        "use_buffered_page_attention": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


# --- construction & default state ------------------------------------------


class TestInit:
    def test_default_state(self):
        m = ModelManager(_config())
        assert m.config.model == "/tmp/model"
        assert m.model is None
        assert m.tokenizer is None
        assert m.device is None
        assert m.model_type is None
        assert m._model_config is None


# --- model path validation -------------------------------------------------


class TestValidateModelPath:
    def test_empty_path_raises(self):
        m = ModelManager(_config(model=""))
        with pytest.raises(ValueError):
            m._validate_model_path()

    def test_nonempty_path_passes(self):
        m = ModelManager(_config(model="/some/path"))
        m._validate_model_path()  # must not raise


# --- get_model_info --------------------------------------------------------


class TestGetModelInfo:
    def test_empty_when_no_model(self):
        m = ModelManager(_config())
        assert m.get_model_info() == {}

    def test_populated_info(self):
        m = ModelManager(_config(dtype="float16", tensor_parallel_size=2))
        m.model = MagicMock()
        m.model_type = "qwen3"
        m.device = torch.device("cpu")
        m._model_config = SimpleNamespace(
            vocab_size=10,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
        )
        info = m.get_model_info()
        assert info["model_type"] == "qwen3"
        assert info["device"] == "cpu"
        assert info["dtype"] == "float16"
        assert info["tensor_parallel_size"] == 2
        assert info["vocab_size"] == 10
        assert info["hidden_size"] == 32
        assert info["num_layers"] == 1
        assert info["num_heads"] == 2


# --- cleanup ---------------------------------------------------------------


class TestCleanup:
    def test_clears_model_and_tokenizer(self):
        m = ModelManager(_config())
        m.model = MagicMock()
        m.tokenizer = MagicMock()
        m.cleanup()
        assert m.model is None
        assert m.tokenizer is None


# --- context manager protocol ---------------------------------------------


class TestContextManager:
    def test_enter_initializes_exit_cleans(self):
        m = ModelManager(_config())
        with patch.object(m, "initialize") as init, patch.object(
            m, "cleanup"
        ) as clean:
            with m:
                init.assert_called_once()
            clean.assert_called_once()


# --- _load_model dtype resolution -----------------------------------------


@pytest.mark.parametrize(
    "dtype_str, hf_torch_dtype, expected",
    [
        ("float16", None, torch.float16),
        ("bfloat16", None, torch.bfloat16),
        ("float32", None, torch.float32),
        ("auto", torch.bfloat16, torch.bfloat16),  # auto follows checkpoint dtype
        ("auto", "bfloat16", torch.float16),  # auto falls back to fp16 for non-dtype
    ],
)
@patch("minivllm.models.manager.load_model")
@patch("minivllm.models.manager.create_model")
def test_load_model_resolves_dtype(
    mock_create, mock_load, dtype_str, hf_torch_dtype, expected
):
    hf_config = SimpleNamespace(
        torch_dtype=hf_torch_dtype if hf_torch_dtype is not None else torch.float16
    )
    m = ModelManager(_config(dtype=dtype_str, hf_config=hf_config))
    m.device = torch.device("cpu")

    model = MagicMock()
    model.to.return_value = model
    mock_create.return_value = model

    m._load_model()

    mock_create.assert_called_once_with(hf_config)
    mock_load.assert_called_once_with(model, m.config.model)
    assert model.to.call_args.kwargs["dtype"] is expected
    # manager bookkeeping updated
    assert m.model is model
    assert m._model_config is model.config

    # hf_config received the runtime flag so layers can read it
    assert hf_config.use_buffered_page_attention is False

    # default dtype restored to whatever it was before the call
    assert torch.get_default_dtype() is not None


if __name__ == "__main__":
    pytest.main([__file__])
