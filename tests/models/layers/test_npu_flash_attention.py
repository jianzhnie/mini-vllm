"""Tests for minivllm.models.layers.npu_flash_attention.

The attention mask helper (``get_attn_mask_npu``) and the ``SPARSE_MODE`` import
validation are pure-Python / CPU, so they are exercised here even without an NPU.
The ``npu_flash_attn_func`` kernel itself is NPU-only and intentionally not run.
"""

from __future__ import annotations

import importlib

import pytest
import torch

from minivllm.models.layers import npu_flash_attention as nfa


@pytest.fixture(autouse=True)
def _restore_mask_cache():
    """Isolate the module-level mask cache around each test."""
    saved = dict(nfa.ATTN_MASK_NPU_CACHE)
    nfa.ATTN_MASK_NPU_CACHE.clear()
    yield
    nfa.ATTN_MASK_NPU_CACHE.clear()
    nfa.ATTN_MASK_NPU_CACHE.update(saved)


# --- get_attn_mask_npu -----------------------------------------------------


class TestGetAttnMaskNpu:
    def test_causal_shape_dtype_and_values(self):
        dev = torch.device("cpu")
        mask = nfa.get_attn_mask_npu(dev, size=8)
        assert mask.dtype == torch.bool
        assert mask.device.type == "cpu"
        assert mask.shape == (8, 8)
        # upper-triangular with zero diagonal: mask[i, j] is True iff j > i
        assert not mask[0, 0]
        assert mask[0, 1]
        assert not mask[1, 0]

    def test_rounds_up_to_pow2_with_2048_minimum(self):
        dev = torch.device("cpu")
        nfa.get_attn_mask_npu(dev, size=8)
        # 8 < 2048, so the cache tensor is allocated at the 2048 minimum
        assert nfa.ATTN_MASK_NPU_CACHE[dev].shape == (2048, 2048)

    def test_reuses_cache_for_smaller_request(self):
        dev = torch.device("cpu")
        nfa.get_attn_mask_npu(dev, size=8)
        first = nfa.ATTN_MASK_NPU_CACHE[dev]
        nfa.get_attn_mask_npu(dev, size=16)
        # No reallocation: same cached tensor is sliced larger
        assert nfa.ATTN_MASK_NPU_CACHE[dev] is first

    def test_grows_cache_when_request_exceeds_cached(self):
        dev = torch.device("cpu")
        nfa.get_attn_mask_npu(dev, size=8)  # 2048
        nfa.get_attn_mask_npu(dev, size=3000)  # 3000 -> next pow2 = 4096
        assert nfa.ATTN_MASK_NPU_CACHE[dev].shape == (4096, 4096)
        mask = nfa.get_attn_mask_npu(dev, size=3000)
        assert mask.shape == (3000, 3000)

    def test_returns_requested_slice_not_full_cache(self):
        dev = torch.device("cpu")
        mask = nfa.get_attn_mask_npu(dev, size=12)
        assert mask.shape == (12, 12)


# --- is_npu_fa2_top_left_aligned_causal_mask ------------------------------


def test_top_left_flag_false_without_npu():
    from transformers.utils import is_torch_npu_available

    if is_torch_npu_available():
        pytest.skip("NPU present; flag reflects real config")
    assert nfa.is_npu_fa2_top_left_aligned_causal_mask() is False


# --- SPARSE_MODE import-time validation -----------------------------------


class TestSparseModeValidation:
    def test_accepts_2_and_3(self, monkeypatch):
        monkeypatch.setenv("NPU_FA2_SPARSE_MODE", "2")
        importlib.reload(nfa)
        assert nfa.SPARSE_MODE == 2

        monkeypatch.setenv("NPU_FA2_SPARSE_MODE", "3")
        importlib.reload(nfa)
        assert nfa.SPARSE_MODE == 3

    def test_rejects_invalid_value(self, monkeypatch):
        monkeypatch.setenv("NPU_FA2_SPARSE_MODE", "5")
        with pytest.raises(ValueError, match="NPU_FA2_SPARSE_MODE"):
            importlib.reload(nfa)

    def test_defaults_to_down_right_when_unset(self, monkeypatch):
        monkeypatch.delenv("NPU_FA2_SPARSE_MODE", raising=False)
        importlib.reload(nfa)
        assert nfa.SPARSE_MODE == nfa.DOWN_RIGHT_ALIGNED_CAUSAL_MASK_MODE


if __name__ == "__main__":
    pytest.main([__file__])
