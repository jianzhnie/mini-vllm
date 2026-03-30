"""Tests for NPU Attention Backend.

This module tests the NPUAttentionBackend implementation including:
- Forward pass with flash attention
- Fallback to standard attention on error
- KV cache storage
- Unified inference API
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from minivllm.models.layers.attention_backend import NPUAttentionBackend


class TestNPUBackendForward:
    """Tests for NPU backend forward pass."""

    @patch('minivllm.models.layers.attention_backend.npu_flash_attn_func')
    def test_forward_transposes_correctly(self, mock_npu_func):
        """Test that NPU backend correctly transposes inputs and outputs.

        NPU expects BSND format but we use BNSD, so transposing is needed.
        """
        backend = NPUAttentionBackend()
        backend._npu_available = True

        batch_size = 2
        num_heads = 4
        seq_len = 8
        head_dim = 16

        # Input in BNSD format
        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim)

        # Mock return in BSND format
        mock_output = torch.randn(batch_size, seq_len, num_heads, head_dim)
        mock_npu_func.return_value = mock_output

        output = backend.forward(q, k, v)

        # Verify inputs were transposed to BSND before calling
        args, kwargs = mock_npu_func.call_args
        q_arg, k_arg, v_arg = args[0], args[1], args[2]

        assert q_arg.shape == (batch_size, seq_len, num_heads, head_dim)
        assert k_arg.shape == (batch_size, seq_len, num_heads, head_dim)
        assert v_arg.shape == (batch_size, seq_len, num_heads, head_dim)

        # Verify output was transposed back to BNSD
        assert output.shape == (batch_size, num_heads, seq_len, head_dim)

    def test_fallback_on_npu_error(self):
        """Test fallback to standard attention when NPU op fails."""
        backend = NPUAttentionBackend()
        backend._npu_available = True
        backend._fallback_backend = MagicMock()

        # Simulate NPU error
        with patch(
                'minivllm.models.layers.attention_backend.npu_flash_attn_func',
                side_effect=RuntimeError('NPU Error'),
        ):
            q = torch.randn(2, 4, 8, 16)
            k = torch.randn(2, 4, 8, 16)
            v = torch.randn(2, 4, 8, 16)

            backend.forward(q, k, v)

            # Should have called fallback
            backend._fallback_backend.forward.assert_called_once()


class TestNPUBackendKVCache:
    """Tests for NPU backend KV cache operations."""

    def test_store_kv_cache_basic(self):
        """Test basic KV cache storage."""
        backend = NPUAttentionBackend()
        backend._npu_available = True

        batch_size = 2
        num_heads = 4
        head_dim = 16
        hidden_size = num_heads * head_dim

        key = torch.randn(batch_size, num_heads, head_dim)
        value = torch.randn(batch_size, num_heads, head_dim)

        total_slots = 100
        k_cache = torch.zeros(total_slots, hidden_size)
        v_cache = torch.zeros(total_slots, hidden_size)

        slot_mapping = torch.tensor([10, 20], dtype=torch.long)

        backend.store_kv_cache(key, value, k_cache, v_cache, slot_mapping)

        # Verify values were stored at correct slots
        flat_key = key.view(batch_size, hidden_size)
        flat_value = value.view(batch_size, hidden_size)

        assert torch.allclose(k_cache[10], flat_key[0])
        assert torch.allclose(v_cache[10], flat_value[0])
        assert torch.allclose(k_cache[20], flat_key[1])
        assert torch.allclose(v_cache[20], flat_value[1])

        # Verify other slots are empty
        assert torch.allclose(k_cache[0], torch.zeros(hidden_size))
        assert torch.allclose(v_cache[0], torch.zeros(hidden_size))

    def test_store_kv_cache_different_slots(self):
        """Test KV cache storage with non-contiguous slots."""
        backend = NPUAttentionBackend()
        backend._npu_available = True

        batch_size = 3
        num_heads = 4
        head_dim = 16
        hidden_size = num_heads * head_dim

        key = torch.randn(batch_size, num_heads, head_dim)
        value = torch.randn(batch_size, num_heads, head_dim)

        k_cache = torch.zeros(50, hidden_size)
        v_cache = torch.zeros(50, hidden_size)

        # Non-contiguous slots
        slot_mapping = torch.tensor([0, 25, 49], dtype=torch.long)

        backend.store_kv_cache(key, value, k_cache, v_cache, slot_mapping)

        # Verify correct slots
        flat_key = key.view(batch_size, hidden_size)
        assert torch.allclose(k_cache[0], flat_key[0])
        assert torch.allclose(k_cache[25], flat_key[1])
        assert torch.allclose(k_cache[49], flat_key[2])


class TestNPUUnifiedInference:
    """Tests for NPU unified inference API."""

    def test_unified_inference_calls_correct_function(self):
        """Test that unified inference API calls the correct function."""
        backend = NPUAttentionBackend()
        backend._npu_available = True

        # Mock the unified inference function
        backend.npu_fused_infer_attention_score = MagicMock()
        backend.npu_fused_infer_attention_score.return_value = torch.randn(
            1, 1, 1)

        query = torch.randn(2, 4, 16)
        key_cache = torch.randn(2, 10, 4, 16)
        value_cache = torch.randn(2, 10, 4, 16)
        seq_len = 10
        num_kv_heads = 4

        backend.unified_inference(query,
                                  key_cache,
                                  value_cache,
                                  seq_len,
                                  num_kv_heads,
                                  scale=1.0)

        # Verify the function was called
        backend.npu_fused_infer_attention_score.assert_called_once()

        # Verify argument shapes
        args = backend.npu_fused_infer_attention_score.call_args[0]
        assert args[0].shape == (2, 4, 1, 16)  # Query expanded
        assert args[1].shape == key_cache.shape
        assert args[2].shape == value_cache.shape
        assert args[4] == seq_len
        assert args[5] == num_kv_heads

        # Verify scale kwarg
        kwargs = backend.npu_fused_infer_attention_score.call_args[1]
        assert kwargs['scale'] == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
