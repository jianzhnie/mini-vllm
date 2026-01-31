import unittest
from unittest.mock import MagicMock, patch

import torch

from minivllm.models.layers.attention_backend import NPUAttentionBackend


class TestNPUBackendMock(unittest.TestCase):

    @patch('minivllm.models.layers.attention_backend.npu_flash_attn_func')
    def test_npu_forward_logic(self, mock_npu_func):
        """Test that NPU backend correctly transposes and calls flash attn."""
        backend = NPUAttentionBackend()
        # Mock availability
        backend._npu_available = True

        batch_size = 2
        num_heads = 4
        seq_len = 8
        head_dim = 16

        # Input BNSD
        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim)

        # Mock return from npu_flash_attn_func (BSND)
        mock_output = torch.randn(batch_size, seq_len, num_heads, head_dim)
        mock_npu_func.return_value = mock_output

        output = backend.forward(q, k, v)

        # Check if inputs were transposed to BSND
        args, kwargs = mock_npu_func.call_args
        q_arg, k_arg, v_arg = args[0], args[1], args[2]

        self.assertEqual(q_arg.shape,
                         (batch_size, seq_len, num_heads, head_dim))
        self.assertEqual(k_arg.shape,
                         (batch_size, seq_len, num_heads, head_dim))
        self.assertEqual(v_arg.shape,
                         (batch_size, seq_len, num_heads, head_dim))

        # Check if output was transposed back to BNSD
        self.assertEqual(output.shape,
                         (batch_size, num_heads, seq_len, head_dim))

    def test_npu_fallback_on_error(self):
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


class TestNPUBackendStoreKVCache(unittest.TestCase):

    def test_store_kv_cache_logic(self):
        """Test NPU store_kv_cache logic."""
        backend = NPUAttentionBackend()
        backend._npu_available = True

        # Setup inputs
        batch_size = 2
        num_heads = 4
        head_dim = 16
        hidden_size = num_heads * head_dim

        key = torch.randn(batch_size, num_heads, head_dim)
        value = torch.randn(batch_size, num_heads, head_dim)

        # Cache: [total_tokens, block_size, num_heads, head_dim] -> simplified to flat [total_tokens, hidden_size]
        # In the implementation, it reshapes cache to [-1, hidden_size]
        total_slots = 100
        k_cache = torch.zeros(total_slots, hidden_size)
        v_cache = torch.zeros(total_slots, hidden_size)

        slot_mapping = torch.tensor([10, 20], dtype=torch.long)

        # Call store_kv_cache
        backend.store_kv_cache(key, value, k_cache, v_cache, slot_mapping)

        # Verify
        # Reshape key/value for comparison
        flat_key = key.view(batch_size, hidden_size)
        flat_value = value.view(batch_size, hidden_size)

        # Check slot 10
        self.assertTrue(torch.allclose(k_cache[10], flat_key[0]))
        self.assertTrue(torch.allclose(v_cache[10], flat_value[0]))

        # Check slot 20
        self.assertTrue(torch.allclose(k_cache[20], flat_key[1]))
        self.assertTrue(torch.allclose(v_cache[20], flat_value[1]))

        # Check other slots are empty
        self.assertTrue(torch.allclose(k_cache[0], torch.zeros(hidden_size)))


class TestNPUUnifiedInference(unittest.TestCase):

    def test_unified_inference_call(self):
        """Test unified inference API call."""
        backend = NPUAttentionBackend()
        backend._npu_available = True

        # Mock the unified inference function
        backend.npu_fused_infer_attention_score = MagicMock()
        backend.npu_fused_infer_attention_score.return_value = torch.randn(
            1, 1, 1)

        # Inputs
        query = torch.randn(
            2, 4,
            16)  # [batch, heads, dim] - wait, query input to unified_inference
        # Attention.forward passes q (B,H,D)

        key_cache = torch.randn(2, 10, 4, 16)
        value_cache = torch.randn(2, 10, 4, 16)
        seq_len = 10
        num_kv_heads = 4

        backend.unified_inference(query, key_cache, value_cache, seq_len,
                                  num_kv_heads)

        # Verify call
        backend.npu_fused_infer_attention_score.assert_called_once()
        args = backend.npu_fused_infer_attention_score.call_args[0]
        self.assertIs(args[0], query)
        self.assertIs(args[1], key_cache)
        self.assertIs(args[2], value_cache)
        self.assertEqual(args[4], seq_len)
        self.assertEqual(args[5], num_kv_heads)


if __name__ == '__main__':
    unittest.main()
