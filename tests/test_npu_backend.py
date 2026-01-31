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


if __name__ == '__main__':
    unittest.main()
