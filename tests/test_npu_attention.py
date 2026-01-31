#!/usr/bin/env python3
"""Test script for NPU Flash Attention integration in mini-vllm.

This script tests the NPU Flash Attention functionality when running on NPU devices.
"""

import unittest

import torch
from transformers.utils import is_torch_npu_available

import minivllm.utils.context as context_module
from minivllm.models.layers.attention import _NPU_FLASH_ATTN_AVAILABLE, Attention
from minivllm.utils.context import Context


class TestNPUAttention(unittest.TestCase):

    def test_npu_availability(self):
        """Test NPU availability and native functions."""
        print('\nTesting NPU availability...')
        print(f'NPU available: {is_torch_npu_available()}')
        print(f'Native NPU functions available: {_NPU_FLASH_ATTN_AVAILABLE}')

        if is_torch_npu_available():
            try:
                import torch_npu

                print(
                    f"torch_npu version: {torch_npu.__version__ if hasattr(torch_npu, '__version__') else 'unknown'}"
                )
                print(
                    f"npu_fusion_attention available: {hasattr(torch_npu, 'npu_fusion_attention')}"
                )
                print(
                    f"npu_incre_flash_attention available: {hasattr(torch_npu, 'npu_incre_flash_attention')}"
                )
            except ImportError as e:
                print(f'Error importing torch_npu: {e}')
        else:
            print('NPU not available - will use CPU fallback')

    def test_attention_creation(self):
        """Test creating an Attention layer."""
        print('Testing Attention layer creation...')
        attention = Attention(
            num_heads=8,
            head_dim=64,
            scale=1.0 / 64.0,
            num_kv_heads=8,
        )
        print('✓ Attention layer created successfully')
        self.assertEqual(attention.num_heads, 8)
        self.assertEqual(attention.head_dim, 64)
        self.assertEqual(attention.num_kv_heads, 8)
        return attention

    def test_attention_forward(self):
        """Test forward pass with synthetic data."""
        device = 'npu' if is_torch_npu_available() else 'cpu'
        print(f'\nTesting attention forward pass on {device}...')

        attention = self.test_attention_creation()

        # Create test tensors
        batch_size = 2
        seq_len = 16
        num_heads = 8
        num_kv_heads = 8
        head_dim = 64

        # Test prefill phase
        print('Testing prefill phase...')
        q = torch.randn(batch_size * seq_len,
                        num_heads,
                        head_dim,
                        device=device)
        k = torch.randn(batch_size * seq_len,
                        num_kv_heads,
                        head_dim,
                        device=device)
        v = torch.randn(batch_size * seq_len,
                        num_kv_heads,
                        head_dim,
                        device=device)

        # Create mock context for prefill
        context = Context(
            is_prefill=True,
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
            cum_seqlens_q=torch.tensor([0, seq_len, seq_len * 2],
                                       device=device),
            cum_seqlens_k=torch.tensor([0, seq_len, seq_len * 2],
                                       device=device),
            block_tables=None,
            slot_mapping=torch.arange(batch_size * seq_len, device=device),
        )

        # Set the context
        context_module._CONTEXT_VAR.set(context)

        try:
            with torch.no_grad():
                output = attention(q, k, v)
                print(f'✓ Prefill output shape: {output.shape}')
                expected_shape = (batch_size * seq_len, num_heads, head_dim)
                self.assertEqual(output.shape, expected_shape)
        except Exception as e:
            self.fail(f'Prefill phase failed: {e}')

    def test_npu_oom_handling(self):
        """Test OOM handling logic (mocked)."""
        if not is_torch_npu_available():
            return

        print('\nTesting NPU OOM handling...')
        # This is a bit hard to test without actually causing OOM,
        # but we can check if the backend has the method.
        attention = Attention(
            num_heads=8,
            head_dim=64,
            scale=1.0 / 64.0,
            num_kv_heads=8,
        )

        from minivllm.models.layers.attention_backend import NPUAttentionBackend

        if isinstance(attention.backend, NPUAttentionBackend):
            # Mock exception
            oom_error = RuntimeError('NPU out of memory')
            handled = attention.backend._handle_oom(oom_error)
            print(f'OOM handled: {handled}')
            self.assertTrue(handled)

            other_error = RuntimeError('Some other error')
            handled = attention.backend._handle_oom(other_error)
            print(f'Other error handled: {handled}')
            self.assertFalse(handled)


if __name__ == '__main__':
    unittest.main()
