"""Tests for NPU Flash Attention integration.

This module tests the NPU Flash Attention functionality when running on NPU devices.
Tests use CPU fallback when NPU is not available.
"""

import pytest
import torch
from transformers.utils import is_torch_npu_available

import minivllm.utils.context as context_module
from minivllm.models.layers.attention import _NPU_FLASH_ATTN_AVAILABLE, Attention
from minivllm.utils.context import Context


class TestNPUAvailability:
    """Tests for NPU availability detection."""

    def test_npu_availability_report(self):
        """Report NPU availability and native function status."""
        print(f'\nNPU available: {is_torch_npu_available()}')
        print(f'Native NPU functions available: {_NPU_FLASH_ATTN_AVAILABLE}')

        if is_torch_npu_available():
            try:
                import torch_npu

                version = getattr(torch_npu, '__version__', 'unknown')
                print(f'torch_npu version: {version}')
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


class TestAttentionCreation:
    """Tests for Attention layer creation."""

    def test_attention_creation_basic(self):
        """Test creating an Attention layer with default parameters."""
        attention = Attention(
            num_heads=8,
            head_dim=64,
            scale=1.0 / 64.0,
            num_kv_heads=8,
        )
        assert attention.num_heads == 8
        assert attention.head_dim == 64
        assert attention.num_kv_heads == 8
        assert attention.scale == 1.0 / 64.0

    def test_attention_creation_gqa(self):
        """Test creating an Attention layer with GQA."""
        attention = Attention(
            num_heads=8,
            head_dim=64,
            scale=1.0 / 64.0,
            num_kv_heads=2,  # GQA
        )
        assert attention.num_kv_heads == 2

    def test_attention_creation_mqa(self):
        """Test creating an Attention layer with MQA."""
        attention = Attention(
            num_heads=8,
            head_dim=64,
            scale=1.0 / 64.0,
            num_kv_heads=1,  # MQA
        )
        assert attention.num_kv_heads == 1


class TestAttentionForward:
    """Tests for Attention forward pass."""

    def test_prefill_forward(self):
        """Test forward pass in prefill phase."""
        device = 'npu' if is_torch_npu_available() else 'cpu'

        attention = Attention(
            num_heads=8,
            head_dim=64,
            scale=1.0 / 64.0,
            num_kv_heads=8,
        )

        batch_size = 2
        seq_len = 16
        num_heads = 8
        num_kv_heads = 8
        head_dim = 64

        # Create test tensors
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

        # Create context for prefill
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

        context_module._CONTEXT_VAR.set(context)

        try:
            with torch.no_grad():
                output = attention(q, k, v)
                expected_shape = (batch_size * seq_len, num_heads, head_dim)
                assert output.shape == expected_shape
        finally:
            context_module.reset_context()

    def test_decode_forward(self):
        """Test forward pass in decode phase (single token)."""
        device = 'npu' if is_torch_npu_available() else 'cpu'

        attention = Attention(
            num_heads=8,
            head_dim=64,
            scale=1.0 / 64.0,
            num_kv_heads=8,
        )

        batch_size = 2
        num_heads = 8
        num_kv_heads = 8
        head_dim = 64

        # Initialize KV cache for decode phase
        attention.k_cache = torch.randn(10,
                                        16,
                                        num_kv_heads,
                                        head_dim,
                                        device=device)
        attention.v_cache = torch.randn(10,
                                        16,
                                        num_kv_heads,
                                        head_dim,
                                        device=device)
        attention._cache_initialized = True

        # Single token decode
        q = torch.randn(batch_size, num_heads, head_dim, device=device)
        k = torch.randn(batch_size, num_kv_heads, head_dim, device=device)
        v = torch.randn(batch_size, num_kv_heads, head_dim, device=device)

        context = Context(
            is_prefill=False,
            max_seqlen_q=1,
            max_seqlen_k=16,
            cum_seqlens_q=None,
            cum_seqlens_k=None,
            block_tables=torch.tensor([[0, 1], [2, 3]], device=device),
            slot_mapping=torch.arange(batch_size, device=device),
            context_lens=torch.tensor([16, 16], device=device),
        )

        context_module._CONTEXT_VAR.set(context)

        try:
            with torch.no_grad():
                output = attention(q, k, v)
                expected_shape = (batch_size, num_heads, head_dim)
                assert output.shape == expected_shape
        finally:
            context_module.reset_context()


class TestNPUOOMHandling:
    """Tests for NPU OOM handling."""

    def test_oom_handling_logic(self):
        """Test OOM handling logic (mocked)."""
        if not is_torch_npu_available():
            pytest.skip('NPU not available')

        from minivllm.models.layers.attention_backend import NPUAttentionBackend

        attention = Attention(
            num_heads=8,
            head_dim=64,
            scale=1.0 / 64.0,
            num_kv_heads=8,
        )

        if isinstance(attention.backend, NPUAttentionBackend):
            # Test OOM error handling
            oom_error = RuntimeError('NPU out of memory')
            handled = attention.backend._handle_oom(oom_error)
            assert handled is True

            # Test non-OOM error handling
            other_error = RuntimeError('Some other error')
            handled = attention.backend._handle_oom(other_error)
            assert handled is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
