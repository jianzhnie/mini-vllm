#!/usr/bin/env python3
"""Test script for NPU Flash Attention integration in mini-vllm.

This script tests the NPU Flash Attention functionality when running on NPU devices.
"""

import torch
from transformers.utils import is_torch_npu_available

from minivllm.models.layers.attention import _NPU_NATIVE_AVAILABLE, Attention
from minivllm.utils.context import Context


def test_attention_creation():
    """Test creating an Attention layer."""
    print('Testing Attention layer creation...')
    attention = Attention(
        num_heads=8,
        head_dim=64,
        scale=1.0 / 64.0,
        num_kv_heads=8,
    )
    print('✓ Attention layer created successfully')
    print(f'  - num_heads: {attention.num_heads}')
    print(f'  - head_dim: {attention.head_dim}')
    print(f'  - scale: {attention.scale}')
    print(f'  - num_kv_heads: {attention.num_kv_heads}')
    return attention


def test_npu_availability():
    """Test NPU availability and native functions."""
    print('\nTesting NPU availability...')
    print(f'NPU available: {is_torch_npu_available()}')
    print(f'Native NPU functions available: {_NPU_NATIVE_AVAILABLE}')

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


def test_attention_forward(attention, device='cpu'):
    """Test forward pass with synthetic data."""
    print(f'\nTesting attention forward pass on {device}...')

    # Create test tensors
    batch_size = 2
    seq_len = 16
    num_heads = 8
    num_kv_heads = 8
    head_dim = 64

    # Test prefill phase
    print('Testing prefill phase...')
    q = torch.randn(batch_size * seq_len, num_heads, head_dim, device=device)
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
        cum_seqlens_q=torch.tensor([0, seq_len, seq_len * 2], device=device),
        cum_seqlens_k=torch.tensor([0, seq_len, seq_len * 2], device=device),
        block_tables=None,
        slot_mapping=torch.arange(batch_size * seq_len, device=device),
    )

    # Set the context
    import minivllm.utils.context as context_module
    context_module._CONTEXT_VAR.set(context)

    try:
        with torch.no_grad():
            output = attention(q, k, v)
            print(f'✓ Prefill output shape: {output.shape}')
            expected_shape = (batch_size * seq_len, num_heads, head_dim)
            assert output.shape == expected_shape, f'Expected {expected_shape}, got {output.shape}'
    except Exception as e:
        print(f'✗ Prefill phase failed: {e}')
    finally:
        context_module._CONTEXT_VAR.set(Context())

    # Test decode phase (requires KV cache setup)
    print('Testing decode phase...')
    try:
        # Initialize KV cache
        max_blocks = 128
        block_size = 16
        attention.k_cache = torch.zeros(max_blocks,
                                        block_size,
                                        num_kv_heads,
                                        head_dim,
                                        device=device)
        attention.v_cache = torch.zeros(max_blocks,
                                        block_size,
                                        num_kv_heads,
                                        head_dim,
                                        device=device)

        # Single token input for decode
        q_decode = torch.randn(batch_size, num_heads, head_dim, device=device)
        k_decode = torch.randn(batch_size,
                               num_kv_heads,
                               head_dim,
                               device=device)
        v_decode = torch.randn(batch_size,
                               num_kv_heads,
                               head_dim,
                               device=device)

        # Set decode context
        decode_context = Context(
            is_prefill=False,
            context_lens=torch.tensor([seq_len, seq_len], device=device),
            block_tables=torch.tensor([[0, 1] + [-1] * (max_blocks - 2),
                                       [2, 3] + [-1] * (max_blocks - 2)],
                                      device=device),
            slot_mapping=torch.arange(batch_size, device=device),
        )
        context_module._CONTEXT_VAR.set(decode_context)

        # Store initial KV
        attention._cache_initialized = True

        with torch.no_grad():
            output_decode = attention(q_decode, k_decode, v_decode)
            print(f'✓ Decode output shape: {output_decode.shape}')
            expected_decode_shape = (batch_size, num_heads, head_dim)
            assert output_decode.shape == expected_decode_shape, f'Expected {expected_decode_shape}, got {output_decode.shape}'

    except Exception as e:
        print(f'✗ Decode phase failed: {e}')


def main():
    """Main test function."""
    print('=== NPU Flash Attention Integration Test ===\n')

    # Test NPU availability
    test_npu_availability()

    # Test attention creation
    attention = test_attention_creation()

    # Test forward pass
    device = 'npu' if is_torch_npu_available() else 'cpu'
    if is_torch_npu_available():
        print(f"\n{'='*50}")
        print('IMPORTANT: NPU is available! Testing on NPU device.')
        print('Note: This requires proper NPU drivers and environment setup.')
        print(f"{'='*50}\n")

    try:
        test_attention_forward(attention, device)
    except Exception as e:
        print(f'Forward pass failed on {device}: {e}')
        if is_torch_npu_available():
            print('Falling back to CPU test...')
            test_attention_forward(attention, 'cpu')

    print('\n=== Test Complete ===')
    if _NPU_NATIVE_AVAILABLE:
        print('✓ Native NPU Flash Attention is available and integrated')
    else:
        print(
            'ℹ Native NPU Flash Attention not available - using fallback implementations'
        )

    if is_torch_npu_available():
        print('✓ NPU device is available')
    else:
        print('ℹ NPU device not available - tests run on CPU')


if __name__ == '__main__':
    main()
