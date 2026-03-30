"""Tests for tensor parallel linear layers.

This module tests the parallel linear layer implementations including:
- ColumnParallelLinear: Splits output dimension across workers
- RowParallelLinear: Splits input dimension across workers
- QKVParallelLinear: Fused QKV projection for attention
"""

import pytest
import torch

from minivllm.models.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
    divide,
)


class TestDivide:
    """Tests for the divide helper function."""

    def test_divide_exact(self):
        """Test division when divisible."""
        assert divide(10, 2) == 5
        assert divide(100, 4) == 25
        assert divide(64, 8) == 8

    def test_divide_uneven_raises(self):
        """Test that uneven division raises ValueError."""
        with pytest.raises(ValueError, match='not divisible'):
            divide(10, 3)
        with pytest.raises(ValueError, match='not divisible'):
            divide(100, 7)

    def test_divide_by_one(self):
        """Test division by 1."""
        assert divide(10, 1) == 10


class TestColumnParallelLinear:
    """Tests for ColumnParallelLinear layer."""

    def test_basic_initialization(self):
        """Test basic initialization without tensor parallelism."""
        linear = ColumnParallelLinear(16, 32, bias=True)
        assert linear.input_size == 16
        assert linear.output_size == 32
        assert linear.weight.shape == (32, 16)
        assert linear.bias is not None
        assert linear.bias.shape == (32, )

    def test_initialization_no_bias(self):
        """Test initialization without bias."""
        linear = ColumnParallelLinear(16, 32, bias=False)
        assert linear.bias is None

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        linear = ColumnParallelLinear(16, 32)
        x = torch.randn(2, 10, 16)
        output = linear(x)
        assert output.shape == (2, 10, 32)

    def test_forward_pass_single_batch(self):
        """Test forward pass with single batch dimension."""
        linear = ColumnParallelLinear(16, 32)
        x = torch.randn(10, 16)
        output = linear(x)
        assert output.shape == (10, 32)


class TestRowParallelLinear:
    """Tests for RowParallelLinear layer."""

    def test_basic_initialization(self):
        """Test basic initialization."""
        linear = RowParallelLinear(16, 32, bias=True)
        assert linear.input_size == 16
        assert linear.output_size == 32
        assert linear.weight.shape == (32, 16)
        assert linear.bias is not None
        assert linear.bias.shape == (32, )

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        linear = RowParallelLinear(16, 32)
        x = torch.randn(2, 10, 16)
        output = linear(x)
        assert output.shape == (2, 10, 32)


class TestQKVParallelLinear:
    """Tests for QKVParallelLinear layer."""

    def test_basic_initialization(self):
        """Test basic initialization with equal heads."""
        # hidden=16, head_dim=4, num_heads=4, num_kv_heads=4
        linear = QKVParallelLinear(16, 4, 4, 4, bias=True)
        # output = (4 + 2*4) * 4 = 12 * 4 = 48
        assert linear.weight.shape == (48, 16)
        assert linear.bias is not None
        assert linear.bias.shape == (48, )

    def test_mqa_initialization(self):
        """Test initialization with MQA (multi-query attention)."""
        # hidden=16, head_dim=4, num_heads=4, num_kv_heads=1
        linear = QKVParallelLinear(16, 4, 4, 1)
        # output = (4 + 2*1) * 4 = 6 * 4 = 24
        assert linear.weight.shape == (24, 16)

    def test_gqa_initialization(self):
        """Test initialization with GQA (grouped-query attention)."""
        # hidden=32, head_dim=8, num_heads=8, num_kv_heads=2
        linear = QKVParallelLinear(32, 8, 8, 2)
        # output = (8 + 2*2) * 8 = 12 * 8 = 96
        assert linear.weight.shape == (96, 32)

    def test_forward_pass(self):
        """Test forward pass."""
        linear = QKVParallelLinear(16, 4, 4, 4)
        x = torch.randn(2, 10, 16)
        output = linear(x)
        # Output should be (batch, seq, q_size + 2 * kv_size)
        # = (2, 10, 16 + 2*16) = (2, 10, 48)
        assert output.shape == (2, 10, 48)

    def test_forward_pass_no_bias(self):
        """Test forward pass without bias."""
        linear = QKVParallelLinear(16, 4, 4, 4, bias=False)
        x = torch.randn(2, 10, 16)
        output = linear(x)
        assert output.shape == (2, 10, 48)

    def test_split_qkv(self):
        """Test that QKV can be split correctly after forward pass."""
        hidden_size = 16
        head_dim = 4
        num_heads = 4
        num_kv_heads = 2

        linear = QKVParallelLinear(hidden_size, head_dim, num_heads,
                                   num_kv_heads)
        x = torch.randn(2, 10, hidden_size)
        output = linear(x)

        # Split the output
        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim

        q = output[..., :q_size]
        k = output[..., q_size:q_size + kv_size]
        v = output[..., q_size + kv_size:]

        assert q.shape == (2, 10, q_size)
        assert k.shape == (2, 10, kv_size)
        assert v.shape == (2, 10, kv_size)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
