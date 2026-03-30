"""Tests for Rotary Positional Embedding (RoPE).

RoPE encodes position information by rotating the query and key vectors
in the attention mechanism based on their absolute positions.
"""

import pytest
import torch

from minivllm.models.layers.rotary_embedding import RotaryEmbedding, get_rope


class TestRotaryEmbeddingBasics:
    """Basic functionality tests for RotaryEmbedding."""

    def test_initialization(self):
        """Test RotaryEmbedding initialization."""
        rope = RotaryEmbedding(
            head_size=64,
            rotary_dim=64,
            max_position_embeddings=2048,
            base=10000.0,
        )
        assert rope.head_size == 64
        assert rope.max_position_embeddings == 2048

    def test_cache_generation(self):
        """Test that cos/sin caches are generated."""
        rope = RotaryEmbedding(
            head_size=16,
            rotary_dim=16,
            max_position_embeddings=100,
            base=10000.0,
        )
        assert hasattr(rope, 'cos_cache')
        assert hasattr(rope, 'sin_cache')
        # Cache shape should be (max_position, rotary_dim // 2)
        expected_shape = (100, 8)
        assert rope.cos_cache.shape == expected_shape
        assert rope.sin_cache.shape == expected_shape

    def test_forward_shape(self):
        """Test that forward pass maintains tensor shapes."""
        batch = 2
        seq_len = 8
        num_heads = 4
        head_size = 16
        rotary_dim = 16
        max_pos = 100

        rope = RotaryEmbedding(head_size, rotary_dim, max_pos, base=10000.0)

        positions = torch.randint(0, max_pos, (batch, seq_len))
        query = torch.randn(batch, num_heads, seq_len, head_size)
        key = torch.randn(batch, num_heads, seq_len, head_size)

        q_rot, k_rot = rope(positions, query, key)

        assert q_rot.shape == query.shape
        assert k_rot.shape == key.shape


class TestRotaryEmbeddingProperties:
    """Tests for RoPE mathematical properties."""

    def test_relative_position_encoding(self):
        """Test that RoPE encodes relative positions correctly.

        The key property of RoPE is that the dot product of rotated vectors
        depends only on their relative positions, not absolute positions.
        """
        head_size = 16
        rotary_dim = 16
        rope = RotaryEmbedding(head_size,
                               rotary_dim,
                               max_position_embeddings=100,
                               base=10000.0)

        # Create identical query and key at different positions
        q = torch.randn(1, 1, 1, head_size)
        k = q.clone()

        # Position 10 for query, position 5 for key -> relative distance 5
        pos_q = torch.tensor([[10]])
        pos_k = torch.tensor([[5]])

        q_rot, _ = rope(pos_q, q, k)
        _, k_rot = rope(pos_k, q, k)

        # Dot product should be the same regardless of absolute positions
        # as long as relative distance is the same
        attn_10_5 = torch.sum(q_rot * k_rot)

        # Now try position 20 and 15 -> same relative distance 5
        pos_q2 = torch.tensor([[20]])
        pos_k2 = torch.tensor([[15]])

        q_rot2, _ = rope(pos_q2, q, k)
        _, k_rot2 = rope(pos_k2, q, k)

        attn_20_15 = torch.sum(q_rot2 * k_rot2)

        # Attention scores should be similar (allowing for small numerical differences)
        assert torch.allclose(attn_10_5, attn_20_15, atol=1e-5)

    def test_long_term_decay(self):
        """Test that attention decays for distant positions.

        RoPE should make attention scores smaller for distant positions.
        """
        head_size = 16
        rotary_dim = 16
        rope = RotaryEmbedding(head_size,
                               rotary_dim,
                               max_position_embeddings=1000,
                               base=10000.0)

        q = torch.randn(1, 1, 1, head_size)
        k = q.clone()

        # Same position -> max attention
        pos_same = torch.tensor([[10]])
        q_rot, k_rot = rope(pos_same, q, k)
        attn_same = torch.sum(q_rot * k_rot).item()

        # Distant positions -> lower attention
        pos_far_q = torch.tensor([[10]])
        pos_far_k = torch.tensor([[100]])
        q_rot_far, _ = rope(pos_far_q, q, k)
        _, k_rot_far = rope(pos_far_k, q, k)
        attn_far = torch.sum(q_rot_far * k_rot_far).item()

        assert attn_far < attn_same


class TestRotaryEmbeddingScaling:
    """Tests for RoPE scaling variants."""

    def test_linear_scaling(self):
        """Test linear position interpolation scaling.

        With factor=2.0, position 2 in scaled should match position 1 in original.
        """
        head_size = 16
        rotary_dim = 16
        max_pos = 100
        base = 10000.0
        scaling = {'type': 'linear', 'factor': 2.0}

        rope_scaled = RotaryEmbedding(head_size,
                                      rotary_dim,
                                      max_pos,
                                      base,
                                      rope_scaling=scaling)
        rope_orig = RotaryEmbedding(head_size, rotary_dim, max_pos, base)

        # Position 2 in scaled should match position 1 in original
        assert torch.allclose(rope_scaled.cos_cache[2],
                              rope_orig.cos_cache[1],
                              atol=1e-5)
        assert torch.allclose(rope_scaled.sin_cache[2],
                              rope_orig.sin_cache[1],
                              atol=1e-5)

    def test_no_scaling(self):
        """Test that no scaling produces expected cache."""
        head_size = 16
        rotary_dim = 16
        max_pos = 100

        rope = RotaryEmbedding(head_size, rotary_dim, max_pos, base=10000.0)

        # Position 0 should have cos=1, sin=0
        assert torch.allclose(rope.cos_cache[0],
                              torch.ones(rotary_dim // 2),
                              atol=1e-6)
        assert torch.allclose(rope.sin_cache[0],
                              torch.zeros(rotary_dim // 2),
                              atol=1e-6)


class TestGetRope:
    """Tests for the get_rope factory function."""

    def test_cache_reuse(self):
        """Test that get_rope returns cached instances for same parameters."""
        head_size = 16
        rotary_dim = 16
        max_pos = 100
        base = 10000.0

        rope1 = get_rope(head_size, rotary_dim, max_pos, base)
        rope2 = get_rope(head_size, rotary_dim, max_pos, base)

        # Should be the same object (cached)
        assert rope1 is rope2

    def test_cache_different_params(self):
        """Test that different parameters create different instances."""
        head_size = 16
        rotary_dim = 16
        max_pos = 100
        base = 10000.0

        rope1 = get_rope(head_size, rotary_dim, max_pos, base)
        rope2 = get_rope(head_size,
                         rotary_dim,
                         max_pos,
                         base,
                         rope_scaling={
                             'type': 'linear',
                             'factor': 2.0
                         })

        # Should be different objects
        assert rope1 is not rope2

    def test_cache_different_max_pos(self):
        """Test that different max_position creates different instances."""
        head_size = 16
        rotary_dim = 16
        base = 10000.0

        rope1 = get_rope(head_size, rotary_dim, 100, base)
        rope2 = get_rope(head_size, rotary_dim, 200, base)

        assert rope1 is not rope2


class TestRotaryEmbeddingEdgeCases:
    """Edge case tests for RotaryEmbedding."""

    def test_partial_rotary_dim(self):
        """Test with rotary_dim < head_size raises error."""
        head_size = 64
        rotary_dim = 32  # Only rotate first half

        # Current implementation requires rotary_dim == head_size
        with pytest.raises(ValueError,
                           match='rotary_dim must equal head_size'):
            RotaryEmbedding(head_size,
                            rotary_dim,
                            max_position_embeddings=100,
                            base=10000.0)

    def test_batch_positions(self):
        """Test with batched positions of different shapes."""
        head_size = 16
        rotary_dim = 16
        rope = RotaryEmbedding(head_size,
                               rotary_dim,
                               max_position_embeddings=100,
                               base=10000.0)

        # 2D positions (batch, seq)
        positions = torch.tensor([[0, 1, 2], [3, 4, 5]])
        query = torch.randn(2, 4, 3, head_size)
        key = torch.randn(2, 4, 3, head_size)

        q_rot, k_rot = rope(positions, query, key)
        assert q_rot.shape == (2, 4, 3, head_size)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
