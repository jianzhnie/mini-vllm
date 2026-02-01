import unittest

import torch

from minivllm.models.layers.rotary_embedding import RotaryEmbedding, get_rope


class TestRotaryEmbedding(unittest.TestCase):
    """Test RotaryEmbedding."""

    def test_rotary_embedding_shape(self):
        batch = 2
        seq_len = 8
        num_heads = 4
        head_size = 16
        rotary_dim = 16
        max_pos = 100
        base = 10000.0

        rope = RotaryEmbedding(head_size, rotary_dim, max_pos, base)

        positions = torch.randint(0, max_pos, (batch, seq_len))
        query = torch.randn(batch, num_heads, seq_len, head_size)
        key = torch.randn(batch, num_heads, seq_len, head_size)

        q_rot, k_rot = rope(positions, query, key)

        self.assertEqual(q_rot.shape, query.shape)
        self.assertEqual(k_rot.shape, key.shape)

    def test_rope_scaling_linear(self):
        head_size = 16
        rotary_dim = 16
        max_pos = 100
        base = 10000.0
        scaling = {'type': 'linear', 'factor': 2.0}

        rope = RotaryEmbedding(head_size,
                               rotary_dim,
                               max_pos,
                               base,
                               rope_scaling=scaling)

        # Check inv_freq
        # Original: 1.0 / (base ** (arange / dim))
        # Scaled: inv_freq / factor

        rope_orig = RotaryEmbedding(head_size, rotary_dim, max_pos, base)

        # Access internal inv_freq if possible, but it's not stored as attribute directly
        # It's used to compute cos_sin_cache

        # We can check cos_cache and sin_cache
        # cache shape: (max_pos, dim/2)
        # linear scaling effectively reduces frequency, so cos(pos * freq/factor)
        # cos(2 * freq/2) == cos(1 * freq)

        # Position 2 in scaled should match position 1 in original

        cos_scaled = rope.cos_cache[2]
        cos_orig = rope_orig.cos_cache[1]
        self.assertTrue(torch.allclose(cos_scaled, cos_orig, atol=1e-5))

        sin_scaled = rope.sin_cache[2]
        sin_orig = rope_orig.sin_cache[1]
        self.assertTrue(torch.allclose(sin_scaled, sin_orig, atol=1e-5))

    def test_get_rope_cache(self):
        head_size = 16
        rotary_dim = 16
        max_pos = 100
        base = 10000.0

        rope1 = get_rope(head_size, rotary_dim, max_pos, base)
        rope2 = get_rope(head_size, rotary_dim, max_pos, base)

        self.assertIs(rope1, rope2)

        scaling = {'type': 'linear', 'factor': 2.0}
        rope3 = get_rope(head_size,
                         rotary_dim,
                         max_pos,
                         base,
                         rope_scaling=scaling)

        self.assertIsNot(rope1, rope3)


if __name__ == '__main__':
    unittest.main()
