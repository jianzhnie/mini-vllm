import pytest
import torch

from minivllm.sampling import (
    FrequencyPenaltySampler, GreedySampler, MinPSampler, NucleusSampler,
    PresencePenaltySampler, RandomSampler, RepetitionPenaltySampler,
    TemperatureMinPTopKSampler, TemperatureSampler, TopKSampler,
    TopKTopPSampler, TopPSampler, TypicalSampler)


class TestBasicSamplers:
    """Test basic sampling methods."""

    def test_greedy_sampler(self):
        """Test that greedy sampler returns the argmax."""
        logits = torch.tensor([[1.0, 5.0, 3.0]])
        sampler = GreedySampler()
        token = sampler(logits)
        assert token.item(
        ) == 1, 'Greedy should select index 1 (highest logit)'

    def test_greedy_sampler_batch(self):
        """Test greedy sampler with batch input."""
        logits = torch.tensor([[1.0, 5.0, 3.0], [2.0, 1.0, 4.0]])
        sampler = GreedySampler()
        tokens = sampler(logits)
        assert tokens.tolist() == [1, 2], 'Batch greedy should work correctly'

    def test_random_sampler(self):
        """Test that random sampler produces valid token indices."""
        logits = torch.randn(1, 100)
        sampler = RandomSampler()

        # Sample multiple times to check distribution
        tokens = [sampler(logits).item() for _ in range(100)]

        # All tokens should be valid indices
        assert all(0 <= t < 100 for t in tokens)

        # Should not always be the same token (with very high probability)
        assert len(set(tokens)) > 1

    def test_temperature_sampler_extreme_temperatures(self):
        """Test temperature sampler with extreme values."""
        logits = torch.tensor([[0.0, 10.0, 0.0]])

        # Very low temperature should heavily prefer the max
        low_temp_sampler = TemperatureSampler(temperature=0.001)
        low_temp_tokens = [low_temp_sampler(logits).item() for _ in range(100)]
        assert low_temp_tokens.count(
            1) > 90, 'Low temp should mostly pick index 1'

        # Very high temperature should be more uniform
        high_temp_sampler = TemperatureSampler(temperature=100.0)
        high_temp_tokens = [
            high_temp_sampler(logits).item() for _ in range(1000)
        ]

        # With high temperature, distribution should be more uniform
        counts = {i: high_temp_tokens.count(i) for i in range(3)}
        # Check that no single token dominates
        assert max(counts.values()) < 900, 'High temp should be more uniform'


class TestFilteringSamplers:
    """Test filtering-based sampling methods."""

    def test_topk_sampler(self):
        """Test top-k sampler."""
        logits = torch.tensor([[1.0, 9.0, 2.0, 8.0, 0.0]])
        inner_sampler = GreedySampler()
        topk_sampler = TopKSampler(k=2, sampler=inner_sampler)

        # With k=2, only top 2 tokens (indices 1 and 3) are valid
        token = topk_sampler(logits)
        assert token.item() in [
            1, 3
        ], f'Top-K should select from top 2 tokens, got {token.item()}'

    def test_topk_sampler_k_larger_than_vocab(self):
        """Test top-k when k > vocabulary size."""
        logits = torch.randn(1, 10)
        inner_sampler = RandomSampler()
        topk_sampler = TopKSampler(k=1000, sampler=inner_sampler)

        # Should still work even when k > vocab_size
        token = topk_sampler(logits)
        assert 0 <= token.item(
        ) < 10, 'Should return valid token even with large k'

    def test_topp_sampler(self):
        """Test top-p (nucleus) sampler."""
        # Create logits where we know the probability distribution
        logits = torch.tensor([[10.0, 5.0, 1.0, 1.0]])
        topp_sampler = TopPSampler(p=0.95, temperature=1.0)

        # Sample multiple times
        tokens = [topp_sampler(logits).item() for _ in range(100)]

        # With p=0.95 and these logits, tokens 0 and 1 should dominate
        valid_tokens = [0, 1]
        assert all(t in valid_tokens
                   for t in tokens), 'Top-P should select from nucleus'

    def test_minp_sampler(self):
        """Test min-p sampler."""
        logits = torch.tensor([[10.0, 5.0, 1.0, 1.0]])
        inner_sampler = RandomSampler()
        minp_sampler = MinPSampler(p=0.1, sampler=inner_sampler)

        token = minp_sampler(logits)
        assert 0 <= token.item() < 4, 'Min-P should return valid token'

    def test_typical_sampler(self):
        """Test typical sampler."""
        logits = torch.randn(1, 100)
        inner_sampler = RandomSampler()
        typical_sampler = TypicalSampler(tau=1.0, sampler=inner_sampler)

        token = typical_sampler(logits)
        assert 0 <= token.item(
        ) < 100, 'Typical sampler should return valid token'

    def test_nucleus_sampler_composite(self):
        """Test nucleus sampler (composite form)."""
        logits = torch.randn(1, 100)
        inner_sampler = TemperatureSampler(temperature=1.0)
        nucleus_sampler = NucleusSampler(p=0.9, sampler=inner_sampler)

        token = nucleus_sampler(logits)
        assert 0 <= token.item(
        ) < 100, 'Nucleus sampler should return valid token'


class TestCombinedSamplers:
    """Test combined sampling methods."""

    def test_topk_topp_combined(self):
        """Test combined top-k and top-p sampler."""
        logits = torch.randn(1, 100)
        sampler = TopKTopPSampler(k=50, p=0.95, temperature=1.0)

        # Sample multiple times
        tokens = [sampler(logits).item() for _ in range(100)]

        # All should be valid
        assert all(
            0 <= t < 100
            for t in tokens), 'Combined sampler should produce valid tokens'
        assert len(set(tokens)) > 1, 'Should have some diversity'

    def test_temperature_minp_topk_combined(self):
        """Test combined temperature, min-p, and top-k sampler."""
        logits = torch.randn(1, 100)
        sampler = TemperatureMinPTopKSampler(temperature=1.0,
                                             min_p=0.05,
                                             top_k=50)

        token = sampler(logits)
        assert 0 <= token.item(
        ) < 100, 'Combined sampler should produce valid token'


class TestPenaltySamplers:
    """Test penalty-based sampling methods."""

    def test_repetition_penalty(self):
        """Test repetition penalty sampler."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        prev_tokens = torch.tensor([1,
                                    2])  # Previously generated tokens 1 and 2

        sampler = RepetitionPenaltySampler(penalty=2.0)
        token = sampler(logits, prev_tokens)

        assert 0 <= token.item() < 4, 'Should return valid token'

    def test_frequency_penalty(self):
        """Test frequency penalty sampler."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        sequence = torch.tensor(
            [0, 1, 0, 2, 0,
             1])  # Token 0 appears 3 times, 1 appears 2 times, etc.

        sampler = FrequencyPenaltySampler(alpha=0.5)
        token = sampler(logits, sequence)

        assert 0 <= token.item() < 4, 'Should return valid token'

    def test_presence_penalty(self):
        """Test presence penalty sampler."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        sequence = torch.tensor([0, 1, 2])  # Tokens 0, 1, 2 have appeared

        sampler = PresencePenaltySampler(penalty=0.5)
        token = sampler(logits, sequence)

        assert 0 <= token.item() < 4, 'Should return valid token'


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_token_logits(self):
        """Test samplers with single token vocabulary."""
        logits = torch.tensor([[5.0]])

        greedy = GreedySampler()
        token = greedy(logits)
        assert token.item() == 0, 'Single token should return 0'

    def test_very_small_logits(self):
        """Test samplers with very small logits."""
        logits = torch.ones(1, 100) * 1e-6

        sampler = TemperatureSampler(temperature=1.0)
        token = sampler(logits)
        assert 0 <= token.item() < 100, 'Should handle small logits'

    def test_mixed_positive_negative_logits(self):
        """Test samplers with mixed positive/negative logits."""
        logits = torch.tensor([[-10.0, 0.0, 10.0, -5.0, 5.0]])

        sampler = TopKSampler(k=3, sampler=GreedySampler())
        token = sampler(logits)
        assert token.item() in [2, 4, 0], 'Should work with mixed logits'

    def test_batch_processing(self):
        """Test samplers with batch inputs."""
        logits = torch.randn(8, 50257)

        sampler = TemperatureSampler(temperature=1.0)
        tokens = sampler(logits)

        assert tokens.shape == (
            8, ), f'Expected shape (8,), got {tokens.shape}'
        assert all(0 <= t < 50257
                   for t in tokens.tolist()), 'All tokens should be valid'

    def test_invalid_parameters(self):
        """Test that samplers reject invalid parameters."""
        with pytest.raises(ValueError):
            TemperatureSampler(temperature=-1.0)

        with pytest.raises(ValueError):
            MinPSampler(p=1.5, sampler=GreedySampler())

        with pytest.raises(ValueError):
            TopKSampler(k=0, sampler=GreedySampler())


class TestNumericalStability:
    """Test numerical stability of samplers."""

    def test_large_logit_values(self):
        """Test samplers with very large logit values."""
        logits = torch.tensor([[1000.0, 1001.0, 999.0]])

        sampler = TemperatureSampler(temperature=1.0)
        # Should not crash or produce NaNs
        token = sampler(logits)
        assert not torch.isnan(torch.tensor(
            float(token))), 'Should not produce NaN'

    def test_identical_logits(self):
        """Test samplers when all logits are identical."""
        logits = torch.ones(1, 100) * 5.0

        sampler = RandomSampler()
        tokens = [sampler(logits).item() for _ in range(100)]

        # Should sample uniformly
        assert len(set(
            tokens)) > 50, 'Should have good diversity with identical logits'


class TestConsistency:
    """Test consistency of samplers."""

    def test_greedy_is_deterministic(self):
        """Test that greedy sampler is always deterministic."""
        logits = torch.randn(1, 100)

        sampler = GreedySampler()
        token1 = sampler(logits)
        token2 = sampler(logits)

        assert token1 == token2, 'Greedy should be deterministic'

    def test_topk_filtering_effectiveness(self):
        """Test that top-k filtering actually reduces vocabulary."""
        logits = torch.randn(1, 1000)

        # Create a counter sampler to track which tokens are reached
        sampled_tokens = set()

        class CounterSampler:

            def __call__(self, filtered_logits):
                # Find non-infinite logits
                valid = ~torch.isinf(filtered_logits)
                valid_indices = valid.nonzero(as_tuple=True)
                if len(valid_indices[0]) > 0:
                    sampled_tokens.add(len(valid_indices[0]))
                return torch.tensor([[0]])

        topk_sampler = TopKSampler(k=50, sampler=CounterSampler())
        topk_sampler(logits)

        # Should have exactly 50 valid tokens
        assert 50 in sampled_tokens, 'Top-K should filter to exactly k tokens'


def run_tests():
    """Run all tests."""
    pytest.main([__file__, '-v'])


if __name__ == '__main__':
    run_tests()
