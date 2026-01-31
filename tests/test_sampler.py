import pytest
import torch

from minivllm.sampling.config import SamplingConfig
from minivllm.sampling.mirostat import MirostatSampler, MirostatV2Sampler
from minivllm.sampling.sampler import Sampler


class TestSamplerConfig:
    """Tests for Sampler initialized with SamplingConfig (global settings)."""

    def setup_method(self):
        self.sampler = Sampler()

    def test_greedy_sampling(self):
        """Test that default config (temp=1.0) with argmax or low temp works."""
        logits = torch.tensor([[10.0, 1.0, 1.0]])
        # With high diff, even temp=1 should pick index 0 almost always.
        torch.manual_seed(42)
        token = self.sampler(logits)
        assert token.item() == 0

        # Explicit low temp via config
        config = SamplingConfig(temperature=1e-5)
        sampler = Sampler(config)
        token = sampler(logits)
        assert token.item() == 0

    def test_temperature_distribution(self):
        """Test that high temperature produces uniform distribution."""
        logits = torch.tensor([[1.0, 1.0, 1.0]])
        config = SamplingConfig(temperature=100.0)
        sampler = Sampler(config)

        torch.manual_seed(42)
        samples = [sampler(logits).item() for _ in range(100)]
        counts = {i: samples.count(i) for i in range(3)}
        # Should be roughly equal
        assert all(c > 20 for c in counts.values())

    def test_top_k_config(self):
        """Test Top-K via config."""
        # logits: 0=10, 1=9, 2=1, 3=0
        logits = torch.tensor([[10.0, 9.0, 1.0, 0.0]])
        config = SamplingConfig(top_k=2)
        sampler = Sampler(config)

        # Should only pick 0 or 1
        torch.manual_seed(42)
        for _ in range(20):
            token = sampler(logits)
            assert token.item() in [0, 1]

    def test_top_p_config(self):
        """Test Top-P via config."""
        # logits: 0=high, 1=high, 2=low
        logits = torch.tensor([[10.0, 10.0, -10.0]])
        config = SamplingConfig(top_p=0.9)  # Should keep 0 and 1
        sampler = Sampler(config)

        torch.manual_seed(42)
        for _ in range(20):
            token = sampler(logits)
            assert token.item() in [0, 1]

    def test_typical_sampling(self):
        """Test Typical Sampling."""
        # Case: [0.9, 0.05, 0.05] -> Entropy low.
        # If tau is small, we might filter out 1 and 2.
        logits = torch.tensor([[5.0, 2.0,
                                2.0]])  # Softmax approx [0.9, 0.05, 0.05]
        config = SamplingConfig(typical_p=0.2)  # Strict typicality
        sampler = Sampler(config)

        # Should likely pick 0
        token = sampler(logits)
        assert token.item() == 0

    def test_top_token_restriction(self):
        """Test Avoid Top-K (Top Token Restriction)."""
        logits = torch.tensor([[10.0, 9.0, 8.0, 7.0]])
        # Avoid top 1 (index 0)
        config = SamplingConfig(avoid_top_k=1)
        sampler = Sampler(config)

        torch.manual_seed(42)
        for _ in range(10):
            token = sampler(logits)
            assert token.item() != 0

    def test_repetition_penalty(self):
        """Test Repetition Penalty."""
        logits = torch.tensor([[1.0, 1.0, 1.0]])
        prev_tokens = torch.tensor([[0]])  # 0 was generated
        config = SamplingConfig(repetition_penalty=2.0)
        sampler = Sampler(config)

        torch.manual_seed(42)
        samples = [
            sampler(logits, prev_tokens=prev_tokens).item() for _ in range(50)
        ]
        # 0 should be rare
        assert samples.count(0) < samples.count(1)


class TestSamplerOverrides:
    """Tests for Sampler with per-batch overrides (temperatures, top_ks, etc.)."""

    def setup_method(self):
        self.sampler = Sampler()

    def test_temperature_scaling_override(self):
        """Test temperature scaling for batch inputs."""
        logits = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]])
        # Seq 0: Low temp (argmax-like), Seq 1: High temp (uniform-like)
        temperatures = torch.tensor([0.01, 100.0])

        # Seq 0: clear winner
        logits[0] = torch.tensor([10.0, 1.0, 1.0, 1.0])

        for _ in range(10):
            tokens = self.sampler(logits, temperatures=temperatures)
            assert tokens[0].item() == 0
            assert 0 <= tokens[1].item() < 4

    def test_top_k_override(self):
        """Test vectorized Top-K filtering."""
        logits = torch.tensor([
            [10.0, 9.0, 8.0, 1.0, 0.0],  # Seq 0
            [10.0, 9.0, 8.0, 1.0, 0.0],  # Seq 1
        ])
        temperatures = torch.tensor([1.0, 1.0])
        # Seq 0: k=2 (keep 10.0, 9.0), Seq 1: k=1 (keep 10.0)
        top_ks = torch.tensor([2, 1])

        for _ in range(20):
            tokens = self.sampler(logits,
                                  temperatures=temperatures,
                                  top_ks=top_ks)
            assert tokens[0].item() in [0, 1]
            assert tokens[1].item() == 0

    def test_top_p_override(self):
        """Test Top-P (nucleus) filtering override."""
        logits = torch.log(
            torch.tensor([
                [0.6, 0.2, 0.1, 0.1],
                [0.6, 0.2, 0.1, 0.1],
            ]))
        temperatures = torch.tensor([1.0, 1.0])
        # Seq 0: p=0.5 (keep 0.6 only), Seq 1: p=0.9 (keep 0.6, 0.2, 0.1)
        top_ps = torch.tensor([0.5, 0.9])

        for _ in range(20):
            tokens = self.sampler(logits,
                                  temperatures=temperatures,
                                  top_ps=top_ps)
            assert tokens[0].item() == 0
            assert tokens[1].item() in [0, 1, 2, 3]

    def test_min_p_override(self):
        """Test Min-P filtering override."""
        logits = torch.log(
            torch.tensor([[0.8, 0.1, 0.05, 0.05], [0.8, 0.1, 0.05, 0.05]]))
        temperatures = torch.tensor([1.0, 1.0])
        # Seq 0: min_p = 0.2 (thresh=0.16) -> keep 0.8
        # Seq 1: min_p = 0.1 (thresh=0.08) -> keep 0.8, 0.1
        min_ps = torch.tensor([0.2, 0.1])

        for _ in range(20):
            tokens = self.sampler(logits,
                                  temperatures=temperatures,
                                  min_ps=min_ps)
            assert tokens[0].item() == 0
            assert tokens[1].item() in [0, 1]


class TestSamplerEdgeCases:
    """Tests for edge cases and validation."""

    def setup_method(self):
        self.sampler = Sampler()

    def test_recovery_from_all_filtered(self):
        """Test recovery when all tokens are filtered out."""
        # Case: min_p > 1.0 should filter everything out
        logits = torch.tensor([[10.0, 5.0, 1.0]])
        temperatures = torch.tensor([1.0])
        min_ps = torch.tensor([2.0])

        # The sampler should detect all -inf and restore original logits or use fallback
        tokens = self.sampler(logits, temperatures=temperatures, min_ps=min_ps)
        assert 0 <= tokens.item() < 3

    def test_nan_handling(self):
        """Test handling of NaN logits."""
        logits = torch.tensor([[float('nan'), 1.0, 0.0], [1.0, 2.0, 3.0]])
        temperatures = torch.tensor([1.0, 1.0])

        # Should not crash and ignore NaNs
        tokens = self.sampler(logits, temperatures=temperatures)
        assert tokens.shape == (2, )

        # If all are NaN
        logits_all_nan = torch.full((1, 5), float('nan'))
        temperatures_nan = torch.tensor([1.0])
        tokens = self.sampler(logits_all_nan, temperatures=temperatures_nan)
        assert 0 <= tokens.item() < 5

    def test_input_validation(self):
        """Test input validation."""
        logits = torch.randn(2, 10)
        temperatures = torch.randn(3)  # Mismatch size

        # Expect runtime error due to broadcasting failure or manual check
        with pytest.raises((ValueError, RuntimeError)):
            self.sampler(logits, temperatures=temperatures)


class TestMirostat:
    """Tests for Mirostat samplers."""

    def test_mirostat_v1(self):
        logits = torch.randn(1, 100)
        sampler = MirostatSampler(target_perplexity=3.0)
        for _ in range(5):
            token = sampler(logits)
            assert token.shape == (1, )
            assert 0 <= token.item() < 100

    def test_mirostat_v2(self):
        logits = torch.randn(1, 100)
        sampler = MirostatV2Sampler(target_perplexity=3.0)
        for _ in range(5):
            token = sampler(logits)
            assert token.shape == (1, )
            assert 0 <= token.item() < 100


if __name__ == '__main__':
    pytest.main([__file__])
