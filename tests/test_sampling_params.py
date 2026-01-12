"""Unit tests for sampling parameters module.

Tests the SamplingParams dataclass to ensure proper validation.
"""

import pytest

from minivllm.sampling_params import SamplingParams


class TestSamplingParamsValidation:
    """Test SamplingParams validation."""

    def test_valid_temperature(self) -> None:
        """Test that valid temperature values are accepted."""
        params = SamplingParams(temperature=0.8, max_tokens=50)
        assert params.temperature == 0.8
        assert params.max_tokens == 50

    def test_temperature_too_low(self) -> None:
        """Test that temperature <= 1e-10 is rejected."""
        with pytest.raises(ValueError, match='temperature'):
            SamplingParams(temperature=0)

    def test_temperature_zero(self) -> None:
        """Test that temperature=0 (greedy sampling) is rejected."""
        with pytest.raises(ValueError, match='temperature'):
            SamplingParams(temperature=0.0)

    def test_max_tokens_positive(self) -> None:
        """Test that max_tokens must be positive."""
        with pytest.raises(ValueError, match='max_tokens'):
            SamplingParams(max_tokens=0)

    def test_max_tokens_negative(self) -> None:
        """Test that max_tokens cannot be negative."""
        with pytest.raises(ValueError, match='max_tokens'):
            SamplingParams(max_tokens=-10)

    def test_ignore_eos_flag(self) -> None:
        """Test ignore_eos flag can be set."""
        params = SamplingParams(temperature=0.7,
                                max_tokens=100,
                                ignore_eos=True)
        assert params.ignore_eos is True

    def test_default_values(self) -> None:
        """Test SamplingParams default values."""
        params = SamplingParams()
        assert params.temperature == 1.0
        assert params.max_tokens == 64
        assert params.ignore_eos is False


class TestSamplingParamsEdgeCases:
    """Test edge cases for SamplingParams."""

    def test_temperature_minimum_valid(self) -> None:
        """Test the minimum valid temperature."""
        params = SamplingParams(temperature=1e-9)  # Just above threshold
        assert params.temperature == 1e-9

    def test_temperature_very_high(self) -> None:
        """Test very high temperature values."""
        params = SamplingParams(temperature=100.0)
        assert params.temperature == 100.0

    def test_max_tokens_very_large(self) -> None:
        """Test very large max_tokens."""
        params = SamplingParams(max_tokens=1000000)
        assert params.max_tokens == 1000000


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
