"""Tests for SamplingConfig dataclass.

This module tests the SamplingConfig class which configures
text generation sampling parameters.
"""

import pytest

from minivllm.sampling.config import SamplingConfig


class TestSamplingConfigDefaults:
    """Tests for SamplingConfig default values."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = SamplingConfig()

        assert config.temperature == 1.0
        assert config.top_k == 0
        assert config.top_p == 1.0
        assert config.min_p == 0.0
        assert config.typical_p == 1.0
        assert config.repetition_penalty == 1.0
        assert config.frequency_penalty == 0.0
        assert config.presence_penalty == 0.0
        assert config.avoid_top_k == 0
        assert config.seed is None


class TestSamplingConfigValidation:
    """Tests for SamplingConfig validation."""

    def test_valid_temperature(self):
        """Test valid temperature values."""
        # Valid temperatures
        SamplingConfig(temperature=0.0)
        SamplingConfig(temperature=0.5)
        SamplingConfig(temperature=1.0)
        SamplingConfig(temperature=2.0)
        SamplingConfig(temperature=100.0)

    def test_negative_temperature_raises(self):
        """Test that negative temperature raises ValueError."""
        with pytest.raises(ValueError,
                           match='Temperature must be non-negative'):
            SamplingConfig(temperature=-0.1)
        with pytest.raises(ValueError,
                           match='Temperature must be non-negative'):
            SamplingConfig(temperature=-1.0)

    def test_valid_top_k(self):
        """Test valid top_k values."""
        SamplingConfig(top_k=0)  # Disabled
        SamplingConfig(top_k=1)
        SamplingConfig(top_k=10)
        SamplingConfig(top_k=100)

    def test_negative_top_k_raises(self):
        """Test that negative top_k raises ValueError."""
        with pytest.raises(ValueError, match='Top-k must be non-negative'):
            SamplingConfig(top_k=-1)
        with pytest.raises(ValueError, match='Top-k must be non-negative'):
            SamplingConfig(top_k=-10)

    def test_valid_top_p(self):
        """Test valid top_p values."""
        SamplingConfig(top_p=0.0)
        SamplingConfig(top_p=0.5)
        SamplingConfig(top_p=1.0)

    def test_invalid_top_p_raises(self):
        """Test that invalid top_p raises ValueError."""
        with pytest.raises(ValueError,
                           match='Top-p must be between 0.0 and 1.0'):
            SamplingConfig(top_p=-0.1)
        with pytest.raises(ValueError,
                           match='Top-p must be between 0.0 and 1.0'):
            SamplingConfig(top_p=1.1)

    def test_valid_min_p(self):
        """Test valid min_p values."""
        SamplingConfig(min_p=0.0)
        SamplingConfig(min_p=0.05)
        SamplingConfig(min_p=1.0)

    def test_invalid_min_p_raises(self):
        """Test that invalid min_p raises ValueError."""
        with pytest.raises(ValueError,
                           match='Min-p must be between 0.0 and 1.0'):
            SamplingConfig(min_p=-0.1)
        with pytest.raises(ValueError,
                           match='Min-p must be between 0.0 and 1.0'):
            SamplingConfig(min_p=1.1)

    def test_valid_typical_p(self):
        """Test valid typical_p values."""
        SamplingConfig(typical_p=0.1)
        SamplingConfig(typical_p=1.0)
        SamplingConfig(typical_p=2.0)

    def test_invalid_typical_p_raises(self):
        """Test that non-positive typical_p raises ValueError."""
        with pytest.raises(ValueError, match='Typical p.*must be positive'):
            SamplingConfig(typical_p=0.0)
        with pytest.raises(ValueError, match='Typical p.*must be positive'):
            SamplingConfig(typical_p=-1.0)

    def test_valid_avoid_top_k(self):
        """Test valid avoid_top_k values."""
        SamplingConfig(avoid_top_k=0)  # Disabled
        SamplingConfig(avoid_top_k=1)
        SamplingConfig(avoid_top_k=10)

    def test_negative_avoid_top_k_raises(self):
        """Test that negative avoid_top_k raises ValueError."""
        with pytest.raises(ValueError,
                           match='Avoid top-k must be non-negative'):
            SamplingConfig(avoid_top_k=-1)


class TestSamplingConfigCreation:
    """Tests for creating SamplingConfig with various parameters."""

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = SamplingConfig(
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            min_p=0.05,
            typical_p=0.5,
            repetition_penalty=1.1,
            frequency_penalty=0.5,
            presence_penalty=0.3,
            avoid_top_k=5,
            seed=42,
        )

        assert config.temperature == 0.7
        assert config.top_k == 50
        assert config.top_p == 0.9
        assert config.min_p == 0.05
        assert config.typical_p == 0.5
        assert config.repetition_penalty == 1.1
        assert config.frequency_penalty == 0.5
        assert config.presence_penalty == 0.3
        assert config.avoid_top_k == 5
        assert config.seed == 42

    def test_partial_custom_values(self):
        """Test creating config with some custom values."""
        config = SamplingConfig(temperature=0.5, top_k=40)

        assert config.temperature == 0.5
        assert config.top_k == 40
        # Rest should be defaults
        assert config.top_p == 1.0
        assert config.min_p == 0.0
        assert config.seed is None


class TestSamplingConfigModification:
    """Tests for SamplingConfig modification behavior."""

    def test_config_is_mutable(self):
        """Test that config is mutable (not frozen)."""
        config = SamplingConfig()

        # Should be able to modify
        config.temperature = 0.5
        assert config.temperature == 0.5

        config.top_k = 50
        assert config.top_k == 50


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
