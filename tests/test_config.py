"""Unit tests for configuration module.

Tests the Config dataclass to ensure proper validation and initialization.
"""

from pathlib import Path

import pytest

from minivllm.config import Config


class TestConfigValidation:
    """Test Config validation and parameter constraints."""

    def test_device_memory_utilization_valid_range(
            self, temp_minimal_model_dir: Path) -> None:
        """Test that device_memory_utilization validation works correctly."""
        # Valid values should not raise
        config = Config(str(temp_minimal_model_dir),
                        device_memory_utilization=0.5)
        assert config.device_memory_utilization == 0.5

    def test_device_memory_utilization_too_low(
            self, temp_minimal_model_dir: Path) -> None:
        """Test that device_memory_utilization validation rejects too-low values."""
        with pytest.raises(ValueError, match='device_memory_utilization'):
            Config(str(temp_minimal_model_dir), device_memory_utilization=0.05)

    def test_device_memory_utilization_too_high(
            self, temp_minimal_model_dir: Path) -> None:
        """Test that device_memory_utilization validation rejects too-high values."""
        with pytest.raises(ValueError, match='device_memory_utilization'):
            Config(str(temp_minimal_model_dir), device_memory_utilization=1.5)

    def test_tensor_parallel_size_valid_range(
            self, temp_minimal_model_dir: Path) -> None:
        """Test that tensor_parallel_size validation works correctly."""
        # Valid values
        config = Config(str(temp_minimal_model_dir), tensor_parallel_size=4)
        assert config.tensor_parallel_size == 4

    def test_tensor_parallel_size_too_high(
            self, temp_minimal_model_dir: Path) -> None:
        """Test that tensor_parallel_size validation rejects too-high values."""
        with pytest.raises(ValueError, match='tensor_parallel_size'):
            Config(str(temp_minimal_model_dir), tensor_parallel_size=16)

    def test_kvcache_block_size_divisibility(
            self, temp_minimal_model_dir: Path) -> None:
        """Test that kvcache_block_size must be divisible by 256."""
        # Valid size (divisible by 256)
        config = Config(str(temp_minimal_model_dir), kvcache_block_size=512)
        assert config.kvcache_block_size == 512

    def test_kvcache_block_size_not_divisible(
            self, temp_minimal_model_dir: Path) -> None:
        """Test that kvcache_block_size validation rejects non-divisible values."""
        with pytest.raises(ValueError, match='kvcache_block_size'):
            Config(str(temp_minimal_model_dir), kvcache_block_size=300)

    def test_max_num_batched_tokens_positive(
            self, temp_minimal_model_dir: Path) -> None:
        """Test that max_num_batched_tokens must be positive."""
        with pytest.raises(ValueError, match='max_num_batched_tokens'):
            Config(str(temp_minimal_model_dir), max_num_batched_tokens=0)

    def test_gpu_memory_utilization_backward_compatibility(
            self, temp_minimal_model_dir: Path) -> None:
        """Test backward compatibility property for gpu_memory_utilization."""
        config = Config(str(temp_minimal_model_dir),
                        device_memory_utilization=0.7)
        assert config.gpu_memory_utilization == 0.7

        # Test setter
        config.gpu_memory_utilization = 0.8
        assert config.device_memory_utilization == 0.8


class TestConfigDefaults:
    """Test Config default values."""

    def test_default_values(self, temp_minimal_model_dir: Path) -> None:
        """Test that Config has reasonable default values."""
        config = Config(str(temp_minimal_model_dir))
        assert config.max_num_batched_tokens == 16384
        assert config.max_num_seqs == 512
        assert config.max_model_len == 4096
        assert config.device_memory_utilization == 0.9
        assert config.tensor_parallel_size == 1
        assert config.enforce_eager is False
        assert config.kvcache_block_size == 64


class TestConfigInvalidPaths:
    """Test Config validation for invalid paths."""

    def test_invalid_model_path(self) -> None:
        """Test that invalid model path raises appropriate error."""
        with pytest.raises(ValueError, match='not a valid directory'):
            Config(model='/nonexistent/path/to/model')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
