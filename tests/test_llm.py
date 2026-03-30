"""Tests for LLM interface.

This module tests the LLM class which is the main user-facing interface
for the mini-vLLM engine.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from minivllm.config import Config
from minivllm.llm import LLM
from minivllm.sampling_params import SamplingParams


class TestLLMBasics:
    """Basic tests for LLM class."""

    def test_llm_is_subclass_of_llmengine(self):
        """Test that LLM is a subclass of LLMEngine."""
        from minivllm.engine.llm_engine import LLMEngine

        assert issubclass(LLM, LLMEngine)

    def test_llm_imports(self):
        """Test that LLM can be imported."""
        assert LLM is not None


class TestLLMInitialization:
    """Tests for LLM initialization."""

    def test_llm_initialization_with_defaults(
        self,
        temp_model_dir: Path,
        mock_tokenizer: MagicMock,
        fully_mocked_model_runner: None,
    ):
        """Test LLM initialization with default parameters."""
        # Test initialization with num_kvcache_blocks
        llm = LLM(str(temp_model_dir), num_kvcache_blocks=100)

        assert llm is not None
        assert hasattr(llm, 'scheduler')
        assert hasattr(llm, 'model_runner')
        assert hasattr(llm, 'tokenizer')

    def test_llm_initialization_with_custom_params(
        self,
        temp_model_dir: Path,
        mock_tokenizer: MagicMock,
        fully_mocked_model_runner: None,
    ):
        """Test LLM initialization with custom parameters."""
        llm = LLM(
            str(temp_model_dir),
            max_num_seqs=256,
            max_num_batched_tokens=8192,
            device_memory_utilization=0.8,
            num_kvcache_blocks=100,
        )

        assert llm is not None
        assert llm.scheduler.max_num_seqs == 256
        assert llm.scheduler.max_num_batched_tokens == 8192


class TestLLMGenerate:
    """Tests for LLM.generate method."""

    def test_sampling_params_validation(self):
        """Test that SamplingParams validates inputs correctly."""
        # Valid params
        params = SamplingParams(temperature=0.7, max_tokens=100)
        assert params.temperature == 0.7
        assert params.max_tokens == 100

        # Invalid temperature (too low)
        with pytest.raises(ValueError):
            SamplingParams(temperature=-1.0)

        # Invalid max_tokens (negative)
        with pytest.raises(ValueError):
            SamplingParams(max_tokens=-10)

    def test_sampling_params_defaults(self):
        """Test SamplingParams default values."""
        params = SamplingParams()

        assert params.temperature == 1.0
        assert params.top_p == 1.0
        assert params.top_k == -1
        assert params.max_tokens == 64
        assert params.ignore_eos is False


class TestLLMConfigPropagation:
    """Tests for config parameter propagation."""

    def test_tensor_parallel_size_propagation(self, temp_model_dir: Path):
        """Test that tensor_parallel_size is passed to config."""
        config = Config(str(temp_model_dir), tensor_parallel_size=4)
        assert config.tensor_parallel_size == 4

    def test_max_model_len_propagation(self, temp_model_dir: Path):
        """Test that max_model_len is passed to config."""
        config = Config(str(temp_model_dir), max_model_len=8192)
        assert config.max_model_len == 8192


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
