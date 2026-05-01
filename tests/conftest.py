"""Pytest configuration and shared fixtures for mini-vLLM tests.

This module provides common fixtures used across multiple test files
to reduce code duplication and improve test maintainability.
"""

from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from minivllm.config import Config
from minivllm.engine.sequence import Sequence
from minivllm.sampling_params import SamplingParams

# =============================================================================
# Model Configuration Fixtures
# =============================================================================


@pytest.fixture
def model_config_json() -> str:
    """Return a standard model configuration JSON string."""
    return ('{"model_type": "llama", '
            '"hidden_size": 768, '
            '"num_hidden_layers": 12, '
            '"num_attention_heads": 12, '
            '"max_position_embeddings": 4096, '
            '"torch_dtype": "float32"}')


@pytest.fixture
def minimal_config_json() -> str:
    """Return a minimal model configuration JSON string."""
    return ('{"model_type": "llama", '
            '"max_position_embeddings": 4096, '
            '"torch_dtype": "float32"}')


@pytest.fixture
def short_model_config_json() -> str:
    """Return a model configuration with short max_position_embeddings."""
    return ('{"model_type": "llama", '
            '"max_position_embeddings": 128, '
            '"torch_dtype": "float32"}')


# =============================================================================
# Temporary Model Directory Fixtures
# =============================================================================


@pytest.fixture
def temp_model_dir(tmp_path: Path, model_config_json: str) -> Path:
    """Create a temporary model directory with config.json.

    Returns:
        Path to the temporary model directory.
    """
    model_dir = tmp_path / 'test_model'
    model_dir.mkdir()
    (model_dir / 'config.json').write_text(model_config_json)
    return model_dir


@pytest.fixture
def temp_model_dir_with_blocks(tmp_path: Path, model_config_json: str) -> Path:
    """Create a temporary model directory with config.json and num_kvcache_blocks.

    This fixture creates a model directory that works with LLMEngine initialization
    by ensuring proper KV cache block configuration.

    Returns:
        Path to the temporary model directory.
    """
    model_dir = tmp_path / 'test_model'
    model_dir.mkdir()
    (model_dir / 'config.json').write_text(model_config_json)
    return model_dir


@pytest.fixture
def temp_minimal_model_dir(tmp_path: Path, minimal_config_json: str) -> Path:
    """Create a minimal temporary model directory with config.json.

    Returns:
        Path to the temporary model directory.
    """
    model_dir = tmp_path / 'test_model'
    model_dir.mkdir()
    (model_dir / 'config.json').write_text(minimal_config_json)
    return model_dir


@pytest.fixture
def temp_short_model_dir(tmp_path: Path, short_model_config_json: str) -> Path:
    """Create a temporary model directory with short max_position_embeddings.

    Returns:
        Path to the temporary model directory.
    """
    model_dir = tmp_path / 'test_model'
    model_dir.mkdir()
    (model_dir / 'config.json').write_text(short_model_config_json)
    return model_dir


# =============================================================================
# Config Fixtures
# =============================================================================


@pytest.fixture
def default_config(temp_model_dir: Path) -> Config:
    """Create a default Config instance with standard settings.

    Returns:
        Config instance with num_kvcache_blocks=100.
    """
    return Config(str(temp_model_dir), num_kvcache_blocks=100)


@pytest.fixture
def small_batch_config(temp_model_dir: Path) -> Config:
    """Create a Config instance with small batch settings for testing.

    Returns:
        Config instance with limited resources.
    """
    return Config(str(temp_model_dir),
                  max_num_seqs=2,
                  max_num_batched_tokens=128,
                  num_kvcache_blocks=10)


@pytest.fixture
def limited_cache_config(temp_model_dir: Path) -> Config:
    """Create a Config instance with very limited KV cache.

    Returns:
        Config instance for testing cache pressure scenarios.
    """
    return Config(str(temp_model_dir),
                  max_num_seqs=10,
                  max_num_batched_tokens=4096,
                  num_kvcache_blocks=2)


# =============================================================================
# Sequence Fixtures
# =============================================================================


@pytest.fixture
def simple_sequence() -> Sequence:
    """Create a simple sequence with default sampling params.

    Returns:
        Sequence with 5 tokens.
    """
    return Sequence(token_ids=[1, 2, 3, 4, 5],
                    sampling_params=SamplingParams())


@pytest.fixture
def short_sequence() -> Sequence:
    """Create a short sequence for testing.

    Returns:
        Sequence with 3 tokens.
    """
    return Sequence(token_ids=[1, 2, 3], sampling_params=SamplingParams())


@pytest.fixture
def long_sequence() -> Sequence:
    """Create a long sequence for testing.

    Returns:
        Sequence with 10 tokens.
    """
    return Sequence(token_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    sampling_params=SamplingParams())


@pytest.fixture
def single_block_sequence() -> Sequence:
    """Create a sequence that fits exactly in one block (256 tokens).

    Returns:
        Sequence with 256 identical tokens.
    """
    return Sequence(token_ids=[1] * 256, sampling_params=SamplingParams())


@pytest.fixture
def multi_block_sequence() -> Sequence:
    """Create a sequence spanning multiple blocks.

    Returns:
        Sequence with 512 tokens (2 blocks).
    """
    return Sequence(token_ids=[i for i in range(512)],
                    sampling_params=SamplingParams())


# =============================================================================
# SamplingParams Fixtures
# =============================================================================


@pytest.fixture
def default_sampling_params() -> SamplingParams:
    """Return default sampling parameters."""
    return SamplingParams()


@pytest.fixture
def greedy_sampling_params() -> SamplingParams:
    """Return sampling parameters for greedy decoding (low temperature)."""
    return SamplingParams(temperature=0.1, max_tokens=50)


@pytest.fixture
def creative_sampling_params() -> SamplingParams:
    """Return sampling parameters for creative generation (high temperature)."""
    return SamplingParams(temperature=1.5, max_tokens=100, top_p=0.9)


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_tokenizer() -> Generator[MagicMock, None, None]:
    """Provide a mocked tokenizer for testing.

    Yields:
        MagicMock: Mocked tokenizer instance.
    """
    with patch('transformers.AutoTokenizer.from_pretrained') as mock:
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer.decode.return_value = 'Generated text'
        mock_tokenizer.eos_token_id = 2
        mock.return_value = mock_tokenizer
        yield mock_tokenizer


@pytest.fixture
def mock_model_runner() -> Generator[MagicMock, None, None]:
    """Provide a mocked ModelRunner for testing.

    Yields:
        MagicMock: Mocked ModelRunner class.
    """
    with patch('minivllm.engine.model_runner.ModelRunner') as mock:
        yield mock


@pytest.fixture
def mock_llm_engine_deps(
        mock_model_runner: MagicMock,
        mock_tokenizer: MagicMock) -> tuple[MagicMock, MagicMock]:
    """Provide both mocked dependencies for LLMEngine testing.

    Returns:
        Tuple of (mock_model_runner, mock_tokenizer).
    """
    return mock_model_runner, mock_tokenizer


@pytest.fixture
def mock_model_manager() -> Generator[MagicMock, None, None]:
    """Provide a mocked ModelManager for testing.

    Yields:
        MagicMock: Mocked ModelManager class.
    """
    with patch('minivllm.engine.model_runner.ModelManager') as mock:
        mock_instance = MagicMock()
        mock_instance.initialize = MagicMock()
        mock_instance.model = MagicMock()
        mock_instance.tokenizer = MagicMock()
        mock_instance.tokenizer.eos_token_id = 2
        mock_instance.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_instance.tokenizer.decode.return_value = 'Generated text'
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_distributed_manager() -> Generator[MagicMock, None, None]:
    """Provide a mocked DistributedManager for testing.

    Yields:
        MagicMock: Mocked DistributedManager class.
    """
    with patch('minivllm.engine.model_runner.DistributedManager') as mock:
        mock_instance = MagicMock()
        mock_instance.initialize = MagicMock()
        mock_instance.synchronize = MagicMock()
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_inference_executor() -> Generator[MagicMock, None, None]:
    """Provide a mocked InferenceExecutor for testing.

    Yields:
        MagicMock: Mocked InferenceExecutor class.
    """
    with patch('minivllm.engine.model_runner.InferenceExecutor') as mock:
        mock_instance = MagicMock()
        mock_instance.initialize = MagicMock()
        mock_instance.capture_device_graphs = MagicMock()
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def fully_mocked_model_runner(
    mock_model_manager: MagicMock,
    mock_distributed_manager: MagicMock,
    mock_inference_executor: MagicMock,
) -> Generator[None, None, None]:
    """Provide fully mocked ModelRunner dependencies.

    This fixture mocks all major components of ModelRunner to prevent
    actual model loading and initialization.

    Yields:
        None
    """
    yield


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        'markers',
        "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line('markers',
                            'integration: marks tests as integration tests')
    config.addinivalue_line('markers', 'cuda: marks tests as requiring CUDA')
    config.addinivalue_line('markers', 'npu: marks tests as requiring NPU')
