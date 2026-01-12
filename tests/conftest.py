"""Pytest configuration and shared fixtures for mini-vLLM tests.

This module provides common test utilities and fixtures for all test modules.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def temp_model_dir(tmp_path):
    """Create a temporary mock HuggingFace model directory.

    Returns a temporary directory with a minimal config.json file
    that can be used for testing Config initialization.
    """
    model_dir = tmp_path / 'test_model'
    model_dir.mkdir()

    # Create minimal config.json
    config_content = '''{
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "max_position_embeddings": 4096,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "torch_dtype": "float32"
    }'''

    (model_dir / 'config.json').write_text(config_content)
    return model_dir


@pytest.fixture
def sample_token_ids():
    """Provide sample token IDs for testing."""
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


@pytest.fixture
def default_sampling_params():
    """Provide default sampling parameters for testing."""
    from minivllm.sampling_params import SamplingParams
    return SamplingParams(temperature=0.8, max_tokens=128)


@pytest.fixture(autouse=True)
def reset_sequence_counter():
    """Reset the global sequence counter before each test.

    This ensures that sequence IDs start from 0 for each test,
    making tests more predictable.
    """
    from itertools import count

    from minivllm.engine.sequence import Sequence

    # Save original counter
    original_counter = Sequence.counter

    # Reset to new counter
    Sequence.counter = count()

    yield

    # Restore original counter
    Sequence.counter = original_counter


# Configure pytest markers
def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        'markers',
        "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line(
        'markers', 'requires_cuda: marks tests that require CUDA device')
    config.addinivalue_line('markers',
                            'integration: marks tests as integration tests')


# Test discovery configuration
def pytest_collection_modifyitems(config, items):
    """Modify test items during collection."""
    for item in items:
        # Add 'integration' marker to integration test modules
        if 'test_integration' in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Add markers based on function names
        if 'slow' in item.nodeid:
            item.add_marker(pytest.mark.slow)
        if 'cuda' in item.nodeid:
            item.add_marker(pytest.mark.requires_cuda)
