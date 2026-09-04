# Mini-vLLM Test Suite

This directory contains comprehensive tests for the mini-vLLM library.

## Test Organization

### Test Modules

Test files mirror the `minivllm/` source layout, so each `tests/<pkg>/test_<name>.py`
covers `minivllm/<pkg>/<name>.py`.

**Top level**
- **test_config.py**: `config.Config` validation and initialization
- **test_llm.py**: `llm.LLM` high-level entry point
- **test_integration.py**: component-interaction integration tests

**engine/**
- **test_sequence.py**: Sequence and token management
- **test_scheduler.py**: two-phase scheduling
- **test_block_manager.py**: block-based KV cache, prefix + CoW
- **test_llm_engine.py**: engine orchestration
- **test_model_runner.py**: rank-0/worker runner
- **test_inference_executor.py**: CUDA-graph batched execution
- **test_distributed_manager.py**: NCCL/HCLL process group sync

**models/**
- **test_registry.py**: `MODEL_REGISTRY` / `create_model` dispatch
- **test_manager.py**: `ModelManager` loading, validation, cleanup, dtype resolution
- **test_qwen_base.py**: shared Qwen backbone (Attention/MLP/layers, `load_weights` packed mapping)
- **test_qwen3_npu.py**: Qwen3 + NPU backend init and kernel dispatch
- **test_qwen3_logit_match.py**: Qwen3 logit reproducibility (vs HuggingFace)

**models/layers/**
- **test_linear.py**: tensor-parallel linear layers
- **test_attention.py**: `Attention` prefill/decode + NPU integration (CPU fallback)
- **test_attention_backend.py**: attention backend dispatch logic
- **test_npu_flash_attention.py**: NPU causal-mask helper caching + `SPARSE_MODE` validation
- **test_page_attention.py**: page attention vs reference
- **test_attention_gather.py**: paged-KV gather
- **test_embed_head.py**: parallel embedding / LM head
- **test_layernorm.py**: RMSNorm
- **test_rotary_embedding.py**: RoPE computation and caching
- **test_activation.py**: SiluAndMul

**sampling/**
- **test_sampler.py**: token sampling pipeline
- **test_mirostat.py**: Mirostat v1/v2 samplers
- **test_functional.py**: stateless sampling ops
- **test_sampling_config.py**: `SamplingConfig`
- **test_sampling_params.py**: user-facing `SamplingParams`

**utils/**
- **test_device.py**: device detection and management
- **test_context.py**: inference context
- **test_loader.py**: safetensors weight loading
- **test_logger_utils.py**: logger configuration
- **test_random_utils.py**: RNG helpers
- **test_example_utils.py**: example-script helpers

### Test Fixtures

See `conftest.py` for shared fixtures including:
- `temp_model_dir`: Temporary HuggingFace model directory
- `sample_token_ids`: Sample token sequences
- `default_sampling_params`: Default sampling parameters
- `reset_sequence_counter`: Resets sequence ID counter

## Running Tests

### Quick Start

```bash
# Run all tests
python -m pytest tests/

# Run with verbose output
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_config.py

# Run tests matching a pattern
python -m pytest -k test_sampling
```

### Using the Test Runner Script

```bash
# Run all tests
python tests/run_tests.py

# Run with coverage
python tests/run_tests.py --coverage

# Run integration tests only
python tests/run_tests.py -m integration

# Run all tests except slow ones
python tests/run_tests.py  # (default behavior)

# Include slow tests
python tests/run_tests.py --slow

# Run tests with verbose output
python tests/run_tests.py -v
```

### Advanced Usage

```bash
# Run tests that were previously failing
python tests/run_tests.py --failed

# Generate JUnit XML report for CI/CD
python tests/run_tests.py --junit

# Combine options
python tests/run_tests.py -v --coverage -m integration
```

## Test Categories

### Unit Tests
- Test individual classes and functions in isolation
- Located in `test_*.py` files

### Integration Tests
- Test interaction between multiple components
- Marked with `@pytest.mark.integration`
- Run with: `pytest -m integration`

### Slow Tests
- Tests that take significant time to run
- Marked with `@pytest.mark.slow`
- Excluded by default, include with: `pytest --slow` or `pytest -m "not slow"` to exclude

## Test Coverage

Generate coverage reports:

```bash
# HTML report
python tests/run_tests.py --coverage

# Terminal report
pytest tests/ --cov=minivllm --cov-report=term-missing
```

## Adding New Tests

1. Create test file in `tests/` directory following naming convention `test_*.py`
2. Use clear, descriptive test function names starting with `test_`
3. Group related tests in test classes
4. Use meaningful assertions with descriptive messages
5. Add docstrings to test classes and functions

Example:

```python
import pytest
from minivllm.module import Class

class TestClassName:
    """Tests for ClassName."""

    def test_feature_description(self):
        """Test that feature works as expected."""
        obj = Class()
        result = obj.method()
        assert result == expected_value

    def test_error_handling(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            Class.invalid_operation()
```

## Pytest Markers

Available markers:

- `@pytest.mark.slow`: Mark test as slow (excluded by default)
- `@pytest.mark.requires_cuda`: Requires CUDA device
- `@pytest.mark.integration`: Integration test

Example:

```python
@pytest.mark.slow
def test_long_running_operation():
    # Long test...
    pass

@pytest.mark.requires_cuda
def test_cuda_functionality():
    # CUDA test...
    pass
```

## Fixtures

Use pytest fixtures for common setup:

```python
def test_config_creation(temp_model_dir):
    """Test using temp_model_dir fixture."""
    from minivllm.config import Config
    config = Config(str(temp_model_dir))
    assert config.model == str(temp_model_dir)
```

## Troubleshooting

### Tests fail with import errors
- Ensure you're running from the project root
- Check that the virtual environment is activated
- Verify minivllm package is installed: `pip install -e .`

### Device-related test failures
- Some tests require specific hardware (CUDA, NPU, etc.)
- Use `-m "not requires_cuda"` to skip device-specific tests

### Slow test suite
- Use `pytest -m "not slow"` to exclude slow tests
- Use `-k` flag to run specific tests: `pytest -k test_fast`

## Continuous Integration

The test suite is designed to work with CI/CD systems:

```bash
# Generate artifacts for CI
python tests/run_tests.py -v --coverage --junit
```

This produces:
- `htmlcov/`: Coverage HTML report
- `test_results.xml`: JUnit XML for CI integration
- Terminal output with test results

## Contributing

When contributing tests:

1. Ensure all tests pass locally
2. Maintain or improve code coverage
3. Follow existing test patterns and naming conventions
4. Add documentation to complex test logic
5. Run full test suite before submitting PR: `pytest tests/ -v`
