"""Unit tests for LLM Engine module.

This module contains comprehensive tests for the LLMEngine class,
including initialization, request management, inference steps,
and sequence scheduling.

Tests cover:
    - Engine initialization with various configurations
    - Adding requests and managing sequences
    - Inference step execution
    - Completion tracking
    - Memory management
    - Error handling and edge cases
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from minivllm.config import Config
from minivllm.engine.llm_engine import LLMEngine
from minivllm.engine.sequence import Sequence
from minivllm.sampling_params import SamplingParams

# Pytest fixtures used from conftest.py:
# - temp_model_dir
# - mock_tokenizer
# - mock_model_manager
# - mock_distributed_manager
# - mock_inference_executor
# - fully_mocked_model_runner


class TestLLMEngineInitialization:
    """Test cases for LLM engine initialization."""

    def test_engine_initialization_with_valid_config(
        self,
        temp_model_dir: Path,
        mock_tokenizer: MagicMock,
        fully_mocked_model_runner: None,
    ) -> None:
        """Test engine initialization with valid configuration."""
        config = Config(str(temp_model_dir), num_kvcache_blocks=100)
        engine = LLMEngine(config)
        assert engine is not None
        assert hasattr(engine, 'scheduler')
        assert hasattr(engine, 'model_runner')
        assert hasattr(engine, 'tokenizer')

    def test_engine_config_parameters_propagation(
        self,
        temp_model_dir: Path,
        mock_tokenizer: MagicMock,
        fully_mocked_model_runner: None,
    ) -> None:
        """Test that configuration parameters are properly propagated to engine."""
        config = Config(str(temp_model_dir),
                        max_num_seqs=256,
                        max_num_batched_tokens=8192,
                        device_memory_utilization=0.8,
                        num_kvcache_blocks=100)
        engine = LLMEngine(config)
        assert engine.scheduler.max_num_seqs == 256
        assert engine.scheduler.max_num_batched_tokens == 8192

    def test_engine_with_tensor_parallelism(
        self,
        temp_model_dir: Path,
        mock_tokenizer: MagicMock,
        fully_mocked_model_runner: None,
    ) -> None:
        """Test engine initialization with tensor parallelism."""
        config = Config(str(temp_model_dir),
                        tensor_parallel_size=4,
                        num_kvcache_blocks=100)
        engine = LLMEngine(config)
        # Check that worker processes attributes exist
        assert hasattr(engine, 'ps')
        assert hasattr(engine, 'events')


class TestLLMEngineSequenceManagement:
    """Test cases for sequence management in LLM engine."""

    def test_add_request_with_string_prompt(
        self,
        temp_model_dir: Path,
        mock_tokenizer: MagicMock,
        fully_mocked_model_runner: None,
    ) -> None:
        """Test adding a request with a string prompt."""
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]

        config = Config(str(temp_model_dir), num_kvcache_blocks=100)
        engine = LLMEngine(config)
        engine.add_request('Hello, world!', SamplingParams())

        # Verify sequence was added to scheduler
        assert len(engine.scheduler.waiting) > 0

    def test_add_request_with_token_ids(
        self,
        temp_model_dir: Path,
        mock_tokenizer: MagicMock,
        fully_mocked_model_runner: None,
    ) -> None:
        """Test adding a request with pre-tokenized input."""
        config = Config(str(temp_model_dir), num_kvcache_blocks=100)
        engine = LLMEngine(config)
        engine.add_request([1, 2, 3, 4, 5], SamplingParams())

        # Verify sequence was added to scheduler
        assert len(engine.scheduler.waiting) > 0

    def test_finished_status_check(
        self,
        temp_model_dir: Path,
        mock_tokenizer: MagicMock,
        fully_mocked_model_runner: None,
    ) -> None:
        """Test checking if engine has finished all sequences."""
        config = Config(str(temp_model_dir), num_kvcache_blocks=100)
        engine = LLMEngine(config)
        # Initially, no sequences so should be finished
        assert engine.is_finished()


class TestLLMEngineInferenceStep:
    """Test cases for inference step execution."""

    def test_step_with_no_sequences(
        self,
        temp_model_dir: Path,
        mock_tokenizer: MagicMock,
        fully_mocked_model_runner: None,
    ) -> None:
        """Test that step() raises RuntimeError when no sequences are scheduled."""
        config = Config(str(temp_model_dir), num_kvcache_blocks=100)
        engine = LLMEngine(config)
        # Without any sequences, step should raise RuntimeError
        with pytest.raises(RuntimeError, match='No sequences scheduled'):
            engine.step()

    def test_step_with_mocked_execution(
        self,
        temp_model_dir: Path,
        mock_tokenizer: MagicMock,
        fully_mocked_model_runner: None,
    ) -> None:
        """Test step execution with mocked model runner."""
        config = Config(str(temp_model_dir), num_kvcache_blocks=100)
        engine = LLMEngine(config)

        # Add a request first
        engine.add_request([1, 2, 3], SamplingParams(max_tokens=10))

        # Mock the model runner's call method to return tokens
        with patch.object(engine.model_runner, 'call', return_value=[4, 5, 6]):
            output, num_tokens = engine.step()
            assert isinstance(output, list)
            assert isinstance(num_tokens, (int, float))


class TestLLMEngineTextGeneration:
    """Test cases for text generation functionality."""

    def test_generate_with_string_prompts(
        self,
        temp_model_dir: Path,
        mock_tokenizer: MagicMock,
        fully_mocked_model_runner: None,
    ) -> None:
        """Test generate() method with string prompts."""
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = 'Generated text'

        config = Config(str(temp_model_dir), num_kvcache_blocks=100)
        engine = LLMEngine(config)

        # Mock the generation loop
        with patch.object(engine, 'is_finished', side_effect=[False, True]):
            with patch.object(engine, 'step') as mock_step:
                seq = Mock(spec=Sequence)
                seq.seq_id = 0
                seq.is_finished = True
                seq.completion_token_ids = [1, 2, 3]
                mock_step.return_value = ([(0, [1, 2, 3])], 3)

                results = engine.generate(['Hello, world!'], use_tqdm=False)

                assert isinstance(results, list)
                if results:
                    assert 'text' in results[0]
                    assert 'token_ids' in results[0]

    def test_generate_with_custom_sampling_params(
        self,
        temp_model_dir: Path,
        mock_tokenizer: MagicMock,
        fully_mocked_model_runner: None,
    ) -> None:
        """Test generate() with custom sampling parameters."""
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = 'Generated text'

        config = Config(str(temp_model_dir), num_kvcache_blocks=100)
        engine = LLMEngine(config)

        # Create custom sampling parameters
        params = SamplingParams(temperature=0.7, max_tokens=100)

        with patch.object(engine, 'is_finished', side_effect=[False, True]):
            with patch.object(engine, 'step') as mock_step:
                mock_step.return_value = ([(0, [1, 2, 3])], 3)

                results = engine.generate(['Hello, world!'],
                                          sampling_params=params,
                                          use_tqdm=False)

                assert isinstance(results, list)

    def test_generate_with_mismatched_params(
        self,
        temp_model_dir: Path,
        mock_tokenizer: MagicMock,
        fully_mocked_model_runner: None,
    ) -> None:
        """Test generate() raises error for mismatched sampling parameters."""
        config = Config(str(temp_model_dir), num_kvcache_blocks=100)
        engine = LLMEngine(config)

        # Mismatched number of prompts and sampling params
        with pytest.raises(ValueError, match='Length of sampling_params'):
            engine.generate(
                ['Prompt 1', 'Prompt 2'],
                sampling_params=[SamplingParams()],  # Only 1, should be 2
                use_tqdm=False)


class TestLLMEngineCleanup:
    """Test cases for engine cleanup and resource management."""

    def test_engine_exit_cleanup(
        self,
        temp_model_dir: Path,
        mock_tokenizer: MagicMock,
        fully_mocked_model_runner: None,
    ) -> None:
        """Test that engine properly cleans up resources on exit."""
        config = Config(str(temp_model_dir), num_kvcache_blocks=100)
        engine = LLMEngine(config)

        # Mock the model runner's call method before exit
        with patch.object(engine.model_runner, 'call') as mock_call:
            # Call exit and verify cleanup was attempted
            engine.exit()

            # Verify model runner was called to exit
            mock_call.assert_called_once_with('exit')

    def test_engine_cleanup_with_multiple_workers(
        self,
        temp_model_dir: Path,
        mock_tokenizer: MagicMock,
        fully_mocked_model_runner: None,
    ) -> None:
        """Test cleanup with multiple worker processes."""
        config = Config(str(temp_model_dir),
                        tensor_parallel_size=4,
                        num_kvcache_blocks=100)
        engine = LLMEngine(config)

        # Mock worker processes
        for p in engine.ps:
            p.is_alive = MagicMock(return_value=False)
            p.join = MagicMock()

        engine.model_runner.call = MagicMock()

        # Call exit
        engine.exit()

        # Verify cleanup was attempted for all workers
        for p in engine.ps:
            p.join.assert_called()


class TestLLMEngineErrorHandling:
    """Test cases for error handling in LLM engine."""

    def test_invalid_model_path(self) -> None:
        """Test that invalid model path raises appropriate error."""
        with pytest.raises(ValueError, match='not a valid directory'):
            Config(model='/nonexistent/path/to/model')

    def test_sampling_params_validation(self) -> None:
        """Test that invalid sampling parameters are caught."""
        with pytest.raises(ValueError, match='temperature'):
            SamplingParams(temperature=-1.0)  # Negative

        with pytest.raises(ValueError, match='max_tokens'):
            SamplingParams(max_tokens=0)  # Must be positive

    def test_config_validation_errors(self, temp_model_dir: Path) -> None:
        """Test configuration validation catches errors."""
        # Test invalid device memory utilization
        with pytest.raises(ValueError, match='device_memory_utilization'):
            Config(str(temp_model_dir), device_memory_utilization=2.0)

        # Test invalid tensor parallel size
        with pytest.raises(ValueError, match='tensor_parallel_size'):
            Config(str(temp_model_dir), tensor_parallel_size=16)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
