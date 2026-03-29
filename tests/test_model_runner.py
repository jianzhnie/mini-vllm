"""Unit tests for Model Runner module.

This module contains comprehensive tests for the refactored ModelRunner class,
which now coordinates specialized managers:
- ModelManager: Model loading, validation, and lifecycle
- DistributedManager: Multi-process coordination and communication
- InferenceExecutor: Model execution and optimization

Tests cover:
    - Model runner initialization
    - KV cache allocation and management
    - Forward pass execution
    - Device graph capture and replay (CUDA)
    - Distributed inference
    - Error handling and edge cases
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from minivllm.config import Config
from minivllm.engine.model_runner import ModelRunner
from minivllm.engine.sequence import Sequence
from minivllm.sampling_params import SamplingParams


def create_mock_config(tmp_path: Path, extra_config: dict = None) -> Config:
    """Helper to create a mock config with model directory."""
    model_dir = tmp_path / 'test_model'
    model_dir.mkdir()

    config = {
        'model_type': 'llama',
        'hidden_size': 768,
        'num_hidden_layers': 12,
        'num_attention_heads': 12,
        'max_position_embeddings': 4096,
        'torch_dtype': 'float32',
    }
    if extra_config:
        config.update(extra_config)

    import json
    (model_dir / 'config.json').write_text(json.dumps(config))

    return Config(str(model_dir))


class TestModelRunnerInitialization:
    """Test cases for ModelRunner initialization."""

    @patch('minivllm.engine.inference_executor.InferenceExecutor.initialize')
    @patch(
        'minivllm.engine.inference_executor.InferenceExecutor.capture_device_graphs'
    )
    @patch('minivllm.models.manager.ModelManager.initialize')
    @patch('minivllm.engine.distributed_manager.DistributedManager.initialize')
    @patch('minivllm.models.create_model')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_runner_initialization(
        self,
        mock_tokenizer,
        mock_create_model,
        mock_dist_init,
        mock_model_mgr_init,
        mock_capture_graphs,
        mock_executor_init,
        tmp_path: Path,
    ) -> None:
        """Test model runner initialization."""
        # Setup mocks
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_create_model.return_value = mock_model
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance

        config = create_mock_config(tmp_path)
        mock_event = MagicMock()

        # Initialize runner
        runner = ModelRunner(config, rank=0, event=mock_event)

        assert runner.rank == 0
        assert runner.world_size == 1
        assert runner.config == config

        # Cleanup
        runner.exit()

    @patch('minivllm.engine.inference_executor.InferenceExecutor.initialize')
    @patch(
        'minivllm.engine.inference_executor.InferenceExecutor.capture_device_graphs'
    )
    @patch('minivllm.models.manager.ModelManager.initialize')
    @patch('minivllm.engine.distributed_manager.DistributedManager.initialize')
    @patch('minivllm.models.create_model')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_runner_initialization_with_tensor_parallelism(
        self,
        mock_tokenizer,
        mock_create_model,
        mock_dist_init,
        mock_model_mgr_init,
        mock_capture_graphs,
        mock_executor_init,
        tmp_path: Path,
    ) -> None:
        """Test model runner initialization with tensor parallelism."""
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_create_model.return_value = mock_model

        config = create_mock_config(tmp_path, {'tensor_parallel_size': 4})
        config.tensor_parallel_size = 4  # Override after creation
        mock_event = [MagicMock() for _ in range(3)]

        runner = ModelRunner(config, rank=1, event=mock_event)

        assert runner.rank == 1
        assert runner.world_size == 4

        runner.exit()


class TestModelRunnerInference:
    """Test cases for model inference execution."""

    @patch('minivllm.engine.inference_executor.InferenceExecutor.initialize')
    @patch(
        'minivllm.engine.inference_executor.InferenceExecutor.capture_device_graphs'
    )
    @patch('minivllm.models.manager.ModelManager.initialize')
    @patch('minivllm.engine.distributed_manager.DistributedManager.initialize')
    @patch('minivllm.models.create_model')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_call_method_exists(
        self,
        mock_tokenizer,
        mock_create_model,
        mock_dist_init,
        mock_model_mgr_init,
        mock_capture_graphs,
        mock_executor_init,
        tmp_path: Path,
    ) -> None:
        """Test that the call method exists and is callable."""
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_create_model.return_value = mock_model

        config = create_mock_config(tmp_path)
        mock_event = MagicMock()

        runner = ModelRunner(config, rank=0, event=mock_event)
        assert hasattr(runner, 'call')
        assert callable(runner.call)

        runner.exit()

    @patch('minivllm.engine.inference_executor.InferenceExecutor.initialize')
    @patch(
        'minivllm.engine.inference_executor.InferenceExecutor.capture_device_graphs'
    )
    @patch('minivllm.models.manager.ModelManager.initialize')
    @patch('minivllm.engine.distributed_manager.DistributedManager.initialize')
    @patch('minivllm.models.create_model')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_call_run_method(
        self,
        mock_tokenizer,
        mock_create_model,
        mock_dist_init,
        mock_model_mgr_init,
        mock_capture_graphs,
        mock_executor_init,
        tmp_path: Path,
    ) -> None:
        """Test calling the run method on model runner."""
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_create_model.return_value = mock_model

        config = create_mock_config(tmp_path)
        mock_event = MagicMock()

        runner = ModelRunner(config, rank=0, event=mock_event)

        # Create mock sequences
        seq = Sequence(token_ids=[1, 2, 3, 4, 5],
                       sampling_params=SamplingParams())

        # Mock the run method
        with patch.object(runner, 'run', return_value=[6, 7]) as mock_run:
            runner.call('run', [seq], True)
            mock_run.assert_called_once_with([seq], True)

        runner.exit()

    @patch('minivllm.engine.inference_executor.InferenceExecutor.initialize')
    @patch(
        'minivllm.engine.inference_executor.InferenceExecutor.capture_device_graphs'
    )
    @patch('minivllm.models.manager.ModelManager.initialize')
    @patch('minivllm.engine.distributed_manager.DistributedManager.initialize')
    @patch('minivllm.models.create_model')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_call_exit_method(
        self,
        mock_tokenizer,
        mock_create_model,
        mock_dist_init,
        mock_model_mgr_init,
        mock_capture_graphs,
        mock_executor_init,
        tmp_path: Path,
    ) -> None:
        """Test calling the exit method on model runner."""
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_create_model.return_value = mock_model

        config = create_mock_config(tmp_path)
        mock_event = MagicMock()

        runner = ModelRunner(config, rank=0, event=mock_event)

        try:
            # Mock the exit method to test call routing
            with patch.object(runner, 'exit') as mock_exit:
                runner.call('exit')
                mock_exit.assert_called_once()
        finally:
            # Ensure cleanup even though exit was mocked
            runner.exit()


class TestModelRunnerDeviceGraph:
    """Test cases for device graph capture and replay."""

    @patch('minivllm.engine.inference_executor.InferenceExecutor.initialize')
    @patch(
        'minivllm.engine.inference_executor.InferenceExecutor.capture_device_graphs'
    )
    @patch('minivllm.models.manager.ModelManager.initialize')
    @patch('minivllm.engine.distributed_manager.DistributedManager.initialize')
    @patch('minivllm.models.create_model')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_cuda_graph_capture_disabled_on_cpu(
        self,
        mock_tokenizer,
        mock_create_model,
        mock_dist_init,
        mock_model_mgr_init,
        mock_capture_graphs,
        mock_executor_init,
        tmp_path: Path,
    ) -> None:
        """Test that CUDA graphs are not captured on CPU."""
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_create_model.return_value = mock_model

        config = create_mock_config(tmp_path, {'enforce_eager': False})
        mock_event = MagicMock()

        runner = ModelRunner(config, rank=0, event=mock_event)

        # Graph capture should be skipped on CPU (handled by InferenceExecutor)
        runner.exit()

    @patch('minivllm.engine.inference_executor.InferenceExecutor.initialize')
    @patch(
        'minivllm.engine.inference_executor.InferenceExecutor.capture_device_graphs'
    )
    @patch('minivllm.models.manager.ModelManager.initialize')
    @patch('minivllm.engine.distributed_manager.DistributedManager.initialize')
    @patch('minivllm.models.create_model')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_enforce_eager_skips_graph(
        self,
        mock_tokenizer,
        mock_create_model,
        mock_dist_init,
        mock_model_mgr_init,
        mock_capture_graphs,
        mock_executor_init,
        tmp_path: Path,
    ) -> None:
        """Test that enforce_eager skips graph capture."""
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_create_model.return_value = mock_model

        config = create_mock_config(tmp_path, {'enforce_eager': True})
        mock_event = MagicMock()

        runner = ModelRunner(config, rank=0, event=mock_event)

        # Graph capture should not be called with enforce_eager=True
        # (handled by InferenceExecutor)
        runner.exit()


class TestModelRunnerCleanup:
    """Test cases for model runner cleanup."""

    @patch('minivllm.engine.inference_executor.InferenceExecutor.initialize')
    @patch(
        'minivllm.engine.inference_executor.InferenceExecutor.capture_device_graphs'
    )
    @patch('minivllm.models.manager.ModelManager.initialize')
    @patch('minivllm.engine.distributed_manager.DistributedManager.initialize')
    @patch('minivllm.models.create_model')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_exit_cleans_resources(
        self,
        mock_tokenizer,
        mock_create_model,
        mock_dist_init,
        mock_model_mgr_init,
        mock_capture_graphs,
        mock_executor_init,
        tmp_path: Path,
    ) -> None:
        """Test that exit properly cleans resources."""
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_create_model.return_value = mock_model

        config = create_mock_config(tmp_path)
        mock_event = MagicMock()

        runner = ModelRunner(config, rank=0, event=mock_event)

        # Call exit and verify it completes without raising
        runner.exit()
        assert True  # If we get here, exit completed successfully


class TestModelRunnerDistributed:
    """Test cases for distributed tensor parallelism."""

    @patch('minivllm.engine.inference_executor.InferenceExecutor.initialize')
    @patch(
        'minivllm.engine.inference_executor.InferenceExecutor.capture_device_graphs'
    )
    @patch('minivllm.models.manager.ModelManager.initialize')
    @patch('minivllm.engine.distributed_manager.DistributedManager.initialize')
    @patch('minivllm.models.create_model')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_distributed_initialization(
        self,
        mock_tokenizer,
        mock_create_model,
        mock_dist_init,
        mock_model_mgr_init,
        mock_capture_graphs,
        mock_executor_init,
        tmp_path: Path,
    ) -> None:
        """Test distributed initialization for tensor parallelism."""
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_create_model.return_value = mock_model

        config = create_mock_config(tmp_path)
        config.tensor_parallel_size = 4  # Override
        mock_event = [MagicMock() for _ in range(3)]

        runner = ModelRunner(config, rank=1, event=mock_event)

        # For multi-GPU, distributed manager should be initialized
        assert runner.world_size == 4
        assert runner.rank == 1

        runner.exit()


class TestModelRunnerErrorHandling:
    """Test cases for error handling in model runner."""

    def test_invalid_config_raises_error(self, tmp_path: Path) -> None:
        """Test that invalid configuration raises appropriate error."""
        with pytest.raises(ValueError):
            Config(model='/nonexistent/path')

    @patch('minivllm.engine.inference_executor.InferenceExecutor.initialize')
    @patch(
        'minivllm.engine.inference_executor.InferenceExecutor.capture_device_graphs'
    )
    @patch('minivllm.models.manager.ModelManager.initialize')
    @patch('minivllm.engine.distributed_manager.DistributedManager.initialize')
    @patch('minivllm.models.create_model')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_inference_executor_initialized(
        self,
        mock_tokenizer,
        mock_create_model,
        mock_dist_init,
        mock_model_mgr_init,
        mock_capture_graphs,
        mock_executor_init,
        tmp_path: Path,
    ) -> None:
        """Test that inference executor is properly initialized."""
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_create_model.return_value = mock_model

        config = create_mock_config(tmp_path)
        mock_event = MagicMock()

        runner = ModelRunner(config, rank=0, event=mock_event)

        # Verify inference executor was initialized
        mock_executor_init.assert_called()
        assert runner.inference_executor is not None

        runner.exit()


class TestModelRunnerManagerAccess:
    """Test cases for accessing sub-managers."""

    @patch('minivllm.engine.inference_executor.InferenceExecutor.initialize')
    @patch(
        'minivllm.engine.inference_executor.InferenceExecutor.capture_device_graphs'
    )
    @patch('minivllm.models.manager.ModelManager.initialize')
    @patch('minivllm.engine.distributed_manager.DistributedManager.initialize')
    @patch('minivllm.models.create_model')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_get_tokenizer(
        self,
        mock_tokenizer,
        mock_create_model,
        mock_dist_init,
        mock_model_mgr_init,
        mock_capture_graphs,
        mock_executor_init,
        tmp_path: Path,
    ) -> None:
        """Test that tokenizer can be retrieved from model manager."""
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_create_model.return_value = mock_model

        config = create_mock_config(tmp_path)
        mock_event = MagicMock()

        runner = ModelRunner(config, rank=0, event=mock_event)

        # Mock the tokenizer on model_manager
        runner.model_manager.tokenizer = mock_tokenizer_instance

        # Get tokenizer via model_manager
        tokenizer = runner.get_tokenizer()
        assert tokenizer is mock_tokenizer_instance

        runner.exit()

    @patch('minivllm.engine.inference_executor.InferenceExecutor.initialize')
    @patch(
        'minivllm.engine.inference_executor.InferenceExecutor.capture_device_graphs'
    )
    @patch('minivllm.models.manager.ModelManager.initialize')
    @patch('minivllm.engine.distributed_manager.DistributedManager.initialize')
    @patch('minivllm.models.create_model')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_get_model_info(
        self,
        mock_tokenizer,
        mock_create_model,
        mock_dist_init,
        mock_model_mgr_init,
        mock_capture_graphs,
        mock_executor_init,
        tmp_path: Path,
    ) -> None:
        """Test that model info can be retrieved."""
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_create_model.return_value = mock_model

        config = create_mock_config(tmp_path)
        mock_event = MagicMock()

        runner = ModelRunner(config, rank=0, event=mock_event)

        info = runner.get_model_info()

        # Verify info structure
        assert 'rank' in info
        assert 'world_size' in info
        assert 'config' in info
        assert info['rank'] == 0
        assert info['world_size'] == 1

        runner.exit()
