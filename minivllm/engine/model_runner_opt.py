"""Model Runner module for coordinating model inference.

This module provides the ModelRunner class which orchestrates the
complete inference process using a modular architecture with specialized managers:

Architecture Overview:
- ModelManager: Handles model loading, validation, and lifecycle management
- DistributedManager: Manages multi-process coordination and communication
- InferenceExecutor: Executes model inference with performance optimization

Design Principles:
- Single Responsibility: Each manager handles a specific aspect of inference
- Modularity: Components can be developed and tested independently
- Extensibility: Easy to add new features or optimizations
- Performance: Optimized for high-throughput inference scenarios

Key Features:
- Unified Model Management: Centralized model loading and validation
- Distributed Coordination: Efficient multi-process communication
- Performance Monitoring: Real-time metrics and statistics collection
- Memory Optimization: Intelligent memory management and pooling
- Error Handling: Comprehensive exception handling and recovery

Usage Pattern:
    >>> runner = ModelRunner(config, rank=0, events=sync_events)
    >>> runner.initialize(max_num_batched_tokens=8192, max_num_seqs=512)
    >>> logits, tokens = runner.execute_inference(sequences, prefill=True)
    >>> info = runner.get_model_info()

Integration Points:
- Works seamlessly with Config for parameter validation
- Integrates with InferenceExecutor for model execution
- Coordinates with DistributedManager for multi-device support
- Supports performance monitoring and error handling

Performance Considerations:
- Model loading is optimized with parallel initialization
- Distributed communication is minimized and efficient
- Memory usage is tracked and optimized
- All operations are instrumented for performance analysis

This refactored design improves maintainability, testability, and
extensibility while maintaining high performance for production use.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from minivllm.config import Config
from minivllm.engine.distributed_manager import DistributedManager
from minivllm.engine.inference_executor import InferenceExecutor
from minivllm.engine.sequence import Sequence
from minivllm.models.manager import ModelManager
from minivllm.utils.logger_utils import get_logger

logger = get_logger(__name__)

__all__ = ['ModelRunner']


class ModelRunner:
    """Orchestrates model inference using specialized managers.

    This refactored ModelRunner serves as a coordinator that delegates
    specific responsibilities to specialized managers:
    - ModelManager: Model loading, validation, and lifecycle
    - DistributedManager: Multi-process coordination and communication
    - InferenceExecutor: Model execution and optimization

    This design follows the Single Responsibility Principle and makes
    the codebase more maintainable and testable.

    Attributes:
        config: Engine configuration.
        rank: Process rank in distributed setup.
        model_manager: Handles model-related operations.
        distributed_manager: Manages distributed coordination.
        inference_executor: Executes model inference.
        is_initialized: Whether the runner has been initialized.
    """

    def __init__(self, config: Config, rank: int,
                 events: Union[Any, List[Any]]) -> None:
        """Initialize the model runner.

        Args:
            config: Engine configuration with model path and settings.
            rank: Process rank (0 = main process in distributed setup).
            events: Synchronization events for distributed coordination.
        """
        self.config = config
        self.rank = rank
        self.is_initialized = False

        # Initialize managers
        self.model_manager = ModelManager(config)
        self.distributed_manager = DistributedManager(config, rank, events)
        self.inference_executor = None  # Will be initialized after model loading

        logger.debug(f'ModelRunner created with rank {rank}')

        # Auto-initialize
        self.initialize(config.max_num_batched_tokens, config.max_num_seqs)

    def initialize(self, max_num_batched_tokens: int,
                   max_num_seqs: int) -> None:
        """Initialize all components and prepare for inference.

        This method:
        1. Initializes the model manager and loads the model
        2. Sets up distributed communication if needed
        3. Creates and initializes the inference executor
        4. Captures optimization graphs if enabled

        Args:
            max_num_batched_tokens: Maximum tokens in a batch.
            max_num_seqs: Maximum sequences in a batch.

        Raises:
            RuntimeError: If initialization fails.
        """
        if self.is_initialized:
            logger.info('ModelRunner already initialized, skipping.')
            return

        try:
            logger.info('Initializing ModelRunner...')

            # Initialize model management
            self.model_manager.initialize()

            # Initialize distributed coordination
            self.distributed_manager.initialize()

            # Wait for all processes to complete model loading
            self.distributed_manager.synchronize()

            # Initialize inference executor
            self.inference_executor = InferenceExecutor(
                self.config, self.model_manager.model)
            self.inference_executor.initialize(max_num_batched_tokens,
                                               max_num_seqs)

            # Capture optimization graphs
            if self.rank == 0:  # Only main process captures graphs
                self.inference_executor.capture_cuda_graphs(max_num_seqs)

            # Final synchronization
            self.distributed_manager.synchronize()

            self.is_initialized = True
            logger.info('ModelRunner initialization completed successfully')

        except Exception as e:
            logger.error(f'ModelRunner initialization failed: {e}')
            self.cleanup()
            raise

    def execute_inference(self,
                          sequences: List[Sequence],
                          prefill: bool = True) -> Tuple[Any, List[int]]:
        """Execute inference for a batch of sequences.

        Args:
            sequences: List of sequences to process.
            prefill: Whether this is a prefill step.

        Returns:
            Tuple of (logits, next_tokens).

        Raises:
            RuntimeError: If runner is not initialized.
        """
        if not self.is_initialized:
            raise RuntimeError(
                'ModelRunner not initialized. Call initialize() first.')

        if not self.inference_executor:
            raise RuntimeError('Inference executor not available')

        return self.inference_executor.execute_batch(sequences, prefill)

    def get_tokenizer(self):
        """Get the tokenizer instance.

        Returns:
            The loaded tokenizer.
        """
        if not self.model_manager:
            raise RuntimeError('Model manager not available')
        return self.model_manager.tokenizer

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model and system information.

        Returns:
            Dictionary containing model, device, and performance info.
        """
        info = {
            'rank': self.rank,
            'config': {
                'model_path': self.config.model_path,
                'tensor_parallel_size': self.config.tensor_parallel_size,
                'dtype': str(self.config.dtype),
                'block_size': self.config.kvcache_block_size,
            },
        }

        # Add model manager info
        if self.model_manager:
            info['model'] = self.model_manager.get_model_info()

        # Add distributed manager info
        if self.distributed_manager:
            info['distributed'] = (
                self.distributed_manager.get_distributed_info())

        # Add performance metrics
        if self.inference_executor:
            info['performance'] = (
                self.inference_executor.get_performance_metrics())

        return info

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        if self.inference_executor:
            self.inference_executor.reset_metrics()
        logger.debug('Performance metrics reset')

    def validate_setup(self) -> bool:
        """Validate that all components are properly set up.

        Returns:
            True if setup is valid, False otherwise.
        """
        try:
            # Check basic initialization
            if not self.is_initialized:
                logger.error('ModelRunner not initialized')
                return False

            # Check model manager
            if not self.model_manager or not self.model_manager.model:
                logger.error('Model manager or model not available')
                return False

            # Check distributed manager
            if not self.distributed_manager:
                logger.error('Distributed manager not available')
                return False

            # Check inference executor
            if not self.inference_executor:
                logger.error('Inference executor not available')
                return False

            # Validate distributed setup if applicable
            if self.config.tensor_parallel_size > 1:
                distributed_info = self.distributed_manager.get_distributed_info(
                )
                if not distributed_info.get('is_distributed'):
                    logger.error('Distributed setup validation failed')
                    return False

            logger.debug('Setup validation passed')
            return True

        except Exception as e:
            logger.error(f'Setup validation failed: {e}')
            return False

    def call(self, method_name: str, *args, **kwargs) -> Any:
        """Call a method on the model runner, handling distributed execution.

        Args:
            method_name: Name of the method to call.
            *args: Positional arguments for the method.
            **kwargs: Keyword arguments for the method.

        Returns:
            Result of the method call.
        """
        if self.distributed_manager and self.distributed_manager.is_distributed:
            # Wake up workers by setting their events
            if isinstance(self.distributed_manager.events, list):
                for event in self.distributed_manager.events:
                    event.set()

            # Broadcast command
            cmd = (method_name, args, kwargs)
            self.distributed_manager.broadcast_data(cmd)

        # Execute locally
        if not hasattr(self, method_name):
            raise AttributeError(
                f"ModelRunner has no attribute '{method_name}'")

        func = getattr(self, method_name)
        return func(*args, **kwargs)

    def run(self,
            sequences: List[Sequence],
            prefill: bool = True) -> Optional[List[int]]:
        """Alias for execute_inference to match LLMEngine interface."""
        output = self.execute_inference(sequences, prefill)
        if output is None:
            return None
        _, tokens = output
        return tokens

    def worker_loop(self) -> None:
        """Execute worker process loop for distributed inference.

        This method is only used for worker processes (rank > 0) in
        distributed setups. The main process (rank 0) uses the
        normal inference interface.
        """
        if self.rank == 0:
            logger.warning('Worker loop called on main process')
            return

        if not self.distributed_manager.is_distributed:
            logger.warning('Worker loop called in non-distributed mode')
            return

        logger.info(f'Starting worker loop for rank {self.rank}')

        # Import Event for type checking and clearing
        from multiprocessing.synchronize import Event

        try:
            while True:
                # Wait for synchronization event
                self.distributed_manager.wait_for_events()

                # Clear event to allow waiting again in next iteration
                if isinstance(self.distributed_manager.events, Event):
                    self.distributed_manager.events.clear()
                elif isinstance(self.distributed_manager.events, list):
                    for e in self.distributed_manager.events:
                        e.clear()

                # Receive command from rank 0
                cmd = self.distributed_manager.broadcast_data(None, src=0)
                method_name, args, kwargs = cmd

                if method_name == 'exit':
                    logger.info(f'Worker {self.rank} received exit command')
                    self.cleanup()
                    break

                # Execute command
                if hasattr(self, method_name):
                    func = getattr(self, method_name)
                    func(*args, **kwargs)
                else:
                    logger.error(
                        f'Worker {self.rank} received unknown method: {method_name}'
                    )

        except KeyboardInterrupt:
            logger.info(f'Worker {self.rank} interrupted')
        except Exception as e:
            logger.error(f'Worker {self.rank} error: {e}', exc_info=True)
        finally:
            logger.info(f'Worker {self.rank} exiting')

    def cleanup(self) -> None:
        """Clean up all resources and managers."""
        logger.info('Cleaning up ModelRunner resources...')

        try:
            # Clean up inference executor
            if self.inference_executor:
                logger.debug('Cleaning up inference executor...')
                self.inference_executor.cleanup()
                self.inference_executor = None

            # Clean up model manager
            if self.model_manager:
                logger.debug('Cleaning up model manager...')
                self.model_manager.cleanup()
                self.model_manager = None

            # Clean up distributed manager
            if self.distributed_manager:
                logger.debug('Cleaning up distributed manager...')
                self.distributed_manager.cleanup()
                self.distributed_manager = None

            self.is_initialized = False
            logger.info('ModelRunner cleanup completed')

        except Exception as e:
            logger.error(f'Error during cleanup: {e}', exc_info=True)

    def exit(self) -> None:
        """Exit the runner and cleanup resources."""
        self.cleanup()

    def __enter__(self):
        """Context manager entry."""
        self.initialize(
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            max_num_seqs=self.config.max_num_seqs)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


# Legacy compatibility functions
def create_model_runner(config: Config,
                        rank: int = 0,
                        events: Union[Any, List[Any]] = None) -> ModelRunner:
    """Create a ModelRunner instance.

    This function provides backward compatibility with the original
    ModelRunner constructor interface.

    Args:
        config: Engine configuration.
        rank: Process rank.
        events: Synchronization events.

    Returns:
        Initialized ModelRunner instance.
    """
    runner = ModelRunner(config, rank, events)
    runner.initialize(max_num_batched_tokens=config.max_num_batched_tokens,
                      max_num_seqs=config.max_num_seqs)
    return runner
