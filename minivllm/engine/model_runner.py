"""Optimized Model Runner module for coordinating model inference.

This module provides the ModelRunner class which orchestrates the complete
inference process using a modular architecture with specialized managers:

Architecture Overview:
- ModelManager: Handles model loading, validation, and lifecycle management
- DistributedManager: Manages multi-process coordination and communication
- InferenceExecutor: Executes model inference with performance optimization

This is a refactored version of the original model_runner.py that splits
functionality into three focused modules for better maintainability.
"""

import os
from multiprocessing.synchronize import Event
from typing import Any

from minivllm.config import Config
from minivllm.engine.distributed_manager import DistributedManager
from minivllm.engine.inference_executor import InferenceExecutor
from minivllm.engine.sequence import Sequence
from minivllm.models.manager import ModelManager
from minivllm.utils.logger_utils import get_logger

logger = get_logger(__name__)

__all__ = ['ModelRunner']


class ModelRunner:
    """Executes model inference with distributed tensor parallelism
    and device graph optimization.

    This refactored ModelRunner coordinates specialized managers:
    - ModelManager: Model loading, validation, and lifecycle
    - DistributedManager: Multi-process coordination and communication
    - InferenceExecutor: Model execution and optimization

    For single-device inference (tensor_parallel_size=1), the ModelRunner runs
    in the main process. For multi-device inference, worker processes are spawned
    that communicate via distributed backend.
    """

    def __init__(self, config: Config, rank: int,
                 event: Event | list[Event]) -> None:
        """Initialize the model runner.

        This method:
        1. Sets up distributed environment variables
        2. Initializes managers for model, distributed coordination, and inference
        3. Loads the model and tokenizer
        4. Allocates KV cache memory
        5. Optionally captures device graphs for efficient inference
        6. Sets up IPC for multi-GPU scenarios

        Args:
            config: Engine configuration with model path and settings.
            rank: Device rank (0 = main device in distributed setup).
            event: Synchronization event(s) for distributed coordination.

        Note:
            For multi-device inference, worker processes (rank > 0) will block
            in the __init__ loop waiting for commands from the main process.
        """
        # Setup distributed environment variables
        if str(rank) != os.environ.get('RANK'):
            os.environ['RANK'] = str(rank)
        if str(rank) != os.environ.get('LOCAL_RANK'):
            os.environ['LOCAL_RANK'] = str(rank)
        if str(config.tensor_parallel_size) != os.environ.get('WORLD_SIZE'):
            os.environ['WORLD_SIZE'] = str(config.tensor_parallel_size)

        self.config: Config = config
        self.rank: int = rank
        self.events: Event | list[Event] = event
        self.world_size: int = config.tensor_parallel_size

        # Initialize managers
        self.model_manager: ModelManager = ModelManager(config)
        self.distributed_manager: DistributedManager = DistributedManager(
            config, rank, event)
        self.inference_executor: InferenceExecutor | None = None

        # Initialize all components
        self._initialize()

        # Setup IPC for distributed inference and enter worker loop if needed
        if self.world_size > 1:
            self.distributed_manager.synchronize()
            if rank != 0:
                self._worker_loop()

    def _initialize(self) -> None:
        """Initialize all components and prepare for inference."""
        try:
            logger.info(f"Rank {self.rank}: Initializing ModelRunner...")

            # Initialize model management (loads model and tokenizer)
            self.model_manager.initialize()

            # Initialize distributed coordination
            self.distributed_manager.initialize()

            # Initialize inference executor with loaded model
            self.inference_executor = InferenceExecutor(
                self.config, self.model_manager.model)
            self.inference_executor.initialize(
                self.config.max_num_batched_tokens, self.config.max_num_seqs)

            # Capture optimization graphs (only on rank 0)
            if self.rank == 0:
                self.inference_executor.capture_device_graphs(
                    self.config.max_num_seqs)

            logger.info(
                f"Rank {self.rank}: ModelRunner initialization completed")

        except Exception as e:
            logger.error(
                f"Rank {self.rank}: ModelRunner initialization failed: {e}")
            self.exit()
            raise

    def _worker_loop(self) -> None:
        """Main event loop for worker processes.

        Worker processes (rank > 0) wait for commands from the main process
        via distributed communication, execute them, and continue until
        receiving 'exit'.
        """
        logger.info(f"Rank {self.rank}: Starting worker loop")

        try:
            while True:
                # Receive command from rank 0 via broadcast
                cmd = self.distributed_manager.broadcast_data(None, src=0)

                # Validate command format
                if not isinstance(cmd, (list, tuple)) or len(cmd) != 3:
                    logger.error(
                        f"Rank {self.rank}: Invalid command format: {cmd}")
                    continue

                method_name, args, kwargs = cmd

                if method_name == 'exit':
                    logger.info(f"Rank {self.rank}: Received exit command")
                    break

                # Execute command locally
                if hasattr(self, method_name):
                    try:
                        method = getattr(self, method_name)
                        method(*args, **kwargs)
                    except Exception as e:
                        logger.error(
                            f"Rank {self.rank}: Error executing {method_name}: {e}"
                        )
                        # Continue processing next command instead of crashing
                else:
                    logger.error(
                        f"Rank {self.rank}: Unknown method received: {method_name}"
                    )
                    # Notify rank 0 about the error if needed

        except KeyboardInterrupt:
            logger.info(f"Rank {self.rank}: Worker interrupted")
        except Exception as e:
            logger.error(f"Rank {self.rank}: Worker error: {e}", exc_info=True)
        finally:
            logger.info(f"Rank {self.rank}: Worker exiting")
            self.exit()

    def call(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Call a method, handling distributed IPC if needed.

        For main process: broadcasts command to workers if distributed,
        then executes locally. For worker processes: executes command
        received via IPC.

        Args:
            method_name: Name of method to call.
            *args: Arguments for the method.
            **kwargs: Keyword arguments for the method.

        Returns:
            Return value of the called method.
        """
        # Broadcast command to workers if in distributed mode
        if self.world_size > 1 and self.rank == 0:
            cmd = (method_name, args, kwargs)
            self.distributed_manager.broadcast_data(cmd)

        # Execute locally
        if not hasattr(self, method_name):
            raise AttributeError(f"ModelRunner has no method '{method_name}'")

        method = getattr(self, method_name)
        return method(*args, **kwargs)

    def run(self, sequences: list[Sequence],
            is_prefill: bool) -> list[int] | None:
        """Execute inference on a batch of sequences.

        Prepares input tensors, runs the model, samples tokens, and updates
        sequence state. For distributed inference, all ranks participate
        in computation but only rank 0 returns tokens.

        Args:
            sequences: Sequences to process.
            is_prefill: Whether this is prefill (True) or decode (False) phase.

        Returns:
            List of sampled token IDs for each sequence (only on rank 0).
            Returns None for worker processes (rank > 0).
        """
        if self.inference_executor is None:
            raise RuntimeError('Inference executor not initialized')

        # Execute inference batch - all ranks participate
        logits, next_tokens = self.inference_executor.execute_batch(
            sequences, is_prefill)

        # Only rank 0 returns tokens
        if self.rank == 0:
            return next_tokens
        return None

    def exit(self) -> None:
        """Cleanup model runner resources.

        This method performs ordered cleanup:
        1. Cleans up inference executor
        2. Cleans up distributed manager
        3. Cleans up model manager
        """
        logger.info(f"Rank {self.rank}: Cleaning up ModelRunner...")

        errors: list[str] = []

        # Step 1: Cleanup inference executor
        if self.inference_executor is not None:
            try:
                self.inference_executor.cleanup()
            except Exception as e:
                errors.append(f"Failed to cleanup inference executor: {e}")
            finally:
                self.inference_executor = None

        # Step 2: Cleanup distributed manager
        try:
            self.distributed_manager.cleanup()
        except Exception as e:
            errors.append(f"Failed to cleanup distributed manager: {e}")

        # Step 3: Cleanup model manager
        try:
            self.model_manager.cleanup()
        except Exception as e:
            errors.append(f"Failed to cleanup model manager: {e}")

        if errors:
            logger.warning(
                f"Rank {self.rank}: Cleanup completed with errors: {errors}")
        else:
            logger.info(f"Rank {self.rank}: ModelRunner cleanup completed")
