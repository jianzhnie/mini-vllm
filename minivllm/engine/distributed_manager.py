"""Distributed Manager module for handling distributed inference coordination.

This module provides the DistributedManager class which handles:
- Multi-process coordination
- Shared memory management (optional, for high-performance IPC)
- Distributed communication setup using PyTorch distributed backend
- Synchronization between processes
- Data broadcast and gather operations
"""

import pickle
from multiprocessing.synchronize import Event
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist

from minivllm.config import Config
from minivllm.utils.device import get_distributed_backend
from minivllm.utils.logger_utils import get_logger

logger = get_logger(__name__)

__all__ = ['DistributedManager']


class DistributedManager:
    """Manages distributed inference coordination and communication.

    This class handles all aspects of distributed inference including
    process coordination, distributed backend initialization, and
    inter-process communication via PyTorch distributed.

    Attributes:
        config: Engine configuration.
        rank: Current process rank (0 = main process).
        world_size: Total number of processes.
        backend: Distributed backend being used.
        events: Synchronization events.
        is_distributed: Whether running in distributed mode.
    """

    def __init__(
        self,
        config: Config,
        rank: int,
        events: Union[Event, List[Event], None] = None,
    ) -> None:
        """Initialize the distributed manager.

        Args:
            config: Engine configuration.
            rank: Process rank in distributed setup.
            events: Synchronization events for coordination.
        """
        self.config = config
        self.rank = rank
        self.world_size = config.tensor_parallel_size
        self.backend: Optional[str] = None
        self.events = events
        self.is_distributed = self.world_size > 1
        self._initialized = False

        logger.debug(f'DistributedManager created with rank {rank}, '
                     f'world_size {self.world_size}')

    def initialize(self) -> None:
        """Initialize distributed communication if needed."""
        if self._initialized:
            return

        if not self.is_distributed:
            logger.debug('Running in single-process mode')
            self._initialized = True
            return

        self._setup_distributed_backend()
        self._validate_setup()

        self._initialized = True
        logger.info(f'Distributed manager initialized. '
                    f'Rank: {self.rank}, World size: {self.world_size}, '
                    f'Backend: {self.backend}')

    def _setup_distributed_backend(self) -> None:
        """Setup the distributed communication backend."""
        try:
            self.backend = get_distributed_backend()
            if not dist.is_initialized():
                dist.init_process_group(backend=self.backend,
                                        rank=self.rank,
                                        world_size=self.world_size)
            logger.debug(f'Distributed backend setup: {self.backend}')
        except Exception as e:
            raise RuntimeError(f'Failed to setup distributed backend: {e}')

    def _validate_setup(self) -> None:
        """Validate distributed setup is working correctly."""
        if not self.is_distributed:
            return

        try:
            # Simple all_reduce test
            tensor = torch.ones(1) * self.rank

            # Move to appropriate device based on backend
            if self.backend == 'nccl':
                tensor = tensor.cuda()
            elif self.backend == 'hccl':
                tensor = tensor.npu()

            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            expected = sum(range(self.world_size))

            if abs(tensor.item() - expected) > 1e-6:
                raise RuntimeError(
                    f'Distributed test failed: got {tensor.item()}, '
                    f'expected {expected}')

            logger.debug('Distributed communication test passed')
        except Exception as e:
            raise RuntimeError(f'Distributed validation failed: {e}')

    def _move_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to appropriate device based on backend.

        Args:
            tensor: The tensor to move.

        Returns:
            Tensor on the appropriate device.
        """
        if self.backend == 'nccl':
            return tensor.cuda()
        elif self.backend == 'hccl':
            return tensor.npu()
        elif self.backend == 'ccl':
            return tensor.xpu()
        return tensor

    def synchronize(self) -> None:
        """Synchronize all processes."""
        if self.is_distributed and dist.is_initialized():
            dist.barrier()
            logger.debug(f'Rank {self.rank}: Distributed barrier completed')

    def broadcast_data(self, data: Any, src: int = 0) -> Any:
        """Broadcast data from source to all processes.

        Uses pickle for serialization and PyTorch distributed for
        efficient data transfer.

        Args:
            data: Data to broadcast (only valid on src process).
            src: Source rank.

        Returns:
            Broadcasted data (same on all processes).
        """
        if not self.is_distributed:
            return data

        try:
            # Serialize data on source
            if self.rank == src:
                data_bytes = pickle.dumps(data)
                data_size = torch.tensor([len(data_bytes)], dtype=torch.long)
            else:
                data_size = torch.tensor([0], dtype=torch.long)

            # Move to appropriate device and broadcast size
            data_size = self._move_to_device(data_size)
            dist.broadcast(data_size, src=src)

            # Prepare buffer
            if self.rank != src:
                data_bytes_len = int(data_size.item())
                data_tensor = torch.empty(data_bytes_len, dtype=torch.uint8)
            else:
                data_tensor = torch.frombuffer(data_bytes, dtype=torch.uint8).clone()

            # Move to device and broadcast data
            data_tensor = self._move_to_device(data_tensor)
            dist.broadcast(data_tensor, src=src)

            # Deserialize on non-source processes
            if self.rank != src:
                data = pickle.loads(data_tensor.cpu().numpy().tobytes())

            return data
        except Exception as e:
            raise RuntimeError(f'Broadcast failed: {e}') from e

    def cleanup(self) -> None:
        """Clean up distributed resources."""
        if not self._initialized:
            return

        try:
            if self.is_distributed and dist.is_initialized():
                dist.barrier()

            if dist.is_initialized():
                dist.destroy_process_group()

            self._initialized = False
            logger.debug(
                f'Rank {self.rank}: Distributed manager cleanup completed')
        except Exception as e:
            logger.warning(
                f'Rank {self.rank}: Error during distributed cleanup: {e}')
