"""Distributed Manager module for handling distributed inference coordination.

This module provides the DistributedManager class which handles:
- Multi-process coordination
- Shared memory management
- Distributed communication setup
- Synchronization between processes
"""

import pickle
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist

from minivllm.config import Config
from minivllm.utils.device import get_dist_info, get_distributed_backend
from minivllm.utils.logger_utils import get_logger

logger = get_logger(__name__)

__all__ = ['DistributedManager']


class DistributedManager:
    """Manages distributed inference coordination and communication.

    This class handles all aspects of distributed inference including
    process coordination, shared memory management, and distributed
    backend initialization.

    Attributes:
        config: Engine configuration.
        rank: Current process rank (0 = main process).
        world_size: Total number of processes.
        backend: Distributed backend being used.
        shm: Shared memory instance for IPC.
        events: Synchronization events.
        is_distributed: Whether running in distributed mode.
    """

    def __init__(self,
                 config: Config,
                 rank: int,
                 events: Union[Event, List[Event], None] = None) -> None:
        """Initialize the distributed manager.

        Args:
            config: Engine configuration.
            rank: Process rank in distributed setup.
            events: Synchronization events for coordination.
        """
        self.config = config
        self.rank = rank
        self.world_size = config.tensor_parallel_size
        self.backend = None
        self.shm: Optional[SharedMemory] = None
        self.events = events or []
        self.is_distributed = self.world_size > 1

        logger.debug(f'DistributedManager created with rank {rank}, '
                     f'world_size {self.world_size}')

    def initialize(self) -> None:
        """Initialize distributed communication if needed."""
        if not self.is_distributed:
            logger.debug('Running in single-process mode')
            return

        self._setup_distributed_backend()
        self._setup_shared_memory()
        self._validate_setup()

        logger.info(f'Distributed manager initialized. '
                    f'Rank: {self.rank}, World size: {self.world_size}')

    def _setup_distributed_backend(self) -> None:
        """Setup the distributed communication backend."""
        try:
            self.backend = get_distributed_backend()
            if not dist.is_initialized():
                # Initialize distributed process group
                dist.init_process_group(backend=self.backend,
                                        rank=self.rank,
                                        world_size=self.world_size)
            logger.debug(f'Distributed backend setup: {self.backend}')
        except Exception as e:
            raise RuntimeError(f'Failed to setup distributed backend: {e}')

    def _setup_shared_memory(self) -> None:
        """Setup shared memory for inter-process communication."""
        try:
            if self.rank == 0:
                # Calculate required shared memory size
                shm_size = self._calculate_shm_size()

                # Create shared memory
                self.shm = SharedMemory(create=True, size=shm_size)
                logger.debug(f'Shared memory created: {shm_size} bytes, '
                             f'name: {self.shm.name}')

                # Broadcast shared memory name to other processes
                self.broadcast_data(self.shm.name)
            else:
                # Receive shared memory name from rank 0
                shm_name = self.broadcast_data(None, src=0)

                # Attach to existing shared memory
                self.shm = SharedMemory(name=shm_name, create=False)
                logger.debug(f'Attached to shared memory: {shm_name}')

        except Exception as e:
            logger.exception('Failed to setup shared memory')
            raise RuntimeError(f'Failed to setup shared memory: {e}')

    def _calculate_shm_size(self) -> int:
        """Calculate required shared memory size.

        Returns:
            Size in bytes for shared memory allocation.
        """
        # This is a simplified calculation - actual implementation
        # should consider model size, batch size, etc.
        base_size = 1024 * 1024 * 100  # 100MB base
        # 4 bytes per token
        model_factor = (self.config.max_num_seqs * self.config.max_model_len *
                        4)
        return base_size + model_factor

    def _validate_setup(self) -> None:
        """Validate distributed setup is working correctly."""
        if not self.is_distributed:
            return

        # Test distributed communication
        try:
            # Simple all_reduce test
            tensor = torch.ones(1).to(torch.device('cpu')) * self.rank
            # Move to correct device if possible, but CPU is safer for general test
            # unless backend requires specific device (like NCCL).
            # For simplicity keeping CPU as most backends support it or handle it.
            # Actually NCCL requires CUDA tensors.
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

    def synchronize(self) -> None:
        """Synchronize all processes."""
        if self.is_distributed and dist.is_initialized():
            dist.barrier()
            logger.debug('Distributed barrier completed')

    def broadcast_data(self, data: Any, src: int = 0) -> Any:
        """Broadcast data from source to all processes.

        Args:
            data: Data to broadcast.
            src: Source rank.

        Returns:
            Broadcasted data (same on all processes).
        """
        if not self.is_distributed:
            return data

        try:
            # Serialize data
            if self.rank == src:
                data_bytes = pickle.dumps(data)
                data_size = torch.tensor([len(data_bytes)], dtype=torch.long)
                if self.backend == 'nccl':
                    data_size = data_size.cuda()
                elif self.backend == 'hccl':
                    data_size = data_size.npu()
            else:
                data_size = torch.tensor([0], dtype=torch.long)
                if self.backend == 'nccl':
                    data_size = data_size.cuda()
                elif self.backend == 'hccl':
                    data_size = data_size.npu()

            # Broadcast size
            dist.broadcast(data_size, src=src)

            # Prepare buffer
            if self.rank != src:
                data_bytes_len = data_size.item()
                data_tensor = torch.empty(data_bytes_len, dtype=torch.uint8)
                if self.backend == 'nccl':
                    data_tensor = data_tensor.cuda()
                elif self.backend == 'hccl':
                    data_tensor = data_tensor.npu()
            else:
                data_tensor = torch.frombuffer(data_bytes, dtype=torch.uint8)
                if self.backend == 'nccl':
                    data_tensor = data_tensor.cuda()
                elif self.backend == 'hccl':
                    data_tensor = data_tensor.npu()

            # Broadcast data
            dist.broadcast(data_tensor, src=src)

            if self.rank != src:
                # Move back to CPU for pickle load
                data = pickle.loads(data_tensor.cpu().numpy().tobytes())

            return data
        except Exception as e:
            raise RuntimeError(f'Broadcast failed: {e}')

    def gather_data(self, data: Any, dst: int = 0) -> Optional[Any]:
        """Gather data from all processes to destination.

        Args:
            data: Data to gather from current process.
            dst: Destination rank.

        Returns:
            Gathered data (only on destination process).
        """
        if not self.is_distributed:
            return [data] if self.rank == dst else None

        try:
            # Serialize local data
            data_bytes = pickle.dumps(data)
            data_size = torch.tensor([len(data_bytes)], dtype=torch.long)

            if self.backend == 'nccl':
                data_size = data_size.cuda()
            elif self.backend == 'hccl':
                data_size = data_size.npu()

            # Gather sizes
            size_list = [
                torch.tensor([0], dtype=torch.long)
                for _ in range(self.world_size)
            ]
            if self.backend == 'nccl':
                size_list = [s.cuda() for s in size_list]
            elif self.backend == 'hccl':
                size_list = [s.npu() for s in size_list]

            dist.all_gather(size_list, data_size)

            # Create tensor for local data
            data_tensor = torch.frombuffer(data_bytes, dtype=torch.uint8)
            if self.backend == 'nccl':
                data_tensor = data_tensor.cuda()
            elif self.backend == 'hccl':
                data_tensor = data_tensor.npu()

            # Gather actual data
            # Note: dist.gather requires a list of tensors on dst
            if self.rank == dst:
                tensor_list = [
                    torch.empty(size.item(), dtype=torch.uint8)
                    for size in size_list
                ]
                if self.backend == 'nccl':
                    tensor_list = [t.cuda() for t in tensor_list]
                elif self.backend == 'hccl':
                    tensor_list = [t.npu() for t in tensor_list]

                dist.gather(data_tensor, gather_list=tensor_list, dst=dst)

                # Unpack
                result = []
                for t in tensor_list:
                    result.append(pickle.loads(t.cpu().numpy().tobytes()))
                return result
            else:
                dist.gather(data_tensor, dst=dst)
                return None
        except Exception as e:
            raise RuntimeError(f'Gather failed: {e}')

    def wait_for_events(self) -> None:
        """Wait for synchronization events."""
        if isinstance(self.events, Event):
            self.events.wait()
        elif isinstance(self.events, list):
            for event in self.events:
                if hasattr(event, 'wait'):
                    event.wait()
        logger.debug('Synchronization events completed')

    def set_events(self) -> None:
        """Set synchronization events to signal completion."""
        if isinstance(self.events, Event):
            self.events.set()
        elif isinstance(self.events, list):
            for event in self.events:
                if hasattr(event, 'set'):
                    event.set()
        logger.debug('Synchronization events set')

    def get_distributed_info(self) -> Dict[str, Any]:
        """Get distributed setup information.

        Returns:
            Dictionary with distributed configuration.
        """
        _, _, local_rank = get_dist_info()
        return {
            'is_distributed': self.is_distributed,
            'rank': self.rank,
            'world_size': self.world_size,
            'backend': self.backend,
            'local_rank': local_rank,
        }

    def cleanup(self) -> None:
        """Clean up distributed resources."""
        if self.shm is not None:
            self.shm.close()
            # Only rank 0 should unlink (delete) the shared memory
            if self.rank == 0:
                try:
                    self.shm.unlink()
                except FileNotFoundError:
                    # Already unlinked
                    pass
            self.shm = None

        if dist.is_initialized():
            dist.destroy_process_group()

        logger.debug('Distributed manager cleanup completed')

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
