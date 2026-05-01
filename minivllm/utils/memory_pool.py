"""Memory pool for efficient tensor allocation.

This module provides a memory pool implementation for managing
device memory allocations efficiently.
"""

import logging

import torch

logger = logging.getLogger(__name__)


class MemoryPool:
    """Simple memory pool implementation with usage tracking."""

    def __init__(self, device: torch.device, max_pool_size_mb: float) -> None:
        self.device = device
        self.max_pool_size_bytes = int(max_pool_size_mb * 1024 * 1024)
        self.current_usage_bytes = 0
        self.allocations: dict[int, int] = {}

    def allocate(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocate a tensor from the pool."""
        tensor = torch.zeros(shape, dtype=dtype, device=self.device)
        size = tensor.element_size() * tensor.numel()

        self.current_usage_bytes += size
        self.allocations[id(tensor)] = size

        if self.current_usage_bytes > self.max_pool_size_bytes:
            logger.warning(
                'Memory pool exceeded limit: %.1fMB > %.1fMB',
                self.current_usage_bytes / 1024**2,
                self.max_pool_size_bytes / 1024**2,
            )

        return tensor

    def deallocate(self, tensor: torch.Tensor) -> None:
        """Deallocate a tensor."""
        if id(tensor) in self.allocations:
            self.current_usage_bytes -= self.allocations[id(tensor)]
            del self.allocations[id(tensor)]

    def cleanup(self) -> None:
        """Clean up all allocations."""
        self.allocations.clear()
        self.current_usage_bytes = 0

    def get_memory_info(self) -> dict[str, float]:
        """Get memory usage information."""
        return {
            'current_usage_mb': self.current_usage_bytes / (1024 * 1024),
            'max_pool_size_mb': self.max_pool_size_bytes / (1024 * 1024),
        }


def get_memory_pool(device: torch.device,
                    max_pool_size_mb: float) -> MemoryPool:
    """Factory function to get a memory pool instance."""
    return MemoryPool(device, max_pool_size_mb)
