"""Linear layer utilities with tensor-parallel support.

This module provides various linear layer implementations optimized for distributed training and inference,
including replicated, column-parallel, row-parallel, and specialized QKV linear layers.
"""

from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

__all__ = [
    'LinearBase',
    'ReplicatedLinear',
    'ColumnParallelLinear',
    'MergedColumnParallelLinear',
    'QKVParallelLinear',
    'RowParallelLinear',
    'divide',
]


def divide(numerator: int, denominator: int) -> int:
    """Divide two integers with divisibility check.

    Args:
        numerator: The dividend.
        denominator: The divisor.

    Returns:
        The quotient.

    Raises:
        ValueError: If numerator is not divisible by denominator.
    """
    if numerator % denominator != 0:
        raise ValueError(f'{numerator} is not divisible by {denominator}')
    return numerator // denominator


def get_tensor_parallel_rank() -> int:
    """Get the current tensor-parallel rank.

    Returns:
        The rank of the current process in the tensor-parallel group.
        Returns 0 if distributed is not initialized.
    """
    try:
        return dist.get_rank() if dist.is_initialized() else 0
    except Exception:
        return 0


def get_tensor_parallel_world_size() -> int:
    """Get the tensor-parallel world size.

    Returns:
        The total number of processes in the tensor-parallel group.
        Returns 1 if distributed is not initialized.
    """
    try:
        return dist.get_world_size() if dist.is_initialized() else 1
    except Exception:
        return 1


class LinearBase(nn.Module):
    """Base class for tensor-parallel linear layers.

    This class provides common functionality for distributed linear layers,
    including tensor-parallel rank/size detection and weight loading.

    Attributes:
        tp_rank: Tensor-parallel rank of current process.
        tp_size: Total number of tensor-parallel processes.
        tp_dim: Dimension along which to shard (0 for column, 1 for row).
        input_size: Input dimension of the linear layer.
        output_size: Output dimension of the linear layer.
        weight: Weight parameter of the linear layer.
        bias: Optional bias parameter of the linear layer.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_dim: Optional[int] = tp_dim
        # Use common helper functions for TP rank/size
        self.tp_rank: int = get_tensor_parallel_rank()
        self.tp_size: int = get_tensor_parallel_world_size()

        self.weight: nn.Parameter = nn.Parameter(
            torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader  # type: ignore

        if bias:
            self.bias: Optional[nn.Parameter] = nn.Parameter(
                torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader  # type: ignore
        else:
            self.register_parameter('bias', None)

    def extra_repr(self) -> str:
        return f'input_size={self.input_size}, output_size={self.output_size}, bias={self.bias is not None}, tp_dim={self.tp_dim}, tp_size={self.tp_size}'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError('Subclasses must implement forward method')

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      *args, **kwargs) -> None:
        """Load weights - to be implemented by subclasses."""
        raise NotImplementedError(
            'Subclasses must implement weight_loader method')


class ReplicatedLinear(LinearBase):
    """Standard dense linear layer replicated across tensor-parallel ranks."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ) -> None:
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      *args, **kwargs) -> None:
        """Load replicated weights (same across all ranks)."""
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear transformation."""
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """Column-parallel linear layer that shards output columns across ranks.

    This layer splits the output dimension across tensor-parallel ranks,
    allowing larger models to fit in distributed memory.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ) -> None:
        tp_size = get_tensor_parallel_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      *args, **kwargs) -> None:
        """Load column-sharded weights."""
        param_data = param.data
        assert self.tp_dim is not None, 'tp_dim must be set for column-parallel layers'
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx,
                                             shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply column-parallel linear transformation."""
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """Merged column-parallel linear that concatenates multiple column
    outputs (e.g., gate and up projections) into a single weight tensor.
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        bias: bool = False,
    ) -> None:
        self.output_sizes: List[int] = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int = 0,
        *args,
        **kwargs,
    ) -> None:
        """Load weights for merged column-parallel layers.

        Args:
            param: The parameter to load into.
            loaded_weight: The weight tensor to load from.
            loaded_shard_id: The index of the shard to load (default: 0).
        """
        param_data = param.data
        assert self.tp_dim is not None, 'tp_dim must be set for column-parallel layers'
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size,
                                            self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """Specialized parallel linear layer that packs Q, K, V projections.

    The output layout is: [Q(=num_heads), K(=num_kv_heads), V(=num_kv_heads)]
    spread across TP ranks. This fusion improves performance by reducing
    the number of kernel launches.
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: Optional[int] = None,
        bias: bool = False,
    ) -> None:
        tp_size = get_tensor_parallel_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size: int = head_size
        self.num_heads: int = divide(total_num_heads, tp_size)
        self.num_kv_heads: int = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads +
                       2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        """Load QKV weights with proper sharding.

        Args:
            param: The parameter to load into.
            loaded_weight: The weight tensor to load from.
            loaded_shard_id: One of 'q', 'k', 'v' to specify which projection to load.
        """
        if loaded_shard_id is None:
            raise ValueError(
                'loaded_shard_id must be provided for QKVParallelLinear')

        param_data = param.data
        assert loaded_shard_id in ['q', 'k',
                                   'v'], f'Invalid shard_id: {loaded_shard_id}'

        if loaded_shard_id == 'q':
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == 'k':
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:  # 'v'
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size

        assert self.tp_dim is not None, 'tp_dim must be set for column-parallel layers'
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size,
                                            self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

    def extra_repr(self) -> str:
        return (
            f'hidden_size={self.input_size}, head_size={self.head_size}, '
            f'num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads}, '
            f'bias={self.bias is not None}, tp_size={self.tp_size}')


class RowParallelLinear(LinearBase):
    """Row-parallel linear layer that shards input features across ranks.

    This layer splits the input dimension across tensor-parallel ranks
    and performs an all-reduce on the output.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ) -> None:
        tp_size = get_tensor_parallel_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      *args, **kwargs) -> None:
        """Load row-sharded weights."""
        param_data = param.data
        assert self.tp_dim is not None, 'tp_dim must be set for row-parallel layers'
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx,
                                             shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply row-parallel linear transformation with all-reduce."""
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
