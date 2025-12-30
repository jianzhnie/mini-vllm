"""Embedding and LM head utilities with tensor-parallel support.

This module provides vocabulary parallel embedding layers and language modelized for distributed inference heads
optim. The implementations support tensor parallelism
by partitioning the vocabulary across multiple devices and aggregating results
during computation.

Key features:
- VocabParallelEmbedding: Distributed token embeddings with automatic sharding
- ParallelLMHead: Language model head with tensor-parallel logit computation
- Support for both embedding lookup and logit computation
- Efficient aggregation of results across tensor-parallel ranks
"""

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from minivllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """Vocabulary-parallel embedding layer for distributed inference.

    This layer partitions the vocabulary across tensor-parallel ranks, with each
    rank storing only its portion of the embedding table. During forward pass,
    it handles local embedding lookup and aggregates results across ranks when
    necessary.

    Args:
        num_embeddings: Total vocabulary size across all ranks
        embedding_dim: Dimension of each embedding vector

    Raises:
        ValueError: If num_embeddings is not divisible by tensor-parallel size

    Attributes:
        tp_rank: Current tensor-parallel rank (0 if not distributed)
        tp_size: Total number of tensor-parallel ranks (1 if not distributed)
        num_embeddings_per_partition: Number of embeddings stored on this rank
        vocab_start_idx: Starting index of vocabulary for this rank
        vocab_end_idx: Ending index of vocabulary for this rank
        weight: Embedding weight matrix for this rank's vocabulary partition

    Shape:
        - Input: Long tensor of token IDs
        - Output: Float tensor of embeddings

    Examples:
        >>> embedding = VocabParallelEmbedding(1000, 512)  # 1000 vocab, 512 dims
        >>> x = torch.tensor([0, 5, 10])  # Token IDs
        >>> output = embedding(x)  # Shape: [3, 512]
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        # Safely get tensor-parallel rank/size. Default to (0, 1) when not
        # running in distributed mode.
        try:
            self.tp_rank: int = dist.get_rank() if dist.is_initialized() else 0
            self.tp_size: int = dist.get_world_size() if dist.is_initialized(
            ) else 1
        except Exception:
            self.tp_rank = 0
            self.tp_size = 1

        if num_embeddings % self.tp_size != 0:
            raise ValueError(
                f'num_embeddings ({num_embeddings}) must be divisible by '
                f'tensor-parallel size ({self.tp_size})')

        self.num_embeddings: int = num_embeddings
        self.num_embeddings_per_partition: int = self.num_embeddings // self.tp_size
        self.vocab_start_idx: int = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx: int = self.vocab_start_idx + self.num_embeddings_per_partition

        # Initialize weight for this rank's vocabulary partition
        self.weight: nn.Parameter = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter,
                      loaded_weight: torch.Tensor) -> None:
        """Load weights for the vocabulary partition.

        Args:
            param: Parameter to load weights into
            loaded_weight: Complete weight tensor from all ranks
        """
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Look up embeddings with optional tensor-parallel aggregation.

        Args:
            x: Long tensor of input token IDs. Token IDs outside this rank's
               vocabulary range will be masked out.

        Returns:
            Embedded tensor with same batch size as input and embedding_dim channels.
        """
        if self.tp_size > 1:
            # Create mask for tokens belonging to this rank's vocabulary
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # Adjust token indices to local vocabulary space
            x = mask * (x - self.vocab_start_idx)

        # Perform embedding lookup
        y = F.embedding(x, self.weight)

        if self.tp_size > 1:
            # Zero out embeddings for tokens not in this rank's vocabulary
            y = mask.unsqueeze(1) * y
            # Aggregate embeddings from all ranks
            dist.all_reduce(y)

        return y


class ParallelLMHead(VocabParallelEmbedding):
    """Parallel language model head for computing logits.

    This class extends VocabParallelEmbedding to provide logit computation
    from hidden states. It handles tensor-parallel aggregation of logits
    across ranks, ensuring that rank 0 receives the complete vocabulary logits.

    Args:
        num_embeddings: Total vocabulary size across all ranks
        embedding_dim: Hidden dimension of the model
        bias: Whether to include bias (not supported, always False)

    Raises:
        ValueError: If bias is requested (not supported)

    Attributes:
        Inherits all attributes from VocabParallelEmbedding

    Shape:
        - Input: Float tensor of hidden states
        - Output: Float tensor of logits (only on rank 0 in tensor-parallel mode)

    Examples:
        >>> lm_head = ParallelLMHead(1000, 512)
        >>> hidden_states = torch.randn(2, 512)
        >>> logits = lm_head(hidden_states)  # Shape: [2, 1000] (rank 0)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ) -> None:
        if bias:
            raise ValueError('bias for ParallelLMHead is not supported')
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Compute logits from hidden states.

        This method handles both prefill and decode phases:
        - During prefill: extracts only the final token for each sequence
        - During decode: computes logits for all provided inputs
        - In tensor-parallel mode: gathers all logits on rank 0

        Args:
            x: Float tensor of hidden states

        Returns:
            Float tensor of logits for the complete vocabulary.
            Returns None on non-zero ranks when using tensor parallelism.
        """
        context = get_context()

        # In prefill mode, only compute logits for the final token of each sequence
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()

        # Compute logits using linear transformation
        logits = F.linear(x, self.weight)

        if self.tp_size > 1:
            # Gather logits from all ranks onto rank 0
            all_logits = [
                torch.empty_like(logits) for _ in range(self.tp_size)
            ] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)

            # Concatenate logits from all ranks on rank 0
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None

        return logits
