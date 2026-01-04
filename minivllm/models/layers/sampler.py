"""Token sampling utilities for text generation.

This module provides the Sampler class for selecting tokens from model logits
using temperature-based sampling. The implementation uses an efficient
approximation of Gumbel sampling for diversity.

Typical usage:
    >>> sampler = Sampler()
    >>> logits = torch.randn(4, 50000)  # (batch_size, vocab_size)
    >>> temperatures = torch.tensor([0.7, 0.8, 0.9, 1.0])
    >>> tokens = sampler(logits, temperatures)
    >>> print(tokens.shape)  # torch.Size([4])
"""

from typing import Final

import torch
from torch import nn


class Sampler(nn.Module):
    """Token sampler that selects tokens using temperature-scaled sampling.

    This module performs temperature scaling on logits followed by an
    efficient approximation of Gumbel sampling using exponential division
    and argmax. The approach provides good diversity while being faster
    than true Gumbel sampling.

    Attributes:
        MIN_TEMPERATURE: Minimum temperature value to prevent numerical issues.
        MIN_PROB: Minimum probability value to prevent log(0) errors.

    Example:
        >>> sampler = Sampler()
        >>> logits = torch.randn(2, 1000)  # batch=2, vocab=1000
        >>> temperatures = torch.tensor([0.7, 0.9])
        >>> tokens = sampler(logits, temperatures)
        >>> print(tokens)  # tensor([42, 156])
    """

    # Class constants for numerical stability
    MIN_TEMPERATURE: Final[float] = 1e-8
    MIN_PROB: Final[float] = 1e-10

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
    ) -> torch.Tensor:
        """Sample tokens from logits with per-sample temperature scaling.

        This method applies temperature scaling to logits and uses an
        approximation of Gumbel sampling to select diverse tokens.

        Args:
            logits: Float tensor of shape (batch_size, vocab_size) containing
                unnormalized log probabilities for each token.
            temperatures: Float tensor of shape (batch_size,) with positive
                temperature values. Lower values (e.g., 0.1) produce more
                focused/deterministic sampling, while higher values (e.g., 1.5)
                produce more diverse/random sampling.

        Returns:
            Long tensor of shape (batch_size,) containing sampled token IDs.

        Raises:
            ValueError: If input dimensions are incorrect or batch sizes don't match.

        Note:
            - Temperature of 1.0 corresponds to sampling from the raw distribution
            - Temperature < 1.0 makes high-probability tokens more likely
            - Temperature > 1.0 flattens the distribution for more diversity
        """
        # Validate input dimensions
        if logits.dim() != 2:
            raise ValueError(
                f'logits must be a 2D tensor of shape (batch, vocab), '
                f'got shape {logits.shape}')
        if temperatures.dim() != 1:
            raise ValueError(
                f'temperatures must be a 1D tensor of shape (batch,), '
                f'got shape {temperatures.shape}')
        if logits.size(0) != temperatures.size(0):
            raise ValueError(
                f'Batch size mismatch: logits has {logits.size(0)} samples '
                f'but temperatures has {temperatures.size(0)} samples')

        # Apply temperature scaling
        # Clamp to prevent divide-by-zero
        temperatures_clamped = temperatures.clamp_min(self.MIN_TEMPERATURE)
        logits_scaled = logits.float().div_(
            temperatures_clamped.unsqueeze(dim=1))

        # Convert scaled logits to probabilities
        probs = torch.softmax(logits_scaled, dim=-1)

        # Approximate Gumbel sampling using exponential noise
        # This is faster than true Gumbel: -log(-log(uniform))
        # but provides similar diversity characteristics
        noise = torch.empty_like(probs).exponential_(1).clamp_min_(
            self.MIN_PROB)
        sample_tokens = probs.div_(noise).argmax(dim=-1)

        return sample_tokens
