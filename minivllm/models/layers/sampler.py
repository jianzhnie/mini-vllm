"""Simple sampling utilities for token selection."""

import torch
from torch import nn


class Sampler(nn.Module):
    """Sampler module that selects tokens given logits and per-sample temperatures.

    Current implementation performs temperature scaling followed by a
    sampling heuristic using Gumbel-like noise via exponential division
    and argmax. Returns a 1-D tensor of token ids (one per batch entry).
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, logits: torch.Tensor,
                temperatures: torch.Tensor) -> torch.Tensor:
        """Sample tokens from logits with per-sample temperatures.

        Args:
            logits: Float tensor of shape (batch, vocab).
            temperatures: Float tensor of shape (batch,) with positive values.

        Returns:
            1-D tensor of sampled token ids (shape: [batch]).
        """
        if logits.dim() != 2:
            raise ValueError(
                'logits must be a 2D tensor of shape (batch, vocab)')
        if temperatures.dim() != 1:
            raise ValueError(
                'temperatures must be a 1D tensor of shape (batch,)')
        if logits.size(0) != temperatures.size(0):
            raise ValueError(
                'batch size of logits and temperatures must match')

        # Safely scale logits by temperature (avoid divide-by-zero)
        # Lower temperature = more focused/deterministic sampling
        # Higher temperature = more random/diverse sampling
        temperatures = temperatures.clamp_min(1e-8)
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))

        # Convert scaled logits to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Approximate sampling using Gumbel-ish trick: divide by exponential noise
        # This provides a sampling approximation that's faster than true Gumbel sampling
        # The noise creates diversity while maintaining some structure
        noise = torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        sample_tokens = probs.div_(noise).argmax(dim=-1)
        return sample_tokens
