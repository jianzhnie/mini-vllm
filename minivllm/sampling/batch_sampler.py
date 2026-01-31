"""Batch token sampling utilities for text generation.

This module provides the BatchSampler class for selecting tokens from model logits
using various sampling strategies including temperature, top-k, top-p (nucleus),
and min-p sampling, optimized for batch processing with per-sequence parameters.
"""

from typing import Final, Optional

import torch
from torch import nn


class BatchSampler(nn.Module):
    """Batch token sampler that selects tokens using combined sampling strategies.

    This module supports per-sequence sampling parameters for:
    1. Temperature scaling
    2. Top-K filtering
    3. Top-P (Nucleus) filtering
    4. Min-P filtering

    Attributes:
        MIN_TEMPERATURE: Minimum temperature value to prevent numerical issues.
        MIN_PROB: Minimum probability value to prevent log(0) errors.

    Example:
        >>> sampler = BatchSampler()
        >>> logits = torch.randn(2, 1000)
        >>> temperatures = torch.tensor([0.7, 0.9])
        >>> tokens = sampler(logits, temperatures)
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
        top_ps: Optional[torch.Tensor] = None,
        top_ks: Optional[torch.Tensor] = None,
        min_ps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample tokens from logits with per-sample sampling parameters.

        Args:
            logits: Float tensor of shape (batch_size, vocab_size).
            temperatures: Float tensor of shape (batch_size,).
            top_ps: Optional float tensor of shape (batch_size,).
            top_ks: Optional long tensor of shape (batch_size,).
            min_ps: Optional float tensor of shape (batch_size,).

        Returns:
            Long tensor of shape (batch_size,) containing sampled token IDs.
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

        # Keep original logits for recovery if all tokens are filtered
        original_logits = logits.clone()

        # Apply temperature scaling
        temperatures_clamped = temperatures.clamp_min(self.MIN_TEMPERATURE)
        logits = logits.float() / temperatures_clamped.unsqueeze(dim=1)

        # Apply Top-K filtering
        if top_ks is not None:
            # Check if any k is active (> 0 and < vocab_size)
            # Vectorized implementation for better performance
            active_mask = (top_ks > 0) & (top_ks < logits.size(-1))
            if active_mask.any():
                max_k = top_ks[active_mask].max().item()
                # Get top-k indices for all sequences
                # Note: We use the max_k across the batch to vectorize
                _, top_k_indices = torch.topk(logits, max_k, dim=-1)

                # Create a mask for valid k positions [batch, max_k]
                # range: [0, 1, ..., max_k-1]
                # top_ks: [k1, k2, ...]
                # mask: [[T, T, F], [T, T, T], ...]
                k_range = torch.arange(max_k,
                                       device=logits.device).unsqueeze(0)
                keep_top_k_mask = k_range < top_ks.unsqueeze(1)

                # Create the final mask for logits [batch, vocab]
                # Initialize with False (filter everything by default for active rows)
                # For inactive rows (top_k <= 0), we will set to True later
                mask = torch.zeros_like(logits, dtype=torch.bool)

                # Scatter the keep mask to the original positions
                # This sets True for the top-k tokens of each sequence
                mask.scatter_(1, top_k_indices, keep_top_k_mask)

                # For sequences where top-k is disabled (active_mask is False),
                # we should keep all tokens (set mask to True)
                mask[~active_mask] = True

                # Apply filter
                logits.masked_fill_(~mask, -float('inf'))

        # Apply Top-P filtering
        if top_ps is not None:
            # Only apply if any p < 1.0
            if (top_ps < 1.0).any():
                # Sort probabilities (descending)
                probs = torch.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs,
                                                          descending=True,
                                                          dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # Remove tokens with cumulative probability above the threshold (nucleus)
                sorted_indices_to_remove = cumulative_probs > top_ps.unsqueeze(
                    1)

                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Scatter sorted tensor to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, -float('inf'))

        # Apply Min-P filtering
        if min_ps is not None:
            # Only apply if any min_p > 0.0
            if (min_ps > 0.0).any():
                probs = torch.softmax(logits, dim=-1)
                top_prob, _ = torch.max(probs, dim=-1, keepdim=True)
                scaled_min_p = min_ps.unsqueeze(1) * top_prob
                tokens_to_remove = probs < scaled_min_p
                logits = logits.masked_fill(tokens_to_remove, -float('inf'))

        # Safety check: if all tokens are filtered out (all -inf), restore original logits
        # This can happen if p is too small or k is too small combined with other filters
        all_filtered = torch.all(torch.isinf(logits), dim=-1)
        if all_filtered.any():
            # Restore original logits for affected sequences (scaled by temperature)
            logits[all_filtered] = original_logits[
                all_filtered] / temperatures_clamped[all_filtered].unsqueeze(
                    dim=1)

        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Handle NaNs (rare but possible with extreme logits)
        if torch.isnan(probs).any():
            probs = torch.nan_to_num(probs, nan=0.0)
            # Renormalize
            sum_probs = probs.sum(dim=-1, keepdim=True)
            probs = probs / sum_probs.clamp_min(self.MIN_PROB)

        # Sample
        num_samples = 1
        sample_tokens = torch.multinomial(probs, num_samples,
                                          replacement=True).squeeze(1)

        return sample_tokens
