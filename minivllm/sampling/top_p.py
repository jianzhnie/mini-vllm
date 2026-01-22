"""
---
title: Top-P Sampling (Nucleus Sampling)
summary: A PyTorch implementation of top-p (nucleus) sampling from language models.
---

# Top-P Sampling (Nucleus Sampling)

This is a standalone implementation of nucleus sampling, also known as top-p sampling.
It is introduced in the paper "The Curious Case of Neural Text Degeneration"
(https://arxiv.org/abs/1904.09751).

The key idea is to select tokens from the smallest set of tokens whose cumulative
probability exceeds threshold p. This creates a dynamic vocabulary that adapts to
the context, filtering out both very common and very rare tokens.

Unlike the composite NucleusSampler that requires another sampler, this implementation
includes the complete sampling logic.
"""

import torch
from torch import Tensor
from torch.distributions import Categorical

from minivllm.sampling.base import Sampler


class TopPSampler(Sampler):
    """
    ## Top-P Sampler (Nucleus Sampling)

    Samples tokens from the smallest set whose cumulative probability exceeds p.
    """

    def __init__(self, p: float = 0.95, temperature: float = 1.0):
        """
        :param p: is the cumulative probability threshold (between 0.0 and 1.0)
        :param temperature: is the temperature for controlling randomness
        """
        if not (0.0 < p <= 1.0):
            raise ValueError(f'p must be between 0.0 and 1.0, got {p}')
        if temperature <= 0:
            raise ValueError(
                f'temperature must be positive, got {temperature}')

        self.p = p
        self.temperature = temperature

    def __call__(self, logits: Tensor) -> Tensor:
        """
        Sample from logits using top-p (nucleus) sampling

        :param logits: are the logits of the distribution of shape `[..., n_tokens]`
        :return: sampled token indices of shape `[...]`
        """
        # Scale logits by temperature
        scaled_logits = logits / self.temperature

        # Get probabilities
        probs = torch.softmax(scaled_logits, dim=-1)

        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs,
                                                  descending=True,
                                                  dim=-1)

        # Calculate cumulative probabilities
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

        # Create mask for tokens in the nucleus (cumsum < p)
        # Prepend True to include at least one token
        nucleus_mask = cumsum_probs <= self.p
        nucleus_mask = torch.cat([
            nucleus_mask.new_ones(nucleus_mask.shape[:-1] +
                                  (1, )), nucleus_mask[..., :-1]
        ],
                                 dim=-1)

        # Create filtered logits with -inf for tokens outside nucleus
        filtered_logits = scaled_logits.gather(-1, sorted_indices)
        filtered_logits[~nucleus_mask] = float('-inf')

        # Sample from the filtered distribution
        dist = Categorical(logits=filtered_logits)
        sampled_sorted_indices = dist.sample()

        # Get the actual token indices (unsort)
        result = torch.gather(sorted_indices, -1,
                              sampled_sorted_indices.unsqueeze(-1))

        return result.squeeze(-1)
