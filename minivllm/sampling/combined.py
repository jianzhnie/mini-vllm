"""
---
title: Combined Sampling
summary: A PyTorch implementation of combined sampling strategies from language models.
---

# Combined Sampling

This module provides combined sampling strategies that apply multiple filtering techniques
in sequence, such as top-k + top-p filtering, or temperature + min-p + top-k filtering.

Combined samplers offer more fine-grained control over token selection and can produce
higher quality outputs by leveraging the strengths of multiple sampling methods.
"""

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from minivllm.sampling.base import Sampler


class TopKTopPSampler(Sampler):
    """
    ## Top-K and Top-P Combined Sampler

    Applies both top-k and top-p (nucleus) filtering to create a more restricted
    sampling distribution.
    """

    def __init__(
        self,
        k: int = 50,
        p: float = 0.95,
        temperature: float = 1.0,
    ):
        """
        :param k: is the number of top tokens to consider
        :param p: is the cumulative probability threshold for nucleus sampling
        :param temperature: is the temperature for controlling randomness
        """
        if k < 1:
            raise ValueError(f'k must be at least 1, got {k}')
        if not (0.0 < p <= 1.0):
            raise ValueError(f'p must be between 0.0 and 1.0, got {p}')
        if temperature <= 0:
            raise ValueError(
                f'temperature must be positive, got {temperature}')

        self.k = k
        self.p = p
        self.temperature = temperature

    def __call__(self, logits: Tensor) -> Tensor:
        """
        Sample from logits using combined top-k and top-p sampling

        :param logits: are the logits of the distribution of shape `[..., n_tokens]`
        :return: sampled token indices of shape `[...]`
        """
        # Scale logits by temperature
        scaled_logits = logits / self.temperature

        # Step 1: Apply top-k filtering
        # Get top-k values and indices
        top_k_values, top_k_indices = torch.topk(scaled_logits,
                                                 min(self.k,
                                                     scaled_logits.size(-1)),
                                                 dim=-1)

        # Create a tensor filled with -inf
        filtered_logits = torch.full_like(scaled_logits, float('-inf'))

        # Scatter the top-k values back
        filtered_logits.scatter_(-1, top_k_indices, top_k_values)

        # Step 2: Apply top-p filtering on the top-k tokens
        probs = torch.softmax(filtered_logits, dim=-1)

        # Sort probabilities of non-filtered tokens
        sorted_probs, sorted_indices = torch.sort(probs,
                                                  descending=True,
                                                  dim=-1)

        # Remove zeros from sorted_probs (tokens with -inf logits)
        # and compute cumulative sum only for valid tokens
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

        # Create nucleus mask
        nucleus_mask = cumsum_probs <= self.p
        nucleus_mask = torch.cat([
            nucleus_mask.new_ones(nucleus_mask.shape[:-1] +
                                  (1, )), nucleus_mask[..., :-1]
        ],
                                 dim=-1)

        # Get the corresponding logits for nucleus tokens
        nucleus_logits = torch.gather(filtered_logits, -1, sorted_indices)
        nucleus_logits[~nucleus_mask] = float('-inf')

        # Sample from the combined distribution
        dist = Categorical(logits=nucleus_logits)
        sampled_sorted_idx = dist.sample()

        # Unsort to get original indices
        result = torch.gather(sorted_indices, -1,
                              sampled_sorted_idx.unsqueeze(-1))

        return result.squeeze(-1)


class TemperatureMinPTopKSampler(Sampler):
    """
    ## Temperature + Min-P + Top-K Combined Sampler

    Applies temperature scaling, min-p filtering, and top-k filtering in sequence.
    This combination often produces high-quality text generation with good balance
    between diversity and coherence.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        min_p: float = 0.0,
        top_k: int = 0,
    ):
        """
        :param temperature: is the temperature for controlling randomness
        :param min_p: is the minimum probability ratio (0.0 to 1.0) relative to max probability
        :param top_k: is the number of top tokens to consider (0 = disabled)
        """
        if temperature <= 0:
            raise ValueError(
                f'temperature must be positive, got {temperature}')
        if not (0.0 <= min_p <= 1.0):
            raise ValueError(f'min_p must be between 0.0 and 1.0, got {min_p}')
        if top_k < 0:
            raise ValueError(f'top_k must be non-negative, got {top_k}')

        self.temperature = temperature
        self.min_p = min_p
        self.top_k = top_k
        self.softmax = nn.Softmax(dim=-1)

    def __call__(self, logits: Tensor) -> Tensor:
        """
        Sample from logits using combined temperature + min-p + top-k sampling

        :param logits: are the logits of the distribution of shape `[..., n_tokens]`
        :return: sampled token indices of shape `[...]`
        """
        # Scale logits by temperature
        scaled_logits = logits / self.temperature

        # Step 1: Apply top-k filtering if enabled
        if self.top_k > 0:
            top_k_values, top_k_indices = torch.topk(
                scaled_logits, min(self.top_k, scaled_logits.size(-1)), dim=-1)
            # Create a tensor filled with -inf
            filtered_logits = torch.full_like(scaled_logits, float('-inf'))
            # Scatter the top-k values back
            filtered_logits.scatter_(-1, top_k_indices, top_k_values)
        else:
            filtered_logits = scaled_logits.clone()

        # Step 2: Apply min-p filtering if enabled
        if self.min_p > 0:
            probs = torch.softmax(filtered_logits, dim=-1)

            # Find the maximum probability
            max_probs = torch.max(probs, dim=-1, keepdim=True).values

            # Calculate the minimum probability threshold
            min_threshold = self.min_p * max_probs

            # Create a mask for tokens above the threshold
            mask = probs >= min_threshold

            # Set below-threshold tokens to -inf
            filtered_logits[~mask] = float('-inf')

        # Handle case where all tokens are filtered
        if torch.any(torch.all(torch.isinf(filtered_logits), dim=-1)):
            # For sequences where all tokens were filtered, use original logits
            all_filtered = torch.all(torch.isinf(filtered_logits),
                                     dim=-1,
                                     keepdim=True)
            filtered_logits = torch.where(all_filtered,
                                          logits / self.temperature,
                                          filtered_logits)

        # Sample from the filtered distribution
        dist = Categorical(logits=filtered_logits)
        return dist.sample()


class ExhaustiveTopKSampler(Sampler):
    """
    ## Exhaustive Top-K Sampler

    A more conservative top-k sampler that ensures the top-k tokens are always selected
    even if some have zero probability after softmax. Useful for preventing all tokens
    from being filtered out.
    """

    def __init__(self, k: int = 10):
        """
        :param k: is the number of top tokens to always include
        """
        if k < 1:
            raise ValueError(f'k must be at least 1, got {k}')
        self.k = k

    def __call__(self, logits: Tensor) -> Tensor:
        """
        Sample from the top-k tokens using exhaustive selection

        :param logits: are the logits of the distribution of shape `[..., n_tokens]`
        :return: sampled token indices of shape `[...]`
        """
        # Get top-k values and indices
        k = min(self.k, logits.size(-1))
        _, top_k_indices = torch.topk(logits, k, dim=-1)

        # Create a one-hot encoding of top-k tokens
        # Create a mask for top-k indices
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(-1, top_k_indices, True)

        # Create filtered logits with only top-k tokens
        filtered_logits = torch.where(
            mask, logits,
            torch.tensor(float('-inf'),
                         device=logits.device,
                         dtype=logits.dtype))

        # Sample from the filtered distribution
        dist = Categorical(logits=filtered_logits)
        return dist.sample()
