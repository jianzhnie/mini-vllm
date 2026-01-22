"""
---
title: Min-P Sampling
summary: A PyTorch implementation of min-p sampling from language models.
---

# Min-P Sampling

Min-P sampling is an alternative to nucleus sampling that maintains a dynamic threshold
based on the maximum probability. It selects tokens that have a probability at least
p * max_prob, where max_prob is the highest probability in the distribution.

This approach is more adaptive than nucleus (top-p) sampling and tends to produce
more consistent results across different prompt lengths and contexts.

Reference: https://github.com/oobabooga/text-generation-webui/pull/5841
"""

import torch
from torch import Tensor, nn

from minivllm.sampling.base import Sampler


class MinPSampler(Sampler):
    """
    ## Min-P Sampler

    Min-P sampling that dynamically filters tokens based on their relative probability
    to the maximum probability in the distribution.
    """

    def __init__(self, p: float, sampler: Sampler):
        """
        :param p: is the minimum probability ratio (0.0 to 1.0) relative to max probability
        :param sampler: is the sampler to use for the filtered tokens
        """
        if not (0.0 <= p <= 1.0):
            raise ValueError(f'p must be between 0.0 and 1.0, got {p}')
        self.p = p
        self.sampler = sampler
        # Softmax to compute probabilities from logits
        self.softmax = nn.Softmax(dim=-1)

    def __call__(self, logits: Tensor) -> Tensor:
        """
        Sample from logits using min-p sampling

        :param logits: are the logits of the distribution of shape `[..., n_tokens]`
        :return: sampled token indices of shape `[...]`
        """
        # Get probabilities from logits
        probs = self.softmax(logits)

        # Find the maximum probability
        max_probs = torch.max(probs, dim=-1, keepdim=True).values

        # Calculate the minimum probability threshold: p * max_prob
        min_threshold = self.p * max_probs

        # Create a mask for tokens above the threshold
        mask = probs >= min_threshold

        # Create filtered logits, setting below-threshold tokens to -inf
        filtered_logits = logits.clone()
        filtered_logits[~mask] = float('-inf')

        # Sample from the filtered logits using the provided sampler
        return self.sampler(filtered_logits)
