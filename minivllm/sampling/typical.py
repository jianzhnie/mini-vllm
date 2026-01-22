"""
---
title: Typical Sampling
summary: A PyTorch implementation of typical sampling from language models.
---

# Typical Sampling

Typical sampling is a decoding strategy based on the Typical Set. Unlike nucleus sampling
which operates on cumulative probabilities, typical sampling filters tokens based on how
typical they are for the given context, measured by their information content (KL divergence).

The key idea is to select tokens with entropy close to the expected entropy of the model.
This can be thought of as selecting tokens that are neither too likely nor too unlikely
given the context, resulting in more natural text generation.

Reference: "Typical Decoding for Natural Language Generation"
"""

from typing import Tensor

import torch
from torch import nn

from minivllm.sampling.base import Sampler


class TypicalSampler(Sampler):
    """
    ## Typical Sampler

    Filters tokens based on how typical they are for the given context,
    measured by KL divergence from the mean entropy.
    """

    def __init__(self, tau: float = 1.0, sampler: Sampler = None):
        """
        :param tau: is the hyperparameter controlling the typical threshold (default: 1.0)
        :param sampler: is the sampler to use for the filtered tokens.
                       If None, will use uniform sampling from the filtered tokens.
        """
        if tau <= 0:
            raise ValueError(f'tau must be positive, got {tau}')
        self.tau = tau
        self.sampler = sampler
        self.softmax = nn.Softmax(dim=-1)

    def __call__(self, logits: Tensor) -> Tensor:
        """
        Sample from logits using typical sampling

        :param logits: are the logits of the distribution of shape `[..., n_tokens]`
        :return: sampled token indices of shape `[...]`
        """
        # Get probabilities from logits
        probs = self.softmax(logits)

        # Compute entropy of the distribution: H = -sum(p * log(p))
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)

        # Compute KL divergence for each token: D_KL = p * (log(p) - log(avg_p))
        # Which simplifies to: p * (log(p) + entropy) when avg_p = exp(entropy)
        neg_entropy = (probs * (log_probs + entropy))

        # Find tokens with entropy close to the distribution's entropy
        # Use scaled absolute difference as the filtering criterion
        entropy_threshold = self.tau * entropy

        # Create mask for tokens within the typical set
        # Select tokens where |log(p) + H| <= tau * H
        mask = neg_entropy.abs() <= entropy_threshold

        # Create filtered logits, setting below-threshold tokens to -inf
        filtered_logits = logits.clone()
        filtered_logits[~mask] = float('-inf')

        # If no tokens are left after filtering, relax the threshold
        if torch.any(torch.all(~mask, dim=-1)):
            # For batches where all tokens were filtered out, use original distribution
            all_filtered = torch.all(~mask, dim=-1, keepdim=True)
            filtered_logits = torch.where(all_filtered, logits,
                                          filtered_logits)

        # Sample from the filtered logits
        if self.sampler is not None:
            return self.sampler(filtered_logits)
        else:
            # Use uniform sampling from the filtered tokens if no sampler provided
            from torch.distributions import Categorical
            dist = Categorical(logits=filtered_logits)
            return dist.sample()
