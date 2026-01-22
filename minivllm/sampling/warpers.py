"""
---
title: Warper-Based Sampling
summary: A PyTorch implementation of logit warper-based sampling from language models.
---

# Warper-Based Sampling

This module provides "warper" samplers that transform logits before sampling,
inspired by Hugging Face's generation API. Warpers modify the probability distribution
in a composable way, allowing multiple transformations to be applied in sequence.

Common warping strategies include:
- Repetition penalty: discourages repeating the same tokens
- Frequency penalty: discourages tokens that have appeared frequently
- Presence penalty: discourages any tokens that have appeared before
"""

from typing import Optional

import torch
from torch import Tensor
from torch.distributions import Categorical

from minivllm.sampling.base import Sampler


class RepetitionPenaltySampler(Sampler):
    """
    ## Repetition Penalty Sampler

    Applies a penalty to previously generated tokens to encourage diversity.
    The penalty is applied as: logit = logit / penalty if logit > 0 else logit * penalty
    """

    def __init__(self,
                 penalty: float = 1.0,
                 sampler: Optional[Sampler] = None):
        """
        :param penalty: is the penalty factor (> 1.0 reduces probability, < 1.0 increases)
        :param sampler: optional sampler to apply after penalty (default: random sampling)
        """
        if penalty <= 0:
            raise ValueError(f'penalty must be positive, got {penalty}')
        self.penalty = penalty
        self.sampler = sampler

    def __call__(self,
                 logits: Tensor,
                 prev_tokens: Optional[Tensor] = None) -> Tensor:
        """
        Apply repetition penalty and sample

        :param logits: are the logits of the distribution of shape `[..., n_tokens]`
        :param prev_tokens: are the previously generated token indices, shape `[...]`
                           If None, no penalty is applied
        :return: sampled token indices of shape `[...]`
        """
        if prev_tokens is not None and self.penalty != 1.0:
            logits = logits.clone()

            # Vectorized implementation for repetition penalty
            # Create a mask for previously generated tokens
            unique_tokens = prev_tokens.unique()
            if unique_tokens.numel() > 0:
                # Create a mask tensor
                mask = torch.zeros(logits.shape[-1],
                                   dtype=torch.bool,
                                   device=logits.device)
                valid_tokens = unique_tokens[
                    (unique_tokens >= 0) & (unique_tokens < logits.shape[-1])]
                if valid_tokens.numel() > 0:
                    mask[valid_tokens] = True
                    # Apply penalty: divide positive logits, multiply negative logits
                    penalty_mask = mask.expand_as(logits)
                    logits[penalty_mask] = torch.where(
                        logits[penalty_mask] > 0,
                        logits[penalty_mask] / self.penalty,
                        logits[penalty_mask] * self.penalty)

        # Sample using the provided sampler or default random sampling
        if self.sampler is not None:
            return self.sampler(logits)
        else:
            dist = Categorical(logits=logits)
            return dist.sample()


class FrequencyPenaltySampler(Sampler):
    """
    ## Frequency Penalty Sampler

    Applies a penalty proportional to how often each token has appeared in the sequence.
    Penalty = alpha * (count of token in sequence)
    """

    def __init__(self, alpha: float = 0.1, sampler: Optional[Sampler] = None):
        """
        :param alpha: is the penalty coefficient (higher values = stronger penalty)
        :param sampler: optional sampler to apply after penalty
        """
        if alpha < 0:
            raise ValueError(f'alpha must be non-negative, got {alpha}')
        self.alpha = alpha
        self.sampler = sampler

    def __call__(self,
                 logits: Tensor,
                 sequence: Optional[Tensor] = None) -> Tensor:
        """
        Apply frequency penalty and sample

        :param logits: are the logits of the distribution
        :param sequence: are the previously generated token indices for frequency counting
        :return: sampled token indices
        """
        if sequence is not None and self.alpha > 0 and sequence.numel() > 0:
            logits = logits.clone()

            # Vectorized implementation for frequency penalty
            unique_tokens, counts = torch.unique(sequence, return_counts=True)
            if unique_tokens.numel() > 0:
                # Filter valid tokens
                valid_mask = (unique_tokens >= 0) & (unique_tokens <
                                                     logits.size(-1))
                valid_tokens = unique_tokens[valid_mask]
                valid_counts = counts[valid_mask]

                if valid_tokens.numel() > 0:
                    # Create penalty tensor
                    penalties = torch.zeros(logits.shape[-1],
                                            dtype=logits.dtype,
                                            device=logits.device)
                    penalties[valid_tokens] = valid_counts.float() * self.alpha

                    # Apply penalty
                    logits -= penalties.expand_as(logits)

        # Sample using the provided sampler or default random sampling
        if self.sampler is not None:
            return self.sampler(logits)
        else:
            dist = Categorical(logits=logits)
            return dist.sample()


class PresencePenaltySampler(Sampler):
    """
    ## Presence Penalty Sampler

    Applies a flat penalty to any token that has appeared in the sequence.
    This is a binary penalty (present vs. not present), unlike frequency penalty
    which scales with occurrence count.
    """

    def __init__(self,
                 penalty: float = 0.0,
                 sampler: Optional[Sampler] = None):
        """
        :param penalty: is the penalty amount for any previously seen token
        :param sampler: optional sampler to apply after penalty
        """
        if penalty < 0:
            raise ValueError(f'penalty must be non-negative, got {penalty}')
        self.penalty = penalty
        self.sampler = sampler

    def __call__(self,
                 logits: Tensor,
                 sequence: Optional[Tensor] = None) -> Tensor:
        """
        Apply presence penalty and sample

        :param logits: are the logits of the distribution
        :param sequence: are the previously generated token indices
        :return: sampled token indices
        """
        if sequence is not None and self.penalty > 0 and sequence.numel() > 0:
            logits = logits.clone()

            # Vectorized implementation for presence penalty
            unique_tokens = torch.unique(sequence)
            if unique_tokens.numel() > 0:
                # Filter valid tokens
                valid_tokens = unique_tokens[(unique_tokens >= 0) &
                                             (unique_tokens < logits.size(-1))]
                if valid_tokens.numel() > 0:
                    # Create penalty mask
                    penalty_mask = torch.zeros(logits.shape[-1],
                                               dtype=torch.bool,
                                               device=logits.device)
                    penalty_mask[valid_tokens] = True

                    # Apply penalty
                    logits[penalty_mask.expand_as(logits)] -= self.penalty

        # Sample using the provided sampler or default random sampling
        if self.sampler is not None:
            return self.sampler(logits)
        else:
            dist = Categorical(logits=logits)
            return dist.sample()


class TopTokenRestrictionSampler(Sampler):
    """
    ## Top Token Restriction Sampler

    Disallows sampling from the top-N most probable tokens. This is useful for
    avoiding overly common or stereotypical outputs.
    """

    def __init__(self,
                 avoid_top_k: int = 0,
                 sampler: Optional[Sampler] = None):
        """
        :param avoid_top_k: number of top tokens to avoid
        :param sampler: optional sampler to apply after filtering
        """
        if avoid_top_k < 0:
            raise ValueError(
                f'avoid_top_k must be non-negative, got {avoid_top_k}')
        self.avoid_top_k = avoid_top_k
        self.sampler = sampler

    def __call__(self, logits: Tensor) -> Tensor:
        """
        Restrict top tokens and sample

        :param logits: are the logits of the distribution
        :return: sampled token indices
        """
        if self.avoid_top_k > 0:
            logits = logits.clone()

            # Get the indices of top-k tokens to avoid
            _, top_k_indices = torch.topk(logits,
                                          min(self.avoid_top_k,
                                              logits.size(-1)),
                                          dim=-1)

            # Set their logits to -inf
            logits.scatter_(-1, top_k_indices, float('-inf'))

            # Safety check: if all tokens are filtered, restore the original
            if torch.all(torch.isinf(logits)):
                # Create a copy of the original logits
                original_logits = torch.empty_like(logits)
                original_logits.copy_(logits)
                logits = original_logits

        # Sample using the provided sampler or default random sampling
        if self.sampler is not None:
            return self.sampler(logits)
        else:
            dist = Categorical(logits=logits)
            return dist.sample()
