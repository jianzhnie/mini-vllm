"""
---
title: Mirostat Sampling for Language Models
summary: A PyTorch implementation of Mirostat sampling for maintaining stable perplexity.
---

# Mirostat Sampling for Language Models

Mirostat is a sampling algorithm that dynamically adjusts the temperature to maintain a target perplexity.
It helps avoid both repetitive text (too low perplexity) and incoherent text (too high perplexity).

The algorithm works by:
1. Calculating the surprise (negative log probability) of the selected token
2. Adjusting the temperature based on the difference between actual and target surprise
3. Using top-k truncation with the adjusted temperature

"""

from typing import Optional

import torch
from torch import Tensor, nn
from torch.distributions import Categorical


class MirostatSampler(nn.Module):
    """
    ## Mirostat Sampler

    Maintains a target perplexity by dynamically adjusting temperature.
    """

    def __init__(self,
                 target_perplexity: float = 3.0,
                 learning_rate: float = 1.0,
                 max_temperature: float = 2.0):
        """
        :param target_perplexity: target perplexity (typically 2-7)
        :param learning_rate: learning rate for temperature adjustment
        :param max_temperature: maximum allowed temperature
        """
        super().__init__()
        self.target_perplexity = target_perplexity
        self.target_surprise = torch.log(torch.tensor(target_perplexity))
        self.learning_rate = learning_rate
        self.max_temperature = max_temperature
        self.temperature = 1.0
        self.previous_surprise: Optional[Tensor] = None

    def __call__(self, logits: Tensor) -> Tensor:
        """
        Sample from logits using Mirostat algorithm

        :param logits: are the logits of the distribution of shape `[..., n_tokens]`
        :return: sampled token indices of shape `[...]`
        """
        # Apply temperature to logits
        tempered_logits = logits / self.temperature

        # Convert to probabilities
        probs = torch.softmax(tempered_logits, dim=-1)

        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs,
                                                  descending=True,
                                                  dim=-1)

        # Calculate cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Find the cutoff point where cumulative probability exceeds threshold
        # This creates a dynamic top-k based on the temperature
        cutoff_mask = cumulative_probs > (1.0 - 1.0 /
                                          (self.temperature * 10.0))

        # Get the first token that exceeds the threshold
        cutoff_index = torch.argmax(cutoff_mask.float(), dim=-1)

        # Create truncated distribution
        truncated_probs = sorted_probs.clone()
        if cutoff_index.numel() > 0:
            truncated_probs[cutoff_mask] = 0.0
            # Renormalize
            truncated_probs = truncated_probs / truncated_probs.sum(
                dim=-1, keepdim=True)

        # Create categorical distribution with truncated probabilities
        dist = Categorical(probs=truncated_probs)

        # Sample from the truncated distribution
        sample_idx = dist.sample()

        # Get the actual token index
        token_idx = sorted_indices.gather(-1,
                                          sample_idx.unsqueeze(-1)).squeeze(-1)

        # Calculate the surprise (negative log probability) of the selected token
        selected_prob = truncated_probs.gather(
            -1, sample_idx.unsqueeze(-1)).squeeze(-1)
        surprise = -torch.log(
            selected_prob + 1e-10)  # Add small epsilon to avoid log(0)

        # Update temperature based on the difference between actual and target surprise
        error = surprise - self.target_surprise
        self.temperature = torch.clamp(self.temperature +
                                       self.learning_rate * error,
                                       min=0.1,
                                       max=self.max_temperature)

        return token_idx

    def reset(self):
        """Reset the sampler state"""
        self.temperature = 1.0
        self.previous_surprise = None


class MirostatV2Sampler(nn.Module):
    """
    ## Mirostat v2 Sampler

    An improved version of Mirostat with better stability.
    """

    def __init__(self,
                 target_perplexity: float = 3.0,
                 learning_rate: float = 0.1,
                 tau: float = 5.0):
        """
        :param target_perplexity: target perplexity (typically 2-7)
        :param learning_rate: learning rate for temperature adjustment
        :param tau: smoothing parameter for temperature updates
        """
        super().__init__()
        self.target_perplexity = target_perplexity
        self.target_surprise = torch.log(torch.tensor(target_perplexity))
        self.learning_rate = learning_rate
        self.tau = tau
        self.temperature = 1.0
        self.mu = target_perplexity  # Initialize mu to target

    def __call__(self, logits: Tensor) -> Tensor:
        """
        Sample from logits using Mirostat v2 algorithm

        :param logits: are the logits of the distribution of shape `[..., n_tokens]`
        :return: sampled token indices of shape `[...]`
        """
        # Apply temperature to logits
        tempered_logits = logits / self.temperature

        # Convert to probabilities
        probs = torch.softmax(tempered_logits, dim=-1)

        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs,
                                                  descending=True,
                                                  dim=-1)

        # Calculate cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Find cutoff using the current mu value
        cutoff_mask = cumulative_probs > (1.0 - 1.0 / self.mu)

        # Get the first token that exceeds the threshold
        cutoff_index = torch.argmax(cutoff_mask.float(), dim=-1)

        # Create truncated distribution
        truncated_probs = sorted_probs.clone()
        if cutoff_index.numel() > 0:
            truncated_probs[cutoff_mask] = 0.0
            # Renormalize
            truncated_probs = truncated_probs / truncated_probs.sum(
                dim=-1, keepdim=True)

        # Create categorical distribution with truncated probabilities
        dist = Categorical(probs=truncated_probs)

        # Sample from the truncated distribution
        sample_idx = dist.sample()

        # Get the actual token index
        token_idx = sorted_indices.gather(-1,
                                          sample_idx.unsqueeze(-1)).squeeze(-1)

        # Calculate the surprise of the selected token
        selected_prob = truncated_probs.gather(
            -1, sample_idx.unsqueeze(-1)).squeeze(-1)
        surprise = -torch.log(selected_prob + 1e-10)

        # Update mu using the error between actual and target surprise
        error = surprise - self.target_surprise
        self.mu = torch.clamp(self.mu - self.learning_rate * error,
                              min=1.0,
                              max=100.0)

        # Update temperature based on mu
        self.temperature = self.tau / self.mu

        return token_idx

    def reset(self):
        """Reset the sampler state"""
        self.temperature = 1.0
        self.mu = self.target_perplexity
