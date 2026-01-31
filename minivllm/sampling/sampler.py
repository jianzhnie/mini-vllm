"""
Unified Sampler class replacing the previous fragmented implementations.
"""

from typing import Optional

import torch
from torch import Tensor, nn

from minivllm.sampling import functional as F
from minivllm.sampling.config import SamplingConfig


class Sampler(nn.Module):
    """
    A unified sampler that supports multiple sampling strategies.

    This replaces the previous hierarchy of individual sampler classes
    (GreedySampler, TopKSampler, etc.) with a single configurable class.

    It supports both stateful usage (initialized with a config) and
    functional usage (passing parameters at call time).
    """

    def __init__(self, config: Optional[SamplingConfig] = None):
        super().__init__()
        self.config = config or SamplingConfig()

    def forward(
        self,
        logits: Tensor,
        config: Optional[SamplingConfig] = None,
        # Optional overrides for batch processing
        temperatures: Optional[Tensor] = None,
        top_ks: Optional[Tensor] = None,
        top_ps: Optional[Tensor] = None,
        min_ps: Optional[Tensor] = None,
        typical_ps: Optional[Tensor] = None,
        avoid_top_ks: Optional[Tensor] = None,
        prev_tokens: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """
        Sample tokens from logits.

        Args:
            logits: [batch_size, vocab_size]
            config: Optional config to override self.config
            temperatures: Optional tensor [batch_size] to override config.temperature
            top_ks: Optional tensor [batch_size] to override config.top_k
            top_ps: Optional tensor [batch_size] to override config.top_p
            min_ps: Optional tensor [batch_size] to override config.min_p
            typical_ps: Optional tensor [batch_size] to override config.typical_p
            avoid_top_ks: Optional tensor [batch_size] to override config.avoid_top_k
            prev_tokens: Optional tensor [batch_size, seq_len] for penalties
            generator: Optional random generator

        Returns:
            sampled_tokens: [batch_size]
        """
        cfg = config or self.config

        # 1. Apply penalties (if any)
        # Note: Penalties are typically applied before other transformations
        if prev_tokens is not None:
            if cfg.repetition_penalty != 1.0:
                logits = F.apply_repetition_penalty(logits, prev_tokens,
                                                    cfg.repetition_penalty)
            if cfg.frequency_penalty != 0.0:
                logits = F.apply_frequency_penalty(logits, prev_tokens,
                                                   cfg.frequency_penalty)
            if cfg.presence_penalty != 0.0:
                logits = F.apply_presence_penalty(logits, prev_tokens,
                                                  cfg.presence_penalty)

        # 2. Apply Top Token Restriction (Avoid Top-K)
        # Useful for watermarking or specific constraints
        avoid_k = avoid_top_ks if avoid_top_ks is not None else cfg.avoid_top_k
        logits = F.apply_top_token_restriction(logits, avoid_k)

        # 3. Apply Temperature
        # Use provided tensor or config value
        temp = temperatures if temperatures is not None else cfg.temperature
        logits = F.apply_temperature(logits, temp)

        # 4. Apply Typical Sampling
        typ_p = typical_ps if typical_ps is not None else cfg.typical_p
        logits = F.apply_typical_filtering(logits, typ_p)

        # 5. Apply Top-K
        k = top_ks if top_ks is not None else cfg.top_k
        logits = F.apply_top_k(logits, k)

        # 6. Apply Top-P
        p = top_ps if top_ps is not None else cfg.top_p
        logits = F.apply_top_p(logits, p)

        # 7. Apply Min-P
        mp = min_ps if min_ps is not None else cfg.min_p
        logits = F.apply_min_p(logits, mp)

        # 8. Sample
        # If temperature is effectively 0, use greedy sampling (argmax)
        # But we handle low temp in apply_temperature by clamping.
        # However, many implementations treat temp=0 as strict greedy.
        # Let's check if we should do strict greedy.

        # If temperatures was a tensor, we can't easily check for 0 globally.
        # But F.apply_temperature clamps min to 1e-8, so it effectively becomes very sharp.
        # So multinomial sampling will act like greedy.

        return F.sample_from_logits(logits, generator=generator)


# Legacy aliases for backward compatibility if needed,
# or specific factory methods can be added here.


class GreedySampler(Sampler):

    def __init__(self):
        super().__init__(
            SamplingConfig(temperature=1e-8))  # Effectively greedy


class RandomSampler(Sampler):

    def __init__(self):
        super().__init__(SamplingConfig(temperature=1.0))
