"""
Unified Sampler class replacing the previous fragmented implementations.
"""

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

    def __init__(self, config: SamplingConfig | None = None):
        super().__init__()
        self.config = config or SamplingConfig()

    def forward(
        self,
        logits: Tensor,
        config: SamplingConfig | None = None,
        # Optional overrides for batch processing
        temperatures: Tensor | None = None,
        top_ks: Tensor | None = None,
        top_ps: Tensor | None = None,
        min_ps: Tensor | None = None,
        typical_ps: Tensor | None = None,
        avoid_top_ks: Tensor | None = None,
        prev_tokens: Tensor | None = None,
        generator: torch.Generator | None = None,
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
                logits = F.apply_repetition_penalty(
                    logits, prev_tokens, cfg.repetition_penalty
                )
            if cfg.frequency_penalty != 0.0:
                logits = F.apply_frequency_penalty(
                    logits, prev_tokens, cfg.frequency_penalty
                )
            if cfg.presence_penalty != 0.0:
                logits = F.apply_presence_penalty(
                    logits, prev_tokens, cfg.presence_penalty
                )

        # 2. Apply Top Token Restriction (Avoid Top-K)
        # Useful for watermarking or specific constraints
        avoid_k = avoid_top_ks if avoid_top_ks is not None else cfg.avoid_top_k
        if not isinstance(avoid_k, Tensor):
            # Per-batch tensor avoid_k not yet supported — silently skip
            logits = F.apply_top_token_restriction(logits, avoid_k)

        # 3. Resolve temperature and identify greedy (T==0) rows. This must happen
        #    before apply_temperature: a per-batch Tensor of zeros would otherwise
        #    hit the MIN_TEMPERATURE clamp and overflow to NaN on fp16/bf16, and
        #    the engine always passes a Tensor (so the old scalar-only check never
        #    fired). Greedy rows bypass the stochastic pipeline and take argmax.
        temp = temperatures if temperatures is not None else cfg.temperature
        if isinstance(temp, (int, float)):
            if temp == 0:
                return torch.argmax(logits, dim=-1)
            greedy = None
        else:
            greedy = temp.to(torch.float32) == 0
            if not greedy.any():
                greedy = None
            elif greedy.all():
                return torch.argmax(logits, dim=-1)

        if greedy is None:
            # No greedy rows: run the full pipeline over the whole batch.
            logits = F.apply_temperature(logits, temp)
            logits = F.apply_typical_filtering(
                logits, typical_ps if typical_ps is not None else cfg.typical_p
            )
            logits = F.apply_top_k(logits, top_ks if top_ks is not None else cfg.top_k)
            logits = F.apply_top_p(logits, top_ps if top_ps is not None else cfg.top_p)
            logits = F.apply_min_p(logits, min_ps if min_ps is not None else cfg.min_p)
            return F.sample_from_logits(logits, generator=generator)

        # Mixed batch: greedy rows argmax, remaining rows run the pipeline.
        non_greedy = ~greedy
        out = logits.new_empty(logits.size(0), dtype=torch.long)
        out[greedy] = torch.argmax(logits[greedy], dim=-1)
        sub = logits[non_greedy]
        sub = F.apply_temperature(sub, temp[non_greedy])
        sub = F.apply_typical_filtering(
            sub, typical_ps[non_greedy] if typical_ps is not None else cfg.typical_p
        )
        sub = F.apply_top_k(sub, top_ks[non_greedy] if top_ks is not None else cfg.top_k)
        sub = F.apply_top_p(sub, top_ps[non_greedy] if top_ps is not None else cfg.top_p)
        sub = F.apply_min_p(sub, min_ps[non_greedy] if min_ps is not None else cfg.min_p)
        out[non_greedy] = F.sample_from_logits(sub, generator=generator)
        return out


# Legacy aliases for backward compatibility if needed,
# or specific factory methods can be added here.


class GreedySampler(Sampler):
    """Greedy sampler that always selects the token with highest logit.

    This is faster than using temperature=0 since it bypasses all sampling logic
    and directly uses argmax.
    """

    def forward(self, logits: Tensor, **kwargs) -> Tensor:
        """Select token with highest logit (argmax).

        Args:
            logits: [batch_size, vocab_size]
            **kwargs: Ignored (for API compatibility)

        Returns:
            Long tensor of shape [batch_size] with selected token IDs
        """
        return torch.argmax(logits, dim=-1)
