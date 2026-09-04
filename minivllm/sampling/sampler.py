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
        repetition_penalties: Tensor | None = None,
        frequency_penalties: Tensor | None = None,
        presence_penalties: Tensor | None = None,
        generators: list[torch.Generator | None] | None = None,
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
            prev_tokens: Optional tensor [batch_size, seq_len] (padded with -1)
                for penalties
            generator: Optional random generator for the no-seed batch path
            repetition_penalties/frequency_penalties/presence_penalties: Optional
                per-row tensor [batch_size] overrides; fall back to the config
                scalar when a row/whole tensor is absent.
            generators: Optional per-row generators (for reproducible sampling);
                ``None`` entries fall back to ``generator``.

        Returns:
            sampled_tokens: [batch_size]
        """
        cfg = config or self.config

        # 1. Apply penalties (if any). The functional penalty ops take a scalar,
        #    so per-row values are applied row-by-row (only over active rows).
        if prev_tokens is not None:
            batch = logits.size(0)
            rep = self._row_list(repetition_penalties, cfg.repetition_penalty, batch)
            freq = self._row_list(frequency_penalties, cfg.frequency_penalty, batch)
            pres = self._row_list(presence_penalties, cfg.presence_penalty, batch)
            for b in range(batch):
                row_logits = logits[b : b + 1]
                row_prev = prev_tokens[b : b + 1]
                if rep[b] != 1.0:
                    row_logits = F.apply_repetition_penalty(
                        row_logits, row_prev, rep[b]
                    )
                if freq[b] != 0.0:
                    row_logits = F.apply_frequency_penalty(row_logits, row_prev, freq[b])
                if pres[b] != 0.0:
                    row_logits = F.apply_presence_penalty(row_logits, row_prev, pres[b])
                logits[b] = row_logits.squeeze(0)

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
            return self._sample(logits, generators, generator)

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
        sub_gens = [generators[i] for i in non_greedy.nonzero().squeeze(-1)] if generators else None
        out[non_greedy] = self._sample(sub, sub_gens, generator)
        return out

    @staticmethod
    def _row_list(values: Tensor | None, default: float, batch: int) -> list[float]:
        """Per-row override list, falling back to the config scalar per row."""
        if values is None:
            return [default] * batch
        return values.tolist()

    def _sample(
        self,
        logits: Tensor,
        generators: list[torch.Generator | None] | None,
        generator: torch.Generator | None,
    ) -> Tensor:
        """Sample a batch, honoring per-row generators when provided."""
        if generators is None:
            return F.sample_from_logits(logits, generator=generator)
        out = logits.new_empty(logits.size(0), dtype=torch.long)
        for b in range(logits.size(0)):
            gen = generators[b] if generators[b] is not None else generator
            out[b] = F.sample_from_logits(logits[b : b + 1], generator=gen).squeeze(0)
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
