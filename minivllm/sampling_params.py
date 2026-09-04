"""Sampling parameters module for text generation.

This module defines the SamplingParams dataclass which controls
the behavior of the text generation process.
"""

from dataclasses import dataclass

__all__ = ["SamplingParams"]


@dataclass
class SamplingParams:
    """Parameters that control text generation sampling behavior.

    This class encapsulates all sampling-related parameters used during
    the text generation process, including temperature and maximum
    token limits.

    Attributes:
        temperature: Controls randomness in sampling. Higher
            values make output more random, lower values more
            deterministic. Set to 0 for greedy (deterministic)
            sampling.
            Must be >= 0. Default: 1.0.
        top_p: Float that controls the cumulative probability of the top tokens to
            consider. Must be in (0, 1]. Set to 1 to consider all tokens.
            Default: 1.0.
        top_k: Integer that controls the number of top tokens to consider.
            Must be -1 or > 0. Set to -1 to consider all tokens. Default: -1.
        min_p: Float that represents the minimum probability for a token to be
            considered, relative to the probability of the most likely token.
            Must be in [0, 1]. Default: 0.0.
        typical_p: Typical-sampling threshold in (0, 1]. Tokens are kept while
            their information content stays within typical_p * entropy. Set to
            1.0 to disable. Default: 1.0.
        repetition_penalty: Multiplies logits of already-seen tokens by 1/p
            (p > 1 discourages repetition). Must be >= 1.0. Default: 1.0 (off).
        frequency_penalty: Subtracts penalty * count(token) from a token's logit.
            Must be >= 0.0. Default: 0.0 (off).
        presence_penalty: Subtracts penalty from a token's logit if it has
            appeared at least once. Must be >= 0.0. Default: 0.0 (off).
        seed: Optional RNG seed for reproducible sampling of this request.
            Default: None (non-deterministic).
        max_tokens: Maximum number of tokens to generate in completion.
            Default: 64.
        ignore_eos: Whether to ignore the end-of-sequence token and
            continue generating until max_tokens is reached.
            Default: False.
    """

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    max_tokens: int = 64
    ignore_eos: bool = False
    typical_p: float = 1.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate sampling parameters after dataclass initialization.

        Raises:
            ValueError: If temperature is negative.
        """
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")

        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")

        if self.top_k < -1 or self.top_k == 0:
            raise ValueError(f"top_k must be -1 (disable) or > 0, got {self.top_k}.")

        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError(f"min_p must be in [0, 1], got {self.min_p}")

        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {self.max_tokens}")

        if self.typical_p <= 0:
            raise ValueError(f"typical_p must be > 0, got {self.typical_p}")

        if self.repetition_penalty < 1.0:
            raise ValueError(
                f"repetition_penalty must be >= 1.0, got {self.repetition_penalty}"
            )

        if self.frequency_penalty < 0.0:
            raise ValueError(
                f"frequency_penalty must be >= 0.0, got {self.frequency_penalty}"
            )

        if self.presence_penalty < 0.0:
            raise ValueError(
                f"presence_penalty must be >= 0.0, got {self.presence_penalty}"
            )
