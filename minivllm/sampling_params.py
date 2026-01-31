"""Sampling parameters module for text generation.

This module defines the SamplingParams dataclass which controls
the behavior of the text generation process.
"""

from dataclasses import dataclass

__all__ = ['SamplingParams']


@dataclass
class SamplingParams:
    """Parameters that control text generation sampling behavior.

    This class encapsulates all sampling-related parameters used during
    the text generation process, including temperature and maximum
    token limits.

    Attributes:
        temperature: Controls randomness in sampling. Higher
            values make output more random, lower values more
            deterministic.
            Must be > 1e-10. Default: 1.0.
        top_p: Float that controls the cumulative probability of the top tokens to consider.
            Must be in (0, 1]. Set to 1 to consider all tokens. Default: 1.0.
        top_k: Integer that controls the number of top tokens to consider.
            Must be -1 or > 0. Set to -1 to consider all tokens. Default: -1.
        min_p: Float that represents the minimum probability for a token to be considered,
            relative to the probability of the most likely token.
            Must be in [0, 1]. Default: 0.0.
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

    def __post_init__(self) -> None:
        """Validate sampling parameters after dataclass initialization.

        This method ensures that temperature is set to a valid value
        (greedy sampling with temperature=0 is not permitted).

        Raises:
            ValueError: If temperature is <= 1e-10 (would enable greedy
                sampling).
        """
        if self.temperature <= 1e-10:
            raise ValueError(
                f'temperature must be > 1e-10, got {self.temperature}. '
                f'Greedy sampling (temperature=0) is not permitted.')

        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f'top_p must be in (0, 1], got {self.top_p}')

        if self.top_k < -1 or self.top_k == 0:
            raise ValueError(
                f'top_k must be -1 (disable) or > 0, got {self.top_k}.')

        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError(f'min_p must be in [0, 1], got {self.min_p}')

        if self.max_tokens <= 0:
            raise ValueError(f'max_tokens must be > 0, got {self.max_tokens}')
