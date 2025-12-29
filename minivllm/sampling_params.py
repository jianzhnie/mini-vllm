"""Sampling parameters module for text generation.

This module defines the SamplingParams dataclass which controls
the behavior of the text generation process.
"""

from dataclasses import dataclass


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
        max_tokens: Maximum number of tokens to generate in completion.
            Default: 64.
        ignore_eos: Whether to ignore the end-of-sequence token and
            continue generating until max_tokens is reached.
            Default: False.
    """

    temperature: float = 1.0
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

        if self.max_tokens <= 0:
            raise ValueError(f'max_tokens must be > 0, got {self.max_tokens}')
