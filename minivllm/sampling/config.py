"""Internal sampling configuration used by the Sampler."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SamplingConfig:
    """Configuration for text generation sampling.

    This is the internal representation used by the Sampler module.
    For the user-facing API, see ``minivllm.sampling_params.SamplingParams``.
    """

    temperature: float = 1.0
    top_k: int = -1  # -1 = disabled
    top_p: float = 1.0
    min_p: float = 0.0
    typical_p: float = 1.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    avoid_top_k: int = 0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError(
                f'temperature must be non-negative, got {self.temperature}')
        if self.top_k < -1:
            raise ValueError(
                f'top_k must be -1 (disabled) or >= 0, got {self.top_k}')
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError(
                f'top_p must be between 0.0 and 1.0, got {self.top_p}')
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError(
                f'min_p must be between 0.0 and 1.0, got {self.min_p}')
        if self.typical_p <= 0:
            raise ValueError(
                f'typical_p must be positive, got {self.typical_p}')
        if self.repetition_penalty < 1.0:
            raise ValueError(
                f'repetition_penalty must be >= 1.0, '
                f'got {self.repetition_penalty}')
        if self.frequency_penalty < 0.0:
            raise ValueError(
                f'frequency_penalty must be >= 0.0, '
                f'got {self.frequency_penalty}')
        if self.presence_penalty < 0.0:
            raise ValueError(
                f'presence_penalty must be >= 0.0, '
                f'got {self.presence_penalty}')
        if self.avoid_top_k < 0:
            raise ValueError(
                f'avoid_top_k must be non-negative, got {self.avoid_top_k}')
