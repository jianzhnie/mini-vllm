from dataclasses import dataclass
from typing import Optional


@dataclass
class SamplingConfig:
    """Configuration for text generation sampling."""

    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    typical_p: float = 1.0  # tau for typical sampling
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    avoid_top_k: int = 0  # for TopTokenRestriction
    seed: Optional[int] = None

    def __post_init__(self):
        if self.temperature < 0:
            raise ValueError(
                f'Temperature must be non-negative, got {self.temperature}')
        if self.top_k < 0:
            raise ValueError(f'Top-k must be non-negative, got {self.top_k}')
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError(
                f'Top-p must be between 0.0 and 1.0, got {self.top_p}')
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError(
                f'Min-p must be between 0.0 and 1.0, got {self.min_p}')
        if self.typical_p <= 0:
            raise ValueError(
                f'Typical p (tau) must be positive, got {self.typical_p}')
        if self.avoid_top_k < 0:
            raise ValueError(
                f'Avoid top-k must be non-negative, got {self.avoid_top_k}')
