"""
Sampling module for language models.

This module provides a unified, configurable Sampler class and stateless functional implementations
of various sampling strategies.

Main Classes:
- Sampler: Unified sampler supporting temperature, top-k, top-p, min-p, typical, penalties, etc.
- SamplingConfig: Configuration dataclass for the Sampler.
- MirostatSampler: Mirostat sampling implementation.
- MirostatV2Sampler: Mirostat v2 sampling implementation.

Functional Interface:
- functional: Module containing stateless sampling functions.
"""

from minivllm.sampling import functional
from minivllm.sampling.config import SamplingConfig
from minivllm.sampling.mirostat import MirostatSampler, MirostatV2Sampler
from minivllm.sampling.sampler import Sampler

__all__ = [
    'Sampler',
    'SamplingConfig',
    'MirostatSampler',
    'MirostatV2Sampler',
    'functional',
]
