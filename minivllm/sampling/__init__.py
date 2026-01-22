"""
Sampling module for language models.

This module provides various sampling strategies for text generation, including:

**Basic Samplers:**
- GreedySampler: Always select the most likely token
- RandomSampler: Uniformly random sampling
- TemperatureSampler: Temperature-scaled random sampling

**Nucleus & Filtering Samplers:**
- NucleusSampler: Composite nucleus sampling (requires inner sampler)
- TopPSampler: Standalone nucleus (top-p) sampling
- TopKSampler: Composite top-k sampling (requires inner sampler)
- MinPSampler: Min-p sampling (dynamic threshold based on max probability)
- TypicalSampler: Typical set sampling (entropy-based filtering)

**Combined Samplers:**
- TopKTopPSampler: Combined top-k and top-p filtering
- TemperatureMinPTopKSampler: Combined temperature, min-p, and top-k
- ExhaustiveTopKSampler: Conservative top-k that ensures selection

**Penalty/Warper Samplers:**
- RepetitionPenaltySampler: Penalize previously generated tokens
- FrequencyPenaltySampler: Penalize tokens by frequency
- PresencePenaltySampler: Penalize any previously seen token
- TopTokenRestrictionSampler: Avoid top-N most probable tokens

**Utilities:**
- TopKTopPFilter: Legacy filter for top-k and top-p
- generate: Full generation function with various parameters
- generate_simple: Simplified generation function
- generate_with_filter: Generation with top-k/top-p filter
"""

from minivllm.sampling.base import Sampler
from minivllm.sampling.combined import (ExhaustiveTopKSampler,
                                        TemperatureMinPTopKSampler,
                                        TopKTopPSampler)
from minivllm.sampling.greedy import GreedySampler
from minivllm.sampling.min_p import MinPSampler
from minivllm.sampling.random import RandomSampler
from minivllm.sampling.temperature import TemperatureSampler
from minivllm.sampling.top_k import TopKSampler
from minivllm.sampling.top_p import TopPSampler
from minivllm.sampling.typical import TypicalSampler
from minivllm.sampling.warpers import (FrequencyPenaltySampler,
                                       PresencePenaltySampler,
                                       RepetitionPenaltySampler,
                                       TopTokenRestrictionSampler)

__all__ = [
    # Base Protocol
    'Sampler',
    # Basic Samplers
    'GreedySampler',
    'RandomSampler',
    'TemperatureSampler',
    # Nucleus & Filtering Samplers
    'TopKSampler',
    'TopPSampler',
    'MinPSampler',
    'TypicalSampler',
    # Combined Samplers
    'TopKTopPSampler',
    'TemperatureMinPTopKSampler',
    'ExhaustiveTopKSampler',
    # Penalty/Warper Samplers
    'RepetitionPenaltySampler',
    'FrequencyPenaltySampler',
    'PresencePenaltySampler',
    'TopTokenRestrictionSampler',
]
