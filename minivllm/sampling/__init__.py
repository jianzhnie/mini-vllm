"""
Sampling module for language models.

This module provides various sampling strategies for text generation, including:
- Greedy sampling
- Random sampling
- Temperature sampling
- Top-k sampling
- Nucleus (top-p) sampling
- Combined top-p and top-k sampling

It also includes text generation utilities that use these sampling methods.
"""

from minivllm.sampling.base import Sampler
from minivllm.sampling.generation import generate as generate
from minivllm.sampling.generation2 import generate as generate_simple
from minivllm.sampling.greedy import GreedySampler
from minivllm.sampling.nucleus import NucleusSampler
from minivllm.sampling.random import RandomSampler
from minivllm.sampling.temperature import TemperatureSampler
from minivllm.sampling.top_k import TopKSampler
from minivllm.sampling.top_p_top_k import TopKTopPFilter
from minivllm.sampling.top_p_top_k import generate as generate_with_filter

__all__ = [
    'Sampler',
    'GreedySampler',
    'RandomSampler',
    'TemperatureSampler',
    'TopKSampler',
    'NucleusSampler',
    'TopKTopPFilter',
    'generate',
    'generate_simple',
    'generate_with_filter',
]
