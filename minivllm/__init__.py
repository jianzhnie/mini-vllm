"""Mini-vLLM: A lightweight language model inference engine.

Mini-vLLM is a streamlined implementation of a language model inference
engine with support for:
- Token-level sequence management
- KV cache with block-level management and prefix caching
- Configurable sampling parameters
- Efficient batching and scheduling
- Multi-GPU tensor parallelism
- CUDA graph optimization for decode phase

The main entry point is the LLM class, which provides a simple interface
for text generation on pre-trained language models.

Quick Start:
    >>> from minivllm import LLM, SamplingParams
    >>>
    >>> llm = LLM("meta-llama/Llama-2-7b")
    >>> outputs = llm.generate(
    ...     ["Once upon a time"],
    ...     SamplingParams(max_tokens=50)
    ... )
    >>> print(outputs[0]['text'])

Core Modules:
    config: Configuration management for the engine
    sampling_params: Text generation sampling parameters
    llm: Main user-facing LLM interface
    engine.sequence: Sequence management and state handling
    engine.scheduler: Sequence scheduling and KV cache management
    engine.block_manager: Block-based KV cache allocation
    engine.llm_engine: Main inference engine orchestration
    engine.model_runner: Model execution and distributed inference
"""

__version__ = '0.1.0'
__author__ = 'Mini-vLLM Contributors'

# Public API - import main classes for convenience
try:
    from minivllm.config import Config  # noqa: F401
    from minivllm.engine.sequence import Sequence, SequenceStatus  # noqa: F401
    from minivllm.llm import LLM  # noqa: F401
    from minivllm.sampling_params import SamplingParams  # noqa: F401

    __all__ = [
        'Config',
        'SamplingParams',
        'LLM',
        'Sequence',
        'SequenceStatus',
    ]
except ImportError as e:
    import warnings
    warnings.warn(f'Failed to import some modules from minivllm: {e}. '
                  f'Make sure all required dependencies are installed: '
                  f'torch, transformers, flash-attn, triton, xxhash. '
                  f'Some functionality may be unavailable.')
    __all__ = []
