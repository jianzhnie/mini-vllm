"""Main LLM interface module.

This module provides the LLM class, which is the primary user-facing
interface for interacting with the mini-vLLM engine.

Quick Start:
    >>> from minivllm import LLM, SamplingParams
    >>> from minivllm.config import Config
    >>>
    >>> config = Config(model="path/to/model")
    >>> llm = LLM(config)
    >>>
    >>> # Generate text with custom parameters
    >>> prompts = ["Hello, world!", "Once upon a time"]
    >>> params = SamplingParams(temperature=0.8, max_tokens=100)
    >>> outputs = llm.generate(prompts, params)
    >>>
    >>> # Access generated text
    >>> for output in outputs:
    ...     print(output['text'])

Advanced Usage:
    >>> config = Config(
    ...     model="path/to/large/model",
    ...     tensor_parallel_size=4,
    ...     max_num_seqs=256,
    ...     device_memory_utilization=0.9,
    ... )
    >>> llm = LLM(config)
    >>>
    >>> # Batch generation with different parameters
    >>> params_list = [
    ...     SamplingParams(temperature=0.7, max_tokens=50),
    ...     SamplingParams(temperature=0.9, max_tokens=100),
    ... ]
    >>> outputs = llm.generate(prompts, params_list)

Performance Tips:
    - Use larger batch sizes for better throughput
    - Increase device_memory_utilization if OOM doesn't occur
    - Enable tensor parallelism for models > 13B parameters
    - Adjust max_num_batched_tokens based on sequence lengths
"""

from minivllm.config import Config
from minivllm.engine.llm_engine import LLMEngine

__all__ = ['LLM']


class LLM(LLMEngine):
    """Main interface for the mini-vLLM language model engine.

    This class serves as a wrapper around the LLMEngine, providing
    a clean and intuitive interface for users to interact with the
    language model inference engine.

    All functionality is inherited from LLMEngine, which handles:
    - Model loading and initialization
    - Sequence scheduling and KV cache management
    - Batch processing and inference
    - Token generation with configurable sampling

    Example Usage:
        >>> from minivllm import LLM, SamplingParams
        >>> from minivllm.config import Config
        >>>
        >>> config = Config(model="meta-llama/Llama-2-7b",
        ...                 max_num_seqs=256,
        ...                 max_num_batched_tokens=8192)
        >>> llm = LLM(config)
        >>>
        >>> # Generate text
        >>> prompts = [
        ...     "Once upon a time",
        ...     "The future of AI is"
        ... ]
        >>> sampling_params = SamplingParams(temperature=0.7, max_tokens=128)
        >>> outputs = llm.generate(prompts, sampling_params)
        >>>
        >>> for output in outputs:
        ...     print(output['text'])

    Attributes:
        All attributes inherited from LLMEngine:
        - model_runner: Handles model inference
        - scheduler: Manages sequence scheduling
        - tokenizer: HuggingFace tokenizer
        - config: Engine configuration

    Note:
        The LLM class is intentionally kept thin, delegating all
        functionality to LLMEngine for maintainability. Subclasses
        can extend this class to add custom behavior if needed.
    """

    def __init__(self, config: Config) -> None:
        """Initialize the LLM.

        Args:
            config: Engine configuration (see Config class).

        Raises:
            TypeError: If config is not a Config instance.
        """
        super().__init__(config)
