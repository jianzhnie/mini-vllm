"""Main LLM interface module.

This module provides the LLM class, which is the primary user-facing
interface for interacting with the mini-vLLM engine.
"""

from minivllm.engine.llm_engine import LLMEngine


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
        >>>
        >>> # Initialize the LLM
        >>> llm = LLM(
        ...     model="meta-llama/Llama-2-7b",
        ...     max_num_seqs=256,
        ...     max_num_batched_tokens=8192
        ... )
        >>>
        >>> # Generate text
        >>> prompts = [
        ...     "Once upon a time",
        ...     "The future of AI is"
        ... ]
        >>> sampling_params = SamplingParams(
        ...     temperature=0.7,
        ...     max_tokens=128
        ... )
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

    def __init__(self, model: str, **kwargs) -> None:
        """Initialize the LLM.

        Creates an LLMEngine instance with the given configuration.
        All arguments are passed through to LLMEngine.__init__.

        Args:
            model: Path to the model directory (HuggingFace format).
            **kwargs: Additional configuration parameters. Common ones:
                - max_num_seqs: Maximum sequences per batch (default: 512)
                - max_num_batched_tokens: Max tokens per batch (default: 16384)
                - gpu_memory_utilization: GPU memory fraction (default: 0.9)
                - tensor_parallel_size: Number of GPUs (default: 1)
                - enforce_eager: Skip CUDA graph optimization (default: False)
                - max_model_len: Maximum sequence length (default: 4096)

        Raises:
            ValueError: If configuration is invalid.
            RuntimeError: If model loading fails.

        Example:
            >>> llm = LLM(
            ...     "meta-llama/Llama-2-7b-hf",
            ...     gpu_memory_utilization=0.8,
            ...     tensor_parallel_size=2
            ... )
        """
        super().__init__(model, **kwargs)
