"""Configuration module for mini-vLLM engine.

This module provides the Config dataclass which defines all configuration
parameters for initializing and running the LLM engine.
"""

import os
from dataclasses import dataclass
from typing import ClassVar, Literal, Optional

from transformers import AutoConfig, PretrainedConfig

# Type aliases for better readability
TensorParallelSize = Literal[1, 2, 3, 4, 5, 6, 7, 8]


@dataclass
class Config:
    """Configuration for the mini-vLLM engine.

    This class encapsulates all configuration parameters needed to initialize
    and run the LLM engine, including model parameters, memory settings, and
    cache configuration.

    Attributes:
        model: Path to the model directory containing model weights
            and configs.
        max_num_batched_tokens: Maximum number of tokens to batch together.
            Range: [1, 1000000]. Default: 16384.
        max_num_seqs: Maximum number of sequences to handle simultaneously.
            Range: [1, 10000]. Default: 512.
        max_model_len: Maximum length of sequences the model can handle.
            Range: [1, 1000000]. Default: 4096.
        gpu_memory_utilization: Fraction of GPU memory to utilize.
            Range: [0.1, 1.0]. Default: 0.9.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
            Range: [1, 8]. Default: 1.
        enforce_eager: Force eager execution mode instead of graph
        mode. Default: False.
        hf_config: HuggingFace model configuration object,
            loaded automatically.
        eos: End-of-sequence token ID. Default: -1 (auto-detect).
        kvcache_block_size: Size of KV cache blocks in tokens.
            Must be divisible by 256. Default: 256.
        num_kvcache_blocks: Number of KV cache blocks. -1 means auto.
            Default: -1.
    """

    # Configuration constraints
    MIN_GPU_MEMORY_UTIL: ClassVar[float] = 0.1
    MAX_GPU_MEMORY_UTIL: ClassVar[float] = 1.0
    MIN_TENSOR_PARALLEL_SIZE: ClassVar[int] = 1
    MAX_TENSOR_PARALLEL_SIZE: ClassVar[int] = 8
    BLOCK_SIZE_DIVISOR: ClassVar[int] = 256

    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: Optional[PretrainedConfig] = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self) -> None:
        """Validate and initialize configuration after dataclass initialization.

        This method performs comprehensive validation of all configuration
        parameters and loads the HuggingFace model configuration. It ensures
        the engine is configured correctly before initialization.

        Validation performed:
        1. Model path validation (must exist and be a directory)
        2. KV cache block size validation (must be divisible by 256 for alignment)
        3. Tensor parallel size validation (1-8 range for efficiency and compatibility)
        4. Memory utilization validation (0.1-1.0 range for safety)
        5. Batch size validation (must accommodate model length)
        6. Loading HuggingFace model configuration
        7. Automatic adjustment of max_model_len based on model capabilities

        Raises:
            ValueError: If any configuration parameter is invalid or
                if the model cannot be loaded.
        """
        # Validate model path exists and is accessible
        self._validate_model_path()

        # Validate numeric ranges
        self._validate_gpu_memory_utilization()
        self._validate_kvcache_block_size()
        self._validate_tensor_parallel_size()
        self._validate_batch_sizes()

        # Load and validate HuggingFace config
        self._load_hf_config()

        # Adjust max_model_len based on model capabilities
        self._adjust_max_model_len()

        # Final validation
        self._validate_batch_token_constraints()

    def _validate_model_path(self) -> None:
        """Validate that model path exists and is accessible."""
        if not os.path.isdir(self.model):
            raise ValueError(
                f"Model path '{self.model}' is not a valid directory. "
                f'Please ensure the model is properly downloaded and accessible.'
            )

    def _validate_gpu_memory_utilization(self) -> None:
        """Validate GPU memory utilization is in reasonable range."""
        if not (self.MIN_GPU_MEMORY_UTIL <= self.gpu_memory_utilization <=
                self.MAX_GPU_MEMORY_UTIL):
            raise ValueError(
                f'gpu_memory_utilization must be between {self.MIN_GPU_MEMORY_UTIL} '
                f'and {self.MAX_GPU_MEMORY_UTIL}, got {self.gpu_memory_utilization}. '
                f'Values outside this range may cause OOM errors or underutilization.'
            )

    def _validate_kvcache_block_size(self) -> None:
        """Validate KV cache block size for optimal performance."""
        if self.kvcache_block_size % self.BLOCK_SIZE_DIVISOR != 0:
            raise ValueError(
                f'kvcache_block_size must be divisible by {self.BLOCK_SIZE_DIVISOR} '
                f'for optimal performance, got {self.kvcache_block_size}. '
                f'Common values: 256, 512, 1024.')

    def _validate_tensor_parallel_size(self) -> None:
        """Validate tensor parallel size for system compatibility."""
        if not (self.MIN_TENSOR_PARALLEL_SIZE <= self.tensor_parallel_size <=
                self.MAX_TENSOR_PARALLEL_SIZE):
            raise ValueError(
                f'tensor_parallel_size must be between {self.MIN_TENSOR_PARALLEL_SIZE} '
                f'and {self.MAX_TENSOR_PARALLEL_SIZE}, got {self.tensor_parallel_size}. '
                f'Higher values may not provide additional benefits and consume more memory.'
            )

    def _validate_batch_sizes(self) -> None:
        """Validate batch size parameters are positive."""
        if self.max_num_batched_tokens <= 0:
            raise ValueError(f'max_num_batched_tokens must be positive, '
                             f'got {self.max_num_batched_tokens}')
        if self.max_num_seqs <= 0:
            raise ValueError(
                f'max_num_seqs must be positive, got {self.max_num_seqs}')
        if self.max_model_len <= 0:
            raise ValueError(
                f'max_model_len must be positive, got {self.max_model_len}')

    def _load_hf_config(self) -> None:
        """Load HuggingFace model configuration."""
        try:
            self.hf_config = AutoConfig.from_pretrained(self.model)
        except Exception as e:
            raise ValueError(
                f'Failed to load HuggingFace model configuration from {self.model}: {e}. '
                f'Please ensure the model is a valid HuggingFace model or local directory.'
            )

    def _adjust_max_model_len(self) -> None:
        """Automatically adjust max_model_len based on model capabilities."""
        if self.hf_config is None:
            return

        model_max_len: int = getattr(self.hf_config, 'max_position_embeddings',
                                     4096)

        if self.max_model_len > model_max_len:
            import warnings
            warnings.warn(
                f'auto-adjusting max_model_len from {self.max_model_len} '
                f'to {model_max_len} (model\'s maximum context length).',
                UserWarning)
            self.max_model_len = model_max_len

    def _validate_batch_token_constraints(self) -> None:
        """Verify batch size is sufficient for model length requirements."""
        if self.max_num_batched_tokens < self.max_model_len:
            raise ValueError(
                f'max_num_batched_tokens ({self.max_num_batched_tokens}) '
                f'must be >= max_model_len ({self.max_model_len}) '
                f'to accommodate the full context length in a single batch.')
