"""Configuration module for mini-vLLM engine.

This module provides the Config dataclass which defines all configuration
parameters for initializing and running the LLM engine.
"""

import os
from dataclasses import dataclass
from typing import Optional

from transformers import AutoConfig, PretrainedConfig


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
            Default: 16384.
        max_num_seqs: Maximum number of sequences to handle simultaneously.
            Default: 512.
        max_model_len: Maximum length of sequences the model can handle.
            Default: 4096.
        gpu_memory_utilization: Fraction of GPU memory to utilize (0.0-1.0).
            Default: 0.9.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
            Must be between 1 and 8. Default: 1.
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
        if not os.path.isdir(self.model):
            raise ValueError(
                f"Model path '{self.model}' is not a valid directory. "
                f'Please ensure the model is properly downloaded and accessible.'
            )

        # Validate GPU memory utilization is in reasonable range
        if not (0.1 <= self.gpu_memory_utilization <= 1.0):
            raise ValueError(
                f'gpu_memory_utilization must be between 0.1 and 1.0, '
                f'got {self.gpu_memory_utilization}. '
                f'Values outside this range may cause OOM errors or underutilization.'
            )

        # Validate KV cache block size for optimal performance
        # 256 is optimal for most modern GPUs (memory alignment, cache efficiency)
        if self.kvcache_block_size % 256 != 0:
            raise ValueError(
                f'kvcache_block_size must be divisible by 256 for optimal performance, '
                f'got {self.kvcache_block_size}. '
                f'Common values: 256, 512, 1024.')

        # Validate tensor parallel size for system compatibility
        # 1-8 is a practical range that balances memory usage and throughput
        if not (1 <= self.tensor_parallel_size <= 8):
            raise ValueError(
                f'tensor_parallel_size must be between 1 and 8, '
                f'got {self.tensor_parallel_size}. '
                f'Higher values may not provide additional benefits and consume more memory.'
            )

        # Validate batch size parameters are positive
        if self.max_num_batched_tokens <= 0:
            raise ValueError(f'max_num_batched_tokens must be positive, '
                             f'got {self.max_num_batched_tokens}')
        if self.max_num_seqs <= 0:
            raise ValueError(f'max_num_seqs must be positive, '
                             f'got {self.max_num_seqs}')

        try:
            # Load HuggingFace model configuration
            # This validates the model format and loads architectural details
            self.hf_config = AutoConfig.from_pretrained(self.model)
        except Exception as e:
            raise ValueError(
                f'Failed to load HuggingFace model configuration from {self.model}: {e}. '
                f'Please ensure the model is a valid HuggingFace model or local directory.'
            )

        # Automatically adjust max_model_len based on model capabilities
        # This prevents requests that exceed the model's maximum context length
        model_max_len = getattr(self.hf_config, 'max_position_embeddings',
                                4096)
        if self.max_model_len > model_max_len:
            # Warn user about automatic adjustment
            import warnings
            warnings.warn(
                f'auto-adjusting max_model_len from {self.max_model_len} '
                f'to {model_max_len} (model\'s maximum context length).',
                UserWarning)
            self.max_model_len = model_max_len

        # Verify batch size is sufficient for model length requirements
        if self.max_num_batched_tokens < self.max_model_len:
            raise ValueError(
                f'max_num_batched_tokens ({self.max_num_batched_tokens}) '
                f'must be >= max_model_len ({self.max_model_len}) '
                f'to accommodate the full context length in a single batch.')
