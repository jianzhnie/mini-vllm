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
        """Validate and initialize configuration after dataclass
        initialization.

        This method:
        - Validates that the model path exists and is a directory
        - Verifies KV cache block size is divisible by 256
        - Validates tensor parallel size is within valid range
        - Loads HuggingFace model configuration
        - Constrains max_model_len based on model's max_position_embeddings
        - Ensures batch token limit meets model requirements

        Raises:
            AssertionError: If any validation check fails.
        """
        # Validate model path exists
        if not os.path.isdir(self.model):
            raise ValueError(
                f"Model path '{self.model}' is not a valid directory")

        # Validate KV cache block size
        if self.kvcache_block_size % 256 != 0:
            raise ValueError(f'kvcache_block_size must be divisible by 256, '
                             f'got {self.kvcache_block_size}')

        # Validate tensor parallel size
        if not (1 <= self.tensor_parallel_size <= 8):
            raise ValueError(f'tensor_parallel_size must be between 1 and 8, '
                             f'got {self.tensor_parallel_size}')

        # Load HuggingFace model configuration
        self.hf_config = AutoConfig.from_pretrained(self.model)

        # Constrain max_model_len to model's maximum
        self.max_model_len = min(self.max_model_len,
                                 self.hf_config.max_position_embeddings)

        # Verify batch size is sufficient for model length
        if self.max_num_batched_tokens < self.max_model_len:
            raise ValueError(
                f'max_num_batched_tokens ({self.max_num_batched_tokens}) '
                f'must be >= max_model_len ({self.max_model_len})')
